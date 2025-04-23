from flask import Flask, request, render_template, jsonify, redirect, url_for, send_from_directory
from flask_sqlalchemy import SQLAlchemy
import os
import cv2
import cvzone
from cvzone.PoseModule import PoseDetector
from datetime import datetime
import numpy as np
import tempfile

app = Flask(__name__)

# Database setup
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///video_metadata.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Temporary folders
UPLOAD_FOLDER = tempfile.mkdtemp()
PROCESSED_FOLDER = os.path.join('static', 'processed')  # make this publicly accessible
SHIRT_FOLDER = "Resources/Shirts"

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['SHIRT_FOLDER'] = SHIRT_FOLDER

# Ensure folders exist
os.makedirs(SHIRT_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# VideoMetadata model
class VideoMetadata(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(100), nullable=False)
    processed_filename = db.Column(db.String(100), nullable=True)
    status = db.Column(db.String(20), default='Processing')
    download_url = db.Column(db.String(200), nullable=True)

    def __repr__(self):
        return f"<VideoMetadata {self.filename}>"

# Initialize DB
with app.app_context():
    db.create_all()

def get_shirt_list():
    return os.listdir(app.config['SHIRT_FOLDER'])

@app.route('/')
def index():
    listShirts = get_shirt_list()
    videos = VideoMetadata.query.all()
    return render_template('index.html', shirts=listShirts, videos=videos)

@app.route('/upload_shirt', methods=['POST'])
def upload_shirt():
    if 'shirt_image' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['shirt_image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        filepath = os.path.join(app.config['SHIRT_FOLDER'], file.filename)
        file.save(filepath)
        return redirect(url_for('index'))

@app.route('/upload', methods=['POST'])
def upload_video():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Save metadata
        video_metadata = VideoMetadata(filename=filename, status='Processing')
        db.session.add(video_metadata)
        db.session.commit()

        shirt_index = int(request.form.get('shirt_index', 0))
        processed_filename = process_video(filepath, filename, shirt_index)

        # Update metadata
        video_metadata.status = 'Completed'
        video_metadata.processed_filename = processed_filename
        video_metadata.download_url = url_for('serve_processed_video', filename=processed_filename)
        db.session.commit()

        return jsonify({
            "message": "Video processing complete!",
            "download_url": video_metadata.download_url
        })

    except Exception as e:
        print("Error in /upload route:", e)
        return jsonify({"error": str(e)}), 500

def process_video(input_path, filename, shirt_index):
    detector = PoseDetector()
    cap = cv2.VideoCapture(input_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    processed_filename = f"processed_{timestamp}_{filename}"
    processed_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(processed_path, fourcc, 30.0, (1280, 720))

    listShirts = get_shirt_list()
    while True:
        success, img = cap.read()
        if not success:
            break

        img = detector.findPose(img)
        lmList, bboxInfo = detector.findPosition(img, bboxWithHands=False, draw=False)

        if lmList and len(lmList) > 24:
            img = overlay_shirt_on_frame(img, lmList, listShirts[shirt_index])

        out_writer.write(img)

    cap.release()
    out_writer.release()
    return processed_filename

def overlay_shirt_on_frame(img, lmList, shirt_filename):
    left_shoulder = np.array(lmList[11][1:3])
    right_shoulder = np.array(lmList[12][1:3])
    left_hip = np.array(lmList[23][1:3])
    right_hip = np.array(lmList[24][1:3])

    imgShirt = cv2.imread(os.path.join(app.config['SHIRT_FOLDER'], shirt_filename), cv2.IMREAD_UNCHANGED)
    height, width = imgShirt.shape[:2]

    source_pts = np.float32([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]
    ])

    collar_offset = 30
    target_pts = np.float32([
        [left_shoulder[0], left_shoulder[1] + collar_offset],
        [right_shoulder[0], right_shoulder[1] + collar_offset],
        [right_hip[0], right_hip[1]],
        [left_hip[0], left_hip[1]]
    ])

    matrix = cv2.getPerspectiveTransform(source_pts, target_pts)
    warped_shirt = cv2.warpPerspective(imgShirt, matrix, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    return overlay_transparent(img, warped_shirt)

def overlay_transparent(background, overlay, alpha_blend=0.7):
    b, g, r, a = cv2.split(overlay)
    green_mask = (g > 150) & (r < 100) & (b < 100)
    a[green_mask] = 0
    alpha = (a / 255.0) * alpha_blend

    for c in range(3):
        background[:, :, c] = (alpha * overlay[:, :, c] + (1 - alpha) * background[:, :, c])

    return background

@app.route('/processed/<filename>')
def serve_processed_video(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
