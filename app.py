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

# Configure temporary directories
UPLOAD_FOLDER = tempfile.mkdtemp()
PROCESSED_FOLDER = tempfile.mkdtemp()
SHIRT_FOLDER = "Resources/Shirts"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['SHIRT_FOLDER'] = SHIRT_FOLDER

os.makedirs(SHIRT_FOLDER, exist_ok=True)

class VideoMetadata(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(100), nullable=False)
    processed_filename = db.Column(db.String(100), nullable=True)
    status = db.Column(db.String(20), default='Processing')
    download_url = db.Column(db.String(200), nullable=True)

    def __repr__(self):
        return f"<VideoMetadata {self.filename}>"

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

        shirt_index = int(request.form.get('shirt_index', 0))
        processed_filename = process_video(filepath, filename, shirt_index)
        processed_url = url_for('download_processed', filename=processed_filename)

        return jsonify({
            "message": "Video processing complete! Click the link below to download.",
            "download_url": processed_url
        })
    
    except Exception as e:
        print("Error in /upload route:", e)
        return jsonify({"error": str(e)}), 500

def process_video(input_path, filename, shirt_index):
    detector = PoseDetector()
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file")

    listShirts = get_shirt_list()
    if not listShirts:
        raise ValueError("No shirts available in the Resources/Shirts directory")
    if shirt_index < 0 or shirt_index >= len(listShirts):
        raise ValueError("Invalid shirt index")

    shirt_filename = listShirts[shirt_index]
    shirt_path = os.path.join(app.config['SHIRT_FOLDER'], shirt_filename)
    imgShirt = cv2.imread(shirt_path, cv2.IMREAD_UNCHANGED)
    if imgShirt is None:
        raise ValueError(f"Could not load shirt image: {shirt_filename}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    processed_filename = f"processed_{timestamp}_{filename}"
    processed_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(processed_path, fourcc, 30.0, (1280, 720))

    while True:
        success, img = cap.read()
        if not success:
            break

        img = detector.findPose(img)
        lmList, _ = detector.findPosition(img, bboxWithHands=False, draw=False)

        if lmList and len(lmList) > 24:
            img = overlay_shirt_on_frame(img, lmList, shirt_filename)
        
        out.write(img)

    cap.release()
    out.release()
    return processed_filename

def overlay_shirt_on_frame(img, lmList, shirt_filename):
    shirt_path = os.path.join(app.config['SHIRT_FOLDER'], shirt_filename)
    imgShirt = cv2.imread(shirt_path, cv2.IMREAD_UNCHANGED)
    if imgShirt is None:
        return img

    left_shoulder = np.array(lmList[11][1:3])
    right_shoulder = np.array(lmList[12][1:3])
    left_hip = np.array(lmList[23][1:3])
    right_hip = np.array(lmList[24][1:3])

    height, width = imgShirt.shape[:2]
    source_pts = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

    center_x = (left_shoulder[0] + right_shoulder[0] + left_hip[0] + right_hip[0]) / 4
    center_y = (left_shoulder[1] + right_shoulder[1] + left_hip[1] + right_hip[1]) / 4
    scaling_factor = 1.5
    shoulder_width = abs(left_shoulder[0] - right_shoulder[0]) * scaling_factor
    hip_height = abs(left_hip[1] - left_shoulder[1]) * scaling_factor
    collar_offset = 30

    target_pts = np.float32([
        [left_shoulder[0], left_shoulder[1] + collar_offset],
        [right_shoulder[0], right_shoulder[1] + collar_offset],
        [right_hip[0], right_hip[1]],
        [left_hip[0], left_hip[1]]
    ])

    matrix = cv2.getPerspectiveTransform(source_pts, target_pts)
    warped_shirt = cv2.warpPerspective(imgShirt, matrix, (img.shape[1], img.shape[0]),
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    return overlay_transparent(img, warped_shirt)

def overlay_transparent(background, overlay, alpha_blend=0.7):
    if overlay.shape[2] != 4:
        raise ValueError("Shirt image must have an alpha channel (PNG format)")

    b, g, r, a = cv2.split(overlay)
    green_mask = (g > 150) & (r < 100) & (b < 100)
    a[green_mask] = 0
    alpha = (a / 255.0) * alpha_blend

    for c in range(3):
        background[:, :, c] = (alpha * overlay[:, :, c] + (1 - alpha) * background[:, :, c])

    return background

@app.route('/processed/<filename>')
def download_processed(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
