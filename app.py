from flask import Flask, request, render_template, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
from cvzone.PoseModule import PoseDetector
import cv2
import numpy as np
import os
import io
import logging
from datetime import datetime
import tempfile

# Initialize Flask app
app = Flask(__name__)
app.logger.setLevel(logging.INFO)

# Configure PostgreSQL database
db_uri = os.environ.get(
    'DATABASE_URL', 
    'postgresql://conference_db_7rej_user:SfCIq9wras1ApfLgGrQcayRy5igtvG7R@dpg-d03frmadbo4c738bml5g-a.virginia-postgres.render.com/conference_db_7rej'
)
app.config['SQLALCHEMY_DATABASE_URI'] = db_uri
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Define database model
class VideoMetadata(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    original_filename = db.Column(db.String(255), nullable=False)
    processed_data = db.Column(db.LargeBinary, nullable=True)
    shirt_used = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    status = db.Column(db.String(20), default='processing')

# Constants
SHIRT_FOLDER = os.path.join(os.path.dirname(__file__), 'Resources', 'Shirts')
app.config['SHIRT_FOLDER'] = SHIRT_FOLDER
os.makedirs(SHIRT_FOLDER, exist_ok=True)

@app.route('/')
def index():
    shirts = get_available_shirts()
    videos = VideoMetadata.query.order_by(VideoMetadata.created_at.desc()).all()
    return render_template('index.html', shirts=shirts, videos=videos)

def get_available_shirts():
    try:
        return [f for f in os.listdir(SHIRT_FOLDER) if f.endswith(('.png', '.jpg', '.jpeg'))]
    except FileNotFoundError:
        app.logger.error("Shirt directory not found at %s", SHIRT_FOLDER)
        return []

@app.route('/upload', methods=['POST'])
def handle_upload():
    try:
        # Validate inputs
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400

        shirt_index = int(request.form.get('shirt_index', 0))
        shirts = get_available_shirts()
        
        if not shirts:
            return jsonify({'error': 'No shirts available'}), 400
        if shirt_index < 0 or shirt_index >= len(shirts):
            return jsonify({'error': 'Invalid shirt selection'}), 400

        # Save original file to temporary storage
        temp_video = tempfile.NamedTemporaryFile(delete=False)
        file.save(temp_video.name)
        
        # Process video
        shirt_path = os.path.join(SHIRT_FOLDER, shirts[shirt_index])
        processed_video = process_video(temp_video.name, shirt_path)
        
        # Store in database
        video_entry = VideoMetadata(
            original_filename=file.filename,
            processed_data=processed_video,
            shirt_used=shirts[shirt_index],
            status='completed'
        )
        db.session.add(video_entry)
        db.session.commit()

        # Cleanup
        os.unlink(temp_video.name)

        return jsonify({
            'message': 'Video processed successfully',
            'download_url': url_for('download_video', video_id=video_entry.id)
        })

    except Exception as e:
        app.logger.error("Processing error: %s", str(e))
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

def process_video(input_path, shirt_path):
    detector = PoseDetector()
    cap = cv2.VideoCapture(input_path)
    shirt_img = cv2.imread(shirt_path, cv2.IMREAD_UNCHANGED)
    
    if shirt_img is None:
        raise ValueError(f"Could not load shirt image: {shirt_path}")

    # Video writer setup
    output = io.BytesIO()
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        temp_output.name, 
        fourcc, 
        30.0, 
        (int(cap.get(3)), int(cap.get(4)))
    
    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame = detector.findPose(frame)
            lmList, _ = detector.findPosition(frame, bboxWithHands=False, draw=False)

            if lmList and len(lmList) > 24:
                frame = apply_shirt(frame, lmList, shirt_img)

            out.write(frame)

        out.release()
        cap.release()

        # Read processed video into memory
        with open(temp_output.name, 'rb') as f:
            output.write(f.read())
            
        return output.getvalue()

    finally:
        if os.path.exists(temp_output.name):
            os.unlink(temp_output.name)

def apply_shirt(frame, landmarks, shirt_img):
    # Shoulder and hip landmarks
    ls, rs = landmarks[11][1:3], landmarks[12][1:3]
    lh, rh = landmarks[23][1:3], landmarks[24][1:3]

    # Calculate shirt position
    shirt_height = int(np.linalg.norm(np.array(ls) - np.array(lh)))
    shirt_width = int(np.linalg.norm(np.array(ls) - np.array(rs)))
    
    # Resize shirt
    resized_shirt = cv2.resize(shirt_img, (shirt_width, shirt_height))
    
    # Position calculation
    y_offset = int(ls[1] - shirt_height * 0.2)
    x_offset = int(ls[0] - shirt_width * 0.1)

    # Overlay shirt
    alpha = resized_shirt[:, :, 3] / 255.0
    for c in range(3):
        frame[y_offset:y_offset+shirt_height, x_offset:x_offset+shirt_width, c] = \
            alpha * resized_shirt[:, :, c] + \
            (1 - alpha) * frame[y_offset:y_offset+shirt_height, x_offset:x_offset+shirt_width, c]

    return frame

@app.route('/download/<int:video_id>')
def download_video(video_id):
    video = VideoMetadata.query.get_or_404(video_id)
    return send_file(
        io.BytesIO(video.processed_data),
        mimetype='video/mp4',
        as_attachment=True,
        download_name=f"processed_{video.original_filename}"
    )

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
