from flask import Flask, request, jsonify, render_template, send_file
import cv2
import numpy as np
from cvzone.PoseModule import PoseDetector
import io
from datetime import datetime

app = Flask(__name__)

# In-memory video processing (no file storage)
@app.route('/')
def index():
    """Render the homepage."""
    return render_template('index.html')  # This assumes a basic HTML frontend exists for uploading.

@app.route('/upload', methods=['POST'])
def upload_video():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        # Read video from the uploaded file (in memory)
        file_bytes = file.read()
        video_stream = io.BytesIO(file_bytes)
        cap = cv2.VideoCapture(video_stream)

        if not cap.isOpened():
            return jsonify({"error": "Unable to open video"}), 400

        # Process the video (e.g., apply PoseDetector)
        detector = PoseDetector()
        processed_video_frames = []

        while True:
            success, frame = cap.read()
            if not success:
                break

            # Detect pose and overlay shirt (you can add your shirt overlay code here)
            frame = detector.findPose(frame)
            lmList, _ = detector.findPosition(frame, bboxWithHands=False, draw=False)

            # If detected pose keypoints, overlay the shirt (or do further processing)
            if lmList and len(lmList) > 24:
                frame = overlay_shirt_on_frame(frame, lmList)  # Assuming overlay_shirt_on_frame is defined as before.

            processed_video_frames.append(frame)

        cap.release()

        # Encode frames into a video (in memory)
        output_video_stream = io.BytesIO()
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(output_video_stream, fourcc, 30.0, (1280, 720))

        for frame in processed_video_frames:
            out_writer.write(frame)

        out_writer.release()

        # Seek to the beginning of the in-memory video stream for download
        output_video_stream.seek(0)

        return send_file(output_video_stream, mimetype='video/mp4', as_attachment=True, download_name="processed_video.mp4")

    except Exception as e:
        print("Error in /upload route:", e)
        return jsonify({"error": str(e)}), 500

def overlay_shirt_on_frame(frame, lmList):
    """Overlay shirt on the frame based on detected pose landmarks."""
    # Example of overlay logic (your overlay logic should be here)
    left_shoulder = np.array(lmList[11][1:3])
    right_shoulder = np.array(lmList[12][1:3])

    # Load the shirt image and overlay it
    # You can modify this based on your actual shirt overlay logic
    shirt = cv2.imread('shirt_image.png', cv2.IMREAD_UNCHANGED)
    shirt_resized = cv2.resize(shirt, (100, 100))  # Resize for the demonstration

    # Assuming you're positioning the shirt based on the shoulders
    shirt_position = (int((left_shoulder[0] + right_shoulder[0]) / 2), int((left_shoulder[1] + right_shoulder[1]) / 2))

    # Overlay shirt on frame logic
    x, y = shirt_position
    h, w, _ = shirt_resized.shape
    frame[y:y+h, x:x+w] = cv2.addWeighted(frame[y:y+h, x:x+w], 0.7, shirt_resized, 0.3, 0)

    return frame

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
