from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
from cvzone.PoseModule import PoseDetector
import tempfile
import os

app = Flask(__name__)

def overlay_shirt_on_frame(img, lmList, shirt_img):
    left_shoulder = np.array(lmList[11][1:3])
    right_shoulder = np.array(lmList[12][1:3])
    left_hip = np.array(lmList[23][1:3])
    right_hip = np.array(lmList[24][1:3])

    source_pts = np.float32([
        [0, 0],
        [shirt_img.shape[1], 0],
        [shirt_img.shape[1], shirt_img.shape[0]],
        [0, shirt_img.shape[0]]
    ])

    collar_offset = 30
    target_pts = np.float32([
        [left_shoulder[0], left_shoulder[1] + collar_offset],
        [right_shoulder[0], right_shoulder[1] + collar_offset],
        [right_hip[0], right_hip[1]],
        [left_hip[0], left_hip[1]]
    ])

    matrix = cv2.getPerspectiveTransform(source_pts, target_pts)
    warped_shirt = cv2.warpPerspective(shirt_img, matrix, (img.shape[1], img.shape[0]),
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    img = overlay_transparent(img, warped_shirt)
    return img

def overlay_transparent(background, overlay, alpha_blend=0.7):
    b, g, r, a = cv2.split(overlay)
    green_mask = (g > 150) & (r < 100) & (b < 100)
    a[green_mask] = 0
    alpha = (a / 255.0) * alpha_blend
    for c in range(3):
        background[:, :, c] = (alpha * overlay[:, :, c] + (1 - alpha) * background[:, :, c])
    return background

@app.route('/upload', methods=['POST'])
def upload_video():
    try:
        if 'file' not in request.files or 'shirt' not in request.files:
            return jsonify({"error": "Missing video or shirt file"}), 400

        video_file = request.files['file']
        shirt_file = request.files['shirt']

        if video_file.filename == '' or shirt_file.filename == '':
            return jsonify({"error": "Missing file name"}), 400

        video_bytes = video_file.read()
        shirt_bytes = np.frombuffer(shirt_file.read(), np.uint8)
        shirt_image = cv2.imdecode(shirt_bytes, cv2.IMREAD_UNCHANGED)

        temp_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        with open(temp_video_path, 'wb') as f:
            f.write(video_bytes)

        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            return jsonify({"error": "Could not open video"}), 400

        detector = PoseDetector()
        processed_frames = []

        while True:
            success, frame = cap.read()
            if not success:
                break

            frame = detector.findPose(frame)
            lmList, _ = detector.findPosition(frame, bboxWithHands=False, draw=False)
            if lmList and len(lmList) > 24:
                frame = overlay_shirt_on_frame(frame, lmList, shirt_image)

            processed_frames.append(frame)

        cap.release()

        height, width = processed_frames[0].shape[:2]
        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(temp_output, fourcc, 30.0, (width, height))

        for f in processed_frames:
            out_writer.write(f)
        out_writer.release()

        return send_file(temp_output, mimetype='video/mp4', as_attachment=True, download_name="processed.mp4")

    except Exception as e:
        print("Processing error:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
