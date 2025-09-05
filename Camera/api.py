from flask import Flask, request, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os
from datetime import datetime
import json
import threading
import time

app = Flask(__name__)
model = YOLO('yolov8n.pt')

class ContinuousVideoWriter:
    def __init__(self, output_dir='./recordings', max_file_duration_hours=1):
        self.output_dir = output_dir
        self.writer = None
        self.current_file = None
        self.lock = threading.Lock()
        self.fps = 30
        self.frame_size = (640, 480)
        self.max_file_duration_hours = max_file_duration_hours
        self.file_start_time = None
        self.frame_count = 0
        
        os.makedirs(output_dir, exist_ok=True)
        
    def get_new_filename(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(self.output_dir, f"continuous_feed_{timestamp}.mp4")
    
    def should_rotate_file(self):
        if self.file_start_time is None:
            return True
        
        elapsed_hours = (datetime.now() - self.file_start_time).total_seconds() / 3600
        return elapsed_hours >= self.max_file_duration_hours
    
    def start_new_file(self):
        with self.lock:
            if self.writer:
                print(f"Finished recording: {self.current_file} ({self.frame_count} frames)")
                self.writer.release()
            
            self.current_file = self.get_new_filename()
            self.file_start_time = datetime.now()
            self.frame_count = 0
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(self.current_file, fourcc, self.fps, self.frame_size)
            print(f"Started new recording: {self.current_file}")
    
    def write_frame(self, frame):
        with self.lock:
            if self.writer is None or self.should_rotate_file():
                self.start_new_file()
            
            if frame.shape[:2][::-1] != self.frame_size:
                frame = cv2.resize(frame, self.frame_size)
            
            self.writer.write(frame)
            self.frame_count += 1
    
    def close(self):
        with self.lock:
            if self.writer:
                print(f"Closing recording: {self.current_file} ({self.frame_count} frames)")
                self.writer.release()
                self.writer = None

video_writer = ContinuousVideoWriter()

@app.route('/upload', methods=['POST'])
def upload_video():
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        video_file = request.files['video']
        camera_id = request.headers.get('Camera-ID', 'unknown')
        sequence = request.headers.get('Sequence', '0')
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            video_file.save(temp_file.name)
            
            cap = cv2.VideoCapture(temp_file.name)
            detections = []
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                annotated_frame = frame.copy()
                results = model(frame)
                
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            confidence = float(box.conf[0])
                            class_name = model.names[int(box.cls[0])]
                            
                            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            
                            label = f"{class_name}: {confidence:.2f}"
                            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                            cv2.rectangle(annotated_frame, (int(x1), int(y1) - label_size[1] - 10), 
                                        (int(x1) + label_size[0], int(y1)), (0, 255, 0), -1)
                            cv2.putText(annotated_frame, label, (int(x1), int(y1) - 5), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                            
                            if frame_count % 30 == 0:
                                detection = {
                                    'frame': frame_count,
                                    'timestamp': datetime.now().isoformat(),
                                    'class': class_name,
                                    'confidence': confidence,
                                    'bbox': [x1, y1, x2, y2]
                                }
                                detections.append(detection)
                
                video_writer.write_frame(annotated_frame)
            
            cap.release()
            os.unlink(temp_file.name)
            
            response = {
                'camera_id': camera_id,
                'sequence': sequence,
                'processed_frames': frame_count,
                'detections': detections,
                'detection_count': len(detections)
            }
            
            return jsonify(response)
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'model_loaded': True})

@app.route('/recording/status', methods=['GET'])
def recording_status():
    with video_writer.lock:
        return jsonify({
            'current_file': video_writer.current_file,
            'frame_count': video_writer.frame_count,
            'start_time': video_writer.file_start_time.isoformat() if video_writer.file_start_time else None,
            'is_recording': video_writer.writer is not None
        })

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=True)
    finally:
        video_writer.close()

