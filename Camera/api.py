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

class ImageSaver:
    def __init__(self, output_dir='./images'):
        self.output_dir = output_dir
        self.lock = threading.Lock()
        
        os.makedirs(output_dir, exist_ok=True)
    
    def save_annotated_image(self, image, camera_id, sequence, detections):
        with self.lock:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # microseconds to milliseconds
            filename = f"{camera_id}_{sequence:06d}_{timestamp}.jpg"
            filepath = os.path.join(self.output_dir, filename)
            
            annotated_image = image.copy()
            
            for detection in detections:
                x1, y1, x2, y2 = detection['bbox']
                confidence = detection['confidence']
                class_name = detection['class']
                
                cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                label = f"{class_name}: {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(annotated_image, (int(x1), int(y1) - label_size[1] - 10), 
                            (int(x1) + label_size[0], int(y1)), (0, 255, 0), -1)
                cv2.putText(annotated_image, label, (int(x1), int(y1) - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            success = cv2.imwrite(filepath, annotated_image)
            if success:
                print(f"Saved annotated image: {filename}")
                return filepath
            else:
                print(f"Failed to save image: {filename}")
                return None

image_saver = ImageSaver()

@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        image_file = request.files['image']
        camera_id = request.headers.get('Camera-ID', 'unknown')
        sequence = request.headers.get('Sequence', '0')
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            image_file.save(temp_file.name)
            
            image = cv2.imread(temp_file.name)
            if image is None:
                os.unlink(temp_file.name)
                return jsonify({'error': 'Invalid image file'}), 400
            
            detections = []
            results = model(image)
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        confidence = float(box.conf[0])
                        class_name = model.names[int(box.cls[0])]
                        
                        detection = {
                            'timestamp': datetime.now().isoformat(),
                            'class': class_name,
                            'confidence': confidence,
                            'bbox': [x1, y1, x2, y2]
                        }
                        detections.append(detection)
            
            saved_path = image_saver.save_annotated_image(image, camera_id, int(sequence), detections)
            
            os.unlink(temp_file.name)
            
            response = {
                'camera_id': camera_id,
                'sequence': sequence,
                'detections': detections,
                'detection_count': len(detections),
                'saved_image': saved_path
            }
            
            return jsonify(response)
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'model_loaded': True})

@app.route('/images/status', methods=['GET'])
def images_status():
    with image_saver.lock:
        image_count = len([f for f in os.listdir(image_saver.output_dir) if f.endswith('.jpg')])
        return jsonify({
            'output_directory': image_saver.output_dir,
            'total_images_saved': image_count,
            'is_active': True
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

