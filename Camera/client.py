import requests
import subprocess
import time
import os
import sys
from datetime import datetime

class PiCameraClient:
    def __init__(self, server_url='http://localhost:5000', camera_id='pi_zero_01'):
        self.server_url = server_url
        self.camera_id = camera_id
        self.upload_endpoint = f"{server_url}/upload"
        self.health_endpoint = f"{server_url}/health"
        
    def check_server_health(self):
        try:
            response = requests.get(self.health_endpoint, timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def capture_image(self, output_path='/tmp/image_capture.jpg'):
        cmd = [
            'libcamera-still',
            '-o', output_path,
            '--width', '640',
            '--height', '480',
            '--quality', '85',
            '--timeout', '1'
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                return output_path
            else:
                print(f"Error capturing image: {result.stderr}")
                return None
        except Exception as e:
            print(f"Exception during image capture: {e}")
            return None
    
    def upload_image(self, image_path, sequence_num):
        if not os.path.exists(image_path):
            print(f"Image file not found: {image_path}")
            return None
            
        headers = {
            'Camera-ID': self.camera_id,
            'Sequence': str(sequence_num)
        }
        
        try:
            with open(image_path, 'rb') as image_file:
                files = {'image': image_file}
                response = requests.post(
                    self.upload_endpoint,
                    files=files,
                    headers=headers,
                    timeout=30
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    print(f"Upload failed with status {response.status_code}: {response.text}")
                    return None
                    
        except Exception as e:
            print(f"Exception during upload: {e}")
            return None
    
    def cleanup_image_file(self, image_path):
        try:
            if os.path.exists(image_path):
                os.remove(image_path)
        except Exception as e:
            print(f"Error cleaning up image file: {e}")
    
    def run_continuous_capture(self, fps=2):
        sequence = 0
        capture_interval = 1.0 / fps  # 0.5 seconds for 2 fps
        
        if not self.check_server_health():
            print("Server health check failed. Please ensure the server is running.")
            return
            
        print(f"Starting continuous image capture (Camera ID: {self.camera_id})")
        print(f"Capture rate: {fps} fps, Server: {self.server_url}")
        
        try:
            while True:
                sequence += 1
                timestamp = datetime.now().isoformat()
                image_path = f'/tmp/image_capture_{sequence}.jpg'
                
                print(f"[{timestamp}] Capturing image #{sequence}...")
                
                captured_file = self.capture_image(image_path)
                if captured_file:
                    print(f"[{timestamp}] Uploading image #{sequence}...")
                    
                    result = self.upload_image(captured_file, sequence)
                    if result:
                        detection_count = result.get('detection_count', 0)
                        print(f"[{timestamp}] Image #{sequence} processed: {detection_count} detections")
                        
                        if detection_count > 0:
                            for detection in result.get('detections', []):
                                print(f"  - {detection['class']} (confidence: {detection['confidence']:.2f})")
                    else:
                        print(f"[{timestamp}] Upload failed for image #{sequence}")
                    
                    self.cleanup_image_file(captured_file)
                else:
                    print(f"[{timestamp}] Failed to capture image #{sequence}")
                
                time.sleep(capture_interval)
                
        except KeyboardInterrupt:
            print("\nStopping capture...")
        except Exception as e:
            print(f"Error in continuous capture: {e}")

def main():
    if len(sys.argv) > 1:
        server_url = sys.argv[1]
    else:
        server_url = 'http://localhost:5000'
    
    camera_id = os.environ.get('CAMERA_ID', 'pi_zero_01')
    
    client = PiCameraClient(server_url, camera_id)
    client.run_continuous_capture()

if __name__ == '__main__':
    main()