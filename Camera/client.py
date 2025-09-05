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
    
    def capture_video_chunk(self, duration=5, output_path='/tmp/video_chunk.mp4'):
        cmd = [
            'libcamera-vid',
            '-t', str(duration * 1000),  # duration in milliseconds
            '-o', output_path,
            '--width', '640',
            '--height', '480',
            '--framerate', '30',
            '--codec', 'h264'
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                return output_path
            else:
                print(f"Error capturing video: {result.stderr}")
                return None
        except Exception as e:
            print(f"Exception during video capture: {e}")
            return None
    
    def upload_video(self, video_path, sequence_num):
        if not os.path.exists(video_path):
            print(f"Video file not found: {video_path}")
            return None
            
        headers = {
            'Camera-ID': self.camera_id,
            'Sequence': str(sequence_num)
        }
        
        try:
            with open(video_path, 'rb') as video_file:
                files = {'video': video_file}
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
    
    def cleanup_video_file(self, video_path):
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
        except Exception as e:
            print(f"Error cleaning up video file: {e}")
    
    def run_continuous_capture(self, chunk_duration=5, delay_between_chunks=1):
        sequence = 0
        
        if not self.check_server_health():
            print("Server health check failed. Please ensure the server is running.")
            return
            
        print(f"Starting continuous video capture (Camera ID: {self.camera_id})")
        print(f"Chunk duration: {chunk_duration}s, Server: {self.server_url}")
        
        try:
            while True:
                sequence += 1
                timestamp = datetime.now().isoformat()
                video_path = f'/tmp/video_chunk_{sequence}.mp4'
                
                print(f"[{timestamp}] Capturing chunk #{sequence}...")
                
                captured_file = self.capture_video_chunk(chunk_duration, video_path)
                if captured_file:
                    print(f"[{timestamp}] Uploading chunk #{sequence}...")
                    
                    result = self.upload_video(captured_file, sequence)
                    if result:
                        detection_count = result.get('detection_count', 0)
                        print(f"[{timestamp}] Chunk #{sequence} processed: {detection_count} detections")
                        
                        if detection_count > 0:
                            for detection in result.get('detections', []):
                                print(f"  - {detection['class']} (confidence: {detection['confidence']:.2f})")
                    else:
                        print(f"[{timestamp}] Upload failed for chunk #{sequence}")
                    
                    self.cleanup_video_file(captured_file)
                else:
                    print(f"[{timestamp}] Failed to capture chunk #{sequence}")
                
                time.sleep(delay_between_chunks)
                
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