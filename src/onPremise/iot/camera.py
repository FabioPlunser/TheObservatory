import cv2
import uuid
import requests
import logging
import threading
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Camera:
    def __init__(self, camera_index=None, video_path=None, edge_server_url="http://localhost:8080"):
        self.camera_id = str(uuid.uuid4())
        self.edge_server_url = edge_server_url
        self.camera_index = camera_index
        self.video_path = video_path
        self.cap = None
        self.is_running = False
        self.stream_thread = None

    def start(self):
        try:
            if self.video_path:
                self.cap = cv2.VideoCapture(self.video_path)
                if not self.cap.isOpened():
                    raise RuntimeError(f"Could not open video file {self.video_path}")
            else:
                self.cap = cv2.VideoCapture(self.camera_index)
                if not self.cap.isOpened():
                    raise RuntimeError(f"Could not open camera {self.camera_index}")

                # Set camera properties
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 30)

            # Register with edge server
            self.register()

            # Start streaming thread
            self.is_running = True
            self.stream_thread = threading.Thread(target=self.stream)
            self.stream_thread.start()
            
            logger.info(f"Camera {self.camera_id} started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start camera: {e}")
            if self.cap:
                self.cap.release()

    def register(self):
        data = {
            "camera_id": self.camera_id,
            "capabilities": {
                "resolution": "640x480",
                "fps": 30
            }
        }
        
        try:
            response = requests.post(
                f"{self.edge_server_url}/register",
                json=data,
                timeout=5
            )
            response.raise_for_status()
            logger.info(f"Camera {self.camera_id} registered successfully")
        except requests.RequestException as e:
            logger.error(f"Registration failed: {e}")
            raise

    def stream(self):
        while self.is_running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("Failed to read frame")
                    continue

                # Encode frame to JPEG
                _, jpeg = cv2.imencode('.jpg', frame)
                
                # Send frame to edge server
                response = requests.post(
                    f"{self.edge_server_url}/frame",
                    files={'frame': ('frame.jpg', jpeg.tobytes(), 'image/jpeg')},
                    data={'camera_id': self.camera_id},
                    timeout=3
                )
                response.raise_for_status()

                # Add a small delay to control frame rate
                time.sleep(0.033)  # ~30 FPS

            except Exception as e:
                logger.error(f"Streaming error: {e}")
                continue

    def stop(self):
        self.is_running = False
        if self.stream_thread:
            self.stream_thread.join()
        if self.cap:
            self.cap.release()
        logger.info(f"Camera {self.camera_id} stopped")

def main():
    cameras = []
    try:
        # Try to start multiple cameras (adjust based on available cameras)
        for i in range(2):  # Try 2 cameras
            camera = Camera(camera_index=i)
            camera.start()
            cameras.append(camera)
            logger.info(f"Started camera {i}")
            time.sleep(1)  # Wait between camera starts
        
        # Keep main thread running
        input("Press Enter to stop streaming...")
    
    except KeyboardInterrupt:
        logger.info("Stopping cameras...")
    finally:
        for camera in cameras:
            camera.stop()

if __name__ == "__main__":
    main()