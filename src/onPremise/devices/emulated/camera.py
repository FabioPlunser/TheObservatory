import cv2
import uuid
import aiohttp
import logging
import asyncio
import websockets
import base64
import json
from datetime import datetime
from edge_server_discover import EdgeServerDiscovery

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Camera:
    def __init__(self):
        self.camera_id = str(uuid.uuid4())
        self.edge_server_url = None
        self.cap = None
        self.is_running = False
        self.discovery = EdgeServerDiscovery()
        self.frame_rate = 30
        self.frame_width = 1920
        self.frame_height = 1080

    async def discover_edge_server(self):
        """Discover edge server"""
        logger.info("Starting edge server discovery")
        self.edge_server_url = await self.discovery.discover_edge_server()

        if not self.edge_server_url:
            logger.error("Failed to discover edge server")
            return False
        logger.info(f"Successfully discovered edge server at {self.edge_server_url}")
        return True

    async def register_with_edge(self):
        """Register camera with edge server"""
        if not self.edge_server_url:
            logger.error("No edge server URL available")
            return False

        try:
            registration_data = {
                "camera_id": self.camera_id,
                "name": f"Camera {self.camera_id[:8]}",
                "capabilities": {
                    "resolution": f"{self.frame_width}x{self.frame_height}",
                    "fps": self.frame_rate,
                    "night_vision": False
                }
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.edge_server_url}/register/camera",
                    json=registration_data,
                    timeout=5
                ) as response:
                    if response.status == 200:
                        await response.json()
                        logger.info(f"Camera {self.camera_id} registered successfully")
                        return True
                    else:
                        logger.error(f"Registration failed with status {response.status}")
                        return False

        except Exception as e:
            logger.error(f"Registration failed: {e}")
            return False

    async def start_streaming(self, video_paths=None, stop_event=None):
        """Start camera and stream to edge server"""
        if not self.edge_server_url:
            logger.error("No edge server URL available")
            return
        
        def configure_capture(video_source):
            self.cap = cv2.VideoCapture(video_source)
            if not self.cap.isOpened():
                raise RuntimeError(f"Could not open video source {video_source}")

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.frame_rate)

            actual_frame_rate = self.cap.get(cv2.CAP_PROP_FPS)
            frame_skip = max(0, int(actual_frame_rate // self.frame_rate) - 1)
            effective_frame_rate = actual_frame_rate / (frame_skip + 1)
            return frame_skip, effective_frame_rate

        try:
            ws_url = self.edge_server_url.replace("http://", "ws://") + f"/ws/camera/{self.camera_id}"
            logger.info(f"Connecting to WebSocket at {ws_url}")

            async with websockets.connect(ws_url, ping_interval=20, ping_timeout=20) as websocket:
                logger.info("WebSocket connection established")
                self.is_running = True

                if video_paths is None:
                    frame_skip, effective_frame_rate = configure_capture(0)
                    await self.stream_frames(stop_event, websocket, frame_skip, effective_frame_rate)
                else:
                    for video_source in video_paths:
                        frame_skip, effective_frame_rate = configure_capture(video_source)
                        await self.stream_frames(stop_event, websocket, frame_skip, effective_frame_rate, video_source)

        except Exception as e:
            logger.error(f"Streaming error: {e}")
        finally:
            if self.cap:
                self.cap.release()
            self.is_running = False
            logger.info(f"WebSocket connection closed for camera {self.camera_id}")

    async def stream_frames(self, stop_event, websocket, frame_skip, effective_frame_rate, video_source=0):
        while self.is_running and (stop_event is None or not stop_event.is_set()):
            try:
                for _ in range(frame_skip):
                    self.cap.grab()

                ret, frame = self.cap.read()
                if not ret:
                    logger.error("Failed to capture frame. Reinitializing camera...")
                    self.cap.release()
                    self.cap = cv2.VideoCapture(video_source)  # Reinitialize camera
                    if not self.cap.isOpened():
                        logger.error("Camera could not be reopened")
                        break
                    continue

                # Resize frame to reduce data size
                frame = cv2.resize(frame, (self.frame_width // 2, self.frame_height // 2))
                # Reduce quality to 50%
                _, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])  
                
                frame_data = base64.b64encode(buffer).decode("utf-8")

                try:
                    await websocket.send(json.dumps({
                        "camera_id": self.camera_id,
                        "timestamp": datetime.now().isoformat(),
                        "frame": frame_data
                    }))
                except Exception as e:
                    logger.error(f"Failed to send frame: {e}")
                    continue  # Skip this frame and try the next one
                
                await asyncio.sleep(1 / effective_frame_rate)
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                break

            await asyncio.sleep(1 / effective_frame_rate)

    async def stop(self):
        """Stop the camera"""
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None

async def main():
    camera = Camera()
    try:
        if not await camera.discover_edge_server():
            logger.error("No edge server found")
            return

        if await camera.register_with_edge():
            await camera.start_streaming()
    except KeyboardInterrupt:
        logger.info("Shutting down camera...")
    except Exception as e:
        logger.error(f"Camera error: {e}")
    finally:
        await camera.stop()

if __name__ == "__main__":
    asyncio.run(main())