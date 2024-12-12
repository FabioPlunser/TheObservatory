import cv2
import uuid
import aiohttp
import logging
import socket 
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
        self.reconnect_delay = 5 
        self.frame_rate = 30
        self.frame_interval = 1.0
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
                    await response.json()
                    logger.info(f"Camera {self.camera_id} registered successfully")
                    return True

        except Exception as e: 
            logger.error(f"Registration failed: {e}")
            return False

    async def start_streaming(self, video_path=None): 
        """Start camera and stream to edge server"""
        if not self.edge_server_url:
            logger.error("No edge server URL available")
            return

        try: 
            if video_path: 
                self.cap = cv2.VideoCapture(video_path)
            else:
                self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened(): 
                raise RuntimeError("Could not open camera")
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.frame_rate)

            ws_url = self.edge_server_url.replace("http://", "ws://") + f"/ws/camera/{self.camera_id}"
            logger.info(f"Connecting to WebSocket at {ws_url}")
            
            async with websockets.connect(ws_url) as websocket: 
                logger.info("WebSocket connection established")
                self.is_running = True 

                while self.is_running: 
                    ret, frame = self.cap.read() 
                    if not ret: 
                        continue 
                
                    _, buffer = cv2.imencode(".jpg", frame)  
                    frame_data = base64.b64encode(buffer).decode("utf-8")

                    try: 
                        await websocket.send(json.dumps({
                            "camera_id": self.camera_id, 
                            "timestamp": datetime.now().isoformat(), 
                            "frame": frame_data
                        }))
                    except Exception as e: 
                        logger.error(f"Failed to send frame: {e}")
                        break

                    await asyncio.sleep(1/self.frame_rate)
        except Exception as e: 
            logger.error(f"Streaming error: {e}")
        finally: 
            if self.cap:
                self.cap.release()
    
    async def stop(self): 
        """Stop the camera"""
        self.is_running = False 
        if self.cap: 
            self.cap.release() 

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