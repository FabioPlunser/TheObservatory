import base64
import socket 
import numpy as np 
import cv2 
import json
import asyncio
import torch 
import platform
import os
from fastapi import FastAPI, WebSocket, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from typing import Dict
from zeroconf import ServiceInfo, Zeroconf
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from database import Database
from ultralytics import YOLO
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def get_device():
    """Determine the best available device for inference"""
    if torch.cuda.is_available():
        return 'cuda'
    elif platform.processor() == 'arm' and platform.system() == 'Darwin':  # Check for M1/M2
        try:
            if torch.backends.mps.is_available():
                if not torch.backends.mps.is_built():
                    logger.warning("MPS not built, falling back to CPU")
                    return 'cpu'
                logger.info("Using M1/M2 Metal GPU")
                return 'mps'
        except AttributeError:
            logger.warning("torch.backends.mps not available, falling back to CPU")
            return 'cpu'
    return 'cpu'
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------

class EdgeServer:
    def __init__(self):
        self.cameras: Dict[str, dict] = {}
        self.alarms: Dict[str, dict] = {}
        self.active_connections: Dict[str, set] = {}
        self.viewers = {}
        self.zeroconf = None
        self.service_info = None
        self.port = 8000
        self.executor = ThreadPoolExecutor(max_workers=1)
        
        # ---------------------------------------------------------------------------- 
        # Initialize YOLO model 
        self.device = get_device()
        logger.info(f"Using device: {self.device} for YOLO inference")
        
        try:
            self.model = YOLO('yolov8n.pt')
            self.model.to(self.device)
                
            logger.info("YOLO model initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing YOLO model: {e}")
            logger.warning("Falling back to CPU")
            self.device = 'cpu'
            self.model = YOLO('yolov8n.pt')
    #----------------------------------------------------------------------------
    #----------------------------------------------------------------------------
    async def start_mdns(self): 
        """Start mDNS service advertising asynchronously"""
        def register_service():
            self.zeroconf = Zeroconf()
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)

            # Fixed service info configuration
            service_name = "EdgeServer"
            service_type = "_edgeserver._tcp.local."
            desc = {'version': '1.0', 'server': 'edge'}

            self.service_info = ServiceInfo(
                type_=service_type,
                name=f"{service_name}.{service_type}",
                addresses=[socket.inet_aton(local_ip)],
                port=self.port,
                properties=desc,
                server=f"{hostname}.local."  
            )

            try:
                self.zeroconf.register_service(self.service_info)
                logger.info(f"mDNS service started on {local_ip}:{self.port}")
                logger.info(f"Service name: {service_name}.{service_type}")
            except Exception as e:
                logger.error(f"Failed to register mDNS service: {e}")

        # Run the blocking operation in a thread pool
        await asyncio.get_event_loop().run_in_executor(self.executor, register_service)
    #----------------------------------------------------------------------------
    #----------------------------------------------------------------------------
    async def stop_mdns(self):
        """Stop mDNS service asynchronously"""
        def unregister_service():
            if self.zeroconf:
                self.zeroconf.unregister_service(self.service_info)
                self.zeroconf.close()
                logger.info("mDNS service stopped")

        if self.zeroconf:
            await asyncio.get_event_loop().run_in_executor(self.executor, unregister_service)
        self.executor.shutdown(wait=False)

    #----------------------------------------------------------------------------
    #----------------------------------------------------------------------------
    async def process_frame(self, frame: np.ndarray):
        """Process frame with YOLO detection using GPU"""
        try:
            # Run inference in thread pool to avoid blocking
            results = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self.model(frame, device=self.device, verbose=False)
            )
            
            detections = []
            person_detected = False
            
            # Process results
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Class 0 is person in COCO dataset
                    if int(box.cls) == 0:
                        person_detected = True
                        conf = float(box.conf)
                        x1, y1, x2, y2 = [int(x) for x in box.xyxy[0].tolist()]
                        
                        # Draw rectangle on frame
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"Person {conf:.2f}", (x1, y1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        detections.append({
                            "confidence": conf,
                            "box": [x1, y1, x2, y2]
                        })
            
            # Encode processed frame
            _, buffer = cv2.imencode('.jpg', frame)
            processed_frame = base64.b64encode(buffer).decode('utf-8')
            
            return {
                "frame": processed_frame,
                "person_detected": person_detected,
                "detections": detections
            }
            
        except Exception as e:
            logger.error(f"Error in YOLO processing: {e}")
            return None

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
edge_server = EdgeServer()
async def check_camera_statuses():
    """Check all cameras and update their status if they haven't sent data in 2 minutes"""
    while True:
        current_time = datetime.now()
        for camera_id, data in edge_server.cameras.items():
            last_seen = data["last_seen"]
            current_status = data["status"]
            
            # Calculate time difference
            time_diff = current_time - last_seen
            
            # If camera hasn't sent data for 2 minutes and isn't already offline
            if time_diff.total_seconds() > 120 and current_status != "offline":
                edge_server.cameras[camera_id]["status"] = "offline"
                await db.update_camera_status(camera_id, "offline")
                logger.info(f"Camera {camera_id} marked as offline")
                
        await asyncio.sleep(60)  # Check every minute

@asynccontextmanager
async def lifespan(app: FastAPI): 
    # Startup 
    await db.init_db()  # Add this line
    await edge_server.start_mdns()
    asyncio.create_task(check_camera_statuses())
    logger.info("Edge server started")
    yield 
    # Shutdown
    await edge_server.stop_mdns()
    logger.info("Edge server stopped")
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
app = FastAPI(lifespan=lifespan)
db = Database()
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
@app.post("/register/camera")
async def register_camera(registration_data: dict): 
    try:
        camera_id = registration_data["camera_id"]

        existing_camera = await db.get_camera(camera_id)
        if existing_camera:
            logger.info(f"Camera {camera_id} already registered, updating status")
            edge_server.cameras[camera_id] = {
                "capabilities": registration_data["capabilities"],
                "last_seen": datetime.now(),
                "status": "registered"
            }
            return {"status": "success", "message": "Camera reconnected"}

        edge_server.cameras[camera_id] = {
            "capabilities": registration_data["capabilities"],
            "last_seen": datetime.now(),
            "status": "registered"
        }

        await db.create_camera(camera_id, registration_data["name"], "registered")

        logger.info(f"Camera {camera_id} registered")
        return {"status": "success", "message": "Camera registered"}
    except Exception as e:
        logger.error(f"Registration failed: {e}")
        raise HTTPException(status_code=500, detail="Registration failed")
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
@app.post("/register/alarm")
async def register_alarm(registration_data: dict):
    try:
        alarm_id = registration_data["alarm_id"]

        existing_alarm = await db.get_alarm_device(alarm_id)
        if existing_alarm:
            logger.info(f"alarm {alarm_id} already registered, updating status")
            edge_server.alarms[alarm_id] = {
                "registration_time": datetime.now(),
                "status": "registered"
            }
            return {"status": "success", "message": "alarm reconnected"}

        edge_server.alarms[alarm_id] = {
            "registration_time": datetime.now(),
            "status": "registered"
        }

        await db.create_alarm_device(alarm_id, registration_data["name"], "registered")

        logger.info(f"alarm {alarm_id} registered")
        return {"status": "success", "message": "alarm registered"}
    except Exception as e:
        logger.error(f"Registration failed: {e}")
        raise HTTPException(status_code=500, detail="Registration failed")
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
@app.websocket("/ws/camera/{camera_id}")
async def camera_websocket(websocket: WebSocket, camera_id: str):
    if camera_id not in edge_server.cameras:
        await websocket.close(code=4000)
        return
    
    await websocket.accept()
    edge_server.active_connections[camera_id] = websocket
    
    edge_server.cameras[camera_id]["status"] = "online"
    await db.update_camera_status(camera_id, "online")
    logger.info(f"Camera WebSocket connection established for camera {camera_id}")
    try:
        while True:
            data = await websocket.receive_text()
            frame_data = json.loads(data)
            
            # Update last seen timestamp
            edge_server.cameras[camera_id]["last_seen"] = datetime.now()

            jpg_data = base64.b64decode(frame_data["frame"])
            nparr = np.frombuffer(jpg_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)


            if frame is not None:
                # Process frame with YOLO
                processed_data = await edge_server.process_frame(frame)
                
                if processed_data:
                    frame_data.update(processed_data)
                    
                    if processed_data["person_detected"]:
                        pass
                        # logger.info(f"Person detected in frame from camera {camera_id} "
                        #           f"with {len(processed_data['detections'])} detections")

                    # Broadcast processed frame to viewers
                    if camera_id in edge_server.viewers:
                        frame_message = json.dumps({
                            "camera_id": camera_id,
                            "timestamp": frame_data["timestamp"],
                            "frame": processed_data["frame"],
                            "detections": processed_data["detections"]
                        })
                        
                        for viewer in edge_server.viewers[camera_id].copy():
                            try:
                                await viewer.send_text(frame_message)
                            except Exception:
                                edge_server.viewers[camera_id].remove(viewer)
                        
    except Exception as e:
        logger.error(f"Camera websocket error: {e}")
    finally:
        edge_server.active_connections.pop(camera_id)
        logger.info(f"Camera WebSocket connection closed for camera {camera_id}")
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
@app.get("/cameras")
async def get_cameras():
    """Get list of connected cameras"""
    return {
        "cameras": [
            {
                "camera_id": camera_id,
                "capabilities": data["capabilities"],
                "last_seen": data["last_seen"].isoformat(),
                "status": data["status"]
            }
            for camera_id, data in edge_server.cameras.items()
        ]
    }
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
@app.get("/api/get-cameras")
async def get_cameras_api(): 
    return await db.get_cameras()
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
@app.get("/api/get-alarms")
async def get_alarms(): 
    return await db.get_alarms()
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
@app.get("/api/get-alarm-devices")
async def get_alarm_devices_api():
    return await db.get_alarm_devices()
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
@app.post("/api/delete-camera/{camera_id}")
async def delete_camera(camera_id: str):
    await db.delete_camera(camera_id)
    return {"status": "success", "message": "Camera deleted"}
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
@app.post("/api/delete-alarm_device/{alarm_device_id}")
async def delete_alarm_device(alarm_device_id: str):
    await db.delete_camera(alarm_device_id)
    return {"status": "success", "message": "Alarm device deleted"}
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
@app.get("/api/get-rooms")
async def get_rooms():
    return await db.get_rooms()
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
@app.post("/api/create-room")
async def create_room(room_name: str = Query(None)):
    await db.create_room(room_name)
    return {"status": "success", "message": "Room created"}
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
@app.put("/api/camera-to-room")
async def assign_camera_room(camera_id: str = Query(None), room_id: str = Query(None)):
    await db.assign_camera_room(camera_id, room_id)
    return {"status": "success", "message": "Camera assigned to room"}
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
@app.put("/api/alarm-device-to-room")
async def assign_alarm_device_room(alarm_device_id: str = Query(None), room_id: str = Query(None)):
    await db.assign_alarm_device_room(alarm_device_id, room_id)
    return {"status": "success", "message": "Camera assigned to room"}
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
@app.websocket("/ws/view/{camera_id}")
async def viewer_websocket(websocket: WebSocket, camera_id: str):
    if camera_id not in edge_server.cameras:
        await websocket.close(code=4000)
        return
    
    await websocket.accept()
    
    # Add this viewer websocket to a list of viewers for this camera
    if camera_id not in edge_server.viewers:
        edge_server.viewers[camera_id] = set()
    edge_server.viewers[camera_id].add(websocket)
    
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except Exception as e:
        logger.error(f"Viewer websocket error: {e}")
    finally:
        edge_server.viewers[camera_id].remove(websocket)
        if not edge_server.viewers[camera_id]:
            edge_server.viewers.pop(camera_id)
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
# After all API endpoints are defined, serve the static website
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(BASE_DIR, "website", "build")
app.mount("/", StaticFiles(directory=BUILD_DIR, html=True), name="static")
if __name__ == "__main__":
    import uvicorn 
    uvicorn.run(app, host="0.0.0.0", port=8000)