import base64
import socket
import numpy as np
import cv2
import json
import asyncio
import torch
import platform
import os
import threading
import subprocess
import aiortc
import queue
import time
from pathlib import Path
from fastapi import FastAPI, WebSocket, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from typing import Dict
from zeroconf import ServiceInfo, Zeroconf
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from database import Database
from ultralytics import YOLO
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.media import MediaPlayer, MediaRelay
from av import VideoFrame
from typing import Optional, Set
from datetime import datetime

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
def get_device():
    """Determine the best available device for inference"""
    if torch.cuda.is_available():
        logger.info("Using CUDA for YOLO inference")
        return "cuda"
    if torch.backends.mps.is_available():
        logger.info("Using MPS for YOLO inference")
        return "mps"
    return "cpu"


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
class RTSPReader:
    def __init__(self, url: str):
        self.url = url
        self.running = False
        self.last_frame = None
        self.lock = threading.Lock()
        self.frame_queue = queue.Queue(maxsize=10)
        self.process: Optional[subprocess.Popen] = None
        self.os_type = platform.system().lower()

    def get_ffmpeg_read_args(self):
        """Get OS-specific FFmpeg reading arguments"""
        base_args = ["ffmpeg", "-rtsp_transport", "tcp", "-stimeout", "5000000"]

        if self.os_type == "darwin":
            # macOS specific settings
            input_args = [
                "-re",  # Read input at native frame rate
                "-vsync",
                "1",  # Strict frame sync for macOS
                "-i",
                self.url,
            ]
        elif self.os_type == "linux":
            # Linux specific settings
            input_args = [
                "-re",
                "-vsync",
                "0",  # Less strict sync for Linux
                "-i",
                self.url,
            ]
        else:
            # Windows specific settings
            input_args = ["-re", "-vsync", "0", "-i", self.url]

        # Common output arguments
        output_args = [
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-flags",
            "low_delay",
            "-fflags",
            "nobuffer+discardcorrupt",
            "-vf",
            "scale=1280:720",
            "-an",
            "pipe:1",
        ]

        return base_args + input_args + output_args

    async def start(self):
        """Start reading RTSP stream using FFmpeg"""
        try:
            self.running = True

            ffmpeg_cmd = self.get_ffmpeg_read_args()
            logger.info(f"Starting FFmpeg reader with command: {' '.join(ffmpeg_cmd)}")

            # Start FFmpeg process
            self.process = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=10**8,  # Large buffer
            )

            # Start frame reading thread
            frame_thread = threading.Thread(target=self._read_frames)
            frame_thread.daemon = True
            frame_thread.start()

            # Start error logging thread
            error_thread = threading.Thread(target=self._log_stderr)
            error_thread.daemon = True
            error_thread.start()

            # Wait for connection
            await asyncio.sleep(2)

            if self.process.poll() is not None:
                self.running = False
                await asyncio.sleep(2)

            # Start frame reading thread
            frame_thread = threading.Thread(target=self._read_frames)
            frame_thread.daemon = True
            frame_thread.start()

            while self.running:
                if self.process.poll() is not None:
                    raise Exception("FFmpeg process terminated unexpectedly")

                try:
                    frame = self.frame_queue.get(timeout=1.0)
                    with self.lock:
                        self.last_frame = frame
                except queue.Empty:
                    continue

                await asyncio.sleep(0.001)

        except Exception as e:
            logger.error(f"Error in RTSP reader: {str(e)}")
            self.stop()

    def _read_frames(self):
        """Read frames from FFmpeg output"""
        try:
            while self.running and self.process and self.process.poll() is None:
                # Read raw video frame (3 bytes per pixel)
                raw_frame = self.process.stdout.read(1280 * 720 * 3)
                if not raw_frame:
                    break

                # Convert to numpy array and reshape
                frame = np.frombuffer(raw_frame, np.uint8)
                frame = frame.reshape((720, 1280, 3))

                # Put frame in queue, drop if queue is full
                try:
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    continue

        except Exception as e:
            logger.error(f"Error reading frames: {str(e)}")

    def _log_stderr(self):
        """Log FFmpeg error output"""
        while self.running and self.process and self.process.poll() is None:
            line = self.process.stderr.readline()
            if line:
                logger.debug(f"FFmpeg: {line.decode().strip()}")

    def get_frame(self):
        """Get the latest frame"""
        with self.lock:
            return self.last_frame.copy() if self.last_frame is not None else None

    def stop(self):
        """Stop the RTSP reader"""
        self.running = False
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
        with self.lock:
            self.last_frame = None


class VideoTransformTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, rtsp_reader, camera_id):
        super().__init__()
        self.rtsp_reader = rtsp_reader
        self.camera_id = camera_id
        self.model = edge_server.model
        self.device = edge_server.device
        self._last_frame_time = (
            time.monotonic() * 1000
        )  # Use monotonic for better timing

        self._start = time.time()
        self._timestamp = 0

    async def next_timestamp(self):
        """Calculate the next frame timestamp"""
        if self._timestamp == 0:
            self._start = time.time
            self._timestamp = 90
        else:
            self._timestamp += int(1 / 30 * 1000000)  # 30fps = 1/30 seconds per frame
        return self._timestamp, 1000000  # timestamp in microseconds, timebase

    async def recv(self):
        # Get frame from RTSP reader
        frame = self.rtsp_reader.get_frame()

        if frame is None:
            # Create black frame if no frame available
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        # Process with YOLO every 100ms
        current_time = time.monotonic() * 1000
        if current_time - self._last_frame_time >= 100:
            try:
                # Run YOLO detection
                results = self.model(frame, device=self.device)

                # Draw detections on frame
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        if int(box.cls) == 0:  # Person class
                            conf = float(box.conf)
                            x1, y1, x2, y2 = [int(x) for x in box.xyxy[0].tolist()]

                            # Draw box and label
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(
                                frame,
                                f"Person {conf:.2f}",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 0),
                                2,
                            )

                self._last_frame_time = current_time

            except Exception as e:
                logger.error(f"Error in YOLO processing: {str(e)}")

        # Convert to VideoFrame for WebRTC
        video_frame = VideoFrame.from_ndarray(frame, format="bgr24")
        pts, time_base = await self.next_timestamp()
        video_frame.pts = pts
        video_frame.time_base = time_base

        return video_frame


class EdgeServer:
    def __init__(self):
        self.cameras: Dict[str, dict] = {}
        self.alarms: Dict[str, dict] = {}
        self.rtsp_readers: Dict[str, RTSPReader] = {}
        self.peer_connections: Dict[str, RTCPeerConnection] = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.relay = MediaRelay()
        self.port = 8000

        self.active_connections: Dict[str, set] = {}
        self.viewers = {}
        self.zeroconf = None
        self.service_info = None

        self.dtection_interval = 100
        self.batch_size = 4
        self.processing_queue = asyncio.Queue(maxsize=32)
        self.result_cahce = {}
        self.processing_lock = asyncio.Lock()
        # Initialize YOLO model
        self.device = get_device()
        logger.info(f"Using device: {self.device} for YOLO inference")
        try:
            self.model = YOLO("yolov8n.pt")
            self.model.to(self.device)
            if self.device == "cuda":
                self.model.export(format="enginge")
            logger.info("YOLO model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing YOLO model: {e}")
            logger.warning("Falling back to CPU")
            self.device = "cpu"
            self.model = YOLO("yolov8n.pt")

    # ----------------------------------------------------------------------------
    # ----------------------------------------------------------------------------
    async def start_mdns(self):
        """Start mDNS service advertising asynchronously"""

        def register_service():
            self.zeroconf = Zeroconf()

            def get_local_ip():
                try:
                    # Create a temporary socket to determine the primary interface
                    temp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    temp_socket.connect(("8.8.8.8", 80))  # Connect to Google DNS
                    local_ip = temp_socket.getsockname()[0]
                    temp_socket.close()
                    return local_ip
                except Exception:
                    # Fallback to getting all non-localhost IPs
                    addresses = []
                    for iface in socket.getaddrinfo(socket.gethostname(), None):
                        ip = iface[4][0]
                        if not ip.startswith("127."):
                            addresses.append(ip)
                    return addresses[0] if addresses else "127.0.0.1"

            local_ip = get_local_ip()
            hostname = platform.node()  # Get clean hostname without domain
            # Fixed service info configuration
            service_name = "EdgeServer"
            service_type = "_edgeserver._tcp.local."
            desc = {"version": "1.0", "server": "edge"}

            self.service_info = ServiceInfo(
                type_=service_type,
                name=f"{service_name}.{service_type}",
                addresses=[socket.inet_aton(local_ip)],
                port=self.port,
                properties=desc,
                server=f"{hostname}.local.",
            )

            logger.info(self.service_info)

            try:
                self.zeroconf.register_service(self.service_info)
                logger.info(f"mDNS service started on {local_ip}:{self.port}")
                logger.info(f"Service name: {service_name}.{service_type}")
            except Exception as e:
                logger.error(f"Failed to register mDNS service: {e}")

        # Run the blocking operation in a thread pool
        await asyncio.get_event_loop().run_in_executor(self.executor, register_service)

    # ----------------------------------------------------------------------------
    # ----------------------------------------------------------------------------
    async def stop_mdns(self):
        """Stop mDNS service asynchronously"""

        def unregister_service():
            if self.zeroconf:
                self.zeroconf.unregister_service(self.service_info)
                self.zeroconf.close()
                logger.info("mDNS service stopped")

        if self.zeroconf:
            await asyncio.get_event_loop().run_in_executor(
                self.executor, unregister_service
            )
        self.executor.shutdown(wait=False)

    # ----------------------------------------------------------------------------
    # ----------------------------------------------------------------------------
    async def start_rtsp_reader(self, camera_id, rtsp_url):
        if camera_id in self.rtsp_readers:
            self.rtsp_readers[camera_id].stop()

        reader = RTSPReader(rtsp_url)
        self.rtsp_readers[camera_id] = reader
        asyncio.create_task(reader.start())
        logger.info(f"Started RTSP reader for camera {camera_id}")

    # ----------------------------------------------------------------------------
    # ----------------------------------------------------------------------------
    async def handle_webrtc_offer(self, camera_id: str, offer: dict) -> dict:
        """Handle incoming WebRTC offer and create answer"""
        if camera_id not in self.rtsp_readers:
            raise ValueError(f"Camera {camera_id} not found")

        # Create new peer connection
        pc = RTCPeerConnection()
        self.peer_connections[camera_id] = pc

        # Create video track from RTSP reader
        video_track = VideoTransformTrack(self.rtsp_readers[camera_id], camera_id)
        pc.addTrack(video_track)

        # Handle the offer
        await pc.setRemoteDescription(
            RTCSessionDescription(offer["sdp"], offer["type"])
        )
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

    async def close_peer_connection(self, camera_id: str):
        """Close WebRTC peer connection"""
        if camera_id in self.peer_connections:
            pc = self.peer_connections[camera_id]
            await pc.close()
            del self.peer_connections[camera_id]


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
edge_server = EdgeServer()
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
app = FastAPI(lifespan=lifespan)
db = Database()


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
@app.post("/register/camera")
async def register_camera(registration_data: dict):
    try:
        camera_id = registration_data["camera_id"]
        rtsp_url = registration_data["rtsp_url"]
        if not rtsp_url:
            raise HTTPException(status_code=400, detail="RTSP URL is required")

        existing_camera = await db.get_camera(camera_id)
        if existing_camera:
            logger.info(f"Camera {camera_id} already registered, updating status")
            edge_server.cameras[camera_id] = {
                "capabilities": registration_data["capabilities"],
                "rstp_url": registration_data["rtsp_url"],
                "last_seen": datetime.now(),
                "status": "registered",
            }
            return {"status": "success", "message": "Camera reconnected"}

        edge_server.cameras[camera_id] = {
            "capabilities": registration_data["capabilities"],
            "rstp_url": registration_data["rtsp_url"],
            "last_seen": datetime.now(),
            "status": "registered",
        }

        await db.create_camera(camera_id, registration_data["name"], "registered")
        await edge_server.start_rtsp_reader(camera_id, rtsp_url)

        logger.info(f"Camera {camera_id} registered with RTSP URL: {rtsp_url}")
        return {"status": "success", "message": "Camera registered"}
    except Exception as e:
        logger.error(f"Registration failed: {e}")
        raise HTTPException(status_code=500, detail="Registration failed")


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
@app.post("/webrtc/{camera_id}/offer")
async def webrtc_offer(camera_id: str, offer: dict):
    try:
        answer = await edge_server.handle_webrtc_offer(camera_id, offer)
        return answer
    except Exception as e:
        logger.error(f"WebRTC offer error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
@app.post("/webrtc/{camera_id}/close")
async def webrtc_close(camera_id: str):
    await edge_server.close_peer_connection(camera_id)
    return {"status": "success"}


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
@app.post("/register/alarm")
async def register_alarm(registration_data: dict):
    try:
        alarm_id = registration_data["alarm_id"]

        existing_alarm = await db.get_alarm_device(alarm_id)
        if existing_alarm:
            logger.info(f"alarm {alarm_id} already registered, updating status")
            edge_server.alarms[alarm_id] = {
                "registration_time": datetime.now(),
                "status": "registered",
            }
            return {"status": "success", "message": "alarm reconnected"}

        edge_server.alarms[alarm_id] = {
            "registration_time": datetime.now(),
            "status": "registered",
        }

        await db.create_alarm_device(alarm_id, registration_data["name"], "registered")

        logger.info(f"alarm {alarm_id} registered")
        return {"status": "success", "message": "alarm registered"}
    except Exception as e:
        logger.error(f"Registration failed: {e}")
        raise HTTPException(status_code=500, detail="Registration failed")


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
@app.get("/cameras")
async def get_cameras():
    """Get list of connected cameras"""
    return {
        "cameras": [
            {
                "camera_id": camera_id,
                "capabilities": data["capabilities"],
                "last_seen": data["last_seen"].isoformat(),
                "status": data["status"],
            }
            for camera_id, data in edge_server.cameras.items()
        ]
    }


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
@app.get("/api/get-cameras")
async def get_cameras_api():
    return await db.get_cameras()


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
@app.get("/api/get-alarms")
async def get_alarms():
    return await db.get_alarms()


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
@app.get("/api/get-alarm-devices")
async def get_alarm_devices_api():
    return await db.get_alarm_devices()


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
@app.post("/api/delete-camera/{camera_id}")
async def delete_camera(camera_id: str):
    await db.delete_camera(camera_id)
    return {"status": "success", "message": "Camera deleted"}


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
@app.post("/api/delete-alarm_device/{alarm_device_id}")
async def delete_alarm_device(alarm_device_id: str):
    await db.delete_camera(alarm_device_id)
    return {"status": "success", "message": "Alarm device deleted"}


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
@app.get("/api/get-rooms")
async def get_rooms():
    return await db.get_rooms()


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
@app.post("/api/create-room")
async def create_room(room_name: str = Query(None)):
    await db.create_room(room_name)
    return {"status": "success", "message": "Room created"}


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
@app.put("/api/camera-to-room")
async def assign_camera_room(camera_id: str = Query(None), room_id: str = Query(None)):
    await db.assign_camera_room(camera_id, room_id)
    return {"status": "success", "message": "Camera assigned to room"}


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
@app.put("/api/alarm-device-to-room")
async def assign_alarm_device_room(
    alarm_device_id: str = Query(None), room_id: str = Query(None)
):
    await db.assign_alarm_device_room(alarm_device_id, room_id)
    return {"status": "success", "message": "Camera assigned to room"}


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# After all API endpoints are defined, serve the static website
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(BASE_DIR, "website", "build")
app.mount("/", StaticFiles(directory=BUILD_DIR, html=True), name="static")
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
