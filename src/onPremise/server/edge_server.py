from zeroconf import ServiceInfo, Zeroconf
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor
from aiortc import RTCPeerConnection, RTCSessionDescription
from datetime import datetime
from rtsp_reader import RTSPReader
from video_transform import VideoTransformTrack
from typing import Dict

import socket
import platform
import torch
import asyncio
import logging

logger = logging.getLogger(__name__)


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class EdgeServer:
    def __init__(self):
        self.cameras = {}
        self.alarms = {}
        self.rtsp_readers = {}
        self.peer_connections = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.port = 8000

        # Initialize YOLO
        self.device = get_device()
        try:
            self.model = YOLO("yolov8n.pt")
            self.model.to(self.device)
        except Exception as e:
            logger.error(f"Error initializing YOLO model: {e}")
            self.device = "cpu"
            self.model = YOLO("yolov8n.pt")

    async def start_mdns(self):
        def register_service():
            self.zeroconf = Zeroconf()

            local_ip = socket.gethostbyname(socket.gethostname())
            hostname = platform.node()

            self.service_info = ServiceInfo(
                type_="_edgeserver._tcp.local.",
                name=f"EdgeServer._edgeserver._tcp.local.",
                addresses=[socket.inet_aton(local_ip)],
                port=self.port,
                properties={"version": "1.0", "server": "edge"},
                server=f"{hostname}.local.",
            )

            self.zeroconf.register_service(self.service_info)

        await asyncio.get_event_loop().run_in_executor(self.executor, register_service)

    async def stop_mdns(self):
        if self.zeroconf:
            self.zeroconf.unregister_service(self.service_info)
            self.zeroconf.close()
        self.executor.shutdown(wait=False)

    async def start_rtsp_reader(self, camera_id, rtsp_url):
        if camera_id in self.rtsp_readers:
            self.rtsp_readers[camera_id].stop()

        reader = RTSPReader(rtsp_url)
        self.rtsp_readers[camera_id] = reader
        asyncio.create_task(reader.start())

    async def handle_webrtc_offer(self, camera_id: str, offer: dict) -> dict:
        if camera_id not in self.rtsp_readers:
            raise ValueError(f"Camera {camera_id} not found")

        pc = RTCPeerConnection()
        self.peer_connections[camera_id] = pc

        video_track = VideoTransformTrack(
            self.rtsp_readers[camera_id], camera_id, self.model, self.device
        )
        pc.addTrack(video_track)

        await pc.setRemoteDescription(
            RTCSessionDescription(offer["sdp"], offer["type"])
        )
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

    async def close_peer_connection(self, camera_id: str):
        if camera_id in self.peer_connections:
            pc = self.peer_connections[camera_id]
            await pc.close()
            del self.peer_connections[camera_id]
