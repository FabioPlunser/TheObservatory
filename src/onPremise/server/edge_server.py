from zeroconf import ServiceInfo, Zeroconf
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from video_processor import VideoProcessor
from typing import Dict, Set
from database import Database

import socket
import platform
import asyncio
import logging
import multiprocessing

logger = logging.getLogger(__name__)


class EdgeServer:
    def __init__(self, port):
        self.cameras = {}
        self.alarms = {}
        self.processors: Dict[str, VideoProcessor] = {}
        self.executor = ThreadPoolExecutor()
        self.port = port
        self.reader_queue = multiprocessing.Queue()
        self.db = Database()

        self.active_streams = set()

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

    async def start_camera_stream(self, camera_id: str, rtsp_url: str):
        """Start streaming for a camera"""
        try:
            # Stop existing processor if any
            await self.stop_camera_stream(camera_id)

            # Create and start new processor
            processor = VideoProcessor(rtsp_url)
            self.processors[camera_id] = processor
            processor.start()

            # Update camera status
            self.cameras[camera_id] = {
                "rtsp_url": rtsp_url,
                "last_seen": datetime.now(),
                "status": "streaming",
            }

            logger.info(f"Started video processor for camera {camera_id}")

        except Exception as e:
            logger.error(f"Error starting camera stream: {e}")
            # Clean up on error
            if camera_id in self.processors:
                self.processors[camera_id].stop()
                del self.processors[camera_id]
            raise

    async def stop_camera_stream(self, camera_id: str):
        """Stop streaming for a camera and clean up all resources"""
        logger.info(f"Stopping camera stream for {camera_id}")

        try:
            # Stop and cleanup video processor
            if camera_id in self.processors:
                try:
                    processor = self.processors[camera_id]
                    processor.stop()  # This will stop both reader and YOLO processes
                    del self.processors[camera_id]
                    logger.info(f"Video processor stopped for camera {camera_id}")
                except Exception as e:
                    logger.error(
                        f"Error stopping video processor for camera {camera_id}: {e}"
                    )
                    raise

            # Update camera status if exists
            if camera_id in self.cameras:
                self.cameras[camera_id]["status"] = "stopped"

            logger.info(f"Successfully stopped camera stream {camera_id}")

        except Exception as e:
            logger.error(f"Error in stop_camera_stream: {e}")
            raise  # Re-raise to handle in the route
        finally:
            # Make absolutely sure we clean up processors
            if camera_id in self.processors:
                try:
                    del self.processors[camera_id]
                except:
                    pass

    async def get_frame(self, camera_id: str) -> bytes:
        """Get the latest frame for a camera"""
        try:
            processor = self.processors.get(camera_id)
            if not processor:
                logger.error(f"No processor found for camera {camera_id}")
                del self.processors[camera_id]
                del self.cameras[camera_id]
                self.db.delete_camera(camera_id)
                return None

            frame = processor.get_latest_frame()
            if frame is not None:
                # Update last seen timestamp
                self.cameras[camera_id]["last_seen"] = datetime.now()

            return frame

        except Exception as e:
            logger.error(f"Error getting frame: {e}")
            return None
