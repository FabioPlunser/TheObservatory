from zeroconf import ServiceInfo, Zeroconf
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from video_processor import VideoProcessor
from typing import Dict, Set
from database import Database
from nats_client import SharedNatsClient, Commands
from logging_config import setup_logger

import socket
import platform
import asyncio
import logging
import multiprocessing
import os
import json

setup_logger()
logger = logging.getLogger("EdgeServer")


class EdgeServer:
    def __init__(self, port):
        self.cameras = {}
        self.alarms = {}
        self.processors: Dict[str, VideoProcessor] = {}
        self.executor = ThreadPoolExecutor()
        self.port = port
        self.reader_queue = multiprocessing.Queue()
        self.db = Database()
        self.nats_client = SharedNatsClient.get_instance()
        self.active_streams = set()

        self._bucket_init_task = None
        self._bucket_init_running = False

    async def init_bucket(self):
        """Start the bucket initialization background task"""
        if self._bucket_init_task is not None:
            logger.info("Bucket initialization task already running")
            return

        self._bucket_init_running = True
        self._bucket_init_task = asyncio.create_task(self._bucket_init_retry())
        logger.info("Started bucket initialization background task")

    async def _bucket_init_retry(self):
        """Background task to retry bucket initialization until successful"""
        retry_delay = 5  # Start with 5 second delay between retries

        while self._bucket_init_running:
            try:
                self.nats_client = SharedNatsClient.get_instance()
                company_id = await self.db.get_company_id()
                company_init_bucket = await self.db.get_company_init_bucket()

                if not company_id:
                    logger.error("No company ID found in database")
                    await asyncio.sleep(retry_delay)
                    continue

                if company_init_bucket:
                    logger.info("Bucket already initialized")
                    self._bucket_init_running = False
                    break

                if not self.nats_client:
                    logger.error("NATS client not initialized bucket")
                    await asyncio.sleep(retry_delay)
                    continue

                response = await self.nats_client.send_message_with_reply(
                    Commands.INIT_BUCKET.value, {"company_id": company_id}
                )

                if response and response.get("success"):
                    logger.info("Successfully initialized bucket")
                    await self.db.update_company_init_bucket(True)
                    self._bucket_init_running = False
                    break
                else:
                    logger.error("Failed to initialize bucket")
                    # Increase retry delay up to max_delay

            except Exception as e:
                logger.error(f"Error in bucket initialization: {e}")

            await asyncio.sleep(retry_delay)

        self._bucket_init_task = None
        logger.info("Bucket initialization task completed")

    async def stop_bucket_init(self):
        """Stop the bucket initialization background task"""
        if self._bucket_init_running:
            self._bucket_init_running = False
            if self._bucket_init_task:
                try:
                    self._bucket_init_task.cancel()
                    await self._bucket_init_task
                except asyncio.CancelledError:
                    pass
                self._bucket_init_task = None

    async def start_mdns(self):
        def register_service():
            try:
                self.zeroconf = Zeroconf()

                # Get all network interfaces
                local_ips = []
                try:
                    hostname = socket.gethostname()
                    local_ips = socket.gethostbyname_ex(hostname)[2]
                except socket.gaierror:
                    # Fallback to a basic IP if hostname resolution fails
                    local_ips = ["127.0.0.1"]

                # Use the first non-localhost IP
                local_ip = next(
                    (ip for ip in local_ips if not ip.startswith("127.")), local_ips[0]
                )

                hostname = platform.node()

                self.service_info = ServiceInfo(
                    type_="_edgeserver._tcp.local.",
                    name=f"EdgeServer._edgeserver._tcp.local.",
                    addresses=[socket.inet_aton(local_ip)],
                    port=self.port,
                    properties={"version": "1.0", "server": "edge"},
                    server=f"{hostname}.local.",
                )

                try:
                    self.zeroconf.register_service(self.service_info)
                    logger.info(
                        f"Successfully registered mDNS service on {local_ip}:{self.port}"
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to register mDNS service: {e}. Server will continue without mDNS."
                    )
                    if hasattr(self, "zeroconf"):
                        self.zeroconf.close()

            except Exception as e:
                logger.warning(
                    f"Failed to start mDNS: {e}. Server will continue without mDNS."
                )
                return

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
            company_id = await self.db.get_company_id()
            # Create and start new processor
            processor = VideoProcessor(rtsp_url, company_id, camera_id)
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
                await self.db.delete_camera(camera_id)
                return None

            frame = processor.get_latest_frame()
            if frame is not None:
                # Update last seen timestamp
                self.cameras[camera_id]["last_seen"] = datetime.now()

            return frame

        except Exception as e:
            logger.error(f"Error getting frame: {e}")
            return None

    async def _setup_aler_subscription(self):
        """Setup subscription for face recognition alerts"""
        nats_client = SharedNatsClient.get_instance()
        try:
            company_id = await self.db.get_company_id()
            if company_id:
                # Subscribe to face recognition alerts
                await nats_client.add_subscription(
                    f"{Commands.ALARM.value}.{company_id}", self._handle_alert
                )
        except Exception as e:
            logger.error(f"Error setting up alert subscription: {e}")

    async def _handle_alert(self, msg):
        """Handle incoming face regonition alerts"""
        nats_client = SharedNatsClient.get_instance()
        try:
            data = json.loads(msg.data.decode())
            alert_type = data.get("type")
            camera_id = data.get("camera_id")
            company_id = data.get("company_id")

            # Verify this aler is for one of our cameras
            if camera_id not in self.cameras:
                # Not our camera, republish for other edge server
                await nats_client.publish_message(
                    f"{Commands.ALARM.value}.{company_id}", msg.data
                )
                return

            if alert_type == "unknown_face":
                face_id = data.get("face_id")
                track_id = data.get("track_id")

                if camera_id in self.processors:
                    processor = self.processors[camera_id]
                    if hasattr(processor, "person_tracker"):
                        if track_id in processor.person_tracker.tracked_person:
                            person = processor.person_tracker.tracked_persons[track_id]
                            person_recogintion_status = "unknonw"

            logger.info(f"Received alert for camera {camera_id}: {alert_type}")

            alarms = self.db.get_all_alarms()
            for alarm in alarms:
                alarm_ws = self.alarms[alarm["id"]].get("websocket")
                if alarm_ws:
                    await alarm_ws.send_json(
                        {
                            "type": "alarm_trigger",
                            "camera_id": camera_id,
                            "timestamp": datetime.now().isoformat(),
                        }
                    )

        except Exception as e:
            logger.error(f"Error handling alert: {e}")
