import socket
import platform
import asyncio
import logging
import json
import os

from zeroconf import ServiceInfo, Zeroconf
from datetime import datetime
from database import Database
from nats_client import SharedNatsClient, Commands
from logging_config import setup_logger
from video_processor import VideoProcessor
from concurrent.futures import ThreadPoolExecutor

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Setup logger for evaluation logs
setup_logger()
logger = logging.getLogger("EdgeServer")

evaluation_logger = logging.getLogger("Evaluation")
evaluation_logger.setLevel(logging.INFO)
evaluation_handler = logging.FileHandler("evaluation.log")
evaluation_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
evaluation_logger.addHandler(evaluation_handler)

class EdgeServer:
    def __init__(self, port):
        self.cameras = {}
        self.alarms = {}
        self.port = port
        self.db = Database()
        self.nats_client = SharedNatsClient.get_instance()
        self.active_streams = set()

        self._bucket_init_task = None
        self._bucket_init_running = False

        self.video_processor = VideoProcessor()

        self.executor = ThreadPoolExecutor()

    async def init_bucket(self):
        """Start the bucket initialization background task"""
        if self._bucket_init_task is not None:
            logger.info("Bucket initialization task already running")
            return

        # Check for NATS connection first
        self._bucket_init_running = True
        self._bucket_init_task = asyncio.create_task(self._bucket_init_retry())

    async def _bucket_init_retry(self):
        """Background task to retry bucket initialization until successful"""
        retry_delay = 5
        max_retries = 10
        retry_count = 0

        while self._bucket_init_running and retry_count < max_retries:
            try:
                company_id = await self.db.get_company_id()
                cloud_url = await self.db.get_cloud_url()
                company_init_bucket = await self.db.get_company_init_bucket()

                if not cloud_url:
                    logger.info("No cloud URL configured, waiting...")
                    await asyncio.sleep(retry_delay)
                    continue

                if company_init_bucket:
                    logger.info("Bucket already initialized")
                    break

                # Initialize NATS client if needed
                if not self.nats_client or not self.nats_client._connected:
                    logger.info(f"Initializing NATS client with URL: {cloud_url}")
                    self.nats_client = await SharedNatsClient.initialize(cloud_url)
                    if not self.nats_client:
                        logger.error("Failed to initialize NATS client")
                        retry_count += 1
                        await asyncio.sleep(retry_delay)
                        continue

                response = await self.nats_client.send_message_with_reply(
                    Commands.INIT_BUCKET.value, {"company_id": company_id}
                )

                if response and response.get("success"):
                    logger.info("Successfully initialized bucket")
                    await self.db.update_company_init_bucket(True)
                    break
                else:
                    logger.error(f"Failed to initialize bucket: {response}")
                    retry_count += 1

            except Exception as e:
                logger.error(f"Error in bucket initialization: {e}")
                retry_count += 1

            await asyncio.sleep(retry_delay)

        if retry_count >= max_retries:
            logger.error("Max retries reached for bucket initialization")

        await self._setup_alert_subscription()

        self._bucket_init_running = False
        self._bucket_init_task = None

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
        """Start mDNS service registration with proper event loop handling"""
        try:
            # Initialize Zeroconf in the executor
            def init_zeroconf():
                try:
                    self.zeroconf = Zeroconf()
                    return True
                except Exception as e:
                    logger.error(f"Failed to initialize Zeroconf: {e}")
                    return False

            # Run Zeroconf initialization in executor since it's blocking
            success = await asyncio.get_running_loop().run_in_executor(
                self.executor, init_zeroconf
            )
            if not success:
                logger.warning("Failed to initialize Zeroconf, continuing without mDNS")
                return

            # Get network interfaces
            def get_network_interfaces():
                local_ips = []
                valid_interfaces = []

                try:
                    # Get all network interfaces
                    import netifaces

                    interfaces = netifaces.interfaces()

                    for iface in interfaces:
                        addrs = netifaces.ifaddresses(iface)
                        # Check for IPv4 addresses
                        if netifaces.AF_INET in addrs:
                            for addr in addrs[netifaces.AF_INET]:
                                ip = addr["addr"]
                                if not ip.startswith("127."):
                                    local_ips.append(ip)
                                    valid_interfaces.append(iface)

                    if not local_ips:
                        # Fallback to hostname resolution if no valid interfaces found
                        hostname = socket.gethostname()
                        local_ips = socket.gethostbyname_ex(hostname)[2]
                except ImportError:
                    # Fallback if netifaces is not available
                    logger.warning(
                        "netifaces not installed, falling back to basic IP detection"
                    )
                    try:
                        hostname = socket.gethostname()
                        local_ips = socket.gethostbyname_ex(hostname)[2]
                    except socket.gaierror:
                        local_ips = ["127.0.0.1"]
                        logger.warning("Hostname resolution failed, using localhost")

                return local_ips, valid_interfaces

            # Get interfaces in executor
            (
                local_ips,
                valid_interfaces,
            ) = await asyncio.get_running_loop().run_in_executor(
                self.executor, get_network_interfaces
            )

            # Filter out localhost and select first valid IP
            valid_ips = [ip for ip in local_ips if not ip.startswith("127.")]
            if not valid_ips:
                logger.warning("No valid non-localhost IPs found, using localhost")
                local_ip = "127.0.0.1"
            else:
                local_ip = valid_ips[0]

            hostname = platform.node()

            # Create service info
            def create_and_register_service():
                try:
                    service_info = ServiceInfo(
                        type_="_edgeserver._tcp.local.",
                        name=f"EdgeServer._edgeserver._tcp.local.",
                        addresses=[socket.inet_aton(local_ip)],
                        port=self.port,
                        properties={
                            "version": "1.0",
                            "server": "edge",
                            "interfaces": (
                                ",".join(valid_interfaces)
                                if valid_interfaces
                                else "default"
                            ),
                        },
                        server=f"{hostname}.local.",
                    )

                    self.service_info = service_info
                    self.zeroconf.register_service(service_info)
                    return True
                except Exception as e:
                    logger.error(f"Error in service registration: {e}")
                    return False

            # Register service with retry logic
            max_retries = 3
            for retry_count in range(max_retries):
                success = await asyncio.get_running_loop().run_in_executor(
                    self.executor, create_and_register_service
                )

                if success:
                    logger.info(
                        f"Successfully registered mDNS service on {local_ip}:{self.port}"
                    )
                    break

                if retry_count == max_retries - 1:
                    logger.error(
                        f"Failed to register mDNS service after {max_retries} attempts"
                    )
                    if hasattr(self, "zeroconf"):
                        await asyncio.get_running_loop().run_in_executor(
                            self.executor, self.zeroconf.close
                        )
                    return

                logger.warning(
                    f"Retry {retry_count + 1}/{max_retries} for mDNS registration"
                )
                await asyncio.sleep(1)

        except Exception as e:
            logger.error(f"Critical error in mDNS registration: {e}")
            # Ensure cleanup on critical failure
            if hasattr(self, "zeroconf"):
                try:
                    await asyncio.get_running_loop().run_in_executor(
                        self.executor, self.zeroconf.close
                    )
                except Exception as cleanup_error:
                    logger.error(f"Error during cleanup: {cleanup_error}")

    async def stop_mdns(self):
        if self.zeroconf:
            self.zeroconf.unregister_service(self.service_info)
            self.zeroconf.close()
        self.executor.shutdown(wait=False)

    async def start_camera_stream(self, camera_id: str, rtsp_url: str):
        """Start streaming for a camera"""
        try:
            # Stop existing processor if any
            company_id = await self.db.get_company_id()
            self.video_processor.add_camera(camera_id, rtsp_url, company_id)

            self.cameras[camera_id] = {
                "rstp_url": rtsp_url,
                "last_seen": datetime.now(),
                "status": "streaming",
            }

            logger.info(f"Started video processor for camera {camera_id}")
            evaluation_logger.info(f"Started video processor for camera {camera_id}")

        except Exception as e:
            logger.error(f"Error starting camera stream: {e}")
            evaluation_logger.error(f"Error starting camera stream: {e}")
            raise

    async def stop_camera_stream(self, camera_id: str):
        """Stop streaming for a camera"""
        logger.info(f"Stopping camera stream {camera_id}")
        evaluation_logger.info(f"Stopping camera stream {camera_id}")

        try:
            if camera_id in self.cameras:
                self.cameras[camera_id]["status"] = "stopped"

            self.video_processor.remove_camera(camera_id)
            # Camera will be stopped when processor is stopped
            logger.info(f"Successfully stopped camera stream {camera_id}")
            evaluation_logger.info(f"Successfully stopped camera stream {camera_id}")

        except Exception as e:
            logger.error(f"Error in stop_camera_stream: {e}")
            evaluation_logger.error(f"Error in stop_camera_stream: {e}")
            raise

    async def get_frame(self, camera_id: str) -> bytes:
        """Get the latest frame for a camera"""
        try:
            return self.video_processor.get_frame(camera_id)

        except Exception as e:
            logger.error(f"Error getting frame: {e}")
            evaluation_logger.error(f"Error getting frame: {e}")
            return None

    async def _setup_alert_subscription(self):
        """Setup subscription for face recognition alerts"""
        logger.info("Setting up alert subscription")
        evaluation_logger.info("Setting up alert subscription")
        nats_client = SharedNatsClient.get_instance()
        try:
            company_id = await self.db.get_company_id()
            if company_id:
                # Subscribe to face recognition alerts
                await nats_client.add_subscription(
                    f"{Commands.ALARM.value}.{company_id}", self._handle_alert
                )
                logger.info("Alert subscription setup successfully")
                evaluation_logger.info("Alert subscription setup successfully")
        except Exception as e:
            logger.error(f"Error setting up alert subscription: {e}")
            evaluation_logger.error(f"Error setting up alert subscription: {e}")

    async def _handle_alert(self, msg):
        """Handle incoming face recognition alerts"""
        nats_client = SharedNatsClient.get_instance()
        try:
            logger.info(f"Received alert: {msg.subject}")
            evaluation_logger.info(f"Received alert: {msg.subject}")
            data = json.loads(msg.data.decode())
            camera_id = data.get("camera_id")
            unknown_face_url = data.get("unknown_face_url")
            track_id = data.get("track_id")
            face_id = data.get("face_id")
            company_id = data.get("company_id")

            # Log detailed information about the alert
            logger.debug(f"Alert details: camera_id={camera_id}, unknown_face_url={unknown_face_url}, track_id={track_id}, face_id={face_id}, company_id={company_id}")
            evaluation_logger.info(f"Alert details: camera_id={camera_id}, track_id={track_id}, face_id={face_id}, company_id={company_id}")

            # Verify this alert is for one of our cameras
            if camera_id not in self.cameras:
                await nats_client.publish_message(
                    f"{Commands.ALARM.value}.{company_id}", msg.data
                )
                return

            # Update that the certain camera has an unknown face
            await self.db.update_camera_unknown_face(camera_id, True, unknown_face_url)

            if camera_id in self.cameras:
                logger.info(f"Received alert for camera {camera_id}")
                evaluation_logger.info(f"Received alert for camera {camera_id}")

                alarms = await self.db.get_all_alarms()

                for alarm in alarms:
                    await self.db.update_alarm_status(
                        alarm["id"], True, None, datetime.now()
                    )

                    alarm_ws = self.alarms[alarm["id"]].get("websocket")
                    if alarm_ws:
                        await alarm_ws.send_json(
                            {
                                "active": "true",
                            }
                        )
                    logger.info(f"Alarm {alarm['id']} activated")
                    evaluation_logger.info(f"Alarm {alarm['id']} activated")

        except Exception as e:
            logger.error(f"Error handling alert: {e}")
            evaluation_logger.error(f"Error handling alert: {e}")