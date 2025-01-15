import nats
import logging
import json
from enum import Enum

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("natsClient.log")],
)

logger = logging.getLogger(__name__)


class NatsClient:
    def __init__(self):
        self.nc = None

    async def connect(self, nats_url):
        try:
            if not nats_url:
                nats_url = "nats://localhost:4222"
            self.nc = await nats.connect(nats_url)
            logger.info(f"Connected to NATS at {nats_url}")
        except Exception as e:
            logger.error(f"Failed to connect to nats with url {nats_url}: {e}")

    async def send_message(self, subject, data):
        if not self.nc:
            logger.error("NATS client not initialized")
            return
        await self.nc.publish(subject, data)

    async def send_message_with_reply(self, subject, data):
        if not self.nc:
            logger.error("NATS client not initialized")
            return
        try:
            # Convert data to JSON and encode
            encoded_data = json.dumps(data).encode()
            # Wait max 5 seconds for response
            response = await self.nc.request(subject, encoded_data, timeout=5.0)
            # Decode response
            return json.loads(response.data.decode())
        except Exception as e:
            logger.error(f"NATS request error: {e}")
            raise

    async def add_subscription(self, subject, callback):
        if not self.nc:
            logger.error("NATS client not initialized")
            return

        await self.nc.subscribe(subject, cb=callback)

    async def close(self):
        if not self.nc:
            logger.error("NATS client not initialized")
            return
        await self.nc.drain()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.nc:
            self.nc.drain()


class Commands(Enum):
    INIT_BUCKET = "INIT_BUCKET"
    GET_PRESIGNED_UPLOAD_KNOWN_FACE_URL = "GET_PRESIGNED_UPLOAD_KNOWN_FACE_URL"
    GET_PRESIGNED_UPLOAD_UNKNOWN_FACE_URL = "GET_PRESIGNED_UPLOAD_UNKNOWN_FACE_URL"
    GET_PRESIGNED_DOWNLOAD_ALL_KNOWN_FACES = "GET_PRESIGNED_DOWNLOAD_ALL_KNOWN_FACES"
    GET_PRESIGNED_DOWNLOAD_ALL_UNKNOWN_FACES = (
        "GET_PRESIGNED_DOWNLOAD_ALL_UNKNOWN_FACES"
    )
    DELETE_KNOWN_FACE = "DELTE_KNOWN_FACE"
    DELETE_ALL_UNKNONW_FACES = "DELETE_UNKNONW_FACE"
    EXECUTE_RECOGNITION = "EXECUTE_RECOGNITION"

    ALARM = "ALARM"
    FAILED = "FAILED"
