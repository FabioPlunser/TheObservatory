import nats
import os
import asyncio
import logging
from enum import Enum

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("natsClient.log")],
)

logger = logging.getLogger(__name__)


class NatsClient:
    def __init__(self, nats_url):
        self.nc = nats.connect(nats_url)
        logger.info(f"Trying to connect to nats with {nats_url}")

    async def send_message(self, subject, data):
        await self.nc.publish(subject, data)

    async def add_subscription(self, subject, callback):
        await self.nc.subscribe(subject, cb=callback)

    async def close(self):
        await self.nc.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.nc.close()


class Commands(Enum):
    INIT_BUCKET = "init_bucket"
    PRESIGNED_UPLOAD_KNOWN_FACE = "PRESIGNED_UPLOAD_KNOWN_FACE"
    PRESIGNED_UPLOAD_RECOGNITION_FACE = "PRESIGNED_UPLOAD_RECOGNITION_FACE"
    PRESIGNED_DOWNLOAD_KNOWN_FACES = "PRESIGNED_DOWNLOAD_KNOWN_FACES"
    PRESIGNED_DOWNLOAD_RECOGNITION_FACES = "PRESIGNED_DOWNLOAD_RECOGNITION_FACES"
    DELETE_KNOWN_FACE = "DELETE_KNOWN_FACE"
    DELETE_RECOGNITION_FACES = "DELETE_RECOGNITION_FACES"
    EXECUTE_RECOGNITION = "EXECUTE_RECOGNITION"
    ALARM = "alarm"
