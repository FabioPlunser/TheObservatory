import nats
import asyncio
import logging
import json

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from enum import Enum
from typing import Optional
from logging_config import setup_logger

setup_logger()
logger = logging.getLogger("NatsClient")


class NatsClient:
    def __init__(self, nats_url=None):
        self.nc = None
        self._connected = False
        self.nats_url = nats_url

    async def connect(self):
        try:
            logger.info(f"Awaiting nats.connect with url: {self.nats_url}")
            self.nc = await nats.connect(self.nats_url)
            self._connected = True
            if self.nc.is_connected:
                logger.info("Succesfully connected to NATS")
            return True
        except Exception as e:
            logger.warning(f"Failed to connect to nats with url {self.nats_url}: {e}")
            self._connected = False
            logger.error(f"Failed to connect to nats: {e}")

    async def send_message(self, subject, data):
        if self.nc is None or not self._connected:
            logger.error("NATS client not initialized")
            return
        await self.nc.publish(subject, json.dumps(data).encode())

    async def send_message_with_reply(self, subject, data):
        if self.nc is None or not self._connected:
            logger.error("NATS client not initialized")
            return
        try:
            # Convert data to JSON and encode
            encoded_data = json.dumps(data).encode()
            # Wait max 5 seconds for response
            response = await self.nc.request(subject, encoded_data, timeout=30)
            # Decode response
            return json.loads(response.data.decode())
        except Exception as e:
            logger.error(f"NATS request error: for {subject} {e}")
            raise

    async def add_subscription(self, subject, callback):
        if self.nc is None:
            logger.error("NATS client not initialized")
            return

        await self.nc.subscribe(subject, cb=callback)

    async def close(self):
        if self.nc is None:
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


class SharedNatsClient:
    _instance: Optional[NatsClient] = None

    @classmethod
    async def initialize(cls, nats_url):
        """Initialize NATS client with improved error handling"""
        if not nats_url:
            logger.error("No NATS URL provided")
            return None

        try:
            logger.info(f"Initializing NATS client with URL: {nats_url}")

            # Close existing instance if it exists
            if cls._instance is not None:
                await cls._instance.close()
                cls._instance = None

            # Create new instance
            cls._instance = NatsClient(nats_url)

            # Try to connect with timeout
            try:
                async with asyncio.timeout(10.0):  # Increased timeout
                    success = await cls._instance.connect()
                    if not success:
                        logger.error("Failed to connect to NATS")
                        cls._instance = None
                        return None
            except asyncio.TimeoutError:
                logger.error("NATS connection timed out")
                cls._instance = None
                return None
            except Exception as e:
                logger.error(f"Failed to connect to NATS: {e}")
                cls._instance = None
                return None

            # Verify connection
            if not cls._instance._connected:
                logger.error("NATS client not properly connected")
                cls._instance = None
                return None

            return cls._instance

        except Exception as e:
            logger.error(f"Error initializing NATS client: {e}")
            if cls._instance:
                await cls._instance.close()
                cls._instance = None
            return None

    @classmethod
    def get_instance(cls) -> Optional[NatsClient]:
        if not cls._instance:
            logger.warning("NATS client not initialized")
        return cls._instance

    @classmethod
    async def update_url(cls, cloud_url: str):
        logger.info(f"Updating NATS client url to: {cloud_url}")
        if cls._instance is not None:
            # Close existing connection if it exists
            await cls._instance.close()

        if cloud_url is None:
            cls._instance = None
            return None

        cls._instance = NatsClient(cloud_url)
        try:
            async with asyncio.timeout(5.0):  # 5 second timeout
                await cls._instance.connect()
        except (asyncio.TimeoutError, Exception) as e:
            logger.error(f"Failed to connect to NATS with new URL: {e}")
            cls._instance = None
            return None

        return cls._instance


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
