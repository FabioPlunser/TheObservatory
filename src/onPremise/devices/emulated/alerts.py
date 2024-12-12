from contextlib import asynccontextmanager

import uuid
from typing import Dict

import aiohttp
import logging
import socket
import asyncio
from fastapi import FastAPI, HTTPException
from starlette.staticfiles import StaticFiles
from uvicorn import Config, Server

from edge_server_discover import EdgeServerDiscovery
import base64
from datetime import datetime
import json


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Alarm:
    def __init__(self):
        self.alarm_id = str(uuid.uuid4())
        self.edge_server_url = None
        self.local_ip = socket.gethostbyname(socket.gethostname())
        self.discovery = EdgeServerDiscovery()
        self.alarms: Dict[str, dict] = {}


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
        """Register alarm with edge server"""
        if not self.edge_server_url:
            logger.error("No edge server URL available")
            return False

        try:
            registration_data = {
                "alarm_id": self.alarm_id,
                "name": f"alarm {self.alarm_id[:8]}",
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                        f"{self.edge_server_url}/register-alarm",
                        json=registration_data,
                        timeout=5
                ) as response:
                    await response.json()
                    logger.info(f"alarm {self.alarm_id} registered successfully")
                    return True

        except Exception as e:
            logger.error(f"Registration failed: {e}")
            return False

    async def stop(self):
        """Stop the alarm"""





@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Alarm server started")
    yield
    # Shutdown
    logger.info("Alarm server stopped")

app = FastAPI(lifespan=lifespan)

@app.post("/alarm")
async def alarm(registration_data: dict):
    try:
        alarm_id = registration_data["alarm_id"]
        camera_id = registration_data["camera_id"]
        image_url = registration_data["prefined_url"]

        logger.info(f"alarm {alarm_id} on camera {camera_id} registered")
        return {"status": "success", "message": "alarm registered"}
    except Exception as e:
        logger.error(f"Registration failed: {e}")
        raise HTTPException(status_code=500, detail="Registration failed")


async def main():
    global alarm;
    alarm = Alarm()
    try:
        if not await alarm.discover_edge_server():
            logger.error("No edge server found")
            return

        if not await alarm.register_with_edge():
            logger.error("Alarm registration failed")

        logger.info("Registration completed. Starting FastAPI server...")

        # Start FastAPI server
        config = Config(app=app, host="0.0.0.0", port=8001)
        server = Server(config)
        await server.serve()

    except KeyboardInterrupt:
        logger.info("Shutting down alarm...")
    except Exception as e:
        logger.error(f"alarm error: {e}")
    finally:
        await alarm.stop()

if __name__ == "__main__":
    asyncio.run(main())