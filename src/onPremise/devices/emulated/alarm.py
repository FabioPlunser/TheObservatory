import asyncio
import uuid
import logging
import aiohttp
import websockets
import platform
from edge_server_discover import EdgeServerDiscovery
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="Alarm: %(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("alarm.log")],
)

logger = logging.getLogger(__name__)


class Alarm:
    def __init__(self):
        self.alarm_id = str(uuid.uuid4())
        self.edge_server_url = None
        self.is_running = False
        self.discovery = EdgeServerDiscovery()
        self.websocket = None
        self.alarm_active = False

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
                "name": f"Alarm {self.alarm_id[:8]}",
                "capabilities": {
                    "sound": True,
                    "light": True,
                },
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.edge_server_url}/register/alarm",
                    json=registration_data,
                    timeout=5,
                ) as response:
                    if response.status == 200:
                        await response.json()
                        logger.info(f"Alarm {self.alarm_id} registered successfully")
                        return True
                    else:
                        logger.error(
                            f"Registration failed with status {response.status}"
                        )
                        return False

        except Exception as e:
            logger.error(f"Registration failed: {e}")
            return False

    async def connect_websocket(self):
        """Connect to edge server websocket"""
        if not self.edge_server_url:
            logger.error("No edge server URL available")
            return

        ws_url = self.edge_server_url.replace("http", "ws")
        full_url = f"{ws_url}/ws/alarms/{self.alarm_id}"

        try:
            self.websocket = await websockets.connect(full_url)
            logger.info(f"WebSocket connected for alarm {self.alarm_id}")

            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    if data.get("type") == "alarm_trigger":
                        await self.trigger_alarm()
                    elif data.get("type") == "alarm_stop":
                        await self.stop_alarm()
                except json.JSONDecodeError:
                    logger.error("Failed to decode WebSocket message")
                except Exception as e:
                    logger.error(f"Error processing WebSocket message: {e}")

        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
        finally:
            if self.websocket:
                await self.websocket.close()
            logger.info("WebSocket connection closed")

    async def trigger_alarm(self):
        """Activate the alarm"""
        self.alarm_active = True
        logger.info("Alarm triggered!")
        # Add hardware-specific alarm activation code here
        # For example, activate buzzer, LED, etc.

    async def stop_alarm(self):
        """Deactivate the alarm"""
        self.alarm_active = False
        logger.info("Alarm stopped")
        # Add hardware-specific alarm deactivation code here

    async def start(self):
        """Start the alarm client"""
        try:
            if not await self.discover_edge_server():
                logger.error("No edge server found")
                return

            if await self.register_with_edge():
                await self.connect_websocket()
        except Exception as e:
            logger.error(f"Error starting alarm: {e}")
        finally:
            if self.websocket:
                await self.websocket.close()


async def main():
    alarm = Alarm()
    try:
        await alarm.start()
    except KeyboardInterrupt:
        logger.info("Shutting down alarm...")
    except Exception as e:
        logger.error(f"Alarm error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
