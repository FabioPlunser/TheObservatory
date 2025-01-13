import asyncio
import uuid
import logging
import aiohttp
import websockets
import platform
import json
import os
import pygame
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
        self.reconnect_attempt = 0
        self.max_reconnect_delay = (
            60  # Maximum delay between reconnection attempts in seconds
        )
        pygame.mixer.init()
        self.sound = None
        try:
            self.sound = pygame.mixer.Sound("alarm.wav")
        except Exception as e:
            logger.error(f"Failed to load alarm sound: {e}")

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

    def calculate_reconnect_delay(self):
        """Calculate delay before next reconnection attempt using exponential backoff"""
        delay = min(2**self.reconnect_attempt, self.max_reconnect_delay)
        self.reconnect_attempt += 1
        return delay

    async def connect_websocket(self):
        """Connect to edge server websocket with automatic reconnection"""
        while True:  # Keep trying to connect
            try:
                if not self.edge_server_url:
                    if not await self.discover_edge_server():
                        delay = self.calculate_reconnect_delay()
                        logger.info(
                            f"Will attempt to rediscover server in {delay} seconds"
                        )
                        await asyncio.sleep(delay)
                        continue

                if not await self.register_with_edge():
                    delay = self.calculate_reconnect_delay()
                    logger.info(f"Will attempt to register again in {delay} seconds")
                    await asyncio.sleep(delay)
                    continue

                ws_url = self.edge_server_url.replace("http", "ws")
                full_url = f"{ws_url}/ws/alarms/{self.alarm_id}"

                async with websockets.connect(full_url) as websocket:
                    self.websocket = websocket
                    logger.info(f"WebSocket connected for alarm {self.alarm_id}")
                    self.reconnect_attempt = (
                        0  # Reset reconnection counter on successful connection
                    )

                    async for message in websocket:
                        try:
                            data = json.loads(message)
                            if data.get("active") == "true":
                                self.alarm_active = True
                                asyncio.create_task(self.trigger_alarm())
                            elif data.get("active") == "false":
                                self.alarm_active = False
                            if data.get("alive"):
                                logger.info("Alive message received")

                        except json.JSONDecodeError:
                            logger.error("Failed to decode WebSocket message")
                        except Exception as e:
                            logger.error(f"Error processing WebSocket message: {e}")

            except websockets.exceptions.ConnectionClosed:
                logger.warning(
                    "WebSocket connection closed, attempting to reconnect..."
                )
                self.websocket = None
                delay = self.calculate_reconnect_delay()
                await asyncio.sleep(delay)

            except Exception as e:
                logger.error(f"WebSocket connection error: {e}")
                self.websocket = None
                delay = self.calculate_reconnect_delay()
                await asyncio.sleep(delay)

            finally:
                if self.websocket:
                    await self.websocket.close()
                    self.websocket = None

    async def trigger_alarm(self):
        """Activate the alarm"""
        logger.info("Alarm triggered!")
        while self.alarm_active:
            if self.sound:
                self.sound.play()
            await asyncio.sleep(1)  # Non-blocking sleep

        # Stop the sound when alarm_active becomes False
        if self.sound:
            self.sound.stop()
        logger.info("Alarm stopped")

    async def stop_alarm(self):
        """Deactivate the alarm"""
        self.alarm_active = False
        logger.info("Alarm stopped")

    async def start(self):
        """Start the alarm client with automatic reconnection"""
        try:
            self.is_running = True
            await self.connect_websocket()
        except Exception as e:
            logger.error(f"Error in alarm main loop: {e}")
        finally:
            self.is_running = False
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
