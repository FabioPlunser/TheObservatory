from fastapi import WebSocket, WebSocketDisconnect
from collections import defaultdict
from datetime import datetime
from edge_server import EdgeServer
from typing import Dict, Set
import asyncio
import time
import logging

logger = logging.getLogger("WebSocketManager")


class WebSocketManager:
    def __init__(self, edge_server: EdgeServer):
        self.active_connections: Dict[str, Set[WebSocket]] = defaultdict(set)
        self.broadcast_tasks: Dict[str, asyncio.Task] = {}
        self.frame_intervals: Dict[str, float] = {}  # Controls FPS per camera
        self.edge_server = edge_server

    async def connect(self, websocket: WebSocket, camera_id: str):
        await websocket.accept()
        self.active_connections[camera_id].add(websocket)

        # Start broadcast task for this camera if not exists
        if camera_id not in self.broadcast_tasks:
            self.frame_intervals[camera_id] = 1 / 30  # 30 FPS target
            self.broadcast_tasks[camera_id] = asyncio.create_task(
                self._broadcast_frames(camera_id)
            )

    async def disconnect(self, camera_id: str):
        """Disconnect all websockets for a camera and cleanup resources"""
        # Get all websockets for this camera
        websockets = self.active_connections[
            camera_id
        ].copy()  # Make a copy to avoid modification during iteration

        # Close each websocket
        for websocket in websockets:
            try:
                await websocket.close()
            except Exception as e:
                logger.error(f"Error closing websocket for camera {camera_id}: {e}")

        # Clear all connections for this camera
        self.active_connections[camera_id].clear()

        # If broadcast task exists, cancel and cleanup
        if camera_id in self.broadcast_tasks:
            try:
                self.broadcast_tasks[camera_id].cancel()
                # Clean up task and interval
                await self.broadcast_tasks[camera_id]
                del self.broadcast_tasks[camera_id]
                del self.frame_intervals[camera_id]
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(
                    f"Error cancelling broadcast task for camera {camera_id}: {e}"
                )

    async def _broadcast_frames(self, camera_id: str):
        """Broadcast frames to all connected clients for a camera"""
        last_frame_time = time.time()

        try:
            while True:
                current_time = time.time()
                if current_time - last_frame_time < self.frame_intervals[camera_id]:
                    await asyncio.sleep(0.001)
                    continue

                # Get new frame
                frame = await self.edge_server.get_frame(camera_id)
                if frame is None:
                    # logger.info(f"No frame for camera {camera_id}")
                    await asyncio.sleep(0.1)
                    continue

                # Update timestamp
                last_frame_time = current_time
                self.edge_server.cameras[camera_id]["last_seen"] = datetime.now()

                # Remove closed connections
                closed = set()
                for websocket in self.active_connections[camera_id]:
                    try:
                        await websocket.send_bytes(b"")  # Ping to check connection
                    except Exception:
                        closed.add(websocket)

                for websocket in closed:
                    self.active_connections[camera_id].discard(websocket)

                # If no connections left, stop broadcasting
                if not self.active_connections[camera_id]:
                    break

                # Broadcast frame to all connected clients
                send_tasks = []
                for websocket in self.active_connections[camera_id]:
                    try:
                        send_tasks.append(websocket.send_bytes(frame))
                    except Exception:
                        continue

                if send_tasks:
                    await asyncio.gather(*send_tasks, return_exceptions=True)

        except asyncio.CancelledError:
            logger.info(f"Broadcast task cancelled for camera {camera_id}")
        except Exception as e:
            logger.error(f"Error in broadcast task for camera {camera_id}: {e}")
