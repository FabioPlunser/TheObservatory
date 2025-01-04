from fastapi import (
    APIRouter,
    HTTPException,
    Query,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from datetime import datetime
from edge_server import EdgeServer
from database import Database

import asyncio
import logging

logger = logging.getLogger(__name__)


class Router:
    def __init__(self, edge_server: EdgeServer, db: Database):
        self.edge_server = edge_server
        self.db = db

    def create_routes(self) -> APIRouter:
        router = APIRouter()

        # ----------------------------------------------------------------------------
        # ----------------------------------------------------------------------------
        @router.post("/register/camera")
        async def register_camera(registration_data: dict):
            try:
                camera_id = registration_data["camera_id"]
                rtsp_url = registration_data["rtsp_url"]
                if not rtsp_url:
                    raise HTTPException(status_code=400, detail="RTSP URL is required")

                existing_camera = await self.db.get_camera(camera_id)
                if existing_camera:
                    logger.info(
                        f"Camera {camera_id} already registered, updating status"
                    )
                    self.edge_server.cameras[camera_id] = {
                        "capabilities": registration_data["capabilities"],
                        "rstp_url": registration_data["rtsp_url"],
                        "last_seen": datetime.now(),
                        "status": "registered",
                    }
                    return {"status": "success", "message": "Camera reconnected"}

                self.edge_server.cameras[camera_id] = {
                    "capabilities": registration_data["capabilities"],
                    "rstp_url": registration_data["rtsp_url"],
                    "last_seen": datetime.now(),
                    "status": "registered",
                }

                await self.db.create_camera(
                    camera_id, registration_data["name"], rtsp_url, "registered"
                )
                await self.edge_server.start_camera_stream(camera_id, rtsp_url)

                logger.info(f"Camera {camera_id} registered with RTSP URL: {rtsp_url}")
                return {"status": "success", "message": "Camera registered"}
            except Exception as e:
                logger.error(f"Registration failed: {e}")
                raise HTTPException(status_code=500, detail="Registration failed")

        # ----------------------------------------------------------------------------
        # ----------------------------------------------------------------------------
        @router.websocket("/ws/camera/{camera_id}")
        async def camera_stream(websocket: WebSocket, camera_id: str):
            if camera_id not in self.edge_server.cameras:
                await websocket.close(code=4004, reason="Camera not found")
                return

            try:
                await websocket.accept()

                while True:
                    try:
                        # Check for client messages (non-blocking)
                        try:
                            data = await asyncio.wait_for(
                                websocket.receive_text(), timeout=0.001
                            )
                            if data == "close":
                                break
                        except asyncio.TimeoutError:
                            pass

                        # Get and send frame
                        frame_bytes = await self.edge_server.get_frame(camera_id)
                        if frame_bytes is not None:
                            await websocket.send_bytes(frame_bytes)

                        # Control frame rate (approximately 30 FPS)
                        await asyncio.sleep(0.033)

                    except WebSocketDisconnect:
                        break
                    except Exception as e:
                        logger.error(f"Error in WebSocket loop: {e}")
                        break

            except Exception as e:
                logger.error(f"WebSocket error: {e}")
            finally:
                logger.info(f"WebSocket connection closed for camera {camera_id}")

        # ----------------------------------------------------------------------------
        # ----------------------------------------------------------------------------
        @router.post("/register/alarm")
        async def register_alarm(registration_data: dict):
            try:
                alarm_id = registration_data["alarm_id"]

                existing_alarm = await self.db.get_alarm_device(alarm_id)
                if existing_alarm:
                    logger.info(f"alarm {alarm_id} already registered, updating status")
                    self.edge_server.alarms[alarm_id] = {
                        "registration_time": datetime.now(),
                        "status": "registered",
                    }
                    return {"status": "success", "message": "alarm reconnected"}

                self.edge_server.alarms[alarm_id] = {
                    "registration_time": datetime.now(),
                    "status": "registered",
                }

                await self.db.create_alarm_device(
                    alarm_id, registration_data["name"], "registered"
                )

                logger.info(f"alarm {alarm_id} registered")
                return {"status": "success", "message": "alarm registered"}
            except Exception as e:
                logger.error(f"Registration failed: {e}")
                raise HTTPException(status_code=500, detail="Registration failed")

        # ----------------------------------------------------------------------------
        # ----------------------------------------------------------------------------
        @router.get("/cameras")
        async def get_cameras():
            """Get list of connected cameras"""
            return {
                "cameras": [
                    {
                        "camera_id": camera_id,
                        "capabilities": data["capabilities"],
                        "last_seen": data["last_seen"].isoformat(),
                        "status": data["status"],
                    }
                    for camera_id, data in self.edge_server.cameras.items()
                ]
            }

        # ----------------------------------------------------------------------------
        # ----------------------------------------------------------------------------
        @router.get("/api/get-cameras")
        async def get_cameras_api():
            return await self.db.get_cameras()

        # ----------------------------------------------------------------------------
        # ----------------------------------------------------------------------------
        @router.get("/api/get-alarms")
        async def get_alarms():
            return await self.db.get_alarms()

        # ----------------------------------------------------------------------------
        # ----------------------------------------------------------------------------
        @router.get("/api/get-alarm-devices")
        async def get_alarm_devices_api():
            return await self.db.get_alarm_devices()

        # ----------------------------------------------------------------------------
        # ----------------------------------------------------------------------------
        @router.post("/api/delete-camera/{camera_id}")
        async def delete_camera(camera_id: str):
            """Delete a camera and clean up all associated resources"""
            try:
                logger.info(f"Starting deletion of camera {camera_id}")

                # First stop the video processor if running
                if camera_id in self.edge_server.processors:
                    logger.info(f"Stopping video processor for camera {camera_id}")
                    await self.edge_server.stop_camera_stream(camera_id)

                # Remove from edge server cameras dict
                if camera_id in self.edge_server.cameras:
                    logger.info(f"Removing camera {camera_id} from edge server")
                    del self.edge_server.cameras[camera_id]

                # Finally remove from database
                logger.info(f"Removing camera {camera_id} from database")
                await self.db.delete_camera(camera_id)

                return {"status": "success", "message": "Camera deleted"}

            except Exception as e:
                error_msg = f"Error deleting camera {camera_id}: {str(e)}"
                logger.error(error_msg)
                raise HTTPException(status_code=500, detail=error_msg)

        # ----------------------------------------------------------------------------
        # ----------------------------------------------------------------------------
        @router.post("/api/delete-alarm_device/{alarm_device_id}")
        async def delete_alarm_device(alarm_device_id: str):
            await self.db.delete_camera(alarm_device_id)
            return {"status": "success", "message": "Alarm device deleted"}

        # ----------------------------------------------------------------------------
        # ----------------------------------------------------------------------------
        @router.get("/api/get-rooms")
        async def get_rooms():
            return await self.db.get_rooms()

        # ----------------------------------------------------------------------------
        # ----------------------------------------------------------------------------
        @router.post("/api/create-room")
        async def create_room(room_name: str = Query(None)):
            await self.db.create_room(room_name)
            return {"status": "success", "message": "Room created"}

        # ----------------------------------------------------------------------------
        # ----------------------------------------------------------------------------
        @router.put("/api/camera-to-room")
        async def assign_camera_room(
            camera_id: str = Query(None), room_id: str = Query(None)
        ):
            await self.db.assign_camera_room(camera_id, room_id)
            return {"status": "success", "message": "Camera assigned to room"}

        # ----------------------------------------------------------------------------
        # ----------------------------------------------------------------------------
        @router.put("/api/alarm-device-to-room")
        async def assign_alarm_device_room(
            alarm_device_id: str = Query(None), room_id: str = Query(None)
        ):
            await self.db.assign_alarm_device_room(alarm_device_id, room_id)
            return {"status": "success", "message": "Camera assigned to room"}

        return router
