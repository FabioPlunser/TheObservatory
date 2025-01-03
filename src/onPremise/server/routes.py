from fastapi import APIRouter, HTTPException, Query, Request
from datetime import datetime
from edge_server import EdgeServer
from database import Database

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
                    camera_id, registration_data["name"], "registered"
                )
                await self.edge_server.start_rtsp_reader(camera_id, rtsp_url)

                logger.info(f"Camera {camera_id} registered with RTSP URL: {rtsp_url}")
                return {"status": "success", "message": "Camera registered"}
            except Exception as e:
                logger.error(f"Registration failed: {e}")
                raise HTTPException(status_code=500, detail="Registration failed")

        # ----------------------------------------------------------------------------
        # ----------------------------------------------------------------------------
        @router.post("/webrtc/{camera_id}/offer")
        async def webrtc_offer(camera_id: str, offer: dict):
            try:
                answer = await self.edge_server.handle_webrtc_offer(camera_id, offer)
                return answer
            except Exception as e:
                logger.error(f"WebRTC offer error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # ----------------------------------------------------------------------------
        # ----------------------------------------------------------------------------
        @router.post("/webrtc/{camera_id}/close")
        async def webrtc_close(camera_id: str):
            await self.edge_server.close_peer_connection(camera_id)
            return {"status": "success"}

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
            await self.db.delete_camera(camera_id)
            return {"status": "success", "message": "Camera deleted"}

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
