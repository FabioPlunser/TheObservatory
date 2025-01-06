from fastapi import (
    APIRouter,
    HTTPException,
    Query,
    Request,
    WebSocket,
    WebSocketDisconnect,
    UploadFile,
    File,
)
from typing import List
from datetime import datetime
from edge_server import EdgeServer
from database import Database
from nats_client import SharedNatsClient, Commands
from logging_config import setup_logger

import asyncio
import aiohttp
import logging
import uuid
import json
import os
import ipaddress

setup_logger()
logger = logging.getLogger("Router")


class Router:
    def __init__(self, edge_server: EdgeServer, db: Database):
        self.edge_server = edge_server
        self.db = db
        self._retry_running = True

    async def get_nats_client(self):
        """Get or wait for NATS client"""
        retry_delay = 5
        while self._retry_running:
            nats_client = SharedNatsClient.get_instance()
            if nats_client and nats_client._connected:
                return nats_client
            await asyncio.sleep(retry_delay)
        return None

    def create_routes(self) -> APIRouter:
        router = APIRouter()

        # ----------------------------------------------------------------------------
        # ----------------------------------------------------------------------------
        @router.get("/api/get-company")
        async def get_company():
            company = await self.db.get_company()
            return {
                "company": company,
            }

        # ----------------------------------------------------------------------------
        # ----------------------------------------------------------------------------
        @router.post("/api/update-cloud-url")
        async def update_cloud_url(cloud_url: str):
            """Update the cloud URL for the NATS client"""

            def is_ip_address(ip_str):
                try:
                    ipaddress.ip_address(ip_str)
                    return True
                except ValueError:
                    return False

            try:
                if not cloud_url:
                    raise HTTPException(status_code=400, detail="Cloud URL is required")

                if "http" in cloud_url:
                    cloud_url = cloud_url.replace("http", "nats")
                    cloud_url = cloud_url + ":4222"
                if "https" in cloud_url:
                    cloud_url = cloud_url.replace("https", "nats")
                    cloud_url = cloud_url + ":4222"
                elif is_ip_address(cloud_url):
                    cloud_url = "nats://" + cloud_url + ":4222"

                await self.db.update_cloud_url(cloud_url)

                await SharedNatsClient.update_url(cloud_url)

                return {"status": "success", "message": "Cloud URL updated"}

            except Exception as e:
                logger.error(f"Error updating cloud URL: {e}")
                raise HTTPException(status_code=500, detail="Error updating cloud URL")

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
                        self.edge_server.cameras[camera_id][
                            "last_seen"
                        ] = datetime.now()

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

                existing_alarm = await self.db.get_alarm(alarm_id)
                if existing_alarm:
                    logger.info(f"alarm {alarm_id} already registered, updating status")
                    self.edge_server.alarms[alarm_id] = {
                        "alarm_data": existing_alarm,
                        "websocket": None,
                        "last_seen": datetime.now(),
                        "status": "registered",
                    }
                    return {"status": "success", "message": "alarm reconnected"}

                alarm = await self.db.create_alarm(alarm_id, "registered")

                self.edge_server.alarms[alarm_id] = {
                    "alarm_data": alarm,
                    "websocket": None,
                    "last_seen": datetime.now(),
                    "status": "registered",
                }

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
            return await self.db.get_all_alarms()

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

        # ----------------------------------------------------------------------------
        # ----------------------------------------------------------------------------
        @router.post("/api/faces/known/upload")
        async def upload_known_face(files: List[UploadFile] = File(...)):
            """Upload known face images to the cloud storage."""
            try:
                results = []
                for file in files:
                    try:
                        image_data = await file.read()
                        face_id = str(uuid.uuid4())

                        nats_client = await self.get_nats_client()
                        if not nats_client:
                            raise HTTPException(
                                status_code=500, detail="NATS client not available"
                            )

                        # Get company ID first
                        company_id = await self.db.get_company_id()
                        if not company_id:
                            raise HTTPException(
                                status_code=400, detail="Company ID not found"
                            )

                        # Get presigned URL from NATS
                        response = await nats_client.send_message_with_reply(
                            Commands.GET_PRESIGNED_UPLOAD_KNOWN_FACE_URL.value,
                            {
                                "company_id": company_id,
                                "face_id": face_id,
                            },
                        )

                        if not response:
                            raise HTTPException(
                                status_code=500, detail="No response from cloud server"
                            )

                        if not response.get("success"):
                            raise HTTPException(
                                status_code=500,
                                detail=f"Error getting upload URL: {response.get('error', 'Unknown error')}",
                            )

                        # Extract URL and any headers we need to include
                        upload_url = response.get("url")
                        if not upload_url:
                            raise HTTPException(
                                status_code=500, detail="No upload URL in response"
                            )

                        # Upload image using presigned URL
                        async with aiohttp.ClientSession() as session:
                            # Make sure to set the correct content type
                            headers = {
                                "Content-Type": "*",
                            }

                            async with session.put(
                                upload_url,
                                data=image_data,
                                headers=headers,
                                timeout=30,  # Increase timeout
                            ) as resp:
                                if resp.status != 200:
                                    error_text = await resp.text()
                                    logger.error(f"S3 upload failed: {error_text}")
                                    raise HTTPException(
                                        status_code=500,
                                        detail=f"Error uploading to S3: {error_text}",
                                    )

                        # Create local database entry
                        s3_key = response.get("s3_key")
                        await self.db.create_known_face(face_id, s3_key=s3_key)

                        results.append(
                            {
                                "face_id": face_id,
                                "s3_key": s3_key,
                                "success": True,
                            }
                        )

                    except Exception as e:
                        logger.error(f"Error processing file {file.filename}: {str(e)}")
                        results.append(
                            {
                                "filename": file.filename,
                                "success": False,
                                "error": str(e),
                            }
                        )

                # Return overall results
                success = any(result["success"] for result in results)
                return {"success": success, "uploaded_files": results}

            except Exception as e:
                logger.error(f"Error in upload_known_face: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # ----------------------------------------------------------------------------
        # ----------------------------------------------------------------------------
        @router.get("/api/faces/known/all")
        async def get_all_known_faces():
            """Download a known face image from the cloud storage."""

            nats_client = await self.get_nats_client()
            if nats_client is None:
                raise HTTPException(
                    status_code=500, detail="NATS client not initialized"
                )

            response = await nats_client.send_message_with_reply(
                Commands.GET_PRESIGNED_DOWNLOAD_ALL_KNOWN_FACES.value,
                {
                    "company_id": await self.db.get_company_id(),
                },
            )

            if not response.get("success"):
                raise HTTPException(status_code=500, detail="Error downloading face")

            return response

        # ----------------------------------------------------------------------------
        # ----------------------------------------------------------------------------
        @router.post("/api/faces/known/delete")
        async def delete_known_face(key: str):
            """Delete a known face image from the cloud storage."""
            nats_client = await self.get_nats_client()
            if nats_client is None:
                raise HTTPException(
                    status_code=500, detail="NATS client not initialized"
                )

            response = await nats_client.send_message_with_reply(
                Commands.DELETE_KNOWN_FACE.value,
                {
                    "key": key,
                },
            )

            if not response.get("success"):
                raise HTTPException(status_code=500, detail="Error deleting face")

            return response

        # ----------------------------------------------------------------------------
        # ----------------------------------------------------------------------------
        @router.post("/api/faces/unknown/delete/all")
        async def delete_all_unknown_faces():
            """Delete all unknown face images from the cloud storage."""
            nats_client = await self.get_nats_client()
            if nats_client is None:
                raise HTTPException(
                    status_code=500, detail="NATS client not initialized"
                )

            response = await nats_client.send_message_with_reply(
                Commands.DELETE_UNKNOWN_FACE.value,
                {
                    "company_id": self.db.get_company_id(),
                },
            )

            if not response.get("success"):
                raise HTTPException(status_code=500, detail="Error deleting faces")

            return response

        # ----------------------------------------------------------------------------
        # ----------------------------------------------------------------------------
        @router.websocket("/ws/alarms/{alarm_id}")
        async def alarm_websocket(websocket: WebSocket, alarm_id: str):
            if alarm_id not in self.edge_server.alarms:
                await websocket.close(code=4004, reason="Alarm not found")
                return

            try:
                await websocket.accept()

                self.edge_server.alarms[alarm_id]["websocket"] = websocket
                while True:
                    try:
                        await asyncio.sleep(0.1)  # Keep connection alive
                    except WebSocketDisconnect:
                        break
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
            finally:
                logger.info(f"WebSocket connection closed for alarm {alarm_id}")

        return router
