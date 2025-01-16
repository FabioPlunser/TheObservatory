import cv2
import numpy as np
import logging
import asyncio
import aiohttp
import uuid

from logging_config import setup_logger
from nats_client import Commands, SharedNatsClient
from person_tracker import TrackedPerson
from database import Database
from datetime import datetime, timedelta

setup_logger()
logger = logging.getLogger("FaceRecognitionHandler")


class AsyncFaceRecognitionHandler:
    def __init__(
        self, company_id: str, camera_id: str, db: Database
    ):
        self.nats_client = SharedNatsClient.get_instance()
        self.company_id = company_id
        self.camera_id = camera_id
        self.db = db
        self.recognition_queue = asyncio.Queue()
        self.running = True
        self.task = None
        self.processed_faces = {}
        self.last_cleanup = datetime.now()
        self.cleanup_interval = timedelta(minutes=5)

        logger.info("Face recognition handler initialized")

    async def start(self):
        """Start the recognition processing loop"""
        if self.task is not None:
            logger.warning("Face recognition handler already started")
            return

        try:
            self.task = asyncio.create_task(self._process_recognition_queue())
            logger.info(f"Face recognition handler started for camera {self.camera_id}")
        except Exception as e:
            logger.error(f"Error starting face recognition handler: {e}")
            self.task = None
            raise

    async def stop(self):
        """Stop the recognition processing"""
        logger.info("Stopping face recognition handler")
        self.running = False
        if self.task:
            try:
                await self.task
            except Exception as e:
                logger.error(f"Error stopping face recognition handler: {e}")
            finally:
                self.task = None
        logger.info("Face recognition handler stopped")

    async def _process_recognition_queue(self):
        """Process queued face recognition requests"""
        logger.info(f"Starting recognition queue processor for camera {self.camera_id}")
        while self.running:
            try:
                # Get next face from queue with timeout
                try:
                    track, face_image = await asyncio.wait_for(
                        self.recognition_queue.get(), timeout=1.0
                    )
                    logger.info(f"Got face for track {track.track_id} from queue")
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"Error getting from recognition queue: {e}")
                    continue

                # Process face recognition
                try:
                    await self._handle_face_recognition(track, face_image)
                except Exception as e:
                    logger.error(f"Error in face recognition: {e}")
                    continue

            except Exception as e:
                logger.error(f"Error in recognition queue processing: {e}")
                await asyncio.sleep(1)

    async def cleanup_processed_faces(self):
        """Clean up processed faces based on configured timeout"""
        try:
            current_time = datetime.now()
            if current_time - self.last_cleanup < self.cleanup_interval:
                return

            self.last_cleanup = current_time
            config = await self.db.get_tracking_config()
            timeout = timedelta(seconds=config["face_recognition_timeout"])

            expired_faces = []
            for face_id, timestamp in self.processed_faces.items():
                if current_time - timestamp > timeout:
                    expired_faces.append(face_id)

            for face_id in expired_faces:
                del self.processed_faces[face_id]

            if expired_faces:
                logger.info(f"Cleaned up {len(expired_faces)} expired face records")

        except Exception as e:
            logger.error(f"Error in face cleanup: {e}")

    async def should_process_face(self, track: TrackedPerson) -> bool:
        """Check if face should be processed based on timeout"""
        try:
            # Skip if no face image
            if track.face_image is None:
                return False

            # Always process if no face ID yet
            if track.face_id is None:
                return True

            # Check timeout for existing face IDs
            if track.face_id in self.processed_faces:
                config = await self.db.get_tracking_config()
                timeout = timedelta(seconds=config["face_recognition_timeout"])
                if datetime.now() - self.processed_faces[track.face_id] < timeout:
                    return False

            return True

        except Exception as e:
            logger.error(f"Error checking face processing: {e}")
            return False

    async def queue_recognition(self, track: TrackedPerson, face_image: np.ndarray):
        """Queue face for recognition with timeout check and error handling"""
        try:
            logger.debug(f"Checking if should process face for track {track.track_id}")

            if face_image is None:
                logger.debug(f"Skipping track {track.track_id}: No face image")
                return

            if track.recognition_status not in ["pending", "unknown"]:
                logger.debug(
                    f"Skipping track {track.track_id}: Status is {track.recognition_status}"
                )
                return

            should_process = await self.should_process_face(track)
            if not should_process:
                logger.debug(
                    f"Skipping track {track.track_id}: Already processed recently"
                )
                return

            logger.debug(f"Queueing face recognition for track {track.track_id}")
            await self.recognition_queue.put((track, face_image))
            logger.debug(
                f"Successfully queued face recognition for track {track.track_id}"
            )

        except asyncio.QueueFull:
            logger.error(
                f"Recognition queue full when trying to queue track {track.track_id}"
            )
        except Exception as e:
            logger.error(
                f"Error queueing recognition for track {track.track_id}: {str(e)}"
            )
            import traceback

            logger.error(traceback.format_exc())

    async def _handle_face_recognition(
        self, track: TrackedPerson, face_image: np.ndarray
    ):
        """Handle face recognition with detailed logging"""
        try:
            if face_image is None:
                logger.error("Face image is None")
                return

            # Convert and compress face image
            try:
                _, img_encoded = cv2.imencode(
                    ".jpg", face_image, [cv2.IMWRITE_JPEG_QUALITY, 85]
                )
                img_bytes = img_encoded.tobytes()
            except Exception as e:
                logger.error(f"Error encoding face image: {e}")
                return

            # Generate unique face ID if none exists
            if track.face_id is None:
                track.face_id = str(uuid.uuid4())
                logger.info(
                    f"Generated new face ID {track.face_id} for track {track.track_id}"
                )

            # Update status to in_progress
            track.recognition_status = "in_progress"
            logger.info(
                f"Starting recognition for track {track.track_id} (face_id: {track.face_id})"
            )

            # Get presigned URL
            try:
                response = await self.nats_client.send_message_with_reply(
                    Commands.GET_PRESIGNED_UPLOAD_UNKNOWN_FACE_URL.value,
                    {
                        "company_id": self.company_id,
                        "face_id": track.face_id,
                    },
                )

                if not response or not response.get("success"):
                    logger.error(f"Failed to get presigned URL: {response}")
                    track.recognition_status = "pending"
                    return

                logger.info(f"Got presigned URL for face_id {track.face_id}")
            except Exception as e:
                logger.error(f"Error getting presigned URL: {e}")
                track.recognition_status = "pending"
                return

            # Upload face image
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.put(
                        response["url"],
                        data=img_bytes,
                        headers={"Content-Type": "*"},
                        timeout=30,
                    ) as resp:
                        if resp.status != 200:
                            error_text = await resp.text()
                            logger.error(f"Failed to upload face image: {error_text}")
                            track.recognition_status = "pending"
                            return

                        logger.info(
                            f"Successfully uploaded face image for face_id {track.face_id}"
                        )
            except Exception as e:
                logger.error(f"Error uploading face image: {e}")
                track.recognition_status = "pending"
                return

            # Execute recognition
            try:
                recognition_response = await self.nats_client.send_message_with_reply(
                    Commands.EXECUTE_RECOGNITION.value,
                    {
                        "company_id": self.company_id,
                        "camera_id": self.camera_id,
                        "face_id": track.face_id,
                        "track_id": track.track_id,
                    },
                )

                logger.info(
                    f"Recognition response for face_id {track.face_id}: {recognition_response}"
                )

                if recognition_response and recognition_response.get("success"):
                    track.face_id = recognition_response.get("face_id")
                    track.recognition_status = recognition_response.get("status")
                    self.processed_faces[track.face_id] = datetime.now()

                    # Update database
                    if track.global_id:  # Only update if we have a global ID
                        await self.db.update_person_track(
                            global_id=track.global_id,
                            camera_id=self.camera_id,
                            face_id=track.face_id,
                            recognition_status=track.recognition_status,
                        )
                else:
                    track.recognition_status = "unknown"
                    logger.error(f"Recognition failed: {recognition_response}")

            except Exception as e:
                logger.error(f"Error in recognition execution: {e}")
                track.recognition_status = "pending"

        except Exception as e:
            logger.error(f"Error in face recognition: {e}")
            track.recognition_status = "pending"
