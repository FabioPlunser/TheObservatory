import numpy as np
import cv2
import time
import colorsys
import torch
import logging
import aiohttp
import mediapipe as mp
import uuid
import os
import asyncio

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from logging_config import setup_logger
from nats_client import NatsClient, Commands
from reid_implementation import Reid
from datetime import datetime, Database


setup_logger()
logger = logging.getLogger("PersonTracker")


@dataclass
class TrackedPerson:
    track_id: int
    bbox: np.ndarray  # [x, y, width, height]
    face_id: Optional[str] = None
    face_image: Optional[np.ndarray] = None
    last_update: float = field(default_factory=time.time)
    recognition_status: str = "pending"  # pending, in_progress, recognized, unknown, tracked
    color: Tuple[int, int, int] = field(default=(0, 255, 0))
    global_id: Optional[str] = None

    def update_position(self, bbox: np.ndarray, current_time: float):
        self.bbox = bbox
        self.last_update = current_time

class OptimizedPersonTracker:
    def __init__(
        self, 
        company_id: str, 
        camera_id: str, 
        device: Optional[torch.device] = None,
        max_image_dimension: int = 1920,
        stale_track_threshold: float = 2.0,
        face_cache_timeout: float = 1.0
    ):
        self.tracked_persons: Dict[int, TrackedPerson] = {}
        self.color_map: Dict[int, Tuple[int, int, int]] = {}
        self.device = device
        self.company_id = company_id
        self.camera_id = camera_id
        self.max_image_dimension = max_image_dimension
        self.stale_track_threshold = stale_track_threshold
        self.face_cache_timeout = face_cache_timeout

        # Initialize components
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self.mp_face_detection = mp.solutions.face_detection
        self.face_cache: Dict[int, Tuple[np.ndarray, float]] = {}
        self.processing_times = deque(maxlen=100)

        self.cross_camera_tracker: Optional[Reid] = None

        self.company_id = company_id
        self.camera_id = camera_id

        self.processed_faces = {}
        self.db = Database()

    def _generate_color(self, track_id: int) -> Tuple[int, int, int]:
        """Generate a unique color using golden ratio"""
        golden_ratio = 0.618033988749895
        h = (track_id * golden_ratio) % 1.0
        s = 0.9
        v = 0.95
        rgb = colorsys.hsv_to_rgb(h, s, v)
        return tuple(int(255 * x) for x in reversed(rgb))

    def _get_color(self, track_id: int) -> Tuple[int, int, int]:
        """Get or create a color for a track ID"""
        if track_id not in self.color_map:
            self.color_map[track_id] = self._generate_color(track_id)
        return self.color_map[track_id]

    def update(
        self,
        frame: np.ndarray,
        yolo_results,
        current_time: Optional[float] = None
    ) -> Tuple[List[TrackedPerson], List[TrackedPerson]]:
        """Update tracker with new detections"""
        if current_time is None:
            current_time = time.time()

        start_time = time.time()
        updated_tracks = []
        new_tracks = []

        try:
            for result in yolo_results:
                boxes = result.boxes
                track_ids = boxes.id.cpu().numpy() if boxes.id is not None else None

                if track_ids is None:
                    continue

                classes = boxes.cls.cpu().numpy()
                xyxys = boxes.xyxy.cpu().numpy()

                for xyxy, class_id, track_id in zip(xyxys, classes, track_ids):
                    if int(class_id) != 0:
                        continue

                    track_id = int(track_id)
                    x1, y1, x2, y2 = xyxy
                    w = x2 - x1
                    h = y2 - y1
                    bbox = np.array([x1, y1, w, h])

                    # Call cross-camera reid
                    if self.cross_camera_tracker:
                        global_ids = self.cross_camera_tracker.update(
                            camera_id=self.camera_id, person_crops=person_crops
                        )
                        # Assign global IDs
                        for t_id, g_id in global_ids.items():
                            if t_id in self.tracked_persons:
                                self.tracked_persons[t_id].global_id = g_id

                    if track_id in self.tracked_persons:
                        person = self.tracked_persons[track_id]
                        person.update_position(bbox, current_time)
                        updated_tracks.append(person)
                    else:
                        color = self._get_color(track_id)
                        person = TrackedPerson(
                            track_id=track_id,
                            bbox=bbox,
                            last_update=current_time,
                            color=color,
                        )
                        self.tracked_persons[track_id] = person
                        new_tracks.append(person)

            self._cleanup_stale_tracks(current_time)
            self.processing_times.append(time.time() - start_time)

            # for new_track in new_tracks:
            #     asyncio.run(self._handle_face_recognition(new_track, frame))

            # Handle cross-camera tracking
            person_crops = []
            for track_id, person in self.tracked_persons.items():
                x, y, w, h = map(int, person.bbox)
                crop = frame[y:y + h, x:x + w]
                person_crops.append((track_id, crop))

        except Exception as e:
            logger.error(f"Error updating tracker: {e}")

        return updated_tracks, new_tracks

    def _cleanup_stale_tracks(self, current_time: float):
        """Remove old tracks and clean up resources"""
        stale_tracks = [
            track_id for track_id, person in self.tracked_persons.items()
            if current_time - person.last_update > self.stale_track_threshold
        ]

        for track_id in stale_tracks:
            del self.tracked_persons[track_id]
            if track_id in self.face_cache:
                del self.face_cache[track_id]

    def get_statistics(self) -> Dict:
        """Get performance statistics"""
        if not self.processing_times:
            return {}

        return {
            "avg_processing_time": sum(self.processing_times) / len(self.processing_times),
            "min_processing_time": min(self.processing_times),
            "max_processing_time": max(self.processing_times),
            "active_tracks": len(self.tracked_persons),
            "tracked_faces": len(self.processed_faces),
        }

    def cleanup(self):
        """Clean up resources"""
        self.thread_pool.shutdown(wait=False)
        self.tracked_persons.clear()
        self.face_cache.clear()
        self.processed_faces.clear()

    async def validate_and_encode_face_image(self, face_image: np.ndarray) -> Optional[bytes]:
        """Validate and encode face image according to AWS Rekognition requirements"""
        try:
            if face_image is None:
                logger.error("Input face image is None")
                return None

            # Check image dimensions
            height, width = face_image.shape[:2]
            if height < 80 or width < 80:
                logger.error(f"Image too small: {width}x{height}, minimum 80x80 required")
                return None
                
            # Resize if image is too large
            if height > self.max_image_dimension or width > self.max_image_dimension:
                scale = self.max_image_dimension / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                face_image = cv2.resize(face_image, (new_width, new_height))
                
            # Encode image with quality check
            _, img_encoded = cv2.imencode('.jpg', face_image, [
                cv2.IMWRITE_JPEG_QUALITY, 85
            ])
            
            if img_encoded is None:
                logger.error("Failed to encode image")
                return None
                
            img_bytes = img_encoded.tobytes()
            
            # Check file size (5MB limit for AWS Rekognition)
            if len(img_bytes) > 5 * 1024 * 1024:
                logger.error("Encoded image exceeds 5MB limit")
                return None
                
            return img_bytes
            
        except Exception as e:
            logger.error(f"Error validating/encoding face image: {e}")
            return None

    async def upload_face_image(
        self,
        session: aiohttp.ClientSession,
        url: str,
        img_bytes: bytes,
        timeout: int = 60
    ) -> None:
        """Upload face image to S3 using presigned URL"""
        try:
            logger.info(f"Uploading to URL: {url}")

            async with session.put(
                url,
                data=img_bytes,
                headers={"Content-Type": "*"},
                skip_auto_headers=['Content-Type'],
                timeout=30
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(f"Upload failed with status {resp.status}")
                    logger.error(f"Response body: {error_text}")
                    logger.error(f"Final URL: {str(resp.url)}")
                    raise Exception(f"Upload failed with status {resp.status}: {error_text}")

                logger.info(f"Successfully uploaded image. Response status: {resp.status}")

        except asyncio.TimeoutError:
            logger.error(f"Upload timed out after {timeout}s")
            raise
        except Exception as e:
            logger.error(f"Upload failed: {str(e)}")
            raise

    async def _handle_face_recognition(self, track: TrackedPerson, face_image: np.ndarray):
        """Handle face recognition with improved error handling"""
        logger.info(f"Handling face recognition for track {track.track_id}")
        nats_client = None
        
        try:
            # Validate and encode image
            img_bytes = await self.validate_and_encode_face_image(face_image)
            if img_bytes is None:
                logger.error(f"Failed to validate/encode face image for track {track.track_id}")
                track.recognition_status = "pending"
                return

            # Generate unique face ID if none exists
            if track.face_id is None:
                track.face_id = str(uuid.uuid4())
                logger.info(f"Generated new face ID {track.face_id} for track {track.track_id}")

            # Update status to in_progress
            track.recognition_status = "in_progress"
            
            # Get NATS connection
            nats_url = await self.db.get_cloud_url()
            nats_client = NatsClient(nats_url)
            await nats_client.connect()

            # Get presigned URL
            response = await nats_client.send_message_with_reply(
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

            presigned_url = response.get("url")
            if not presigned_url:
                logger.error("No presigned URL in response")
                track.recognition_status = "pending"
                return

            logger.info(f"Got presigned URL for face_id {track.face_id}")

            # Upload image
            async with aiohttp.ClientSession() as session:
                for attempt in range(3):  # Try up to 3 times
                    try:
                        await self.upload_face_image(session, presigned_url, img_bytes)
                        logger.info(f"Successfully uploaded image for face_id {track.face_id}")
                        break
                    except Exception as e:
                        if attempt == 2:  # Last attempt
                            logger.error(f"All upload attempts failed for face_id {track.face_id}")
                            track.recognition_status = "pending"
                            return
                        logger.warning(f"Upload attempt {attempt + 1} failed: {str(e)}")
                        await asyncio.sleep(1)  # Wait before retry

            # Execute recognition
            await nats_client.send_message(
                Commands.EXECUTE_RECOGNITION.value,
                {
                    "company_id": self.company_id,
                    "camera_id": self.camera_id,
                    "face_id": track.face_id,
                    "track_id": track.track_id,
                },
            )

            track.recognition_status = "tracked"
            self.processed_faces[track.face_id] = datetime.now()

            # Update database if global ID exists
            if track.global_id:
                try:
                    await self.db.update_person_track(
                        global_id=track.global_id,
                        camera_id=self.camera_id,
                        face_id=track.face_id,
                        recognition_status=track.recognition_status,
                    )
                except Exception as e:
                    logger.error(f"Failed to update database: {str(e)}")
            else:
                track.recognition_status = "unknown"

        except Exception as e:
            logger.error(f"Error in face recognition: {str(e)}")
            track.recognition_status = "pending"
        finally:
            if nats_client:
                try:
                    await nats_client.close()
                except Exception as e:
                    logger.error(f"Error closing NATS client: {str(e)}")