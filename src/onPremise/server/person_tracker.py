import numpy as np
import cv2
import time
import colorsys
import torch
import threading
import queue
import logging
import aiohttp
import mediapipe as mp
import asyncio
import uuid

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from logging_config import setup_logger
from nats_client import SharedNatsClient, Commands
from reid_implementation import OptimizedCrossCameraTracker
from datetime import datetime
from database import Database


setup_logger()
logger = logging.getLogger("PersonTracker")


@dataclass
class TrackedPerson:
    track_id: int
    bbox: np.ndarray  # [x, y, width, height]
    face_id: Optional[str] = None
    face_image: Optional[np.ndarray] = None
    last_update: float = field(default_factory=time.time)
    recognition_status: str = "pending"  # pending, in_progress, recognized, unknown
    color: Tuple[int, int, int] = field(default=(0, 255, 0))
    global_id: Optional[str] = None

    def update_position(self, bbox: np.ndarray, current_time: float):
        self.bbox = bbox
        self.last_update = current_time


class OptimizedPersonTracker:
    def __init__(
        self, company_id: str, camera_id: str, device: Optional[torch.device] = None
    ):
        self.tracked_persons: Dict[int, TrackedPerson] = {}
        self.color_map: Dict[int, Tuple[int, int, int]] = {}
        self.device = device

        # Thread pool for parallel processing
        self.thread_pool = ThreadPoolExecutor(max_workers=4)

        # Face detection queue and processing thread
        self.face_queue = queue.Queue(maxsize=20)
        self.face_processing = True
        self.face_thread = threading.Thread(target=self._process_face_queue)
        self.face_thread.daemon = True

        # Remove the face detector initialization from __init__
        self.mp_face_detection = mp.solutions.face_detection

        # Cache for recently detected faces
        self.face_cache: Dict[int, Tuple[np.ndarray, float]] = {}
        self.cache_timeout = 1.0  # 1 second cache timeout

        # Performance monitoring
        self.processing_times = deque(maxlen=100)

        # Start face processing thread
        self.face_thread.start()

        self.cross_camera_tracker: Optional[OptimizedCrossCameraTracker] = None

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

    def _process_face_queue(self):
        """Background thread for processing face detection queue"""
        while self.face_processing:
            try:
                track_id, frame, bbox = self.face_queue.get(timeout=0.1)

                current_time = time.time()
                if track_id in self.face_cache:
                    cached_face, cache_time = self.face_cache[track_id]
                    if current_time - cache_time < self.cache_timeout:
                        continue

                face_img = self._extract_face(frame, bbox)

                if face_img is not None:
                    if track_id in self.tracked_persons:
                        self.tracked_persons[track_id].face_image = face_img
                        self.face_cache[track_id] = (face_img, current_time)
                        self.face_queue.task_done()
                else:
                    logger.error(f"Face extraction failed for track {track_id}")
                    self.face_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in face processing: {e}")

    def _extract_face(
        self, frame: np.ndarray, bbox: np.ndarray
    ) -> Optional[np.ndarray]:
        """Extract face from person bounding box using MediaPipe"""
        import os

        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        try:
            x, y, w, h = map(int, bbox)

            # Validate input frame
            if frame is None or frame.size == 0:
                logger.error("Invalid input frame")
                return None

            # Add padding with boundary checks
            pad_x = int(w * 0.1)
            pad_y = int(h * 0.1)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(frame.shape[1], x + w + pad_x)
            y2 = min(frame.shape[0], y + h + pad_y)

            person_roi = frame[y1:y2, x1:x2]
            if (
                person_roi.size == 0
                or person_roi.shape[0] == 0
                or person_roi.shape[1] == 0
            ):
                logger.error("Invalid ROI dimensions")
                return None

            # Create a new detector instance for each image
            with self.mp_face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.5
            ) as face_detector:
                # Convert BGR to RGB
                rgb_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)

                # Ensure the image is contiguous
                if not rgb_roi.flags["C_CONTIGUOUS"]:
                    rgb_roi = np.ascontiguousarray(rgb_roi)

                # Process the image
                results = face_detector.process(rgb_roi)

                if not results or not results.detections:
                    return None

                # Get the first detected face
                detection = results.detections[0]
                bbox_rel = detection.location_data.relative_bounding_box

                # Convert relative coordinates to absolute
                roi_h, roi_w = rgb_roi.shape[:2]
                fx = int(bbox_rel.xmin * roi_w)
                fy = int(bbox_rel.ymin * roi_h)
                fw = int(bbox_rel.width * roi_w)
                fh = int(bbox_rel.height * roi_h)

                # Boundary checks
                fx = max(0, fx)
                fy = max(0, fy)
                fw = min(roi_w - fx, fw)
                fh = min(roi_h - fy, fh)

                if fw <= 0 or fh <= 0:
                    return None

                # Extract face region
                face_img = rgb_roi[fy : fy + fh, fx : fx + fw]

                # Convert back to BGR
                face_img = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)

                # Resize face image
                try:
                    if face_img.size > 0:
                        face_img = cv2.resize(face_img, (160, 160))
                        return face_img
                except cv2.error as e:
                    logger.error(f"OpenCV resize error: {e}")
                    return None

        except Exception as e:
            logger.error(f"Error extracting face: {e}")
            import traceback

            logger.error(traceback.format_exc())

        return None

    def update(
        self, frame: np.ndarray, yolo_results, current_time: Optional[float] = None
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

                futures = []
                for xyxy, class_id, track_id in zip(xyxys, classes, track_ids):
                    if int(class_id) != 0:  # Skip non-person detections
                        continue

                    track_id = int(track_id)
                    x1, y1, x2, y2 = xyxy
                    w = x2 - x1
                    h = y2 - y1
                    bbox = np.array([x1, y1, w, h])

                    if track_id in self.tracked_persons:
                        person = self.tracked_persons[track_id]
                        person.update_position(bbox, current_time)

                        if (
                            person.face_image is None
                            or person.recognition_status == "pending"
                        ):
                            try:
                                self.face_queue.put_nowait((track_id, frame, bbox))
                            except queue.Full:
                                pass

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

                        try:
                            self.face_queue.put_nowait((track_id, frame, bbox))
                        except queue.Full:
                            pass

            self._cleanup_stale_tracks(current_time)
            self.processing_times.append(time.time() - start_time)

            for new_track in new_tracks:
                while self.face_queue.qsize() > 0:
                    logger.info("Waiting for face queue to clear")
                    time.sleep(0.1)

                asyncio.run(
                    self._handle_face_recognition(
                        new_track, self.tracked_persons[new_track.track_id].face_image
                    )
                )

            # After creating or updating person tracks, gather person crops
            person_crops = []
            for track_id, person in self.tracked_persons.items():
                x, y, w, h = map(int, person.bbox)
                crop = frame[y : y + h, x : x + w]
                person_crops.append((track_id, crop))

            # Call cross-camera reid
            if self.cross_camera_tracker:
                global_ids = self.cross_camera_tracker.update(
                    camera_id=self.camera_id, person_crops=person_crops
                )
                # Assign global IDs
                for t_id, g_id in global_ids.items():
                    if t_id in self.tracked_persons:
                        self.tracked_persons[t_id].global_id = g_id

        except Exception as e:
            logger.error(f"Error updating tracker: {e}")

        return updated_tracks, new_tracks

    def _cleanup_stale_tracks(self, current_time: float):
        """Remove old tracks and clean up resources"""
        stale_threshold = 2.0  # seconds
        stale_tracks = []

        for track_id, person in self.tracked_persons.items():
            if current_time - person.last_update > stale_threshold:
                stale_tracks.append(track_id)

        for track_id in stale_tracks:
            del self.tracked_persons[track_id]
            if track_id in self.face_cache:
                del self.face_cache[track_id]

    def get_statistics(self) -> Dict:
        """Get performance statistics"""
        if not self.processing_times:
            return {}

        return {
            "avg_processing_time": sum(self.processing_times)
            / len(self.processing_times),
            "min_processing_time": min(self.processing_times),
            "max_processing_time": max(self.processing_times),
            "active_tracks": len(self.tracked_persons),
            "face_queue_size": self.face_queue.qsize(),
        }

    def cleanup(self):
        """Clean up resources"""
        self.face_processing = False
        if self.face_thread.is_alive():
            self.face_thread.join(timeout=1)
        self.thread_pool.shutdown(wait=False)
        self.tracked_persons.clear()
        self.face_cache.clear()

    async def _handle_face_recognition(
        self, track: TrackedPerson, face_image: np.ndarray
    ):
        """Handle face recognition with detailed logging"""
        logger.info(
            f"\033[0;31m Handling face recognition for track {track.track_id} \033[0m"
        )
        try:
            if face_image is None:
                logger.error("\033[0;31m Face image is None  \033[0m")
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
            nats_client = SharedNatsClient.get_instance()
            if not nats_client:
                logger.error("Reid handle face recogntion nats client to avaiable")
                nats_url = await self.db.get_cloud_url()
                nats_client = await SharedNatsClient.initialize(nats_url)
            try:
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

            except Exception as e:
                logger.error(f"Error in recognition execution: {e}")
                track.recognition_status = "pending"

        except Exception as e:
            logger.error(f"Error in face recognition: {e}")
            track.recognition_status = "pending"
