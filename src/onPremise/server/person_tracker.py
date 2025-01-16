from dataclasses import dataclass, field
import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
import time
import colorsys
import torch
import threading
from concurrent.futures import ThreadPoolExecutor
import queue
from collections import deque
import logging
from logging_config import setup_logger

# Setup logger
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
    def __init__(self, device: Optional[torch.device] = None):
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

        # Initialize standard face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        # Check if face cascade loaded successfully
        if self.face_cascade.empty():
            raise RuntimeError("Error loading face cascade classifier")

        # Cache for recently detected faces
        self.face_cache: Dict[int, Tuple[np.ndarray, float]] = {}
        self.cache_timeout = 1.0  # 1 second cache timeout

        # Performance monitoring
        self.processing_times = deque(maxlen=100)

        # Start face processing thread
        self.face_thread.start()

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
                track_id, frame, bbox = self.face_queue.get(timeout=1)

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

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in face processing: {e}")

    def _extract_face(
        self, frame: np.ndarray, bbox: np.ndarray
    ) -> Optional[np.ndarray]:
        """Extract face from person bounding box"""
        try:
            x, y, w, h = map(int, bbox)

            # Add padding with boundary checks
            pad_x = int(w * 0.1)
            pad_y = int(h * 0.1)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(frame.shape[1], x + w + pad_x)
            y2 = min(frame.shape[0], y + h + pad_y)

            person_roi = frame[y1:y2, x1:x2]
            if person_roi.size == 0:
                return None

            # Convert to grayscale for face detection
            gray_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)

            # Use standard CPU face detection
            try:
                faces = self.face_cascade.detectMultiScale(
                    gray_roi,
                    scaleFactor=1.1,
                    minNeighbors=4,
                    minSize=(10, 10),
                    flags=cv2.CASCADE_SCALE_IMAGE,
                )
            except cv2.error as e:
                logger.error(f"OpenCV face detection error: {e}")
                return None

            if len(faces) > 0:
                face = faces[0]

                largest_face = max(faces, key=lambda x: x[2] * x[3])
                fx, fy, fw, fh = largest_face

                face_img = person_roi[fy : fy + fh, fx : fx + fw]
                if face_img.size == 0:
                    return None

                try:
                    face_img = cv2.resize(face_img, (160, 160))
                except cv2.error as e:
                    logger.error(f"OpenCV resize error: {e}")
                    return None

                return face_img

        except Exception as e:
            logger.error(f"Error extracting face: {e}")
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

            # After creating or updating person tracks, gather person crops
            person_crops = []
            for track_id, person in self.tracked_persons.items():
                x, y, w, h = map(int, person.bbox)
                crop = frame[y : y + h, x : x + w]
                person_crops.append((track_id, crop))

            # Call cross-camera reid
            if hasattr(self, "cross_camera_tracker"):
                global_ids = self.cross_camera_tracker.update(
                    camera_id="myCamera",  # Use your actual camera_id here
                    person_crops=person_crops,
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
