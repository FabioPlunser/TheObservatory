from dataclasses import dataclass, field
import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
import time
import colorsys

import logging
from logging_config import setup_logger

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

    def update_position(self, bbox: np.ndarray, current_time: float):
        self.bbox = bbox
        self.last_update = current_time


class PersonTracker:
    def __init__(self):
        self.tracked_persons: Dict[int, TrackedPerson] = {}
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.color_map: Dict[int, Tuple[int, int, int]] = {}

    def _generate_color(self, track_id: int) -> Tuple[int, int, int]:
        """Generate a unique color for a track ID using golden ratio"""
        golden_ratio = 0.618033988749895
        h = (track_id * golden_ratio) % 1.0
        # Use high saturation and value for vibrant colors
        s = 0.9
        v = 0.95
        # Convert to RGB
        rgb = colorsys.hsv_to_rgb(h, s, v)
        # Convert to BGR (OpenCV format) and scale to 0-255
        return tuple(int(255 * x) for x in reversed(rgb))

    def _get_color(self, track_id: int) -> Tuple[int, int, int]:
        """Get or create a color for a track ID"""
        if track_id not in self.color_map:
            self.color_map[track_id] = self._generate_color(track_id)
        return self.color_map[track_id]

    def _extract_face(
        self, frame: np.ndarray, bbox: np.ndarray
    ) -> Optional[np.ndarray]:
        """Extract face from person bounding box"""
        logger.info("Extracting face")
        try:
            x, y, w, h = map(int, bbox)

            # Add padding to the bounding box
            pad_x = int(w * 0.1)  # 10% padding
            pad_y = int(h * 0.1)

            # Calculate padded coordinates ensuring they stay within frame bounds
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(frame.shape[1], x + w + pad_x)
            y2 = min(frame.shape[0], y + h + pad_y)

            person_roi = frame[y1:y2, x1:x2]

            # Detect faces in the person ROI
            gray_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray_roi, 1.1, 4, minSize=(30, 30)
            )

            if len(faces) > 0:
                # Get the largest face
                largest_face = max(faces, key=lambda x: x[2] * x[3])
                fx, fy, fw, fh = largest_face

                # Extract and resize face image
                face_img = person_roi[fy : fy + fh, fx : fx + fw]
                face_img = cv2.resize(
                    face_img, (160, 160)
                )  # Standard size for face recognition
                return face_img

        except Exception as e:
            print(f"Error extracting face: {e}")
        return None

    def update(
        self, frame: np.ndarray, yolo_results, current_time: float
    ) -> Tuple[List[TrackedPerson], List[TrackedPerson]]:
        """
        Update tracker with new detections from YOLOv8
        Returns: (updated_tracks, new_tracks)
        """
        updated_tracks = []
        new_tracks = []
        logger.info("Updating tracker")

        try:
            # Process YOLOv8 tracking results
            for result in yolo_results:
                boxes = result.boxes

                # Get track IDs if available
                track_ids = boxes.id.cpu().numpy() if boxes.id is not None else None

                if track_ids is None:
                    continue

                # Get class IDs
                classes = boxes.cls.cpu().numpy()

                # Get boxes in x1,y1,x2,y2 format
                xyxys = boxes.xyxy.cpu().numpy()

                for idx, (xyxy, class_id, track_id) in enumerate(
                    zip(xyxys, classes, track_ids)
                ):
                    # Only process if it's a person (class 0)
                    if int(class_id) != 0:
                        continue

                    track_id = int(track_id)

                    # Convert to x,y,w,h format
                    x1, y1, x2, y2 = xyxy
                    w = x2 - x1
                    h = y2 - y1
                    bbox = np.array([x1, y1, w, h])

                    if track_id in self.tracked_persons:
                        # Update existing person
                        person = self.tracked_persons[track_id]
                        person.update_position(bbox, current_time)

                        # Only extract face if we don't have one or if status is pending
                        if (
                            person.face_image is None
                            or person.recognition_status == "pending"
                        ):
                            face_img = self._extract_face(frame, bbox)
                            if face_img is not None:
                                person.face_image = face_img

                        updated_tracks.append(person)
                    else:
                        # Create new tracked person with unique color
                        face_img = self._extract_face(frame, bbox)
                        color = self._get_color(track_id)
                        person = TrackedPerson(
                            track_id=track_id,
                            bbox=bbox,
                            face_image=face_img,
                            last_update=current_time,
                            color=color,
                        )
                        self.tracked_persons[track_id] = person
                        new_tracks.append(person)

            # Remove stale tracks (not updated recently)
            stale_threshold = 2.0  # seconds
            stale_tracks = []
            for track_id, person in self.tracked_persons.items():
                if current_time - person.last_update > stale_threshold:
                    stale_tracks.append(track_id)

            for track_id in stale_tracks:
                del self.tracked_persons[track_id]
                # Keep color in color_map for consistency if person reappears

        except Exception as e:
            print(f"Error updating tracker: {e}")

        return updated_tracks, new_tracks
