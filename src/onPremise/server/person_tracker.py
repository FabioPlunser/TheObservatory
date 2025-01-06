import multiprocessing as mp
import cv2
import numpy as np
import logging
import time
import torch
import asyncio
import uuid
import aiohttp

from bytetracker import BYTETracker
from typing import Dict, List, Optional, Tuple
from nats_client import SharedNatsClient, Commands
from logging_config import setup_logger

setup_logger()
logger = logging.getLogger("PersonTracker")


class TrackedPerson:
    def __init__(self, bbox, track_id, face_image=None):
        self.bbox = bbox
        self.track_id = track_id
        self.last_seen = 0
        self.face_image = face_image
        self.face_id = None
        self.recognition_status = "pending"

    def shoudl_detect_face(self, current_time: float) -> bool:
        if self.recognition_status == "pending":
            return current_time - self.last_face_detection > 5  # check every 5 seconds
        return (
            current_time - self.recognition_time > 30
        )  # Update recognized faces every 30 seconds

    def update_face(self, face_image: np.ndarray, current_time: float):
        self.face_image = face_image
        self.last_face_detection = current_time
        if self.recognition_status not in ["in_progress", "recognized"]:
            self.recognition_status = "pending"


class PersonTracker:
    def __init__(self):
        self.tracker = BYTETracker(
            track_thresh=0.25, track_buffer=30, match_thresh=0.8, frame_rate=30
        )

        self.tracked_persons: Dict[int, TrackedPerson] = {}
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def _detect_face(
        self, frame: np.ndarray, person_bbox: np.ndarray
    ) -> Optional[np.ndarray]:
        """Extract face from person bounding box"""
        x1, y1, x2, y2 = map(int, person_bbox)
        # Ensure coordinates are within frame bounds
        y1, y2 = max(0, y1), min(frame.shape[0], y2)
        x1, x2 = max(0, x1), min(frame.shape[1], x2)

        if y2 <= y1 or x2 <= x1:
            return None

        person_roi = frame[y1:y2, x1:x2]

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            # Get the largest face
            largest_face = max(faces, key=lambda x: x[2] * x[3])
            fx, fy, fw, fh = largest_face
            face_img = person_roi[fy : fy + fh, fx : fx + fw]
            return cv2.resize(face_img, (160, 160))  # Standardize size

        return None

    def update(
        self,
        frame: np.ndarray,
        boxes: np.ndarray,
        scores: np.ndarray,
        current_time: float,
    ) -> Tuple[List[TrackedPerson], List[TrackedPerson]]:
        """Update tracker with new detections and return updated and new tracks"""
        if len(boxes) == 0:
            return [], []

        # BYTETracker expects boxes and scores
        online_targets = self.tracker.update(boxes, scores)

        updated_tracks = []
        new_tracks = []

        for t in online_targets:
            track_id = t.track_id
            bbox = t.tlbr  # [top, left, bottom, right]

            if track_id in self.tracked_persons:
                person = self.tracked_persons[track_id]
                person.bbox = bbox

                # Check if we need to update face detection
                if person.should_detect_face(current_time):
                    face_img = self._detect_face(frame, bbox)
                    if face_img is not None:
                        person.update_face(face_img, current_time)
                        updated_tracks.append(person)
            else:
                # New person detected
                person = TrackedPerson(track_id, bbox)
                face_img = self._detect_face(frame, bbox)
                if face_img is not None:
                    person.update_face(face_img, current_time)
                    new_tracks.append(person)
                self.tracked_persons[track_id] = person

        # Clean up old tracks
        active_track_ids = {t.track_id for t in online_targets}
        self.tracked_persons = {
            k: v for k, v in self.tracked_persons.items() if k in active_track_ids
        }

        return updated_tracks, new_tracks
