import torch
import torch.nn as nn
import cv2
import numpy as np
import asyncio
import aiohttp
import uuid
import logging
import threading
import mediapipe as mp
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from torchvision import transforms
from collections import deque
from nats_client import NatsClient, Commands
from database import Database

logger = logging.getLogger("ReID")


@dataclass
class PersonFeature:
    global_id: str
    reid_features: np.ndarray
    appearance_history: List[np.ndarray] = field(default_factory=list)
    last_seen: datetime = field(default_factory=datetime.now)
    camera_history: Dict[str, List[datetime]] = field(default_factory=dict)
    confidence: float = 0.0
    face_id: Optional[str] = None
    recognition_status: str = "pending"


class Reid:
    def __init__(
        self,
        feature_history_size: int = 20,
        reid_threshold: float = 0.5,
        max_feature_age: int = 7200,
        device=None,
    ):
        self.device = device if device else self._init_device()
        self.model = self._init_model()
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((256, 128)),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # Feature management
        self.person_features: Dict[str, PersonFeature] = {}
        self.camera_to_global: Dict[Tuple[str, int], str] = {}
        self.feature_history_size = feature_history_size
        self.reid_threshold = reid_threshold
        self.max_feature_age = timedelta(seconds=max_feature_age)

        # Performance tracking
        self.metrics = {
            "feature_extraction_time": deque(maxlen=100),
            "matching_time": deque(maxlen=100),
            "reid_success_rate": deque(maxlen=100),
        }

        # Face detection and upload
        self.processed_faces = {}
        self.db = Database()
        self.processing_lock = threading.Lock()

    def _init_device(self) -> torch.device:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        else:
            device = torch.device("cpu")
        return device

    def _init_model(self):
        """Initialize a simple but effective CNN model for ReID"""
        from torchvision.models import resnet18, ResNet18_Weights

        # Load pretrained ResNet18
        model = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Modify the last layer for ReID
        num_features = model.fc.in_features
        model.fc = torch.nn.Linear(num_features, 512)

        model = model.to(self.device).eval()
        return model

    def _extract_features(self, frame: np.ndarray) -> np.ndarray:
        """Extract features from a single frame"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = self.transform(frame_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.model(img)
        return features.cpu().numpy()[0]

    def _compute_similarity(self, feat1: np.ndarray, feat2: List[np.ndarray]) -> float:
        """Compute similarity between features"""
        if not feat2:
            return 0.0

        similarities = []
        for hist_feat in feat2:
            sim = float(
                np.dot(feat1, hist_feat)
                / (np.linalg.norm(feat1) * np.linalg.norm(hist_feat))
            )
            similarities.append(sim)

        return max(similarities)

    def _extract_face(self, person_crop: np.ndarray) -> Optional[np.ndarray]:
        """Extract face from person crop using MediaPipe"""
        try:
            if (
                person_crop is None
                or person_crop.shape[0] < 32
                or person_crop.shape[1] < 32
            ):
                return None

            with mp.solutions.face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=0.5,
            ) as face_detector:
                results = face_detector.process(
                    cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
                )

                if results.detections:
                    detection = results.detections[0]
                    bbox = detection.location_data.relative_bounding_box

                    h, w = person_crop.shape[:2]
                    x = max(0, int(bbox.xmin * w))
                    y = max(0, int(bbox.ymin * h))
                    width = min(w - x, int(bbox.width * w))
                    height = min(h - y, int(bbox.height * h))

                    face_crop = person_crop[y : y + height, x : x + width]
                    if face_crop.size == 0:
                        return None

                    return cv2.resize(face_crop, (112, 112))

            return None
        except Exception as e:
            logger.error(f"Error extracting face: {e}")
            return None

    async def handle_face_upload(
        self, person: PersonFeature, camera_id: str, face_crop: np.ndarray
    ) -> None:
        """Handle face upload and recognition"""
        try:
            # Generate face ID if none exists
            if person.face_id is None:
                person.face_id = str(uuid.uuid4())

            # Update status
            person.recognition_status = "in_progress"

            # Connect to NATS
            nats_url = await self.db.get_cloud_url()
            nats_client = NatsClient(nats_url)
            await nats_client.connect()

            try:
                company_id = await self.db.get_company_id()

                # Get upload URL
                response = await nats_client.send_message_with_reply(
                    Commands.GET_PRESIGNED_UPLOAD_UNKNOWN_FACE_URL.value,
                    {
                        "company_id": company_id,
                        "face_id": person.face_id,
                    },
                )

                if not response or not response.get("success"):
                    raise Exception("Failed to get presigned URL")

                presigned_url = response.get("url")

                # Upload image
                _, img_encoded = cv2.imencode(".jpg", face_crop)
                img_bytes = img_encoded.tobytes()

                async with aiohttp.ClientSession() as session:
                    async with session.put(
                        presigned_url,
                        data=img_bytes,
                        headers={"Content-Type": "*"},
                        timeout=30,
                    ) as resp:
                        if resp.status != 200:
                            raise Exception(f"Upload failed with status {resp.status}")

                # Trigger recognition
                await nats_client.send_message(
                    Commands.EXECUTE_RECOGNITION.value,
                    {
                        "company_id": company_id,
                        "camera_id": camera_id,
                        "face_id": person.face_id,
                        "global_id": person.global_id,
                    },
                )

                person.recognition_status = "tracked"
                self.processed_faces[person.face_id] = datetime.now()

                # Update database
                await self.db.update_person_track(
                    global_id=person.global_id,
                    camera_id=camera_id,
                    face_id=person.face_id,
                    recognition_status=person.recognition_status,
                )

            finally:
                await nats_client.close()

        except Exception as e:
            logger.error(f"Error in face upload: {str(e)}")
            person.recognition_status = "pending"

    def update(
        self, camera_id: str, person_crops: List[Tuple[int, np.ndarray]]
    ) -> Dict[int, str]:
        """Update ReID system with new detections"""
        current_time = datetime.now()
        track_to_global = {}

        with self.processing_lock:
            for track_id, crop in person_crops:
                # Extract features
                features = self._extract_features(crop)

                # Find best match
                best_match = None
                best_score = 0

                for global_id, person in self.person_features.items():
                    if current_time - person.last_seen > self.max_feature_age:
                        continue

                    similarity = self._compute_similarity(
                        features, person.appearance_history
                    )
                    if similarity > best_score and similarity > self.reid_threshold:
                        best_score = similarity
                        best_match = global_id

                if best_match:
                    # Update existing identity
                    person = self.person_features[best_match]
                    person.appearance_history.append(features)
                    if len(person.appearance_history) > self.feature_history_size:
                        person.appearance_history.pop(0)
                    person.last_seen = current_time
                    person.confidence = min(1.0, person.confidence + 0.1)
                    track_to_global[track_id] = best_match
                else:
                    # Create new identity
                    new_id = f"person_{len(self.person_features)}"
                    new_person = PersonFeature(
                        global_id=new_id,
                        reid_features=features,
                        appearance_history=[features],
                        camera_history={camera_id: [current_time]},
                        confidence=0.3,
                    )
                    self.person_features[new_id] = new_person
                    track_to_global[track_id] = new_id

                    # Handle face detection and upload
                    face_crop = self._extract_face(crop)
                    if face_crop is not None:
                        asyncio.create_task(
                            self.handle_face_upload(new_person, camera_id, face_crop)
                        )

        return track_to_global
