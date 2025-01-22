import torch
import torch.nn as nn
import cv2
import numpy as np
import asyncio
import aiohttp
import uuid
import logging
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from torchvision import transforms
from collections import OrderedDict
from nats_client import NatsClient, Commands
from database import Database
from torch.cuda import amp

logger = logging.getLogger("ReID")


class LRUCache:
    """Thread-safe LRU Cache for feature storage"""

    def __init__(self, maxsize: int = 128):
        self.cache = OrderedDict()
        self.maxsize = maxsize
        self._lock = threading.Lock()

    def get(self, key, default=None):
        with self._lock:
            try:
                value = self.cache.pop(key)
                self.cache[key] = value
                return value
            except KeyError:
                return default

    def put(self, key, value):
        with self._lock:
            try:
                self.cache.pop(key)
            except KeyError:
                if len(self.cache) >= self.maxsize:
                    self.cache.popitem(last=False)
            self.cache[key] = value


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
        batch_size: int = 32,
        device=None,
    ):
        self.device = device if device else self._init_device()
        self.model = self._init_model()
        self.scaler = amp.GradScaler('cuda')

        # Optimize transform pipeline
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(
                    (128, 64), antialias=True
                ),  # Smaller size, faster processing
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

        # Performance optimizations
        self.batch_size = batch_size
        self.feature_cache = LRUCache(maxsize=1000)
        self.processing_lock = threading.Lock()

        # Batch processing
        self.batch_buffer = []
        self.batch_metadata = []

        # Face detection and upload
        self.processed_faces = {}
        self.db = Database()

    def _init_device(self) -> torch.device:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            # Enable TF32 on Ampere GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            # Set memory allocator
            torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of GPU memory
        else:
            device = torch.device("cpu")
            torch.set_num_threads(8)  # Optimize CPU threads
        return device

    def _init_model(self):
        """Initialize an efficient CNN model for ReID"""
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

        # Load pretrained MobileNetV3 (smaller and faster than ResNet)
        model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)

        # Modify for ReID
        num_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_features, 256)  # Smaller embedding size

        model = model.to(self.device).eval()

        # Optimize for inference
        if self.device.type == "cuda":
            model = model.half()  # Convert to FP16 for faster inference

        return model

    @torch.no_grad()
    def _extract_features_batch(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Optimized batch feature extraction"""
        if not frames:
            return []

        # Process images in parallel
        processed_frames = []
        for frame in frames:
            processed_frames.append(self._preprocess_frame(frame))

        # Stack and move to GPU
        batch = torch.stack(processed_frames).to(self.device)
        if self.device.type == "cuda":
            batch = batch.half()

        # Extract features with automatic mixed precision
        with torch.amp.autocast(self.device.type):
            features = self.model(batch)
            features = nn.functional.normalize(features, dim=1)

        return features.float().cpu().numpy()

    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Optimized frame preprocessing"""
        frame = cv2.resize(frame, (64, 128), interpolation=cv2.INTER_AREA)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32) / 255.0
        frame = self.transform(frame)
        return frame

    def _compute_similarity(self, feat1: np.ndarray, feat2: List[np.ndarray]) -> float:
        """Fast similarity computation using vectorized operations"""
        if not feat2:
            return 0.0

        # Convert to tensor for faster computation
        feat1_tensor = torch.from_numpy(feat1).to(self.device)
        feat2_tensor = torch.from_numpy(np.stack(feat2)).to(self.device)

        # Compute similarities in one go
        similarities = torch.nn.functional.cosine_similarity(
            feat1_tensor.unsqueeze(0), feat2_tensor
        )

        return float(similarities.max().cpu())

    async def process_faces(
        self,
        person: PersonFeature,
        camera_id: str,
        person_crop,
    ):
        """Process faces in parallel"""
        try:
            logger.info("Uploading face for recognition")
            await self.handle_face_upload(person, camera_id, person_crop)
        except Exception as e:
            logger.error(f"Error processing face: {e}")

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
        """Update ReID system with batch processing"""
        current_time = datetime.now()
        track_to_global = {}

        with self.processing_lock:
            # Extract features in batch
            crops = [crop for _, crop in person_crops]
            features = self._extract_features_batch(crops)

            # Process each detection
            for (track_id, person_crop), feat in zip(person_crops, features):
                # Try to get from cache first
                cached_id = self.feature_cache.get(track_id)
                if cached_id and cached_id in self.person_features:
                    person = self.person_features[cached_id]
                    if current_time - person.last_seen <= self.max_feature_age:
                        track_to_global[track_id] = cached_id
                        continue

                # Find best match
                best_match = None
                best_score = 0

                for global_id, person in self.person_features.items():
                    if current_time - person.last_seen > self.max_feature_age:
                        continue

                    similarity = self._compute_similarity(
                        feat, person.appearance_history
                    )
                    if similarity > best_score and similarity > self.reid_threshold:
                        best_score = similarity
                        best_match = global_id

                if best_match:
                    # Update existing identity
                    person = self.person_features[best_match]
                    person.appearance_history.append(feat)
                    if len(person.appearance_history) > self.feature_history_size:
                        person.appearance_history.pop(0)
                    person.last_seen = current_time
                    person.confidence = min(1.0, person.confidence + 0.1)
                    track_to_global[track_id] = best_match
                    self.feature_cache.put(track_id, best_match)
                else:
                    # Create new identity
                    new_id = f"person_{len(self.person_features)}"
                    new_person = PersonFeature(
                        global_id=new_id,
                        reid_features=feat,
                        appearance_history=[feat],
                        camera_history={camera_id: [current_time]},
                        confidence=0.3,
                    )
                    self.person_features[new_id] = new_person
                    track_to_global[track_id] = new_id
                    self.feature_cache.put(track_id, new_id)

                    asyncio.run(self.process_faces(new_person, camera_id, person_crop))

        return track_to_global

    def cleanup(self):
        """Cleanup resources"""
        self.feature_cache = None
        torch.cuda.empty_cache()
