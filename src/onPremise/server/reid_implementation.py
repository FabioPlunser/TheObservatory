from dataclasses import dataclass
import numpy as np
import torch
from torchreid import models
import cv2
from dataclasses import dataclass
from datetime import datetime, timedelta
import threading
from typing import Dict, List, Optional, Tuple
from scipy.spatial.distance import cdist
from concurrent.futures import ThreadPoolExecutor
import queue
import logging
from collections import deque
import time
from contextlib import nullcontext
from multiprocessing import cpu_count
from functools import lru_cache

logger = logging.getLogger("ReID")


@dataclass
class PersonFeature:
    reid_features: np.ndarray
    appearance_features: List[np.ndarray]
    temporal_features: Dict[str, float]
    spatial_features: Dict[str, Tuple[float, float]]
    last_seen: datetime
    camera_id: str
    track_id: int
    face_id: Optional[str] = None
    recognition_status: str = "pending"
    confidence: float = 0.0


class OptimizedCrossCameraTracker:
    def __init__(
        self,
        max_feature_age: int = 300,
        feature_history_size: int = 5,
        reid_threshold: float = 0.6,
        spatial_weight: float = 0.3,
        temporal_weight: float = 0.2,
        batch_size: int = 16,  # Increased batch size for better throughput
    ):
        # Initialize device
        self.device = self._init_device()

        # Initialize ReID model with optimizations
        self.reid_model = self._init_reid_model()

        # Feature management
        self.person_features: Dict[str, PersonFeature] = {}
        self.camera_to_global: Dict[Tuple[str, int], str] = {}

        # Configuration
        self.max_feature_age = timedelta(seconds=max_feature_age)
        self.feature_history_size = feature_history_size
        self.reid_threshold = reid_threshold
        self.spatial_weight = spatial_weight
        self.temporal_weight = temporal_weight
        self.batch_size = batch_size

        # Feature extraction queue for batch processing
        self.feature_queue = queue.Queue(maxsize=30)
        self.processing_thread = threading.Thread(target=self._process_feature_queue)
        self.processing_thread.daemon = True
        self.processing_thread.start()

        # Enhanced thread and process pools
        self.thread_pool = ThreadPoolExecutor(max_workers=4)

        # Optimized feature cache with LRU
        self.feature_cache: Dict[str, Tuple[np.ndarray, datetime]] = {}
        self.cache_timeout = timedelta(seconds=30)

        # Preprocessing optimizations
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # Enhanced performance monitoring
        self.metrics = {
            "feature_extraction_time": deque(maxlen=100),
            "matching_time": deque(maxlen=100),
            "preprocessing_time": deque(maxlen=100),
            "cache_hits": 0,
            "cache_misses": 0,
            "batch_sizes": deque(maxlen=100),
        }

    def _cleanup_old_features(self, current_time: datetime):
        """Remove features that are too old"""
        expired_ids = []
        expired_camera_tracks = []

        for global_id, feature_data in self.person_features.items():
            if current_time - feature_data.last_seen > self.max_feature_age:
                expired_ids.append(global_id)
                # Find associated camera tracks
                for camera_track, g_id in self.camera_to_global.items():
                    if g_id == global_id:
                        expired_camera_tracks.append(camera_track)

        # Remove expired entries
        for global_id in expired_ids:
            del self.person_features[global_id]
        for camera_track in expired_camera_tracks:
            del self.camera_to_global[camera_track]

    def _log_performance_metrics(self):
        """Log performance metrics"""
        metrics = {
            "avg_feature_extraction": sum(self.metrics["feature_extraction_time"])
            / len(self.metrics["feature_extraction_time"]),
            "avg_matching_time": sum(self.metrics["matching_time"])
            / len(self.metrics["matching_time"]),
            "avg_preprocessing": sum(self.metrics["preprocessing_time"])
            / len(self.metrics["preprocessing_time"]),
            "cache_hit_ratio": (
                self.metrics["cache_hits"]
                / (self.metrics["cache_hits"] + self.metrics["cache_misses"])
                if (self.metrics["cache_hits"] + self.metrics["cache_misses"]) > 0
                else 0
            ),
            "avg_batch_size": (
                sum(self.metrics["batch_sizes"]) / len(self.metrics["batch_sizes"])
                if self.metrics["batch_sizes"]
                else 0
            ),
        }
        logger.info(f"Performance metrics: {metrics}")

    def _init_device(self) -> torch.device:
        """Initialize optimal device for processing"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            # Enhanced CUDA optimizations
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            torch.set_float32_matmul_precision("high")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
            torch.set_num_threads(cpu_count())
        return device

    def _init_reid_model(self):
        """Initialize ReID model with optimizations"""
        model = models.build_model(
            name="osnet_ain_x1_0", num_classes=1000, loss="softmax", pretrained=True
        )
        model = model.eval().to(self.device)

        if torch.cuda.is_available():
            # Use updated autocast syntax
            self.amp_context = torch.amp.autocast('cuda')
            torch.cuda.empty_cache()
        else:
            self.amp_context = nullcontext()

        return model

    @lru_cache(maxsize=128)
    def _preprocess_image(self, img_bytes: bytes) -> np.ndarray:
        """Cached and optimized image preprocessing"""
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

        # Fast resize with optimal interpolation
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)

        # Optimized CLAHE on L channel only
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab[..., 0] = self.clahe.apply(lab[..., 0])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # Vectorized normalization
        img = img.astype(np.float32) / 255.0
        img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])

        return img

    def _process_feature_queue(self):
        """Enhanced background thread for batch processing features"""
        batch_crops = []
        batch_ids = []
        last_process_time = time.time()

        while True:
            try:
                # Dynamic batch collection with timeout
                try:
                    while len(batch_crops) < self.batch_size:
                        # Process batch if it's been too long
                        if time.time() - last_process_time > 0.5 and batch_crops:
                            break

                        crop, track_id = self.feature_queue.get(timeout=0.1)
                        batch_crops.append(crop)
                        batch_ids.append(track_id)
                except queue.Empty:
                    if batch_crops:  # Process partial batch
                        pass
                    else:
                        continue

                if not batch_crops:
                    continue

                # Record batch size for metrics
                self.metrics["batch_sizes"].append(len(batch_crops))

                # Extract features for batch
                start_time = time.time()
                features = self._extract_reid_features_batch(batch_crops)
                self.metrics["feature_extraction_time"].append(time.time() - start_time)

                # Update feature cache
                current_time = datetime.now()
                for feature, track_id in zip(features, batch_ids):
                    self.feature_cache[track_id] = (feature, current_time)

                batch_crops = []
                batch_ids = []
                last_process_time = time.time()

            except Exception as e:
                logger.error(f"Error in feature processing: {e}")
                batch_crops = []
                batch_ids = []

    def _extract_reid_features_batch(self, crops: List[np.ndarray]) -> List[np.ndarray]:
        """Optimized batch feature extraction"""
        try:
            if not crops:
                return []

            start_time = time.time()

            # Process images in current process
            preprocessed = []
            for crop in crops:
                if crop.shape[0] < 64 or crop.shape[1] < 32:
                    continue

                try:
                    # Preprocess image directly
                    img = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_LINEAR)

                    # Optimize contrast
                    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                    lab[..., 0] = self.clahe.apply(lab[..., 0])
                    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

                    # Normalize
                    img = img.astype(np.float32) / 255.0
                    img = (img - np.array([0.485, 0.456, 0.406])) / np.array(
                        [0.229, 0.224, 0.225]
                    )

                    # Convert to tensor
                    img = torch.from_numpy(img).float()
                    img = img.permute(2, 0, 1)
                    preprocessed.append(img)
                except Exception as e:
                    logger.error(f"Preprocessing error: {e}")
                    continue

            if not preprocessed:
                return []

            # Stack into batch
            batch = torch.stack(preprocessed).to(self.device)

            # Extract features with optimizations
            with torch.no_grad(), self.amp_context:
                features = self.reid_model(batch)
                features = features.cpu().numpy()

            # Vectorized L2 normalization
            features = features / np.linalg.norm(features, axis=1, keepdims=True)

            self.metrics["preprocessing_time"].append(time.time() - start_time)
            return features

        except Exception as e:
            logger.error(f"Error in batch feature extraction: {e}")
            return []

    def _compute_similarity_matrix(
        self, query_features: np.ndarray, gallery_features: List[np.ndarray]
    ) -> np.ndarray:
        """Optimized similarity matrix computation"""
        if not gallery_features:
            return np.array([])

        # Vectorized operations with pre-allocation
        gallery_features = np.vstack(gallery_features)
        similarities = 1 - cdist(
            query_features.reshape(1, -1), gallery_features, metric="cosine"
        )
        return similarities.flatten()

    def _compute_spatial_similarity(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        dist = np.linalg.norm(np.array(pos1) - np.array(pos2))
        return 1 / (1 + dist)

    def _compute_temporal_similarity(self, t1: float, t2: float) -> float:
        dt = abs(t1 - t2)
        return np.exp(-dt / 2.0)

    def update(
        self,
        camera_id: str,
        person_crops: List[Tuple[int, np.ndarray]],
        positions: Optional[Dict[int, Tuple[float, float]]] = None,
        face_ids: Optional[Dict[int, str]] = None,
        recognition_statuses: Optional[Dict[int, str]] = None,
    ) -> Dict[int, str]:
        """Optimized update with batch processing"""
        start_time = time.time()
        track_to_global = {}
        current_time = datetime.now()

        try:
            # Clean up old features
            self._cleanup_old_features(current_time)

            # Process in optimal batches
            for i in range(0, len(person_crops), self.batch_size):
                batch = person_crops[i : i + self.batch_size]
                self._process_batch(
                    batch,
                    camera_id,
                    positions,
                    face_ids,
                    recognition_statuses,
                    current_time,
                    track_to_global,
                )

        except Exception as e:
            logger.error(f"Error in update: {e}")

        # Update performance metrics
        self.metrics["matching_time"].append(time.time() - start_time)

        # Log metrics periodically
        # if len(self.metrics["matching_time"]) == 100:
        #     self._log_performance_metrics()

        return track_to_global

    def _process_batch(
        self,
        batch: List[Tuple[int, np.ndarray]],
        camera_id: str,
        positions: Optional[Dict[int, Tuple[float, float]]],
        face_ids: Optional[Dict[int, str]],
        recognition_statuses: Optional[Dict[int, str]],
        current_time: datetime,
        track_to_global: Dict[int, str],
    ):
        """Process a batch of detections"""
        for track_id, crop in batch:
            camera_track = (camera_id, track_id)
            position = positions.get(track_id) if positions else None

            # Handle existing tracks
            if camera_track in self.camera_to_global:
                self._update_existing_track(
                    camera_track,
                    track_id,
                    crop,
                    position,
                    camera_id,
                    current_time,
                    face_ids,
                    recognition_statuses,
                    track_to_global,
                )
                continue

            # Handle new tracks
            self._process_new_track(
                track_id,
                crop,
                position,
                camera_id,
                current_time,
                face_ids,
                recognition_statuses,
                track_to_global,
                camera_track,
            )

    def _update_existing_track(
        self,
        camera_track: Tuple[str, int],
        track_id: int,
        crop: np.ndarray,
        position: Optional[Tuple[float, float]],
        camera_id: str,
        current_time: datetime,
        face_ids: Optional[Dict[int, str]],
        recognition_statuses: Optional[Dict[int, str]],
        track_to_global: Dict[int, str],
    ):
        """Update an existing track"""
        global_id = self.camera_to_global[camera_track]
        track_to_global[track_id] = global_id

        if global_id in self.person_features:
            person = self.person_features[global_id]

            try:
                self.feature_queue.put_nowait((crop, track_id))
            except queue.Full:
                pass

            if position:
                person.spatial_features[camera_id] = position
            person.temporal_features[camera_id] = current_time.timestamp()
            person.last_seen = current_time

            if face_ids and track_id in face_ids:
                person.face_id = face_ids[track_id]
            if recognition_statuses and track_id in recognition_statuses:
                person.recognition_status = recognition_statuses[track_id]

    def _process_new_track(
        self,
        track_id: int,
        crop: np.ndarray,
        position: Optional[Tuple[float, float]],
        camera_id: str,
        current_time: datetime,
        face_ids: Optional[Dict[int, str]],
        recognition_statuses: Optional[Dict[int, str]],
        track_to_global: Dict[int, str],
        camera_track: Tuple[str, int],
    ):
        """Process a new track"""
        try:
            self.feature_queue.put_nowait((crop, track_id))
        except queue.Full:
            return

        # Wait for features
        max_wait = 0.1
        start_wait = time.time()
        while time.time() - start_wait < max_wait:
            if track_id in self.feature_cache:
                reid_features, _ = self.feature_cache[track_id]
                break
            time.sleep(0.01)
        else:
            return

        # Call _match_and_update after we have features
        self._match_and_update(
            reid_features,
            track_id,
            position,
            camera_id,
            current_time,
            face_ids,
            recognition_statuses,
            track_to_global,
        )

    def _match_and_update(
        self,
        reid_features: np.ndarray,
        track_id: int,
        position: Optional[Tuple[float, float]],
        camera_id: str,
        current_time: datetime,
        face_ids: Optional[Dict[int, str]],
        recognition_statuses: Optional[Dict[int, str]],
        track_to_global: Dict[int, str],
    ):
        """Match and update person features with optimized processing"""
        try:
            # Compare with existing features
            candidates = []
            for global_id, feature_data in self.person_features.items():
                # Remove the skip logic for same camera/time

                # Compute appearance similarity
                appearance_sim = self._compute_similarity_matrix(
                    reid_features, feature_data.appearance_features
                ).max()

                # Compute spatial-temporal similarity
                spatial_sim = 0.0
                temporal_sim = 0.0

                if position and camera_id in feature_data.spatial_features:
                    spatial_sim = self._compute_spatial_similarity(
                        position, feature_data.spatial_features[camera_id]
                    )

                if camera_id in feature_data.temporal_features:
                    temporal_sim = self._compute_temporal_similarity(
                        current_time.timestamp(),
                        feature_data.temporal_features[camera_id],
                    )

                # Weighted similarity score
                total_sim = (
                    appearance_sim * (1 - self.spatial_weight - self.temporal_weight)
                    + spatial_sim * self.spatial_weight
                    + temporal_sim * self.temporal_weight
                )

                if total_sim > self.reid_threshold:
                    candidates.append((global_id, total_sim, feature_data))

            # Handle matching
            if candidates:
                candidates.sort(key=lambda x: x[1], reverse=True)
                best_match, similarity, matched_feature = candidates[0]
                global_id = best_match
                logger.info(
                    f"Matched person across cameras with similarity {similarity:.2f}"
                )

                # Update matched person's features
                matched_feature.appearance_features.append(reid_features)
                if len(matched_feature.appearance_features) > self.feature_history_size:
                    matched_feature.appearance_features.pop(0)

                # Update spatial-temporal info
                if position:
                    matched_feature.spatial_features[camera_id] = position
                matched_feature.temporal_features[camera_id] = current_time.timestamp()
                matched_feature.last_seen = current_time

                # Update recognition info if available
                if face_ids and track_id in face_ids:
                    matched_feature.face_id = face_ids[track_id]
                if recognition_statuses and track_id in recognition_statuses:
                    matched_feature.recognition_status = recognition_statuses[track_id]

            else:
                # Create new identity
                global_id = f"person_{len(self.person_features)}"
                logger.info(f"Created new person {global_id}")

                self.person_features[global_id] = PersonFeature(
                    reid_features=reid_features,
                    appearance_features=[reid_features],
                    temporal_features={camera_id: current_time.timestamp()},
                    spatial_features={camera_id: position} if position else {},
                    last_seen=current_time,
                    camera_id=camera_id,
                    track_id=track_id,
                    face_id=face_ids.get(track_id) if face_ids else None,
                    recognition_status=(
                        recognition_statuses.get(track_id, "pending")
                        if recognition_statuses
                        else "pending"
                    ),
                    confidence=1.0,
                )

            # Update mappings
            camera_track = (camera_id, track_id)
            self.camera_to_global[camera_track] = global_id
            track_to_global[track_id] = global_id

        except Exception as e:
            logger.error(f"Error in match and update: {e}")
            return None

        def get_person_info(self, global_id: str) -> Optional[PersonFeature]:
            """Get information about a person by their global ID"""
            return self.person_features.get(global_id)

        def _cleanup_old_features(self, current_time: datetime):
            """Remove features that are too old"""
            expired_ids = []
            expired_camera_tracks = []

            for global_id, feature_data in self.person_features.items():
                if current_time - feature_data.last_seen > self.max_feature_age:
                    expired_ids.append(global_id)
                    # Find and mark associated camera tracks for removal
                    for camera_track, g_id in self.camera_to_global.items():
                        if g_id == global_id:
                            expired_camera_tracks.append(camera_track)

            # Remove expired entries
            for global_id in expired_ids:
                del self.person_features[global_id]
            for camera_track in expired_camera_tracks:
                del self.camera_to_global[camera_track]
