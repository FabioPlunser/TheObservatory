import numpy as np
import asyncio
import torch
import cv2
import threading
import logging
import os
import mediapipe as mp
import aiohttp
import uuid
from nats_client import NatsClient, Commands

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from functools import lru_cache
from typing import Dict, List, Optional, Tuple
from scipy.spatial.distance import cdist
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict, OrderedDict
from logging_config import setup_logger
from database import Database

setup_logger()
logger = logging.getLogger("ReID")


@dataclass
class PersonFeature:
    global_id: str
    reid_features: np.ndarray
    appearance_history: List[np.ndarray] = field(default_factory=list)
    temporal_features: Dict[str, float] = field(default_factory=dict)
    spatial_features: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    last_seen: datetime = field(default_factory=datetime.now)
    camera_history: Dict[str, List[datetime]] = field(default_factory=dict)
    confidence: float = 0.0
    track_id_history: List[int] = field(default_factory=list)
    face_id: Optional[str] = None
    inactive_time: timedelta = field(default_factory=lambda: timedelta(seconds=0))
    recognition_status: str = "pending"


class LRUCache:
    """Thread-safe LRU Cache implementation"""

    def __init__(self, maxsize: int = 128):
        self.maxsize = maxsize
        self.cache = OrderedDict()
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

    def clear(self):
        with self._lock:
            self.cache.clear()


class Reid:
    metrics = {
        "feature_extraction_time": deque(maxlen=100),
        "matching_time": deque(maxlen=100),
        "reid_success_rate": deque(maxlen=100),
        "track_lengths": deque(maxlen=100),
        "active_tracks": 0,
    }

    # Feature extraction optimization
    feature_extraction_batch = []
    feature_extraction_lock = threading.Lock()

    def __init__(
        self,
        device: torch.device = None,
        max_feature_age: int = 7200,
        feature_history_size: int = 20,
        reid_threshold: float = 0.75,
        spatial_weight: float = 0.3,
        temporal_weight: float = 0.2,
        appearance_weight: float = 0.5,
        movement_pattern_weight: float = 0.0,
        batch_size: int = 32,
        max_image_dimension: int = 1920,
    ):
        # Use provided device or initialize a new one
        self.device = device if device is not None else self._init_device()
        self.reid_model = self._init_reid_model()

        # Enhanced feature management
        self.person_features: Dict[str, PersonFeature] = {}
        self.camera_to_global: Dict[Tuple[str, int], str] = {}
        self.track_id_to_global: Dict[int, str] = {}

        # Improved configuration - Fix timedelta initialization
        self.max_feature_age = timedelta(
            seconds=int(max_feature_age)
        )  # Ensure int conversion
        self.feature_history_size = feature_history_size
        self.reid_threshold = reid_threshold
        self.weights = {
            "appearance": appearance_weight,
            "spatial": spatial_weight,
            "temporal": temporal_weight,
            "movement": movement_pattern_weight,
        }

        # Performance optimizations
        self.batch_size = batch_size
        self.feature_cache = {}
        self.processing_lock = threading.Lock()
        self.appearance_buffer = deque(maxlen=100)

        self.max_image_dimension = max_image_dimension
        self.processed_faces = {}
        self.db = Database()

        # Enhanced tracking stability
        self.min_track_confidence = 0.3
        self.confidence_decay_rate = 0.95
        self.reactivation_threshold = 0.8
        self.movement_patterns: Dict[
            str, List[Tuple[datetime, Tuple[float, float]]]
        ] = {}

        # Improve temporal consistency
        self.id_timeout = timedelta(hours=24)  # Keep IDs for 24 hours
        self.feature_update_rate = 0.3  # Rate to update feature history
        self.confidence_threshold = 0.7  # Minimum confidence for ID reuse

        # Enhanced movement tracking
        self.velocity_history = defaultdict(list)
        self.position_history = defaultdict(list)
        self.max_history_size = 100

        # Performance optimization settings
        self.feature_cache_size = 100
        self.min_detection_size = (32, 32)
        self.target_size = (224, 224)  # Reduced size for ReID
        self.batch_process_timeout = 0.1

        # Add feature caching
        self.feature_cache = LRUCache(maxsize=self.feature_cache_size)

        self.max_spatial_distance = 100
        self.min_reuse_confidence = 0.6
        self.active_ids_per_frame = set()

    def _init_device(self) -> torch.device:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        else:
            device = torch.device("cpu")
        return device

    def _init_reid_model(self):
        # Initialize with a more robust backbone like ResNet50-IBN
        model = torch.hub.load("XingangPan/IBN-Net", "resnet50_ibn_a", pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, 2048)
        model = model.eval().to(self.device)
        return model

    def _extract_features_batch(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Optimized batch feature extraction"""
        if not frames:
            return []

        try:
            # Process in smaller sub-batches for better memory usage
            sub_batch_size = 8
            all_features = []

            for i in range(0, len(frames), sub_batch_size):
                sub_batch = frames[i : i + sub_batch_size]

                # Parallel preprocessing
                with ThreadPoolExecutor(max_workers=2) as executor:
                    processed_frames = list(
                        executor.map(self._preprocess_frame, sub_batch)
                    )

                # Stack and process sub-batch
                batch = torch.stack(processed_frames).to(self.device)

                # Ensure the input tensor is of the same type as the model weights
                if self.device.type == "cuda":
                    batch = batch.half()  # Convert to half precision if using CUDA

                with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True):
                    features = self.reid_model(batch)
                    features = torch.nn.functional.normalize(features, dim=1)
                    all_features.extend(features.cpu().numpy())

            return all_features

        except Exception as e:
            logger.error(f"Error in batch feature extraction: {e}")
            return []

    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Optimized frame preprocessing"""
        # Resize to target size
        frame = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_AREA)
        # Convert to RGB and normalize
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32) / 255.0
        frame = (frame - np.array([0.485, 0.456, 0.406])) / np.array(
            [0.229, 0.224, 0.225]
        )
        # Convert to tensor
        frame = torch.from_numpy(frame).permute(2, 0, 1)
        return frame

    def _extract_features(self, frame: np.ndarray) -> np.ndarray:
        """Extract robust feature embeddings"""
        with torch.no_grad():
            # Preprocessing
            img = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_AREA)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255.0
            img = (img - np.array([0.485, 0.456, 0.406])) / np.array(
                [0.229, 0.224, 0.225]
            )
            img = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)
            img = img.to(self.device)

            # Feature extraction with attention
            features = self.reid_model(img)
            features = torch.nn.functional.normalize(features, dim=1)
            return features.cpu().numpy()[0]

    def _compute_appearance_similarity(
        self, query_features: np.ndarray, gallery_features: List[np.ndarray]
    ) -> float:
        """Compute appearance similarity with improved matching"""
        if not gallery_features:
            return 0.0

        # Compute similarities with each historical feature
        similarities = []
        weights = np.linspace(0.5, 1.0, len(gallery_features))

        for feat, weight in zip(gallery_features, weights):
            # Use cosine similarity
            sim = (
                1
                - cdist(
                    query_features.reshape(1, -1), feat.reshape(1, -1), metric="cosine"
                )[0][0]
            )

            # Apply non-linear transformation to make matching more strict
            sim = np.power(sim, 1.5)  # This makes high similarities more important
            similarities.append(sim * weight)

        # Use both mean and max similarity for better discrimination
        mean_sim = np.mean(similarities)
        max_sim = np.max(similarities)

        return 0.7 * max_sim + 0.3 * mean_sim  # Weighted combination

    def _analyze_movement_patterns(
        self, pattern: List[Tuple[datetime, Tuple[float, float]]]
    ) -> Dict:
        """Analyze movement patterns to extract behavioral characteristics"""
        if len(pattern) < 2:
            return {}

        analysis = {
            "avg_velocity": 0,
            "direction_changes": 0,
            "stopping_points": 0,
            "activity_zones": set(),
            "typical_paths": [],
        }

        # Calculate velocities and accelerations
        velocities = []
        accelerations = []

        for i in range(1, len(pattern)):
            dt = (pattern[i][0] - pattern[i - 1][0]).total_seconds()
            if dt > 0:
                dx = pattern[i][1][0] - pattern[i - 1][1][0]
                dy = pattern[i][1][1] - pattern[i - 1][1][1]
                velocity = (dx / dt, dy / dt)
                velocities.append(velocity)

                if i > 1:
                    prev_velocity = velocities[-2]
                    ax = (velocity[0] - prev_velocity[0]) / dt
                    ay = (velocity[1] - prev_velocity[1]) / dt
                    accelerations.append((ax, ay))

                    # Detect direction changes
                    if self._is_direction_change(velocity, prev_velocity):
                        analysis["direction_changes"] += 1

        # Calculate average velocity
        if velocities:
            avg_velocity = np.mean([np.sqrt(vx**2 + vy**2) for vx, vy in velocities])
            analysis["avg_velocity"] = avg_velocity

        # Detect stopping points (low velocity areas)
        for velocity in velocities:
            if np.sqrt(velocity[0] ** 2 + velocity[1] ** 2) < 0.1:  # threshold
                analysis["stopping_points"] += 1

        # Identify activity zones
        for pos in pattern:
            zone = self._quantize_position(pos[1])
            analysis["activity_zones"].add(zone)

        # Extract typical paths
        if len(pattern) >= 3:
            analysis["typical_paths"] = self._extract_typical_paths(pattern)

        return analysis

    def _is_direction_change(
        self, v1: Tuple[float, float], v2: Tuple[float, float], threshold: float = 45
    ) -> bool:
        """Detect significant direction changes"""
        angle = np.arctan2(v1[1], v1[0]) - np.arctan2(v2[1], v2[0])
        angle = np.degrees(angle)
        return abs(angle) > threshold

    def _quantize_position(
        self, pos: Tuple[float, float], grid_size: float = 1.0
    ) -> Tuple[int, int]:
        """Quantize position into discrete zones"""
        return (int(pos[0] / grid_size), int(pos[1] / grid_size))

    def _extract_typical_paths(
        self, pattern: List[Tuple[datetime, Tuple[float, float]]]
    ) -> List[List[Tuple[float, float]]]:
        """Extract common movement paths"""
        paths = []
        current_path = []

        for i in range(1, len(pattern)):
            current_path.append(pattern[i][1])

            # Check for path break conditions (long time gap or stopping point)
            time_gap = (pattern[i][0] - pattern[i - 1][0]).total_seconds()
            if time_gap > 5.0 or self._is_stopping_point(
                pattern[i][1], pattern[i - 1][1]
            ):
                if len(current_path) >= 3:  # Minimum path length
                    paths.append(current_path.copy())
                current_path = []

        return paths

    def _compute_movement_pattern_similarity(
        self, track_id: int, candidate_id: str
    ) -> float:
        """Enhanced movement pattern comparison"""
        if (
            track_id not in self.velocity_history
            or candidate_id not in self.velocity_history
        ):
            return 0.0

        current_velocities = self.velocity_history[track_id][-10:]
        candidate_velocities = self.velocity_history[candidate_id][-10:]

        if not current_velocities or not candidate_velocities:
            return 0.0

        # Compare velocity patterns
        current_pattern = np.array(current_velocities)
        candidate_pattern = np.array(candidate_velocities)

        # DTW distance between velocity patterns
        distance = self._dtw_distance(current_pattern, candidate_pattern)
        similarity = 1.0 / (1.0 + distance)

        return similarity

    def _dtw_distance(self, pattern1: np.ndarray, pattern2: np.ndarray) -> float:
        """Calculate Dynamic Time Warping distance between patterns"""
        n, m = len(pattern1), len(pattern2)
        dtw_matrix = np.inf * np.ones((n + 1, m + 1))
        dtw_matrix[0, 0] = 0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = np.linalg.norm(pattern1[i - 1] - pattern2[j - 1])
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1]
                )

        return dtw_matrix[n, m]

    def _update_movement_patterns(
        self, track_id: int, position: Tuple[float, float], timestamp: datetime
    ):
        """Update movement patterns with velocity calculation"""
        self.position_history[track_id].append((position, timestamp))
        if len(self.position_history[track_id]) > self.max_history_size:
            self.position_history[track_id].pop(0)

        if len(self.position_history[track_id]) >= 2:
            pos1, t1 = self.position_history[track_id][-2]
            pos2, t2 = self.position_history[track_id][-1]

            dt = (t2 - t1).total_seconds()
            if dt > 0:
                velocity = ((pos2[0] - pos1[0]) / dt, (pos2[1] - pos1[1]) / dt)
                self.velocity_history[track_id].append(velocity)
                if len(self.velocity_history[track_id]) > self.max_history_size:
                    self.velocity_history[track_id].pop(0)

    def update(
        self,
        camera_id: str,
        person_crops: List[Tuple[int, np.ndarray]],
        positions: Optional[Dict[int, Tuple[float, float]]] = None,
    ) -> Dict[int, str]:
        """Updated main update function with improved tracking logic"""
        current_time = datetime.now()
        track_to_global = {}

        with self.processing_lock:
            # Update movement patterns
            if positions:
                for track_id, pos in positions.items():
                    self._update_movement_patterns(track_id, pos, current_time)

            # Process detections in batch for efficiency
            batch_features = self._extract_features_batch(
                [crop for _, crop in person_crops]
            )

            for (track_id, _), features in zip(person_crops, batch_features):
                global_id = self._match_or_create_identity(
                    track_id,
                    features,
                    camera_id,
                    positions,
                    current_time,
                    person_crops[0][1],
                )
                if global_id:
                    track_to_global[track_id] = global_id

        return track_to_global

    def _update_existing_track(
        self,
        global_id: str,
        track_id: int,
        crop: np.ndarray,
        camera_id: str,
        positions: Optional[Dict[int, Tuple[float, float]]],
        current_time: datetime,
    ):
        """Update existing track with new information"""
        person = self.person_features[global_id]

        # Extract and update features
        new_features = self._extract_features(crop)
        person.appearance_history.append(new_features)
        if len(person.appearance_history) > self.feature_history_size:
            person.appearance_history.pop(0)

        # Update temporal and spatial information
        person.last_seen = current_time
        person.temporal_features[camera_id] = current_time.timestamp()

        if positions and track_id in positions:
            person.spatial_features[camera_id] = positions[track_id]

        # Update camera history
        if camera_id not in person.camera_history:
            person.camera_history[camera_id] = []
        person.camera_history[camera_id].append(current_time)

        # Update movement patterns
        if track_id in positions:
            self.movement_patterns[global_id].append(
                (current_time, positions[track_id])
            )

        # Increase confidence with consistent tracking
        person.confidence = min(1.0, person.confidence + 0.1)

    def _handle_new_detection(
        self,
        track_id: int,
        crop: np.ndarray,
        camera_id: str,
        positions: Optional[Dict[int, Tuple[float, float]]],
        current_time: datetime,
        person_crop,
    ) -> Optional[str]:
        """Handle new detection with improved feature handling"""
        features = self._extract_features(crop)
        best_match = None
        best_score = 0

        # Check all existing identities including recently inactive ones
        for global_id, person in self.person_features.items():
            if current_time - person.last_seen > self.max_feature_age:
                continue

            # Compute comprehensive similarity score
            appearance_sim = self._compute_appearance_similarity(
                features, person.appearance_history
            )
            spatial_sim = (
                self._compute_spatial_similarity(
                    positions.get(track_id), person.spatial_features.get(camera_id)
                )
                if positions
                else 0
            )
            temporal_sim = self._compute_temporal_similarity(
                current_time.timestamp(), person.temporal_features.get(camera_id, 0)
            )
            movement_sim = self._compute_movement_pattern_similarity(
                track_id, global_id
            )

            # Weighted combination of similarities
            total_sim = (
                self.weights["appearance"] * appearance_sim
                + self.weights["spatial"] * spatial_sim
                + self.weights["temporal"] * temporal_sim
                + self.weights["movement"] * movement_sim
            )

            if (
                person.inactive_time > timedelta(0)
                and total_sim > self.reactivation_threshold
            ):
                total_sim *= 1.2

            if total_sim > best_score and total_sim > self.reid_threshold:
                best_score = total_sim
                best_match = global_id

        if best_match:
            self._update_existing_track(
                best_match, track_id, crop, camera_id, positions, current_time
            )
            self.camera_to_global[(camera_id, track_id)] = best_match
            return best_match
        else:
            # Create new identity - pass the actual image crop, not features
            new_id = f"person_{len(self.person_features)}"
            new_person = PersonFeature(
                global_id=new_id,
                reid_features=features,
                appearance_history=[features],
                temporal_features={camera_id: current_time.timestamp()},
                spatial_features=(
                    {camera_id: positions.get(track_id)} if positions else {}
                ),
                camera_history={camera_id: [current_time]},
                track_id_history=[track_id],
                confidence=0.3,
            )
            self.person_features[new_id] = new_person
            self.camera_to_global[(camera_id, track_id)] = new_id
            self.movement_patterns[new_id] = []
            if track_id in positions:
                self.movement_patterns[new_id].append(
                    (current_time, positions[track_id])
                )

            # Extract face and initiate upload process
            try:
                face_crop = self._extract_face(person_crop)
                if face_crop is not None:
                    # Create task for face upload
                    asyncio.run(
                        self.handle_face_upload(new_person, camera_id, face_crop)
                    )
            except Exception as e:
                logger.error(f"Error handling face upload for new person: {e}")

            return new_id

    def _extract_face(self, person_crop: np.ndarray) -> Optional[np.ndarray]:
        """Extract face from person crop using MediaPipe with improved format handling"""
        try:
            # Ensure input image is properly formatted
            if person_crop is None or not isinstance(person_crop, np.ndarray):
                logger.warning("Invalid input for face detection - not a numpy array")
                return None

            if len(person_crop.shape) != 3 or person_crop.shape[2] != 3:
                logger.warning(
                    f"Invalid image format for face detection - shape: {person_crop.shape}"
                )
                return None

            # Ensure input image is uint8 with proper scaling
            if person_crop.dtype != np.uint8:
                if np.issubdtype(person_crop.dtype, np.floating):
                    if person_crop.max() <= 1.0:  # Assuming [0,1] range
                        person_crop = (person_crop * 255).clip(0, 255).astype(np.uint8)
                    else:  # Assuming actual pixel values
                        person_crop = person_crop.clip(0, 255).astype(np.uint8)
                else:
                    person_crop = person_crop.astype(np.uint8)

            # Check minimum size requirement
            if person_crop.shape[0] < 32 or person_crop.shape[1] < 32:
                logger.debug("Person crop too small for face detection")
                return None

            # Create face detector with explicit model selection
            with mp.solutions.face_detection.FaceDetection(
                model_selection=1,  # Use the full-range model
                min_detection_confidence=0.5,
            ) as face_detector:
                # Convert to RGB if necessary (MediaPipe expects RGB)
                rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)

                # Process the image
                results = face_detector.process(rgb_crop)

                if results.detections:
                    detection = results.detections[0]  # Get the first face detected
                    bbox = detection.location_data.relative_bounding_box

                    # Convert relative coordinates to absolute
                    h, w = rgb_crop.shape[:2]
                    x = max(0, int(bbox.xmin * w))
                    y = max(0, int(bbox.ymin * h))
                    width = min(w - x, int(bbox.width * w))
                    height = min(h - y, int(bbox.height * h))

                    # Validate dimensions
                    if width <= 0 or height <= 0:
                        logger.warning("Invalid face crop dimensions")
                        return None

                    # Extract and process face region
                    face_crop = person_crop[y : y + height, x : x + width]

                    # Final validation
                    if face_crop.size == 0:
                        logger.warning("Empty face crop")
                        return None

                    # Resize to standard size
                    face_crop = cv2.resize(face_crop, (112, 112))
                    return face_crop

            logger.debug("No face detected in person crop")
            return None

        except Exception as e:
            logger.error(f"Error extracting face: {e}")
            return None

    def _cleanup_old_features(self, current_time: datetime):
        """Clean up old features with improved logic"""
        expired_ids = []
        for global_id, person in self.person_features.items():
            time_since_last_seen = current_time - person.last_seen

            if time_since_last_seen > self.max_feature_age:
                expired_ids.append(global_id)
            elif time_since_last_seen > timedelta(minutes=5):
                # Mark as inactive but don't remove yet
                person.inactive_time = time_since_last_seen
                person.confidence *= self.confidence_decay_rate

        # Remove expired identities
        for global_id in expired_ids:
            del self.person_features[global_id]
            # Clean up related mappings
            self.camera_to_global = {
                k: v for k, v in self.camera_to_global.items() if v != global_id
            }
            if global_id in self.movement_patterns:
                del self.movement_patterns[global_id]

    def _match_or_create_identity(
        self,
        track_id: int,
        features: np.ndarray,
        camera_id: str,
        positions: Optional[Dict[int, Tuple[float, float]]],
        current_time: datetime,
        person_crop,
    ) -> Optional[str]:
        """Match to existing identity or create new one"""
        # Check if we already have a mapping for this track_id
        if (camera_id, track_id) in self.camera_to_global:
            global_id = self.camera_to_global[(camera_id, track_id)]
            if global_id in self.person_features:
                self._update_existing_track(
                    global_id, track_id, features, camera_id, positions, current_time
                )
                return global_id

        # Try to match with existing identities
        return self._handle_new_detection(
            track_id, features, camera_id, positions, current_time, person_crop
        )

    @lru_cache(maxsize=128)
    def _compute_spatial_similarity(
        self,
        current_pos: Optional[Tuple[float, float]],
        last_pos: Optional[Tuple[float, float]],
    ) -> float:
        """Cached spatial similarity computation"""
        if current_pos is None or last_pos is None:
            return 0.0

        distance = np.sqrt(
            (current_pos[0] - last_pos[0]) ** 2 + (current_pos[1] - last_pos[1]) ** 2
        )
        return 1.0 / (1.0 + distance)

    def _compute_temporal_similarity(
        self, current_time: float, last_time: float
    ) -> float:
        """Compute temporal similarity between timestamps"""
        if last_time == 0:
            return 0.0

        time_diff = abs(current_time - last_time)
        return np.exp(-time_diff / 3600.0)  # Decay over 1 hour

    def _is_stopping_point(
        self,
        pos1: Tuple[float, float],
        pos2: Tuple[float, float],
        threshold: float = 0.1,
    ) -> bool:
        """Determine if two positions indicate a stopping point"""
        distance = np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)
        return distance < threshold

    async def validate_and_encode_face_image(
        self, face_image: np.ndarray
    ) -> Optional[bytes]:
        """Validate and encode face image according to AWS Rekognition requirements"""
        try:
            if face_image is None:
                logger.error("Input face image is None")
                return None

            # Check image dimensions
            height, width = face_image.shape[:2]
            if height < 80 or width < 80:
                logger.error(
                    f"Image too small: {width}x{height}, minimum 80x80 required"
                )
                return None

            # Resize if image is too large
            if height > self.max_image_dimension or width > self.max_image_dimension:
                scale = self.max_image_dimension / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                face_image = cv2.resize(face_image, (new_width, new_height))

            # Encode image with quality check
            _, img_encoded = cv2.imencode(
                ".jpg", face_image, [cv2.IMWRITE_JPEG_QUALITY, 85]
            )

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
        timeout: int = 60,
    ) -> None:
        """Upload face image to S3 using presigned URL"""
        try:
            logger.debug(f"Uploading to URL: {url}")

            async with session.put(
                url,
                data=img_bytes,
                headers={"Content-Type": "*"},
                skip_auto_headers=["Content-Type"],
                timeout=30,
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(f"Upload failed with status {resp.status}")
                    logger.error(f"Response body: {error_text}")
                    logger.error(f"Final URL: {str(resp.url)}")
                    raise Exception(
                        f"Upload failed with status {resp.status}: {error_text}"
                    )

                logger.info(
                    f"Successfully uploaded image. Response status: {resp.status}"
                )

        except asyncio.TimeoutError:
            logger.error(f"Upload timed out after {timeout}s")
            raise
        except Exception as e:
            logger.error(f"Upload failed: {str(e)}")
            raise

    async def handle_face_upload(
        self, person: PersonFeature, camera_id: str, face_crop: np.ndarray
    ) -> None:
        """Handle face upload and recognition for new person"""
        logger.info(f"Handling face upload for person {person.global_id}")
        nats_client = None

        try:
            # Validate and encode image
            img_bytes = await self.validate_and_encode_face_image(face_crop)
            if img_bytes is None:
                logger.error(
                    f"Failed to validate/encode face image for person {person.global_id}"
                )
                person.recognition_status = "pending"
                return

            # Generate unique face ID if none exists
            if person.face_id is None:
                person.face_id = str(uuid.uuid4())
                logger.info(
                    f"Generated new face ID {person.face_id} for person {person.global_id}"
                )

            # Update status to in_progress
            person.recognition_status = "in_progress"

            # Get NATS connection
            nats_url = await self.db.get_cloud_url()
            nats_client = NatsClient(nats_url)
            await nats_client.connect()

            company_id = await self.db.get_company_id()

            # Get presigned URL
            response = await nats_client.send_message_with_reply(
                Commands.GET_PRESIGNED_UPLOAD_UNKNOWN_FACE_URL.value,
                {
                    "company_id": company_id,
                    "face_id": person.face_id,
                },
            )

            if not response or not response.get("success"):
                logger.error(f"Failed to get presigned URL: {response}")
                person.recognition_status = "pending"
                return

            presigned_url = response.get("url")
            if not presigned_url:
                logger.error("No presigned URL in response")
                person.recognition_status = "pending"
                return

            # Upload image
            async with aiohttp.ClientSession() as session:
                for attempt in range(3):  # Try up to 3 times
                    try:
                        await self.upload_face_image(session, presigned_url, img_bytes)
                        logger.info(
                            f"Successfully uploaded image for face_id {person.face_id}"
                        )
                        break
                    except Exception as e:
                        if attempt == 2:  # Last attempt
                            logger.error(
                                f"All upload attempts failed for face_id {person.face_id}"
                            )
                            person.recognition_status = "pending"
                            return
                        logger.warning(f"Upload attempt {attempt + 1} failed: {str(e)}")
                        await asyncio.sleep(1)

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
            try:
                await self.db.update_person_track(
                    global_id=person.global_id,
                    camera_id=camera_id,
                    face_id=person.face_id,
                    recognition_status=person.recognition_status,
                )
            except Exception as e:
                logger.error(f"Failed to update database: {str(e)}")

        except Exception as e:
            logger.error(f"Error in face upload: {str(e)}")
            person.recognition_status = "pending"
        finally:
            if nats_client:
                try:
                    await nats_client.close()
                except Exception as e:
                    logger.error(f"Error closing NATS client: {str(e)}")
