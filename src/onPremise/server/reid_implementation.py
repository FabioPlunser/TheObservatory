import logging
from logging_config import setup_logger
import numpy as np
import torch
from torchreid import models, utils
import cv2
from dataclasses import dataclass
from datetime import datetime, timedelta
import os
from typing import Dict, List, Optional, Tuple
from scipy.spatial.distance import cdist

logger = logging.getLogger("ReID")


@dataclass
class PersonFeature:
    reid_features: np.ndarray  # ReID feature vector
    appearance_features: List[np.ndarray]  # List of historical appearance features
    temporal_features: Dict[str, float]  # Camera-specific temporal features
    spatial_features: Dict[str, Tuple[float, float]]  # Last known positions per camera
    last_seen: datetime
    camera_id: str
    track_id: int
    face_id: Optional[str] = None
    recognition_status: str = "pending"
    confidence: float = 0.0


class CrossCameraTracker:
    def __init__(
        self,
        max_feature_age: int = 300,  # 5 minutes default
        feature_history_size: int = 5,
        reid_threshold: float = 0.6,
        spatial_weight: float = 0.3,
        temporal_weight: float = 0.2,
    ):

        # Initialize ReID model
        self.reid_model = models.build_model(
            name="osnet_ain_x1_0",  # Using a more advanced OSNet architecture
            num_classes=1000,
            loss="softmax",
            pretrained=True,
        )
        self.reid_model = self.reid_model.eval()

        # Set up hardware acceleration
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.reid_model = self.reid_model.cuda()
            logger.info("Using CUDA for ReID model")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.reid_model = self.reid_model.to(self.device)
            logger.info("Using Apple M1 MPS acceleration for ReID model")
        else:
            self.device = torch.device("cpu")
            torch.set_num_threads(os.cpu_count())
            logger.info(f"Using CPU for ReID model with {os.cpu_count()} threads")

        self.person_features: Dict[str, PersonFeature] = {}
        self.camera_to_global: Dict[Tuple[str, int], str] = {}

        # Configuration parameters
        self.max_feature_age = timedelta(seconds=max_feature_age)
        self.feature_history_size = feature_history_size
        self.reid_threshold = reid_threshold
        self.spatial_weight = spatial_weight
        self.temporal_weight = temporal_weight

    def _extract_reid_features(self, person_crop: np.ndarray) -> Optional[np.ndarray]:
        """Extract ReID features with improved preprocessing"""
        try:
            if person_crop.shape[0] < 64 or person_crop.shape[1] < 32:
                return None

            # Enhanced preprocessing pipeline
            img = cv2.resize(person_crop, (256, 256))  # Larger size for better detail

            # Apply CLAHE for better contrast
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[..., 0] = clahe.apply(lab[..., 0])
            img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

            # Normalize
            img = img.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = (img - mean) / std

            # Convert to tensor
            img = torch.from_numpy(img).float()
            img = img.permute(2, 0, 1).unsqueeze(0)
            img = img.to(self.device)

            # Extract features with gradient accumulation for stability
            features_list = []
            with torch.no_grad():
                # Multiple forward passes with slight augmentation
                features = self.reid_model(img)
                features_list.append(features)

                # Horizontal flip
                img_flip = torch.flip(img, [3])
                features_flip = self.reid_model(img_flip)
                features_list.append(features_flip)

            # Average the features
            features = torch.mean(torch.stack(features_list), dim=0)
            features = features.cpu().numpy()

            # L2 normalization
            features = features / np.linalg.norm(features)

            return features

        except Exception as e:
            logger.error(f"Error extracting ReID features: {e}")
            return None

    def _compute_similarity_matrix(
        self, query_features: np.ndarray, gallery_features: List[np.ndarray]
    ) -> np.ndarray:
        """Compute similarity matrix between query and gallery features"""
        if not gallery_features:
            return np.array([])

        gallery_features = np.vstack(gallery_features)
        similarities = 1 - cdist(
            query_features.reshape(1, -1), gallery_features, metric="cosine"
        )
        return similarities.flatten()

    def _compute_spatial_similarity(
        self, pos1: Tuple[float, float], pos2: Tuple[float, float]
    ) -> float:
        """Compute spatial similarity based on positions"""
        if not all([pos1, pos2]):
            return 0.0
        dist = np.sqrt(((pos1[0] - pos2[0]) ** 2) + ((pos1[1] - pos2[1]) ** 2))
        return np.exp(-dist / 100)  # Decay factor of 100 pixels

    def _compute_temporal_similarity(self, time1: float, time2: float) -> float:
        """Compute temporal similarity based on timestamps"""
        time_diff = abs(time1 - time2)
        return np.exp(-time_diff / 30)  # Decay factor of 30 seconds

    def update(
        self,
        camera_id: str,
        person_crops: List[Tuple[int, np.ndarray]],
        positions: Optional[Dict[int, Tuple[float, float]]] = None,
        face_ids: Optional[Dict[int, str]] = None,
        recognition_statuses: Optional[Dict[int, str]] = None,
    ) -> Dict[int, str]:
        """Update cross-camera tracking with improved matching"""
        current_time = datetime.now()
        track_to_global = {}

        # Clean up old features first
        self._cleanup_old_features(current_time)

        for track_id, crop in person_crops:
            camera_track = (camera_id, track_id)
            position = positions.get(track_id) if positions else None

            # If we've seen this camera-track combo before
            if camera_track in self.camera_to_global:
                global_id = self.camera_to_global[camera_track]
                track_to_global[track_id] = global_id

                # Update features and metadata
                if global_id in self.person_features:
                    person = self.person_features[global_id]
                    reid_features = self._extract_reid_features(crop)

                    if reid_features is not None:
                        # Update feature history
                        person.appearance_features.append(reid_features)
                        if len(person.appearance_features) > self.feature_history_size:
                            person.appearance_features.pop(0)

                        # Update spatial-temporal information
                        if position:
                            person.spatial_features[camera_id] = position
                        person.temporal_features[camera_id] = current_time.timestamp()
                        person.last_seen = current_time

                    # Update recognition info
                    if face_ids and track_id in face_ids:
                        person.face_id = face_ids[track_id]
                    if recognition_statuses and track_id in recognition_statuses:
                        person.recognition_status = recognition_statuses[track_id]

                continue

            # Extract features for new detection
            reid_features = self._extract_reid_features(crop)
            if reid_features is None:
                continue

            # Compare with existing features
            candidates = []
            for global_id, feature_data in self.person_features.items():
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
            self.camera_to_global[camera_track] = global_id
            track_to_global[track_id] = global_id

        return track_to_global

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

    def get_person_info(self, global_id: str) -> Optional[PersonFeature]:
        """Get information about a person by their global ID"""
        return self.person_features.get(global_id)


# Singleton implementation
class SharedCrossCameraTracker:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = CrossCameraTracker()
        return cls._instance
