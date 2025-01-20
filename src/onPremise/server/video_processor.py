import cv2
import numpy as np
import torch
import logging
import time
import os
import threading
import multiprocessing as mp
import redis
import pickle
from collections import deque, defaultdict
import asyncio
import colorsys

from redis.exceptions import RedisError
from typing import Dict, List, Optional, Tuple, Set
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor
from logging_config import setup_logger
from contextlib import nullcontext
from reid_implementation import Reid

setup_logger()
logger = logging.getLogger("VideoProcessor")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class VideoProcessor:
    def __init__(self, batch_size: int = 16):
        # GPU initialization with optimized settings
        self.device = self._init_device()
        self.stream = torch.cuda.Stream() if torch.cuda.is_available() else None

        # Initialize YOLO model with optimizations
        self.model = self._init_model()

        # Processing settings
        self.batch_size = batch_size
        self.frame_skip = 3  # Process every 3rd frame
        self.target_fps = 15  # Target processing FPS
        self.min_frame_interval = 1.0 / self.target_fps
        self.output_scale = (640, 480)
        self.max_batch_age = 0.1

        # Thread pools for preprocessing and postprocessing
        self.preprocess_pool = ThreadPoolExecutor(max_workers=4)
        self.postprocess_pool = ThreadPoolExecutor(max_workers=4)

        # Redis configuration with connection pooling
        self.redis_pool = redis.ConnectionPool(
            host="localhost",
            port=6379,
            db=0,
            decode_responses=False,
            max_connections=50,
        )
        self.redis_client = redis.Redis(connection_pool=self.redis_pool)

        # Redis keys
        self.FRAME_QUEUE_KEY = "frame_queue:{camera_id}"
        self.PROCESSED_FRAME_KEY = "processed_frame:{camera_id}"
        self.FRAME_EXPIRE_TIME = 5

        # Batch processing
        self.batch_frames = []
        self.batch_metadata = []
        self.last_batch_time = time.time()
        self.active_cameras: Set[str] = set()

        # Performance tracking
        self.processing_times = defaultdict(deque)
        self.fps_counters = defaultdict(lambda: {"count": 0, "last_time": time.time()})

        # State management
        self.camera_threads = {}
        self.stop_events = {}
        self.reid_manager = Reid()
        self.stop_event = threading.Event()

        # Start processing threads
        self._start_processing_threads()

        monitor_thread = threading.Thread(target=self._monitor_performance, daemon=True)
        monitor_thread.start()

    def _init_device(self) -> torch.device:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            # Optimize CUDA settings
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            torch.set_float32_matmul_precision("high")
            if torch.cuda.get_device_capability()[0] >= 7:
                torch.backends.cuda.enable_mem_efficient_sdp(True)
            # Pre-allocate CUDA memory
            torch.cuda.empty_cache()
            torch.cuda.memory.empty_cache()
            return device
        return torch.device("cpu")

    def _start_processing_threads(self):
        """Start processing threads with different responsibilities"""
        # Batch collection thread
        self.batch_thread = threading.Thread(target=self._batch_collector, daemon=True)
        self.batch_thread.start()

        # Processing threads
        num_processing_threads = 2
        self.processing_threads = []
        for _ in range(num_processing_threads):
            thread = threading.Thread(target=self._batch_processor, daemon=True)
            thread.start()
            self.processing_threads.append(thread)

        # Performance monitoring thread
        self.monitor_thread = threading.Thread(
            target=self._monitor_performance, daemon=True
        )
        self.monitor_thread.start()

    def _init_model(self) -> YOLO:
        model = YOLO("yolov8n.pt", verbose=False)
        model.to(self.device)
        if torch.cuda.is_available():
            model.fuse()
        return model

    def _batch_collector(self):
        """Collect frames into batches efficiently"""
        while not self.stop_event.is_set():
            try:
                batch_frames = []
                batch_metadata = []

                # Collect frames from all active cameras
                for camera_id in self.active_cameras.copy():
                    try:
                        pipe = self.redis_client.pipeline()
                        # Get multiple frames at once
                        for _ in range(2):  # Try to get 2 frames per camera
                            pipe.rpop(self.FRAME_QUEUE_KEY.format(camera_id=camera_id))
                        frame_datas = pipe.execute()

                        for frame_data in frame_datas:
                            if frame_data:
                                data = self._deserialize_frame_data(frame_data)
                                if data:
                                    frame = self._preprocess_frame(data)
                                    if frame is not None:
                                        batch_frames.append(frame)
                                        batch_metadata.append(data)

                                if len(batch_frames) >= self.batch_size:
                                    break

                    except RedisError as e:
                        logger.error(f"Redis error in batch collection: {e}")

                if batch_frames:
                    # Store batch in Redis for processing
                    batch_id = str(time.time())
                    try:
                        self.redis_client.setex(
                            f"batch:{batch_id}",
                            self.FRAME_EXPIRE_TIME,
                            self._serialize_frame_data(
                                {"frames": batch_frames, "metadata": batch_metadata}
                            ),
                        )
                    except RedisError as e:
                        logger.error(f"Redis error storing batch: {e}")

                time.sleep(0.001)  # Prevent CPU overuse

            except Exception as e:
                logger.error(f"Error in batch collector: {e}")
                time.sleep(0.1)

    def _preprocess_frame(self, data: dict) -> Optional[np.ndarray]:
        """Preprocess frame with optimization"""
        try:
            frame = cv2.imdecode(
                np.frombuffer(data["frame"], np.uint8), cv2.IMREAD_COLOR
            )
            if frame is None:
                return None

            # Efficient CPU preprocessing
            frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)

            return frame

        except Exception as e:
            logger.error(f"Error preprocessing frame: {e}")
            return None

    def _batch_processor(self):
        """Process batches with optimized GPU utilization"""
        while not self.stop_event.is_set():
            try:
                # Look for available batches
                for key in self.redis_client.scan_iter("batch:*"):
                    try:
                        batch_data = self.redis_client.get(key)
                        if batch_data:
                            batch = self._deserialize_frame_data(batch_data)
                            if batch:
                                self._process_batch(batch["frames"], batch["metadata"])
                            self.redis_client.delete(key)
                    except RedisError as e:
                        logger.error(f"Redis error in batch processing: {e}")

                time.sleep(0.001)

            except Exception as e:
                logger.error(f"Error in batch processor: {e}")
                time.sleep(0.1)

    def _serialize_frame_data(self, frame_data: dict) -> bytes:
        try:
            return pickle.dumps(frame_data)
        except Exception as e:
            logger.error(f"Error serializing frame data: {e}")
            return b""

    def _deserialize_frame_data(self, frame_data: bytes) -> dict:
        try:
            return pickle.loads(frame_data)
        except Exception as e:
            logger.error(f"Error deserializing frame data: {e}")
            return {}

    def add_camera(self, camera_id: str, rtsp_url: str, company_id: str):
        """Add camera with optimized frame reader"""

        logger.info(f"Adding camera {camera_id} with RTSP: {rtsp_url}")
        self.stop_events[camera_id] = threading.Event()
        self.active_cameras.add(camera_id)

        process = threading.Thread(
            target=self._frame_reader_process,
            args=(camera_id, rtsp_url, company_id, self.stop_events[camera_id]),
            daemon=True,
        )
        process.start()
        self.camera_threads[camera_id] = process

    def remove_camera(self, camera_id: str):
        """Remove camera and clean up resources"""
        if camera_id in self.stop_events:
            self.stop_events[camera_id].set()
            self.active_cameras.discard(camera_id)

            if camera_id in self.camera_threads:
                self.camera_threads[camera_id].join(timeout=5.0)
                del self.camera_threads[camera_id]
            del self.stop_events[camera_id]

            # Clean up Redis keys
            try:
                self.redis_client.delete(
                    self.FRAME_QUEUE_KEY.format(camera_id=camera_id),
                    self.PROCESSED_FRAME_KEY.format(camera_id=camera_id),
                )
            except RedisError as e:
                logger.error(f"Redis error cleaning up camera: {e}")

    def _frame_reader_process(
        self,
        camera_id: str,
        rtsp_url: str,
        company_id: str,
        stop_event: threading.Event,
    ):
        cap = None
        frame_counter = 0
        last_frame_time = time.time()
        frame_queue_key = self.FRAME_QUEUE_KEY.format(camera_id=camera_id)

        max_retries: int = 10
        retry_count = 0

        logger.info("Starting frame reader for camera {camera_id}")

        def initialize_capture():
            nonlocal cap, retry_count

            if cap is not None:
                cap.release()
                cap = None
            try:
                logger.info(f"Initializing capture for camera {camera_id}")
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
                cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                logger.info(f"Opened RTSP stream: {rtsp_url}")

                if not cap.isOpened():
                    retry_count += 1
                    logger.warning(
                        f"Attempt {retry_count}/{max_retries} failed to open RTSP stream: {rtsp_url}"
                    )
                    if retry_count < max_retries:
                        return False
                    else:
                        logger.error(
                            f"Failed to open RTSP stream after {max_retries} attempts: {rtsp_url}"
                        )
                        return False

                logger.info(f"Successfully opened RTSP stream: {rtsp_url}")
                retry_count = 0  # Reset retry count on successful connection
                return True

            except Exception as e:
                retry_count += 1
                logger.error(
                    f"Error initializing capture (attempt {retry_count}/{max_retries}): {e}"
                )
                if retry_count < max_retries:
                    return False
                return False

        while not stop_event.is_set():
            try:
                if cap is None or not cap.isOpened():
                    if not initialize_capture():
                        if retry_count >= max_retries:
                            logger.error(
                                "Maximum retry attempts reached. Stopping frame reader."
                            )
                            break
                        continue

                # Use read timeout to prevent hanging
                read_success = False
                read_start = time.time()
                while time.time() - read_start < 1.0:  # 1 second timeout for frame read
                    ret, frame = cap.read()
                    if ret:
                        read_success = True
                        break
                    time.sleep(0.01)  # Small sleep to prevent CPU spinning

                if not read_success:
                    logger.warning(f"Frame read timeout for camera {camera_id}")
                    cap.release()
                    cap = None
                    continue

                frame_counter += 1
                if frame_counter % self.frame_skip != 0:
                    continue

                current_time = time.time()
                if current_time - last_frame_time < self.min_frame_interval:
                    continue

                last_frame_time = current_time

                # Process frame
                try:
                    frame = cv2.resize(frame, (640, 480))
                    _, buffer = cv2.imencode(
                        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80]
                    )

                    # Use pipeline for atomic Redis operations
                    pipe = self.redis_client.pipeline()
                    frame_data = {
                        "frame": buffer.tobytes(),
                        "camera_id": camera_id,
                        "company_id": company_id,
                        "timestamp": current_time,
                    }

                    pipe.lpush(frame_queue_key, self._serialize_frame_data(frame_data))
                    pipe.ltrim(frame_queue_key, 0, 5)  # Keep only last 5 frames
                    pipe.expire(frame_queue_key, self.FRAME_EXPIRE_TIME)
                    pipe.execute()

                except RedisError as e:
                    logger.error(f"Redis error for camera {camera_id}: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Frame processing error for camera {camera_id}: {e}")
                    continue

            except Exception as e:
                logger.error(f"Error in frame reader for camera {camera_id}: {e}")
                if cap is not None:
                    cap.release()
                    cap = None
                time.sleep(1)

        # Cleanup
        if cap is not None:
            cap.release()
        logger.info(f"Frame reader stopped for camera {camera_id}")

    def _process_batch(self, frames: List[np.ndarray], metadata: List[dict]):
        """Process batch with optimized GPU memory usage"""
        try:
            # Process through YOLO with mixed precision
            with torch.amp.autocast("cuda", enabled=True):
                results = self.model.track(
                    source=frames,
                    conf=0.5,
                    iou=0.7,
                    persist=True,
                    tracker="bytetrack.yaml",
                    device=self.device,
                    verbose=False,
                )

            # Process results in parallel
            futures = []
            for result, meta in zip(results, metadata):
                future = self.postprocess_pool.submit(
                    self._process_detections,
                    frames[metadata.index(meta)],
                    result,
                    meta["camera_id"],
                )
                futures.append((future, meta["camera_id"]))

            # Store results as they complete
            for future, camera_id in futures:
                try:
                    processed_frame = future.result(timeout=1.0)
                    if processed_frame is not None:
                        _, buffer = cv2.imencode(
                            ".jpg", processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 80]
                        )

                        self.redis_client.setex(
                            self.PROCESSED_FRAME_KEY.format(camera_id=camera_id),
                            self.FRAME_EXPIRE_TIME,
                            buffer.tobytes(),
                        )

                        # Update FPS counter
                        self._update_fps(camera_id)

                except Exception as e:
                    logger.error(f"Error processing result: {e}")

        except Exception as e:
            logger.error(f"Error in batch processing: {e}")

    def _update_fps(self, camera_id: str):
        """Update FPS counter for camera"""
        counter = self.fps_counters[camera_id]
        counter["count"] += 1
        current_time = time.time()
        elapsed = current_time - counter["last_time"]

        if elapsed >= 5.0:  # Log FPS every 5 seconds
            fps = counter["count"] / elapsed
            logger.info(f"Camera {camera_id} - FPS: {fps:.2f}")
            counter["count"] = 0
            counter["last_time"] = current_time

    def _monitor_performance(self):
        """Monitor system performance"""
        while not self.stop_event.is_set():
            try:
                if torch.cuda.is_available():
                    # Log GPU memory usage
                    allocated = torch.cuda.memory_allocated() / 1024**2
                    reserved = torch.cuda.memory_reserved() / 1024**2
                    logger.info(
                        f"GPU Memory - Allocated: {allocated:.1f}MB, Reserved: {reserved:.1f}MB"
                    )

                # Log processing times
                for camera_id, times in self.processing_times.items():
                    if times:
                        avg_time = sum(times) / len(times)
                        logger.info(
                            f"Camera {camera_id} - Avg processing time: {avg_time*1000:.1f}ms"
                        )

                time.sleep(10)  # Log every 10 seconds

            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")
                time.sleep(1)

    def _process_detections(self, frame, result, camera_id):
        """Process detections with ReID"""
        try:
            person_crops = []
            positions = {}
            boxes = result.boxes
            track_ids = boxes.id.cpu().numpy() if boxes.id is not None else None

            if track_ids is None:
                return frame

            for box, track_id in zip(boxes.xyxy.cpu().numpy(), track_ids):
                x1, y1, x2, y2 = map(int, box)
                crop = frame[y1:y2, x1:x2]
                person_crops.append((int(track_id), crop))
                positions[int(track_id)] = ((x1 + x2) / 2, (y1 + y2) / 2)

            # Get global IDs from ReID
            global_ids = self.reid_manager.update(
                camera_id=camera_id, person_crops=person_crops, positions=positions
            )

            # Draw results
            draw_frame = frame.copy()
            for box, track_id in zip(boxes.xyxy.cpu().numpy(), track_ids):
                x1, y1, x2, y2 = map(int, box)
                global_id = global_ids.get(int(track_id))
                if global_id:
                    color = self._get_id_color(global_id)
                    cv2.rectangle(draw_frame, (x1, y1), (x2, y2), color, 2)
                    text = f"ID: {global_id}"
                    cv2.putText(
                        draw_frame,
                        text,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2,
                    )

            return cv2.resize(draw_frame, self.output_scale)

        except Exception as e:
            logger.error(f"Error processing detections: {e}")
            return frame

    def _get_id_color(self, global_id: str) -> Tuple[int, int, int]:
        """Generate consistent color for global ID"""
        hash_val = hash(global_id)
        hue = (hash_val % 360) / 360.0
        rgb = [int(x * 255) for x in colorsys.hsv_to_rgb(hue, 0.9, 0.9)]
        return (rgb[2], rgb[1], rgb[0])  # BGR for OpenCV

    def get_frame(self, camera_id: str) -> Optional[bytes]:
        """Get latest processed frame from Redis"""
        try:
            return self.redis_client.get(
                self.PROCESSED_FRAME_KEY.format(camera_id=camera_id)
            )
        except RedisError as e:
            logger.error(f"Redis error getting frame: {e}")
            return None

    def stop(self):
        """Stop all processing"""
        self.stop_event.set()
        for camera_id in list(self.stop_events.keys()):
            self.remove_camera(camera_id)
