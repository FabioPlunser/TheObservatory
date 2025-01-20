import cv2
import numpy as np
import torch
import logging
import time
import queue
import os
import threading
import multiprocessing as mp
import colorsys

from typing import Dict, List, Optional, Tuple
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict, deque
from logging_config import setup_logger
from contextlib import nullcontext
from reid_implementation import Reid

setup_logger()
logger = logging.getLogger("VideoProcessor")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class VideoProcessor:
    """
    Shared video processor that handles multiple cameras efficiently
    """

    def __init__(self, batch_size: int = 16):  # Increased batch size
        # Unified timing settings at the start
        self.min_frame_interval = 1.0 / 30.0  # Target 30 FPS
        self.batch_timeout = 0.033  # 30 FPS equivalent
        self.max_batch_age = 0.1  # Maximum age of frames in batch
        self.min_batch_wait = 0.001  # Minimum wait time between batches

        # Add frame ordering settings
        self.ordered_frames: Dict[str, List[Tuple[float, np.ndarray]]] = defaultdict(list)
        self.frame_buffer_time = 0.5  # Buffer 500ms of frames
        self.max_frame_delay = 1.0    # Maximum delay before dropping frames
        self.frame_reorder_threshold = 0.1  # Threshold for reordering (100ms)

        # Initlialize GPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            torch.set_float32_matmul_precision("high")
            if torch.cuda.get_device_capability()[0] >= 7:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cuda.enable_mem_efficient_sdp(True)
            logger.info("Using CUDA GPU")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
            logger.info("Using Apple M1/M2 GPU (MPS)")
        else:
            self.device = torch.device("cpu")
            torch.set_num_threads(mp.cpu_count())
            logger.info("Using CPU")

        self.model = YOLO("yolov8n.pt", verbose=False)
        self.model.to(self.device)
        if torch.cuda.is_available():
            self.model.fuse()

        logger.info("Device: {}".format(self.device))

        self.batch_size = batch_size
        self.frame_skip = 3  # Skip more frames
        self.max_queue_size = 30
        self.processing_scale = 0.5  # Scale factor for processing
        self.output_scale = (640, 480)  # Output resolution

        # Batch processing optimization
        self.min_batch_size = 4
        self.max_batch_size = 16
        self.batch_timeout = 0.066  # ~15 FPS equivalent

        # Add worker pool for frame preprocessing
        self.preprocess_pool = ThreadPoolExecutor(max_workers=2)
        self.postprocess_pool = ThreadPoolExecutor(max_workers=2)

        # Frame ordering and buffering
        self.frame_timestamps: Dict[str, deque] = {}
        self.frame_buffer_size = 5
        self.last_processed_time: Dict[str, float] = {}

        self.frame_queue = mp.Queue(maxsize=self.max_queue_size)
        self.output_queues: Dict[str, queue.Queue] = {}
        self.frame_buffers: Dict[str, deque] = {}

        self.stop_event = threading.Event()

        self.camera_threads = {}
        self.stop_events = {}
        self.person_trackers: Dict[str, Reid] = {}

        self.fps_counters = {}
        self.monitoring_thread = threading.Thread(
            target=self._monitor_performance, daemon=True
        )
        self.monitoring_thread.start()

        self.motion_threshold = 30
        self.min_motion_area = 500

        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=100, varThreshold=16, detectShadows=False
        )

        self._resource_lock = threading.Lock()

        self.num_workers = max(1, mp.cpu_count() - 1)
        self.workers = []
        for _ in range(self.num_workers):
            worker = threading.Thread(target=self._detection_worker, daemon=True)
            worker.start()
            self.workers.append(worker)

        # Initialize Reid for cross-camera tracking with shared instance
        self.reid_manager = Reid()  # Single Reid instance for all cameras
        self.person_trackers = {}  # No longer need individual trackers per camera

        # Enhance batch processing
        self.batch_timeout = 0.033  # 30 FPS equivalent
        self.max_batch_age = 0.1  # Maximum age of frames in batch

        # Performance optimizations
        torch.backends.cudnn.benchmark = True
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.stream = torch.cuda.Stream()
            self.stream_ctx = lambda: torch.cuda.stream(self.stream)
        else:
            self.stream_ctx = nullcontext

    def add_camera(self, camera_id: str, rtsp_url: str, company_id: str):
        """Add a new camera stream."""
        self.stop_events[camera_id] = threading.Event()

        self.output_queues[camera_id] = mp.Queue(maxsize=self.max_queue_size)
        self.frame_buffers[camera_id] = deque(maxlen=3)
        self.fps_counters[camera_id] = {"frames": 0, "last_check": time.time()}

        # Initialize person tracker
        self.person_trackers[camera_id] = Reid(company_id, camera_id, self.device)

        # Start frame reader process with cleanup callback
        process = threading.Thread(
            target=self._frame_reader_process,
            args=(
                camera_id,
                rtsp_url,
                company_id,
                self.frame_queue,
                self.stop_events[camera_id],
            ),
            daemon=True,
        )
        process.start()
        self.camera_threads[camera_id] = process

    def remove_camera(self, camera_id: str):
        """Remove a camera"""
        # Thread will stop at next iteration due to stop_event
        if camera_id in self.stop_events:
            # Just signal the thread to stop and return immediately
            self.stop_events[camera_id].set()

            # Start a cleanup thread that won't block the server
            cleanup_thread = threading.Thread(
                target=self._cleanup_camera_resources, args=(camera_id,), daemon=True
            )
            cleanup_thread.start()

        if camera_id in self.output_queues:
            del self.output_queues[camera_id]

        if camera_id in self.frame_buffers:
            del self.frame_buffers[camera_id]

        if camera_id in self.person_trackers:
            del self.person_trackers[camera_id]

        if camera_id in self.fps_counters:
            del self.fps_counters[camera_id]

        for collection in (
            self.output_queues,
            self.frame_buffers,
            self.person_trackers,
            self.fps_counters,
        ):
            collection.pop(camera_id, None)

    def _cleanup_camera_resources(self, camera_id: str):
        """Handle camera cleanup in a separate thread"""
        try:
            # Wait for the frame reader thread to finish with timeout
            if camera_id in self.camera_threads:
                self.camera_threads[camera_id].join(timeout=5.0)

            # Clean up resources
            with self._resource_lock:  # Add a lock to prevent race conditions
                if camera_id in self.camera_threads:
                    del self.camera_threads[camera_id]
                if camera_id in self.stop_events:
                    del self.stop_events[camera_id]
                for collection in (
                    self.output_queues,
                    self.frame_buffers,
                    self.person_trackers,
                    self.fps_counters,
                ):
                    collection.pop(camera_id, None)

        except Exception as e:
            print(f"Error during camera {camera_id} cleanup: {e}")

    def _frame_reader_process(
        self,
        camera_id: str,
        rtsp_url: str,
        company_id: str,
        frame_queue: mp.Queue,
        stop_event: threading.Event,
    ):
        # Fix the tuple syntax error
        max_retries = 3  # Remove the tuple syntax
        retry_delay = 2.0  # Remove the tuple syntax
        exponential_backoff = True

        cap = None
        frame_counter = 0
        last_frame_time = time.time()
        frame_interval = 1.0 / 30.0  # Target 30 FPS
        retry_count = 0

        logger.info(f"Starting frame reader for camera {camera_id}")

        def initialize_capture():
            nonlocal cap, retry_count

            if cap is not None:
                cap.release()
                cap = None

            current_delay = retry_delay * (2**retry_count if exponential_backoff else 1)

            try:
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
                        logger.info(f"Retrying in {current_delay:.1f} seconds...")
                        time.sleep(current_delay)
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
                    logger.info(f"Retrying in {current_delay:.1f} seconds...")
                    time.sleep(current_delay)
                    return False
                return False

        last_frame_time = time.time()
        # Remove this line as we're using self.min_frame_interval now
        # frame_interval = 1.0 / 30.0  # Target 30 FPS

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

                ret, frame = cap.read()
                if not ret:
                    if cap is not None:
                        cap.release()
                        cap = None
                    time.sleep(0.1)
                    continue

                frame_counter += 1
                if frame_counter % self.frame_skip != 0:
                    continue

                current_time = time.time()
                if current_time - last_frame_time < self.min_frame_interval:
                    continue

                frame = cv2.resize(frame, (640, 480))
                _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

                try:
                    # alwas push newest frame to the queue
                    frame_queue.put_nowait(
                        {
                            "frame": buffer.tobytes(),
                            "camera_id": camera_id,
                            "company_id": company_id,
                            "timestamp": current_time,
                        }
                    )
                    last_frame_time = current_time
                except Exception as e:
                    continue

                pass

            except Exception as e:
                logger.error(f"Error reading camera {camera_id}: {e}")
                if cap is not None:
                    cap.release()
                    cap = None
                time.sleep(1)

            finally:
                # Clean up any frame reader specific resources
                try:
                    # Close any open video captures, streams, etc.
                    if hasattr(self, "cap") and self.cap is not None:
                        self.cap.release()
                except Exception as e:
                    print(
                        f"Error cleaning up frame reader resources for camera {camera_id}: {e}"
                    )

        if cap is not None:
            cap.release()
        logger.info(f"Frame reader stopped for camera {camera_id}")

    def _detection_worker(self):
        """Optimized worker thread for processing frames"""
        batch_frames = []
        batch_metadata = []

        while not self.stop_event.is_set():
            try:
                # Collect and preprocess frames asynchronously
                futures = []
                while len(batch_frames) < self.max_batch_size:
                    try:
                        data = self.frame_queue.get_nowait()
                        future = self.preprocess_pool.submit(self._preprocess_frame, data)
                        futures.append((future, data))
                    except queue.Empty:
                        break

                # Wait for preprocessing to complete
                for future, data in futures:
                    try:
                        frame = future.result(timeout=0.1)
                        if frame is not None:
                            batch_frames.append(frame)
                            batch_metadata.append(data)
                    except Exception as e:
                        logger.error(f"Preprocessing error: {e}")

                if len(batch_frames) < self.min_batch_size:
                    time.sleep(0.001)
                    continue

                # Process batch
                with self.stream_ctx():
                    results = self.model.track(
                        source=batch_frames,
                        conf=0.5,
                        iou=0.7,
                        persist=True,
                        tracker="bytetrack.yaml",
                        device=self.device,
                        verbose=False,
                    )

                    # Process results in parallel
                    post_futures = []
                    for result, metadata in zip(results, batch_metadata):
                        future = self.postprocess_pool.submit(
                            self._process_detection_result,
                            result, metadata, batch_metadata, batch_frames
                        )
                        post_futures.append(future)

                    # Wait for postprocessing
                    for future in post_futures:
                        try:
                            future.result(timeout=0.1)
                        except Exception as e:
                            logger.error(f"Postprocessing error: {e}")

                batch_frames.clear()
                batch_metadata.clear()

            except Exception as e:
                logger.error(f"Error in detection worker: {e}")
                time.sleep(0.1)

    def _preprocess_frame(self, data):
        """Preprocess frame with optimized settings"""
        try:
            frame_data = np.frombuffer(data["frame"], np.uint8)
            frame = cv2.imdecode(frame_data, cv2.IMREAD_COLOR)
            if frame is None:
                return None

            # Scale down for processing
            height, width = frame.shape[:2]
            new_width = int(width * self.processing_scale)
            new_height = int(height * self.processing_scale)
            frame = cv2.resize(frame, (new_width, new_height), 
                             interpolation=cv2.INTER_AREA)

            return frame
        except Exception as e:
            logger.error(f"Frame preprocessing error: {e}")
            return None

    def _monitor_performance(self):
        """Monitor FPS for each camera"""
        while not self.stop_event.is_set():
            try:
                current_time = time.time()
                for camera_id, counter in self.fps_counters.items():
                    elapsed = current_time - counter["last_check"]
                    if elapsed >= 5.0:  # Report every 5 seconds
                        fps = counter["frames"] / elapsed
                        logger.info(f"Camera {camera_id} - FPS: {fps:.2f}")
                        counter["frames"] = 0
                        counter["last_check"] = current_time
                time.sleep(1)
            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")

    def _process_detections(self, frame, result, camera_id):
        """Process detections for visualization"""
        try:
            person_crops = []
            positions = {}
            boxes = result.boxes
            track_ids = boxes.id.cpu().numpy() if boxes.id is not None else None

            if track_ids is None:
                return frame

            # Get person crops and positions for Reid
            for box, track_id in zip(boxes.xyxy.cpu().numpy(), track_ids):
                x1, y1, x2, y2 = map(int, box)
                crop = frame[y1:y2, x1:x2]
                person_crops.append((int(track_id), crop))
                positions[int(track_id)] = (
                    (x1 + x2) / 2,
                    (y1 + y2) / 2,
                )  # center position

            # Get global IDs from Reid manager
            global_ids = self.reid_manager.update(
                camera_id=camera_id, person_crops=person_crops, positions=positions
            )

            # Draw results
            draw_frame = frame.copy()
            for box, track_id in zip(boxes.xyxy.cpu().numpy(), track_ids):
                x1, y1, x2, y2 = map(int, box)
                global_id = global_ids.get(int(track_id))

                if global_id:
                    # Use consistent color based on global_id
                    color = self._get_id_color(global_id)

                    # Draw bounding box
                    cv2.rectangle(draw_frame, (x1, y1), (x2, y2), color, 2)

                    # Draw global ID with background
                    text = f"ID: {global_id}"
                    (text_width, text_height), _ = cv2.getTextSize(
                        text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                    )

                    cv2.rectangle(
                        draw_frame,
                        (x1, y1 - text_height - 10),
                        (x1 + text_width + 10, y1),
                        color,
                        -1,
                    )

                    cv2.putText(
                        draw_frame,
                        text,
                        (x1 + 5, y1 - 7),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2,
                    )

            return draw_frame

        except Exception as e:
            logger.error(f"Error processing detections: {e}")
            return frame

    def _get_id_color(self, global_id: str) -> Tuple[int, int, int]:
        """Generate consistent color for global ID"""
        # Hash the global_id to get a consistent color
        hash_val = hash(global_id)
        hue = (hash_val % 360) / 360.0  # Convert hash to hue value between 0 and 1
        rgb = colorsys.hsv_to_rgb(hue, 0.9, 0.9)  # Convert HSV to RGB
        return tuple(int(x * 255) for x in rgb)

    def get_frame(self, camera_id: str) -> Optional[bytes]:
        """Get the latest processed frame for a camera"""
        try:
            if camera_id not in self.output_queues:
                return None
            return self.output_queues[camera_id].get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        """Stop all processing"""
        self.stop_event.set()
        # No need to join threads as they are daemon threads

        # Clear queues
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except:
                break

        for q in self.output_queues.values():
            while not q.empty():
                try:
                    q.get_nowait()
                except:
                    break

    def _process_detection_result(self, result, metadata, batch_metadata, batch_frames):
        """Process single detection result with frame ordering"""
        try:
            camera_id = metadata["camera_id"]
            timestamp = metadata["timestamp"]
            frame_idx = batch_metadata.index(metadata)
            frame = batch_frames[frame_idx]

            if camera_id not in self.person_trackers:
                return

            # Add frame to ordered buffer
            self._add_to_ordered_frames(camera_id, timestamp, frame)
            
            # Process frames in order
            self._process_ordered_frames(camera_id, result)

        except Exception as e:
            logger.error(f"Error processing detection result: {e}")

    def _add_to_ordered_frames(self, camera_id: str, timestamp: float, frame: np.ndarray):
        """Add frame to ordered buffer"""
        self.ordered_frames[camera_id].append((timestamp, frame))
        self.ordered_frames[camera_id].sort(key=lambda x: x[0])  # Sort by timestamp

        # Remove old frames
        current_time = time.time()
        self.ordered_frames[camera_id] = [
            (ts, f) for ts, f in self.ordered_frames[camera_id]
            if current_time - ts < self.max_frame_delay
        ]

    def _process_ordered_frames(self, camera_id: str, result):
        """Process frames in order"""
        if not self.ordered_frames[camera_id]:
            return

        current_time = time.time()
        frames_to_process = []
        
        # Collect frames ready for processing
        while self.ordered_frames[camera_id]:
            timestamp, frame = self.ordered_frames[camera_id][0]
            
            # Check if frame is too old
            if current_time - timestamp > self.max_frame_delay:
                self.ordered_frames[camera_id].pop(0)
                continue
                
            # Check if enough time has passed to ensure proper ordering
            if current_time - timestamp < self.frame_reorder_threshold:
                break
                
            frames_to_process.append((timestamp, frame))
            self.ordered_frames[camera_id].pop(0)

        # Process frames in order
        for timestamp, frame in frames_to_process:
            processed_frame = self._process_detections(frame, result, camera_id)
            if processed_frame is not None:
                self._enqueue_processed_frame(camera_id, processed_frame)

    def _enqueue_processed_frame(self, camera_id: str, frame: np.ndarray):
        """Optimized frame encoding"""
        try:
            # Scale to output size
            frame = cv2.resize(frame, self.output_scale, interpolation=cv2.INTER_AREA)

            # Use hardware-accelerated JPEG encoding if available
            if hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, 80]
            else:
                encode_params = [
                    cv2.IMWRITE_JPEG_QUALITY, 80,
                    cv2.IMWRITE_JPEG_OPTIMIZE, 1,
                ]

            _, buffer = cv2.imencode('.jpg', frame, encode_params)

            if (
                camera_id in self.output_queues
                and not self.output_queues[camera_id].full()
            ):
                self.output_queues[camera_id].put_nowait(buffer.tobytes())
                self.fps_counters[camera_id]["frames"] += 1

        except Exception as e:
            logger.error(f"Error enqueueing processed frame: {e}")
