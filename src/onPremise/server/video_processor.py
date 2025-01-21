import cv2
import numpy as np
import torch
import logging
import time
import queue
import os
import threading
import colorsys
import multiprocessing as mp

from typing import Optional, Tuple
from ultralytics import YOLO
from logging_config import setup_logger
from contextlib import nullcontext
from reid_implementation import Reid
from concurrent.futures import ThreadPoolExecutor

setup_logger()
logger = logging.getLogger("VideoProcessor")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
torch.backends.cudnn.benchmark = True


class ThreadSafeDict(dict):
    def __init__(self, *args, **kwargs):
        self.lock = threading.Lock()
        super().__init__(*args, **kwargs)

    def __getitem__(self, key):
        with self.lock:
            return super().__getitem__(key)

    def __setitem__(self, key, value):
        with self.lock:
            super().__setitem__(key, value)

    def __delitem__(self, key):
        with self.lock:
            super().__delitem__(key)


class VideoProcessor:
    """
    Shared video processor that handles multiple cameras efficiently
    """

    def __init__(self, num_detection_threads: int = 4):
        # Initialize collections
        self.result_queues = ThreadSafeDict()
        self.stop_events = ThreadSafeDict()
        self.camera_threads = ThreadSafeDict()
        self.fps_counters = ThreadSafeDict()

        self.detection_queues = ThreadSafeDict()

        self.frame_skip = 2
        self.max_queue_size = 3
        self.target_fps = 15
        self.frame_interval = 1.0 / self.target_fps
        self.batch_size = 8

        self.thread_pool = ThreadPoolExecutor(max_workers=mp.cpu_count())

        self._color_cache = {}

        # GPU Optimizations
        self._setup_gpu()

        # Initialize model with optimizations
        self.model = self._init_model()

        # Initialize Reid
        self.reid_manager = Reid()

        # Global stop event
        self.stop_event = threading.Event()

        # Start detection threads
        self.detection_threads = []
        for _ in range(num_detection_threads):
            thread = threading.Thread(target=self._detection_worker, daemon=True)
            thread.start()
            self.detection_threads.append(thread)

        # Start monitoring
        self.monitoring_thread = threading.Thread(
            target=self._monitor_performance, daemon=True
        )
        self.monitoring_thread.start()

        # Global stop event
        self.stop_event = threading.Event()

    def _setup_gpu(self):
        """Enhanced GPU setup with memory optimizations"""
        if torch.cuda.is_available():
            # Enable TF32 and other optimizations
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.deterministic = False
            torch.set_float32_matmul_precision("high")
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            torch.set_num_threads(mp.cpu_count())
            self.scaler = None

    def _init_model(self):
        """Initialize model with additional optimizations"""
        model = YOLO("yolov8n.pt")
        if self.device.type == "cuda":
            model.to(self.device)
            model.fuse()
        else:
            model.to(self.device)
        return model

    def add_camera(self, camera_id: str, rtsp_url: str, company_id: str):
        """Add a new camera stream."""
        try:
            self.result_queues[camera_id] = queue.Queue(maxsize=self.max_queue_size)
            self.detection_queues[camera_id] = queue.Queue(maxsize=self.max_queue_size)
            self.fps_counters[camera_id] = {"frames": 0, "last_check": time.time()}
            self.stop_events[camera_id] = threading.Event()
            logger.info(f"Adding camera {camera_id} with RTSP URL: {rtsp_url}")

            reader_thread = threading.Thread(
                target=self._frame_reader_thread,
                args=(camera_id, rtsp_url, company_id),
                daemon=True,
            )
            reader_thread.start()
            self.camera_threads[camera_id] = reader_thread

        except Exception as e:
            logger.error(f"Error adding camera {camera_id}: {e}")
            self._cleanup_camera_resources(camera_id)
            raise

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

        if camera_id in self.fps_counters:
            del self.fps_counters[camera_id]

        for collection in (self.fps_counters,):
            collection.pop(camera_id, None)

    def _cleanup_camera_resources(self, camera_id: str):
        """Handle camera cleanup in a separate thread"""
        try:
            # Wait for the frame reader thread to finish with timeout
            if camera_id in self.camera_threads:
                self.camera_threads[camera_id].join(timeout=5.0)

            if camera_id in self.camera_threads:
                del self.camera_threads[camera_id]
            if camera_id in self.stop_events:
                del self.stop_events[camera_id]
            for collection in (self.fps_counters,):
                collection.pop(camera_id, None)

        except Exception as e:
            print(f"Error during camera {camera_id} cleanup: {e}")

    def _frame_reader_thread(self, camera_id: str, rtsp_url: str, company_id: str):
        """Optimized frame reader with better error handling and performance"""
        retry_count = 0
        max_retries = 5
        cap = None
        frame_counter = 0
        last_frame_time = time.time()

        # Pre-allocate frame buffer
        frame_buffer = None

        while not self.stop_events[camera_id].is_set():
            try:
                if cap is None or not cap.isOpened():
                    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
                    if not cap.isOpened():
                        retry_count += 1
                        if retry_count >= max_retries:
                            logger.error(f"Failed to open camera {camera_id}")
                            break
                        time.sleep(min(2**retry_count, 30))  # Cap max sleep time
                        continue
                    retry_count = 0

                    # Set optimal buffer size
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                ret, frame = cap.read()
                if not ret:
                    if cap is not None:
                        cap.release()
                        cap = None
                    time.sleep(0.1)
                    continue

                current_time = time.time()
                if current_time - last_frame_time < self.frame_interval:
                    time.sleep(0.001)
                    continue

                if frame_buffer is None or frame_buffer.shape != frame.shape:
                    frame_buffer = np.empty_like(frame)
                np.copyto(frame_buffer, frame)

                try:
                    self.detection_queues[camera_id].put_nowait(
                        {
                            "frame": frame_buffer,
                            "camera_id": camera_id,
                            "company_id": company_id,
                            "timestamp": current_time,
                        }
                    )
                    last_frame_time = current_time
                except queue.Full:
                    continue

            except Exception as e:
                logger.error(f"Error in frame reader for camera {camera_id}: {str(e)}")
                if cap is not None:
                    cap.release()
                    cap = None
                time.sleep(0.1)

        if cap is not None:
            cap.release()

    def _detection_worker(self):
        """Optimized detection worker with improved batch processing"""
        batch_frames = []
        batch_metadata = []
        last_batch_time = time.time()

        while not self.stop_event.is_set():
            try:
                current_time = time.time()

                if current_time - last_batch_time < self.frame_interval:
                    time.sleep(0.001)
                    continue

                for camera_id in list(self.detection_queues.keys()):
                    try:
                        while len(batch_frames) < self.batch_size:
                            try:
                                data = self.detection_queues[camera_id].get_nowait()
                                batch_frames.append(data["frame"])
                                batch_metadata.append(data)
                            except queue.Empty:
                                break
                    except Exception as e:
                        logger.error(f"Error processing camera {camera_id}: {e}")
                        continue

                if batch_frames:
                    try:
                        with (
                            torch.amp.autocast("cuda")
                            if self.device.type == "cuda"
                            else nullcontext()
                        ):
                            results = self.model.track(
                                source=batch_frames,
                                conf=0.5,
                                iou=0.7,
                                persist=True,
                                tracker="bytetrack.yaml",
                                verbose=False,
                            )

                        processing_futures = []
                        for result, metadata in zip(results, batch_metadata):
                            camera_id = metadata["camera_id"]
                            if camera_id not in self.result_queues:
                                continue

                            frame = batch_frames[batch_metadata.index(metadata)]
                            future = self.thread_pool.submit(
                                self._process_detections, frame, result, camera_id
                            )
                            processing_futures.append((future, camera_id))

                        # Handle processed frames
                        for future, camera_id in processing_futures:
                            try:
                                processed_frame = future.result()
                                if processed_frame is not None:
                                    _, buffer = cv2.imencode(
                                        ".jpg",
                                        processed_frame,
                                        [cv2.IMWRITE_JPEG_QUALITY, 50],
                                    )

                                    try:
                                        self.result_queues[camera_id].put_nowait(
                                            buffer.tobytes()
                                        )
                                        if camera_id in self.fps_counters:
                                            self.fps_counters[camera_id]["frames"] += 1
                                    except queue.Full:
                                        continue
                            except Exception as e:
                                logger.error(f"Error processing frame result: {e}")
                                continue

                    finally:
                        batch_frames.clear()
                        batch_metadata.clear()
                        last_batch_time = current_time

            except Exception as e:
                logger.error(f"Error in detection worker: {e}")
                time.sleep(0.1)

            time.sleep(0.001)

    def _process_detections(self, frame, result, camera_id):
        """Process detections with improved error handling and type checking"""
        try:
            # Ensure result.boxes exists and has necessary attributes
            if not hasattr(result, "boxes") or result.boxes is None:
                return frame

            if not hasattr(result.boxes, "cls") or not hasattr(result.boxes, "id"):
                return frame

            boxes_cls = (
                result.boxes.cls.cpu().numpy() if result.boxes.cls.numel() > 0 else []
            )
            if len(boxes_cls) == 0:
                return frame  
            person_mask = boxes_cls == 0
            if not np.any(person_mask):  
                return frame

            person_mask = boxes_cls == 0
            if not np.any(person_mask): 
                return frame

            # Filter boxes and IDs
            boxes = result.boxes[person_mask]
            if not hasattr(boxes, "id") or boxes.id.numel() == 0:
                return 

            track_ids = boxes.id.cpu().numpy().astype(int)
            boxes_xyxy = boxes.xyxy.cpu().numpy()
            if len(track_ids) == 0 or len(boxes_xyxy) == 0:
                return frame

            # Process person detections
            person_crops = []
            positions = {}

            for box, track_id in zip(boxes_xyxy, track_ids):
                x1, y1, x2, y2 = map(int, box)

                # Validate bounding box coordinates
                if x1 < 0 or y1 < 0 or x2 >= frame.shape[1] or y2 >= frame.shape[0]:
                    continue

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0 or crop.shape[0] < 30 or crop.shape[1] < 30:
                    continue

                person_crops.append((track_id, crop.copy()))
                positions[track_id] = ((x1 + x2) / 2, (y1 + y2) / 2)

            if not person_crops:
                return frame

            # Update Reid with detections
            global_ids = self.reid_manager.update(camera_id, person_crops)
            draw_frame = frame.copy()

            for box, track_id in zip(boxes_xyxy, track_ids):
                global_id = global_ids.get(int(track_id))
                if not global_id:
                    continue

                x1, y1, x2, y2 = map(int, box)
                color = self._get_id_color(global_id)

                # Draw bounding box
                cv2.rectangle(draw_frame, (x1, y1), (x2, y2), color, 2)

                # Draw text with background
                text = f"ID: {global_id}"
                (text_width, text_height), baseline = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                )

                # Text background
                cv2.rectangle(
                    draw_frame,
                    (x1, y1 - text_height - 10),
                    (x1 + text_width + 10, y1),
                    color,
                    -1,
                )

                # Text
                cv2.putText(
                    draw_frame,
                    text,
                    (x1 + 5, y1 - 5),
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
        """Fast color generation"""
        hash_val = hash(global_id)
        hue = (hash_val % 360) / 360.0
        rgb = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
        return tuple(int(x * 255) for x in rgb)

    def get_frame(self, camera_id: str) -> Optional[bytes]:
        """Get the latest processed frame for a camera"""
        try:
            if camera_id not in self.result_queues:
                return None
            return self.result_queues[camera_id].get_nowait()
        except queue.Empty:
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
