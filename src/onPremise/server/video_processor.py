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
import psutil

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
        self.frame_queues = ThreadSafeDict()
        self.result_queues = ThreadSafeDict()
        self.stop_events = ThreadSafeDict()
        self.camera_threads = ThreadSafeDict()
        self.fps_counters = ThreadSafeDict()

        self.detection_queues = ThreadSafeDict()

        self.frame_skip = 3
        self.max_queue_size = 5
        self.target_fps = 15
        self.frame_interval = 1.0 / self.target_fps
        self.batch_size = 16

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
        for _ in range(4):
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
        """Setup GPU optimizations"""
        if torch.cuda.is_available():
            # Enable TF32 on Ampere
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            torch.set_float32_matmul_precision("high")

            # Set per-GPU memory fraction
            for gpu_id in range(torch.cuda.device_count()):
                torch.cuda.set_per_process_memory_fraction(0.8, gpu_id)

            self.device = torch.device("cuda")
            self.scaler = torch.amp.GradScaler("cuda")
        else:
            self.device = torch.device("cpu")
            torch.set_num_threads(4)  # Limit CPU threads
            self.scaler = None

    def _init_model(self):
        """Initialize model with optimizations"""
        model = YOLO("yolov8n.pt")
        model.to(self.device)
        if torch.cuda.is_available():
            model.fuse()
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

    def _frame_reader_thread(
        self,
        camera_id: str,
        rtsp_url: str,
        company_id: str,
    ):
        """Frame reader thread that sends frames to shared detection queue"""
        retry_count = 0
        max_retries = 10
        cap = None
        frame_counter = 0
        last_frame_time = time.time()

        while not self.stop_events[camera_id].is_set():
            try:
                if cap is None or not cap.isOpened():
                    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
                    if not cap.isOpened():
                        retry_count += 1
                        if retry_count >= max_retries:
                            logger.error(f"Failed to open camera {camera_id}")
                            break
                        time.sleep(2**retry_count)
                        continue
                    retry_count = 0

                ret, frame = cap.read()
                if not ret:
                    if cap is not None:
                        cap.release()
                        cap = None
                    continue

                current_time = time.time()
                if current_time - last_frame_time < self.frame_interval:
                    continue

                frame_counter += 1
                if frame_counter % self.frame_skip != 0:
                    continue

                if frame.shape[0] > 480 or frame.shape[1] > 640:
                    frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
                try:
                    self.detection_queues[camera_id].put_nowait(
                        {
                            "frame": frame,
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
                time.sleep(1)

        if cap is not None:
            cap.release()

    def _detection_worker(self):
        """GPU-specific detection worker with improved batch processing and error handling"""
        batch_frames = []
        batch_metadata = []
        last_batch_time = time.time()

        while not self.stop_event.is_set():
            try:
                current_time = time.time()

                # Check if enough time has passed for next batch
                if current_time - last_batch_time < self.frame_interval:
                    time.sleep(0.001)  # Small sleep to prevent CPU spinning
                    continue

                # Check all camera queues round-robin
                for camera_id in list(self.detection_queues.keys()):
                    try:
                        while len(batch_frames) < self.batch_size:
                            try:
                                data = self.detection_queues[camera_id].get_nowait()
                                batch_frames.append(data["frame"])
                                batch_metadata.append(data)
                            except queue.Empty:
                                break  # Move to next camera if queue is empty

                    except Exception as e:
                        logger.error(f"Error processing camera {camera_id}: {e}")
                        continue

                # Process batch if we have frames
                if batch_frames:
                    try:
                        # Use appropriate acceleration based on device
                        context = (
                            torch.amp.autocast(self.device.type, enabled=True)
                            if self.device.type in ["cuda", "cpu"]
                            else nullcontext()
                        )

                        with context:
                            results = self.model.track(
                                source=batch_frames,
                                conf=0.5,
                                iou=0.7,
                                persist=True,
                                tracker="bytetrack.yaml",
                                device=self.device,
                                verbose=False,
                            )

                        # Process results for each frame
                        for result, metadata in zip(results, batch_metadata):
                            camera_id = metadata["camera_id"]
                            frame = batch_frames[batch_metadata.index(metadata)]

                            if camera_id not in self.result_queues:
                                continue

                            processed_frame = self._process_detections(
                                frame, result, camera_id
                            )

                            if processed_frame is not None:
                                try:
                                    # Encode frame with lower quality for better performance
                                    _, buffer = cv2.imencode(
                                        ".jpg",
                                        processed_frame,
                                        [cv2.IMWRITE_JPEG_QUALITY, 80],
                                    )

                                    self.result_queues[camera_id].put_nowait(
                                        buffer.tobytes()
                                    )

                                    # Update FPS counter if it exists
                                    if camera_id in self.fps_counters:
                                        self.fps_counters[camera_id]["frames"] += 1

                                except queue.Full:
                                    continue  # Skip if output queue is full
                                except Exception as e:
                                    logger.error(
                                        f"Error encoding frame for camera {camera_id}: {e}"
                                    )
                                    continue

                    except Exception as e:
                        logger.error(f"Error processing batch: {e}")

                    finally:
                        # Clear batch data
                        batch_frames.clear()
                        batch_metadata.clear()
                        last_batch_time = current_time

            except Exception as e:
                logger.error(f"Error in detection worker: {e}")
                time.sleep(0.1)  # Sleep longer on error

            # Prevent CPU spinning
            time.sleep(0.001)

    def _process_detections(self, frame, result, camera_id):
        """Process detections with improved error handling and type checking"""
        try:
            # Initial validation
            if not hasattr(result.boxes, "cls") or not hasattr(result.boxes, "id"):
                return frame

            # Get person detections (class 0 in COCO)
            boxes_cls = result.boxes.cls.cpu().numpy()
            person_mask = boxes_cls == 0
            if not np.any(person_mask):  # More explicit numpy boolean reduction
                return frame

            boxes = result.boxes[person_mask]
            if boxes.id is None:
                return frame

            # Get IDs and boxes for person detections
            track_ids = boxes.id.cpu().numpy().astype(int)
            boxes_xyxy = boxes.xyxy.cpu().numpy()

            # Process person detections
            person_crops = []
            positions = {}

            for box, track_id in zip(boxes_xyxy, track_ids):
                x1, y1, x2, y2 = map(int, box)

                # Validate box coordinates
                if (
                    x1 < 0
                    or y1 < 0
                    or x2 >= frame.shape[1]
                    or y2 >= frame.shape[0]
                    or x2 <= x1
                    or y2 <= y1
                ):
                    continue

                try:
                    # Add padding for better detection
                    pad = 10
                    x1_pad = max(0, x1 - pad)
                    y1_pad = max(0, y1 - pad)
                    x2_pad = min(frame.shape[1], x2 + pad)
                    y2_pad = min(frame.shape[0], y2 + pad)

                    crop = frame[y1_pad:y2_pad, x1_pad:x2_pad]

                    # Validate crop size
                    if crop.size == 0 or crop.shape[0] < 30 or crop.shape[1] < 30:
                        continue

                    person_crops.append((int(track_id), crop.copy()))
                    positions[int(track_id)] = ((x1 + x2) / 2, (y1 + y2) / 2)

                except Exception as e:
                    logger.error(f"Error cropping detection: {e}")
                    continue

            if not person_crops:
                return frame

            # Update Reid with detections
            global_ids = self.reid_manager.update(camera_id, person_crops, positions)

            # Draw results
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
