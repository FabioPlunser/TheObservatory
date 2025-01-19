import cv2
import numpy as np
import torch
import logging
import time
import queue
import os
import threading
import multiprocessing as mp

from typing import Dict, List, Optional
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict, deque
from logging_config import setup_logger
from person_tracker import OptimizedPersonTracker
from rtsp_reader import RTSPReader

setup_logger()
logger = logging.getLogger("VideoProcessor")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

class VideoProcessor:
    """
    Shared video processor that handles multiple cameras efficiently
    """

    def __init__(self, batch_size: int = 8):
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
        self.frame_skip = 2
        self.max_queue_size = 10
        self.min_batch_wait = 0.01

        self.frame_queue = mp.Queue(maxsize=self.max_queue_size)
        self.output_queues: Dict[str, queue.Queue] = {}
        self.frame_buffers: Dict[str, deque] = {}

        self.stop_event = threading.Event()

        self.camera_threads = {}
        self.stop_events = {}
        self.person_trackers = {}

        self.num_workers = max(1, mp.cpu_count() - 1)
        self.workers = []
        for _ in range(self.num_workers):
            worker = threading.Thread(target=self._detection_worker, daemon=True)
            worker.start()
            self.workers.append(worker)

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

    def add_camera(self, camera_id: str, rtsp_url: str, company_id: str):
        """Add a new camera stream."""
        self.stop_events[camera_id] = threading.Event()

        self.output_queues[camera_id] = mp.Queue(maxsize=self.max_queue_size)
        self.frame_buffers[camera_id] = deque(maxlen=3)
        self.fps_counters[camera_id] = {"frames": 0, "last_check": time.time()}

        # Initialize person tracker
        self.person_trackers[camera_id] = OptimizedPersonTracker(
            company_id, camera_id, self.device
        )

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
        cap = None
        frame_counter = 0
        last_frame_time = time.time()
        frame_interval = 1.0 / 30.0  # Target 30 FPS

        logger.info(f"Starting frame reader for camera {camera_id}")

        while not stop_event.is_set():
            try:
                if cap is None or not cap.isOpened():
                    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
                    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                    logger.info(f"Opened RTSP stream: {rtsp_url}")

                    if not cap.isOpened():
                        logger.error(f"Failed to open RTSP stream: {rtsp_url}")
                        time.sleep(1)
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
                if current_time - last_frame_time < frame_interval:
                    continue

                frame = cv2.resize(frame, (640, 480))
                _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

                try:
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
                    logger.error(f"Error queuing frame: {e}")
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
        """Worker thread for processing frames"""
        batch_frames = []
        batch_metadata = []
        last_batch_time = time.time()

        while not self.stop_event.is_set():
            try:
                # Collect batch
                while len(batch_frames) < self.batch_size:
                    try:
                        data = self.frame_queue.get_nowait()
                        frame_data = np.frombuffer(data["frame"], np.uint8)
                        frame = cv2.imdecode(frame_data, cv2.IMREAD_COLOR)
                        if frame is not None:
                            batch_frames.append(frame)
                            batch_metadata.append(data)
                    except queue.Empty:
                        break

                if not batch_frames:
                    time.sleep(0.001)
                    continue

                current_time = time.time()
                if not batch_frames:
                    time.sleep(0.001)
                    continue

                if self.device.type == "cuda":
                    with torch.autocast("cuda", enabled=True):
                        results = self.model.track(
                            source=batch_frames,
                            conf=0.5,
                            iou=0.7,
                            persist=True,
                            tracker="bytetrack.yaml",
                            device=self.device,
                            verbose=False,
                        )
                else:
                    with torch.autocast("cpu", enabled=True):
                        logger.info("Using CPU for detection")
                        results = self.model.track(
                            source=batch_frames,
                            conf=0.5,
                            iou=0.7,
                            persist=True,
                            tracker="bytetrack.yaml",
                            device=self.device,
                            verbose=False,
                        )

                # Process results
                for result, metadata in zip(results, batch_metadata):
                    camera_id = metadata["camera_id"]
                    frame = batch_frames[batch_metadata.index(metadata)]

                    if camera_id not in self.person_trackers:
                        continue

                    tracker = self.person_trackers[camera_id]
                    processed_frame = self._process_detections(
                        frame, result, tracker, camera_id
                    )

                    if processed_frame is not None:
                        processed_frame = cv2.resize(processed_frame, (640, 480))

                        # Use optimized JPEG encoding parameters
                        encode_params = [
                            cv2.IMWRITE_JPEG_QUALITY,
                            80,
                            cv2.IMWRITE_JPEG_OPTIMIZE,
                            1,
                            cv2.IMWRITE_JPEG_PROGRESSIVE,
                            1,
                        ]

                        # Fast JPEG compression
                        _, buffer = cv2.imencode(".jpg", processed_frame, encode_params)

                        if camera_id in self.output_queues:
                            try:
                                self.output_queues[camera_id].put_nowait(
                                    buffer.tobytes()
                                )
                                self.fps_counters[camera_id]["frames"] += 1
                            except queue.Full:
                                continue

                # Reset batch
                batch_frames.clear()
                batch_metadata.clear()
                last_batch_time = current_time

            except Exception as e:
                logger.error(f"Error in detection worker: {e}")
                batch_frames.clear()
                batch_metadata.clear()
                time.sleep(0.1)

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

    def _process_detections(self, frame, result, tracker, camera_id):
        """Process detections for visualization"""
        try:
            # Update tracker
            updated_tracks, new_tracks = tracker.update(frame, result)

            # Draw results
            draw_frame = frame.copy()

            all_tracks = updated_tracks + new_tracks
            for track in all_tracks:
                x, y, w, h = map(int, track.bbox)

                # Draw bounding box
                cv2.rectangle(draw_frame, (x, y), (x + w, y + h), track.color, 2)

                # Draw text with background
                text = f"ID: {track.track_id}"
                if track.recognition_status:
                    text += f" ({track.recognition_status})"

                (text_width, text_height), _ = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                )

                cv2.rectangle(
                    draw_frame,
                    (x, y - text_height - 10),
                    (x + text_width + 10, y),
                    track.color,
                    -1,
                )

                cv2.putText(
                    draw_frame,
                    text,
                    (x + 5, y - 7),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )

            return draw_frame

        except Exception as e:
            logger.error(f"Error processing detections: {e}")
            return frame

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
