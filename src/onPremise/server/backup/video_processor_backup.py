import multiprocessing as mp
import cv2
import numpy as np
import torch
import logging
import time
import queue
import os
import asyncio

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from person_tracker import OptimizedPersonTracker
from reid_implementation import OptimizedCrossCameraTracker
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Any
from ultralytics import YOLO
from logging_config import setup_logger

setup_logger()
logger = logging.getLogger("VideoProcessor")


# def process_frames_process(
#     frame_queue: mp.Queue,
#     output_queue: mp.Queue,
#     company_id: str,
#     camera_id: str,
#     stop_event: mp.Event,
# ):
#     """Separate process for frame processing"""
#     thread_pool = None
#     import os

#     os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
#     try:
#         # Initialize device
#         if torch.cuda.is_available():
#             device = torch.device("cuda")
#             torch.backends.cuda.matmul.allow_tf32 = True
#             torch.backends.cudnn.benchmark = True
#         elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
#             device = torch.device("mps")
#         else:
#             device = torch.device("cpu")
#             torch.set_num_threads(mp.cpu_count())

#         # Initialize YOLO model with correct settings
#         model = YOLO("yolov8n.pt")
#         model.to(device)

#         logger.info("Company_id: %s, Camera_id: %s", company_id, camera_id)
#         # Initialize trackers
#         person_tracker = OptimizedPersonTracker(company_id, camera_id, device)
#         cross_camera_tracker = OptimizedCrossCameraTracker()
#         # Attach cross-camera tracker to person_tracker
#         person_tracker.cross_camera_tracker = cross_camera_tracker

#         # Initialize thread pool
#         thread_pool = ThreadPoolExecutor(max_workers=8)

#         # Batch processing setup
#         batch_size = 4
#         batch_frames = []
#         batch_timestamps = []

#         while not stop_event.is_set():
#             try:
#                 # Collect batch
#                 while len(batch_frames) < batch_size:
#                     try:
#                         frame_data, timestamp = frame_queue.get_nowait()

#                         # Decompress frame
#                         nparr = np.frombuffer(frame_data, np.uint8)
#                         frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

#                         if frame is None:
#                             continue

#                         batch_frames.append(frame)
#                         batch_timestamps.append(timestamp)
#                     except queue.Empty:
#                         break

#                 if not batch_frames:
#                     time.sleep(0.001)
#                     continue

#                 # Process batch with YOLO - Updated model call
#                 try:
#                     results = model.track(
#                         source=batch_frames,
#                         conf=0.5,
#                         iou=0.7,
#                         persist=True,
#                         tracker="bytetrack.yaml",
#                         device=device,
#                         verbose=False,
#                     )
#                 except RuntimeError as e:
#                     if "torchvision::nms" in str(e):
#                         logger.error("NMS error on CUDA. Falling back to CPU.")
#                         raise

#                 # Process detections in parallel
#                 futures = []
#                 for frame, timestamp, result in zip(
#                     batch_frames, batch_timestamps, results
#                 ):
#                     future = thread_pool.submit(
#                         process_detections,
#                         frame,
#                         result,
#                         person_tracker,
#                         cross_camera_tracker,
#                         camera_id,
#                         timestamp,
#                     )
#                     futures.append((future, frame))

#                 # Handle processed results
#                 for future, frame in futures:
#                     try:
#                         processed_frame = future.result(timeout=1)
#                         if processed_frame is not None:
#                             # Compress frame for output
#                             _, buffer = cv2.imencode(
#                                 ".jpg",
#                                 processed_frame,
#                                 [cv2.IMWRITE_JPEG_QUALITY, 50],
#                             )
#                             frame_bytes = buffer.tobytes()

#                             if not output_queue.full():
#                                 output_queue.put_nowait(frame_bytes)
#                     except Exception as e:
#                         logger.error(f"Error processing detection: {e}")

#                 batch_frames = []
#                 batch_timestamps = []

#             except Exception as e:
#                 logger.error(f"Error in frame processor: {e}")
#                 batch_frames = []
#                 batch_timestamps = []

#     except Exception as e:
#         logger.error(f"Error initializing processor: {e}")
#     finally:
#         if thread_pool:
#             thread_pool.shutdown()


# def read_frames_process(rtsp_url: str, frame_queue: mp.Queue, stop_event: mp.Event):
#     """Separate process for frame reading with enhanced RTSP handling"""
#     cap = None
#     retry_delay = 1.0
#     max_retry_delay = 5.0
#     max_consecutive_failures = 10
#     failure_count = 0
#     import os

#     os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

#     while not stop_event.is_set():
#         try:
#             if cap is None:
#                 logger.info(f"Attempting to connect to RTSP stream: {rtsp_url}")

#                 # Configure RTSP settings
#                 os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
#                 time.sleep(2)
#                 # Create capture with optimized settings
#                 cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

#                 cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
#                 cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

#                 # Check connection
#                 if not cap.isOpened():
#                     raise RuntimeError("Failed to open RTSP stream")

#                 logger.info("Successfully connected to RTSP stream")
#                 retry_delay = 1.0  # Reset retry delay on successful connection
#                 failure_count = 0  # Reset failure count

#             frame_counter = 0
#             skip_frames = 2  # Process every 3rd frame

#             while not stop_event.is_set():
#                 ret, frame = cap.read()
#                 if not ret:
#                     failure_count += 1
#                     if failure_count > max_consecutive_failures:
#                         logger.error("Too many consecutive frame read failures")
#                         raise RuntimeError("Stream connection lost")
#                     continue

#                 failure_count = 0  # Reset on successful read
#                 frame_counter += 1

#                 if frame_counter % skip_frames != 0:
#                     continue

#                 # Optimize frame size
#                 frame = cv2.resize(frame, (680, 480), interpolation=cv2.INTER_AREA)

#                 # Compress frame for queue
#                 try:
#                     _, buffer = cv2.imencode(
#                         ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 50]
#                     )
#                     frame_data = buffer.tobytes()

#                     if not frame_queue.full():
#                         frame_queue.put_nowait((frame_data, time.time()))
#                 except queue.Full:
#                     continue  # Skip frame if queue is full
#                 except Exception as e:
#                     logger.error(f"Error compressing frame: {e}")
#                     continue

#         except Exception as e:
#             logger.error(f"RTSP stream error: {e}")
#             if cap is not None:
#                 cap.release()
#                 cap = None

#             # Progressive backoff with maximum
#             time.sleep(retry_delay)
#             retry_delay = min(retry_delay * 1.5, max_retry_delay)
#             continue

#     # Cleanup
#     if cap is not None:
#         cap.release()


# def process_detections(
#     frame: np.ndarray,
#     result,
#     person_tracker: OptimizedPersonTracker,
#     cross_camera_tracker: OptimizedCrossCameraTracker,
#     camera_id: str,
#     timestamp: float,
# ) -> Optional[np.ndarray]:
#     """Process detections for a single frame"""
#     import os

#     os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
#     try:
#         # Update person tracker
#         updated_tracks, new_tracks = person_tracker.update(frame, result)

#         # Create a copy of frame for drawing
#         draw_frame = frame.copy()

#         # Prepare data for cross-camera tracking
#         person_crops = []
#         face_ids = {}
#         recognition_statuses = {}
#         positions = {}

#         all_tracks = updated_tracks + new_tracks
#         for track in all_tracks:
#             # Get person crop
#             x, y, w, h = map(int, track.bbox)
#             person_crop = frame[y : y + h, x : x + w]
#             person_crops.append((track.track_id, person_crop))

#             if track.face_id:
#                 face_ids[track.track_id] = track.face_id
#             recognition_statuses[track.track_id] = track.recognition_status
#             positions[track.track_id] = (
#                 track.bbox[0] + track.bbox[2] / 2,
#                 track.bbox[1] + track.bbox[3] / 2,
#             )

#             # Draw bounding box and information
#             cv2.rectangle(draw_frame, (x, y), (x + w, y + h), track.color, 2)

#             # Draw filled rectangle behind text
#             text = f"ID: {track.track_id} ({track.recognition_status})"
#             (text_width, text_height), _ = cv2.getTextSize(
#                 text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
#             )
#             cv2.rectangle(
#                 draw_frame,
#                 (x, y - text_height - 10),
#                 (x + text_width + 10, y),
#                 track.color,
#                 -1,
#             )

#             # Draw text
#             cv2.putText(
#                 draw_frame,
#                 text,
#                 (x + 5, y - 7),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.5,
#                 (255, 255, 255),  # White text
#                 2,
#             )

#             # Add status indicator dot
#             status_color = {
#                 "pending": (0, 255, 255),  # Yellow
#                 "in_progress": (0, 165, 255),  # Orange
#                 "recognized": (0, 255, 0),  # Green
#                 "unknown": (0, 0, 255),  # Red
#             }.get(track.recognition_status, (255, 255, 255))

#             dot_radius = 5
#             dot_position = (x + text_width + 20, y - text_height // 2 - 5)
#             cv2.circle(draw_frame, dot_position, dot_radius, status_color, -1)

#         # Update cross-camera tracking
#         if person_crops:
#             global_ids = cross_camera_tracker.update(
#                 camera_id=camera_id,
#                 person_crops=person_crops,
#                 positions=positions,
#                 face_ids=face_ids,
#                 recognition_statuses=recognition_statuses,
#             )

#             # Update tracked persons with global IDs
#             for track in all_tracks:
#                 if track.track_id in global_ids:
#                     track.global_id = global_ids[track.track_id]

#         return draw_frame

#     except Exception as e:
#         logger.error(f"Error processing detections: {e}")
#         return frame


def create_shared_dict():
    return {"processing_times": [], "last_cleanup": time.time(), "fps": 0.0}


class FrameReader(mp.Process):
    def __init(
        self,
        rtsp_url: str,
        frame_queue: mp.Queue,
        stop_event: mp.Event,
        pause_event: mp.Event,
        shared_dict,
    ):
        super().__init__()
        self.rtsp_url = rtsp_url
        self.frame_queue = frame_queue
        self.stop_event = stop_event
        self.pause_event = pause_event
        self.shared_dict = shared_dict

        self.frame_skip = 2
        self.max_fame_size = (680, 480)
        self.retry_delay = 1.0
        self.max_retry_delay = 5.0

    def run(self):
        """Main process loop"""
        cap = None
        frame_count = 0
        last_frame_time = time.time()
        retry_delay = self.retry_delay

        while not self.stop_event.is_set():
            try:
                if cap is None or not cap.isOpenend():
                    cap = self.__init_capture()
                    if cap is None:
                        time.sleep(retry_delay)
                        retry_delay = min(retry_delay * 1.5, self.max_retry_delay)
                        continue
                    retry_delay = self.retry_delay

                if self.pause_event.is_set():
                    time.sleep(0.1)
                    continue

                ret, frame = cap.read()
                if not ret:
                    logger.warning("Faile to read frame")
                    continue

                frame_count += 1
                if frame_count % self.frame_skip != 0:
                    continue

                processed_frame = self._preprocess_frame(frame)
                if processed_frame is None:
                    logger.error("Failed to preprocess frame")
                    continue

                # Rate limiting
                current_time = time.time()
                if current_time - last_frame_time < 0.033:
                    time.sleep(0.001)
                    continue

                last_frame_time = current_time

                if not self.frame_queue.full():
                    self.frame_queue.put_nowait((processed_frame, current_time))

            except Exception as e:
                logger.error(f"Error in frame reading: {e}")
                if cap is not None:
                    cap.realease()
                    cap = None
                time.sleep(retry_delay)

        if cap is not None:
            cap.release()

    def _init_capture(self) -> Optional[cv2.VideoCapture]:
        """Initialize capture with optimized settings"""
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)

        if not cap.isOpened():
            return None

        # Optimize buffer size and codec
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        return cap

    def _preprocess_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Optimized frame preprocessing"""
        try:
            # Resize frame
            frame = cv2.resize(frame, self.max_frame_size, interpolation=cv2.INTER_AREA)

            # Compress frame
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
            frame_data = buffer.tobytes()

            return frame_data
        except Exception as e:
            logger.error(f"Error preprocessing frame: {e}")
            return None


class FrameProcessor(mp.Process):
    def __init__(
        self,
        frame_queue: mp.Queue,
        output_queue: mp.Queue,
        company_id: str,
        camera_id: str,
        stop_event: mp.Event,
        pause_event: mp.Event,
        shared_dict,
        latest_frame: mp.Value,
    ):
        super().__init__()
        self.frame_queue = frame_queue
        self.output_queue = output_queue
        self._latest_frame = latest_frame
        self.company_id = company_id
        self.camera_id = camera_id
        self.stop_event = stop_event
        self.pause_event = pause_event
        self.shared_dict = shared_dict

        # Configuration
        self.batch_size = 8
        self.min_batch_wait = 0.1
        self.cleanup_interval = 60

    def run(self):
        """Main process loop"""
        try:
            # Initialize device and model
            self.device = self._init_device()
            self.model = self._init_model()

            # Initialize trackers
            self.person_tracker = self._init_trackers()

            # Batch processing variables
            batch_frames: List[np.ndarray] = []
            batch_timestamps: List[float] = []
            last_batch_time = time.time()
            last_cleanup = time.time()

            while not self.stop_event.is_set():
                if self.pause_event.is_set():
                    time.sleep(0.1)
                    continue

                # Collect batch
                while len(batch_frames) < self.batch_size:
                    try:
                        frame_data, timestamp = self.frame_queue.get_nowait()
                        frame = self._decompress_frame(frame_data)
                        if frame is not None:
                            batch_frames.append(frame)
                            batch_timestamps.append(timestamp)
                    except queue.Empty:
                        break

                # Process if we have frames and either batch is full or enough time has passed
                current_time = time.time()
                if batch_frames and (
                    len(batch_frames) >= self.batch_size
                    or current_time - last_batch_time > self.min_batch_wait
                ):

                    # Process batch
                    processed_frames = self._process_batch(
                        batch_frames, batch_timestamps
                    )

                    # Update output queue and latest frame
                    for frame in processed_frames:
                        if frame is not None:
                            # Update latest frame (non-blocking)
                            with self._latest_frame.get_lock():
                                self._latest_frame.value = frame
                            # Add to output queue if space available
                            if not self.output_queue.full():
                                self.output_queue.put_nowait(frame)

                    # Reset batch
                    batch_frames = []
                    batch_timestamps = []
                    last_batch_time = current_time

                    # Periodic cleanup
                    if current_time - last_cleanup > self.cleanup_interval:
                        self._perform_cleanup()
                        last_cleanup = current_time

                else:
                    time.sleep(0.001)  # Prevent busy waiting

        except Exception as e:
            logger.error(f"Error in frame processing: {e}")
        finally:
            self._cleanup_resources()

    def _init_device(self) -> torch.device:
        """Initialize optimal processing device"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
            torch.set_num_threads(mp.cpu_count())
        return device

    def _init_model(self) -> YOLO:
        """Initialize YOLO model with optimizations"""
        model = YOLO("yolov8n.pt")
        model.to(self.device)
        return model

    def _init_trackers(self) -> Any:
        """Initialize tracking system"""
        # Implement your tracker initialization here
        pass

    def _process_batch(
        self, batch_frames: List[np.ndarray], batch_timestamps: List[float]
    ) -> List[Optional[bytes]]:
        """Process a batch of frames"""
        try:
            # Run YOLO detection on batch
            results = self.model.track(
                source=batch_frames,
                conf=0.5,
                iou=0.7,
                persist=True,
                tracker="bytetrack.yaml",
                device=self.device,
                verbose=False,
            )

            # Process detections for each frame
            processed_frames = []
            for frame, timestamp, result in zip(
                batch_frames, batch_timestamps, results
            ):
                try:
                    processed_frame = self._process_detections(frame, result, timestamp)
                    if processed_frame is not None:
                        # Compress frame
                        _, buffer = cv2.imencode(
                            ".jpg", processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 50]
                        )
                        processed_frames.append(buffer.tobytes())
                    else:
                        processed_frames.append(None)
                except Exception as e:
                    logger.error(f"Error processing detection: {e}")
                    processed_frames.append(None)

            return processed_frames

        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            return [None] * len(batch_frames)

    def _process_detections(
        self, frame: np.ndarray, result: Any, timestamp: float
    ) -> Optional[np.ndarray]:
        """Process detections for a single frame"""
        # Implement your detection processing here
        pass

    @staticmethod
    def _decompress_frame(frame_data: bytes) -> Optional[np.ndarray]:
        """Decompress frame from bytes"""
        try:
            nparr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return frame
        except Exception as e:
            logger.error(f"Error decompressing frame: {e}")
            return None

    def _perform_cleanup(self):
        """Perform periodic cleanup tasks"""
        torch.cuda.empty_cache()
        gc.collect()

    def _cleanup_resources(self):
        """Clean up resources on shutdown"""
        torch.cuda.empty_cache()
        gc.collect()


class VideoProcessor:
    def __init__(self, rtsp_url: str, company_id: str, camera_id: str):
        self.rtsp_url = rtsp_url
        self.company_id = company_id
        self.camera_id = camera_id

        self.manager = mp.Manager()
        self.shared_dict = self.manager.dict()

        self.frame_queue = mp.Queue(maxsize=10)
        self.output_queue = mp.Queue(maxsize=5)

        self.stop_event = mp.Event()
        self.pause_event = mp.Event()

        self._latest_frame = self.manager.Value("c", b"")

        self.frame_skip = 2
        self.max_frame_size = (680, 480)
        self.batch_size = 9
        self.min_batch_wait = 0.1

        self.fps_counter = mp.Value("d", 0.0)
        self.processing_times = []
        self.last_cleanup = time.time()
        self.cleanup_interval = 60

        self.processes = []

        self.frame_reader = FrameReader(
            rtsp_url,
            self.frame_queue,
            self.stop_event,
            self.pause_event,
            self.shared_dict,
        )

        self.frame_processor = FrameProcessor(
            self.frame_queue,
            self.output_queue,
            company_id,
            camera_id,
            self.stop_event,
            self.pause_event,
            self.shared_dict,
        )

    def start(self):
        """Start the processing pipeline"""
        try:
            self.frame_reader.start()
            self.frame_processor.start()
            logger.info("Started video processor")
        except Exception as e:
            logger.error(f"Error starting video processor: {e}")
            self.stop()
            raise

    def stop(self):
        """Stop all processes and clean up resources"""
        logger.info("Stopping video processor")
        try:
            self.stop_event.set()

            for process in [self.frame_reader, self.frame_processor]:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=1)
                    if process.is_alive():
                        process.kill()

            # Clear queues
            self._clear_queue(self.frame_queue)
            self._clear_queue(self.output_queue)
        except Exception as e:
            logger.error(f"Error stopping video processor: {e}")
            # Emergency cleanup
            for process in [self.frame_reader, self.frame_processor]:
                if process.is_alive():
                    try:
                        process.kill()
                    except:
                        pass

    def get_latest_frame(self) -> Optional[bytes]:
        """Get the latest processed frame (non-blocking)

        Returns:
            bytes: The latest frame data or None if no frame is available
        """
        try:
            frame_data = self.output_queue.get_nowait()
            return frame_data if frame_data else None
        except Exception as e:
            logger.error(f"Error getting latest frame: {e}")
            return None

    @staticmethod
    def _clear_queue(q: mp.Queue):
        """Safely clear a queue"""
        while not q.empty():
            try:
                q.get_nowait()
            except queue.Empty:
                break
