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
from typing import Optional
from ultralytics import YOLO
from logging_config import setup_logger

setup_logger()
logger = logging.getLogger("VideoProcessor")


def process_frames_process(
    frame_queue: mp.Queue,
    output_queue: mp.Queue,
    company_id: str,
    camera_id: str,
    stop_event: mp.Event,
):
    """Separate process for frame processing"""
    thread_pool = None
    import os

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    try:
        # Initialize device
        if torch.cuda.is_available():
            device = torch.device("cuda")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
            torch.set_num_threads(mp.cpu_count())

        # Initialize YOLO model with correct settings
        model = YOLO("yolov8n.pt")
        model.to(device)

        logger.info("Company_id: %s, Camera_id: %s", company_id, camera_id)
        # Initialize trackers
        person_tracker = OptimizedPersonTracker(company_id, camera_id, device)
        cross_camera_tracker = OptimizedCrossCameraTracker()
        # Attach cross-camera tracker to person_tracker
        person_tracker.cross_camera_tracker = cross_camera_tracker

        # Initialize thread pool
        thread_pool = ThreadPoolExecutor(max_workers=4)

        # Batch processing setup
        batch_size = 4
        batch_frames = []
        batch_timestamps = []

        while not stop_event.is_set():
            try:
                # Collect batch
                while len(batch_frames) < batch_size:
                    try:
                        frame_data, timestamp = frame_queue.get_nowait()

                        # Decompress frame
                        nparr = np.frombuffer(frame_data, np.uint8)
                        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                        if frame is None:
                            continue

                        batch_frames.append(frame)
                        batch_timestamps.append(timestamp)
                    except queue.Empty:
                        break

                if not batch_frames:
                    time.sleep(0.001)
                    continue

                # Process batch with YOLO - Updated model call
                try:
                    results = model.track(
                        source=batch_frames,
                        conf=0.5,
                        iou=0.7,
                        persist=True,
                        tracker="bytetrack.yaml",
                        device=device,
                        verbose=False,
                    )
                except RuntimeError as e:
                    if "torchvision::nms" in str(e):
                        logger.error("NMS error on CUDA. Falling back to CPU.")
                        model.to("cpu")
                        results = model.track(
                            source=batch_frames,
                            conf=0.5,
                            iou=0.7,
                            persist=True,
                            tracker="bytetrack.yaml",
                            device="cpu",
                            verbose=False,
                        )
                    else:
                        raise

                # Process detections in parallel
                futures = []
                for frame, timestamp, result in zip(
                    batch_frames, batch_timestamps, results
                ):
                    future = thread_pool.submit(
                        process_detections,
                        frame,
                        result,
                        person_tracker,
                        cross_camera_tracker,
                        camera_id,
                        timestamp,
                    )
                    futures.append((future, frame))

                # Handle processed results
                for future, frame in futures:
                    try:
                        processed_frame = future.result(timeout=1)
                        if processed_frame is not None:
                            # Compress frame for output
                            _, buffer = cv2.imencode(
                                ".jpg",
                                processed_frame,
                                [cv2.IMWRITE_JPEG_QUALITY, 85],
                            )
                            frame_bytes = buffer.tobytes()

                            if not output_queue.full():
                                output_queue.put_nowait(frame_bytes)
                    except Exception as e:
                        logger.error(f"Error processing detection: {e}")

                batch_frames = []
                batch_timestamps = []

            except Exception as e:
                logger.error(f"Error in frame processor: {e}")
                batch_frames = []
                batch_timestamps = []

    except Exception as e:
        logger.error(f"Error initializing processor: {e}")
    finally:
        if thread_pool:
            thread_pool.shutdown()


def read_frames_process(rtsp_url: str, frame_queue: mp.Queue, stop_event: mp.Event):
    """Separate process for frame reading with enhanced RTSP handling"""
    cap = None
    retry_delay = 1.0
    max_retry_delay = 5.0
    max_consecutive_failures = 10
    failure_count = 0
    import os

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    while not stop_event.is_set():
        try:
            if cap is None:
                logger.info(f"Attempting to connect to RTSP stream: {rtsp_url}")

                # Configure RTSP settings
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
                time.sleep(2)
                # Create capture with optimized settings
                cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

                # Check connection
                if not cap.isOpened():
                    raise RuntimeError("Failed to open RTSP stream")

                logger.info("Successfully connected to RTSP stream")
                retry_delay = 1.0  # Reset retry delay on successful connection
                failure_count = 0  # Reset failure count

            frame_counter = 0
            skip_frames = 2  # Process every 3rd frame

            while not stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    failure_count += 1
                    if failure_count > max_consecutive_failures:
                        logger.error("Too many consecutive frame read failures")
                        raise RuntimeError("Stream connection lost")
                    continue

                failure_count = 0  # Reset on successful read
                frame_counter += 1

                if frame_counter % skip_frames != 0:
                    continue

                # Optimize frame size
                frame = cv2.resize(frame, (680, 480), interpolation=cv2.INTER_AREA)

                # Compress frame for queue
                try:
                    _, buffer = cv2.imencode(
                        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70]
                    )
                    frame_data = buffer.tobytes()

                    if not frame_queue.full():
                        frame_queue.put_nowait((frame_data, time.time()))
                except queue.Full:
                    continue  # Skip frame if queue is full
                except Exception as e:
                    logger.error(f"Error compressing frame: {e}")
                    continue

        except Exception as e:
            logger.error(f"RTSP stream error: {e}")
            if cap is not None:
                cap.release()
                cap = None

            # Progressive backoff with maximum
            time.sleep(retry_delay)
            retry_delay = min(retry_delay * 1.5, max_retry_delay)
            continue

    # Cleanup
    if cap is not None:
        cap.release()


def process_detections(
    frame: np.ndarray,
    result,
    person_tracker: OptimizedPersonTracker,
    cross_camera_tracker: OptimizedCrossCameraTracker,
    camera_id: str,
    timestamp: float,
) -> Optional[np.ndarray]:
    """Process detections for a single frame"""
    import os

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    try:
        # Update person tracker
        updated_tracks, new_tracks = person_tracker.update(frame, result)

        # Create a copy of frame for drawing
        draw_frame = frame.copy()

        # Prepare data for cross-camera tracking
        person_crops = []
        face_ids = {}
        recognition_statuses = {}
        positions = {}

        all_tracks = updated_tracks + new_tracks
        for track in all_tracks:
            if track.face_image is None:
                track.face_image = person_tracker._extract_face(frame, track.bbox)

            # Get person crop
            x, y, w, h = map(int, track.bbox)
            person_crop = frame[y : y + h, x : x + w]
            person_crops.append((track.track_id, person_crop))

            if track.face_id:
                face_ids[track.track_id] = track.face_id
            recognition_statuses[track.track_id] = track.recognition_status
            positions[track.track_id] = (
                track.bbox[0] + track.bbox[2] / 2,
                track.bbox[1] + track.bbox[3] / 2,
            )

            # Draw bounding box and information
            cv2.rectangle(draw_frame, (x, y), (x + w, y + h), track.color, 2)

            # Draw filled rectangle behind text
            text = f"ID: {track.track_id} ({track.recognition_status})"
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

            # Draw text
            cv2.putText(
                draw_frame,
                text,
                (x + 5, y - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),  # White text
                2,
            )

            # Add status indicator dot
            status_color = {
                "pending": (0, 255, 255),  # Yellow
                "in_progress": (0, 165, 255),  # Orange
                "recognized": (0, 255, 0),  # Green
                "unknown": (0, 0, 255),  # Red
            }.get(track.recognition_status, (255, 255, 255))

            dot_radius = 5
            dot_position = (x + text_width + 20, y - text_height // 2 - 5)
            cv2.circle(draw_frame, dot_position, dot_radius, status_color, -1)

        # Update cross-camera tracking
        if person_crops:
            global_ids = cross_camera_tracker.update(
                camera_id=camera_id,
                person_crops=person_crops,
                positions=positions,
                face_ids=face_ids,
                recognition_statuses=recognition_statuses,
            )

            # Update tracked persons with global IDs
            for track in all_tracks:
                if track.track_id in global_ids:
                    track.global_id = global_ids[track.track_id]

        return draw_frame

    except Exception as e:
        logger.error(f"Error processing detections: {e}")
        return frame


class VideoProcessor:
    def __init__(self, rtsp_url: str, company_id: str, camera_id: str):
        self.rtsp_url = rtsp_url
        self.company_id = company_id
        self.camera_id = camera_id

        # Initialize queues with reasonable sizes
        self.frame_queue = mp.Queue(maxsize=30)
        self.output_queue = mp.Queue(maxsize=30)

        # Control event
        self.stop_event = mp.Event()

        # Store processes
        self.processes = []

    def start(self):
        """Start the processing pipeline"""
        try:
            # Start frame reader process
            reader_process = mp.Process(
                target=read_frames_process,
                args=(self.rtsp_url, self.frame_queue, self.stop_event),
            )
            self.processes.append(reader_process)

            # Start processor process
            processor_process = mp.Process(
                target=process_frames_process,
                args=(
                    self.frame_queue,
                    self.output_queue,
                    self.company_id,
                    self.camera_id,
                    self.stop_event,
                ),
            )
            self.processes.append(processor_process)

            # Start all processes
            for process in self.processes:
                process.start()

        except Exception as e:
            logger.error(f"Error starting video processor: {e}")
            self.stop()
            raise

    def get_latest_frame(self) -> Optional[bytes]:
        """Get the latest processed frame"""
        try:
            return self.output_queue.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        """Stop all processes"""
        logger.info("Stopping video processor")
        try:
            # Set stop event
            self.stop_event.set()

            # Stop all processes
            for process in self.processes:
                if process and process.is_alive():
                    process.terminate()
                    process.join(timeout=1)
                    if process.is_alive():
                        process.kill()

            # Clear queues
            self._clear_queue(self.frame_queue)
            self._clear_queue(self.output_queue)

            logger.info("Video processor stopped successfully")

        except Exception as e:
            logger.error(f"Error stopping video processor: {e}")
            # Emergency cleanup
            for process in self.processes:
                if process and process.is_alive():
                    try:
                        process.kill()
                    except:
                        pass

    @staticmethod
    def _clear_queue(q: mp.Queue):
        """Safely clear a queue"""
        while not q.empty():
            try:
                q.get_nowait()
            except:
                break
