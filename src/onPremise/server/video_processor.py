import multiprocessing as mp
import cv2
import numpy as np
import logging
import time
import torch
import uuid
import aiohttp
import asyncio

from concurrent.futures import ThreadPoolExecutor
from logging_config import setup_logger
from person_tracker import TrackedPerson, PersonTracker
from nats_client import NatsClient, SharedNatsClient, Commands
from reid_implementation import SharedCrossCameraTracker

setup_logger()
logger = logging.getLogger("VideoProcessor")


def read_frames(url: str, frame_queue: mp.Queue, stop_event: mp.Event):
    """Process function to read frames from RTSP stream"""
    logger.info(f"Starting frame reader for {url}")
    time.sleep(5)  # Wait for 5 seconds before starting the frame reader
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        logger.error("Failed to open RTSP stream")
        return

    frame_counter = 0
    skip_frames = 2

    while not stop_event.is_set():
        frame_counter += 1

        # Skip frames to reduce processing load
        if frame_counter % skip_frames != 0:
            logger.info("Skipping frame")
            continue

        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to read frame")
            time.sleep(0.1)
            continue

        # Resize frame to 720p to reduce processing load
        frame = cv2.resize(frame, (640, 480))

        # If queue is full, remove oldest frame
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except:
                pass

        try:
            frame_queue.put_nowait(frame)
        except:
            logger.error("Failed to put frame in queue")
            pass

    cap.release()
    logger.info("Frame reader stopped")


def draw_on_frame(
    frame: np.ndarray, person: TrackedPerson, global_id: str
) -> np.ndarray:
    """Draw bounding box and status for a tracked person"""
    x, y, w, h = map(int, person.bbox)

    # Use person's unique color for bounding box
    box_color = person.color

    # Draw bounding box with person's unique color
    cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)

    # Draw filled rectangle behind text for better visibility
    text = f"Global: {global_id} ({person.recognition_status})"
    (text_width, text_height), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
    )
    cv2.rectangle(
        frame, (x, y - text_height - 10), (x + text_width + 10, y), box_color, -1
    )

    # Draw text in white for better contrast
    cv2.putText(
        frame,
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
    }.get(person.recognition_status, (255, 255, 255))

    dot_radius = 5
    dot_position = (x + text_width + 20, y - text_height // 2 - 5)
    cv2.circle(frame, dot_position, dot_radius, status_color, -1)

    return frame


async def handle_face_recognition(
    nats_client,
    company_id: str,
    camera_id: str,
    track: "TrackedPerson",
    face_image: np.ndarray,
):
    """Handle face recognition through AWS Rekognition"""
    try:
        # Skip if we don't have a face image
        if face_image is None:
            return

        # Convert face image to jpg bytes
        _, img_encoded = cv2.imencode(".jpg", face_image)
        img_bytes = img_encoded.tobytes()

        # Get presigned URL for unknown face upload
        response = await nats_client.send_message_with_reply(
            Commands.GET_PRESIGNED_UPLOAD_UNKNOWN_FACE_URL.value,
            {
                "company_id": company_id,
                "face_id": track.face_id,
            },
        )

        if not response or not response.get("success"):
            logger.error("Failed to get presigned URL for face upload")
            return

        # Upload the face image using presigned URL
        async with aiohttp.ClientSession() as session:
            async with session.put(response["url"], data=img_bytes) as resp:
                if resp.status != 200:
                    logger.error("Failed to upload face image")
                    return

        # Execute recognition
        track.recognition_status = "in_progress"
        recognition_response = await nats_client.send_message_with_reply(
            Commands.EXECUTE_RECOGNITION.value,
            {
                "company_id": company_id,
                "camera_id": camera_id,
                "face_id": track.face_id,
                "track_id": track.track_id,
            },
        )

        if recognition_response and recognition_response.get("success"):
            track.face_id = recognition_response.get("face_id")
            track.recognition_status = recognition_response.get("status")
        else:
            track.recognition_status = "unknown"

    except Exception as e:
        logger.error(f"Error in face recognition: {e}")
        track.recognition_status = "pending"


def process_frames(
    frame_queue: mp.Queue,
    output_queue: mp.Queue,
    company_id: str,
    camera_id: str,
    stop_event: mp.Event,
):
    loop = None
    thread_pool = None

    try:
        import torch
        import cv2
        from ultralytics import YOLO
        from reid_implementation import SharedCrossCameraTracker

        # Get the shared cross-camera tracker instance
        cross_camera_tracker = SharedCrossCameraTracker.get_instance()

        # Check for hardware acceleration
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using Apple M1 MPS acceleration")
        else:
            device = torch.device("cpu")
            logger.warning("No GPU acceleration available. Running on CPU")

        # Initialize model
        model = YOLO("yolov8n.pt")
        model.to(device)

        # Create thread pool
        thread_pool = ThreadPoolExecutor(max_workers=4)

        # Initialize tracker
        person_tracker = PersonTracker()

        # Initialize NATS client
        nats_client = SharedNatsClient.get_instance()

        # Create event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        while not stop_event.is_set():
            try:
                frame = frame_queue.get(timeout=1)
            except:
                continue

            # Run YOLO detection with tracking
            results = model.track(
                source=frame,
                conf=0.5,
                iou=0.7,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False,
                device=device,
            )

            if results and len(results) > 0:
                # Update tracker with results
                current_time = time.time()
                updated_tracks, new_tracks = person_tracker.update(
                    frame, results, current_time
                )

                # Prepare data for cross-camera tracking
                person_crops = []
                face_ids = {}
                recognition_statuses = {}

                # Process all tracks
                for track in updated_tracks + new_tracks:
                    # Generate face ID if not exists
                    if not track.face_id:
                        track.face_id = str(uuid.uuid4())

                    # Get person crop
                    x, y, w, h = map(int, track.bbox)
                    person_crop = frame[y : y + h, x : x + w]
                    person_crops.append((track.track_id, person_crop))

                    if track.face_id:
                        face_ids[track.track_id] = track.face_id
                    if track.recognition_status:
                        recognition_statuses[track.track_id] = track.recognition_status

                positions = {
                    track.track_id: (
                        track.bbox[0] + track.bbox[2] / 2,  # center x
                        track.bbox[1] + track.bbox[3] / 2,  # center y
                    )
                    for track in (new_tracks + updated_tracks)
                }

                # Update cross-camera tracking
                global_ids = cross_camera_tracker.update(
                    camera_id=camera_id,
                    person_crops=person_crops,
                    positions=positions,  # Add this
                    face_ids=face_ids,
                    recognition_statuses=recognition_statuses,
                )

                # Process face recognition
                face_tasks = []
                for track in new_tracks + [
                    t for t in updated_tracks if t.recognition_status == "pending"
                ]:
                    if track.face_image is not None:
                        # Get global ID
                        global_id = global_ids.get(track.track_id)

                        # Check if already recognized in another camera
                        if global_id:
                            person_info = cross_camera_tracker.get_person_info(
                                global_id
                            )
                            if (
                                person_info
                                and person_info.recognition_status != "pending"
                            ):
                                track.face_id = person_info.face_id
                                track.recognition_status = (
                                    person_info.recognition_status
                                )
                                continue

                        # If not recognized, send for recognition
                        if nats_client and nats_client._connected:
                            face_task = loop.create_task(
                                handle_face_recognition(
                                    nats_client,
                                    company_id,
                                    camera_id,
                                    track,
                                    track.face_image,
                                )
                            )
                            face_tasks.append(face_task)

                # Draw results
                draw_frame = frame.copy()
                futures = []
                for person in person_tracker.tracked_persons.values():
                    global_id = global_ids.get(person.track_id, "unknown")
                    future = thread_pool.submit(
                        draw_on_frame, draw_frame, person, global_id
                    )
                    futures.append(future)

                # Wait for drawing
                for future in futures:
                    try:
                        future.result(timeout=0.1)
                    except TimeoutError:
                        logger.warning("Drawing operation timed out")
                        continue

            else:
                draw_frame = frame

            # Convert to bytes for WebSocket
            _, buffer = cv2.imencode(".jpg", draw_frame)
            frame_bytes = buffer.tobytes()

            if output_queue.full():
                try:
                    output_queue.get_nowait()
                except:
                    pass
            try:
                output_queue.put_nowait(frame_bytes)
            except:
                pass

            # Process pending tasks
            loop.run_until_complete(asyncio.sleep(0))

    except Exception as e:
        logger.error(f"Error in frame processor: {e}", exc_info=True)
    finally:
        if loop:
            loop.close()
        if thread_pool:
            thread_pool.shutdown(wait=False)
        logger.info("Frame processor stopped")


class VideoProcessor:
    def __init__(self, rtsp_url: str, company_id: str, camera_id: str):
        self.rtsp_url = rtsp_url
        self.company_id = company_id
        self.camera_id = camera_id
        self.frame_queue = mp.Queue(maxsize=10)
        self.output_queue = mp.Queue(maxsize=10)
        self.stop_event = mp.Event()
        self.reader_process = None
        self.processor_process = None

    def start(self):
        """Start both reader and processor processes"""
        try:
            self.reader_process = mp.Process(
                target=read_frames,
                args=(self.rtsp_url, self.frame_queue, self.stop_event),
            )

            self.processor_process = mp.Process(
                target=process_frames,
                args=(
                    self.frame_queue,
                    self.output_queue,
                    self.company_id,
                    self.camera_id,
                    self.stop_event,
                ),
            )

            self.reader_process.start()
            self.processor_process.start()
            logger.info("Started video processor processes")

        except Exception as e:
            logger.error(f"Error starting processes: {e}")
            self.stop()
            raise

    def get_latest_frame(self) -> bytes:
        """Get the latest processed frame"""
        try:
            return self.output_queue.get_nowait()
        except:
            return None

    def stop(self):
        logger.info("Stopping video processor")
        try:
            self.stop_event.set()

            for process in [self.reader_process, self.processor_process]:
                if process and process.is_alive():
                    process.terminate()
                    process.join(timeout=1)
                    if process.is_alive():
                        process.kill()

            logger.info("Video processor stopped successfully")

        except Exception as e:
            logger.error(f"Error stopping video processor: {e}")
            # Emergency cleanup
            for process in [self.reader_process, self.processor_process]:
                if process and process.is_alive():
                    try:
                        process.kill()
                    except:
                        pass
