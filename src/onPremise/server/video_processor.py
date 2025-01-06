import multiprocessing as mp
import cv2
import numpy as np
import logging
import time
import torch
import uuid
import aiohttp
import asyncio

from logging_config import setup_logger
from person_tracker import TrackedPerson, PersonTracker
from nats_client import NatsClient, SharedNatsClient, Commands

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

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to read frame")
            time.sleep(0.1)
            continue

        # Resize frame to 720p to reduce processing load
        frame = cv2.resize(frame, (1280, 720))

        # If queue is full, remove oldest frame
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except:
                pass

        try:
            frame_queue.put_nowait(frame)
        except:
            pass

    cap.release()
    logger.info("Frame reader stopped")


async def handle_face_recognition(
    nats_client: NatsClient,
    company_id: str,
    camera_id: str,
    person: TrackedPerson,
    face_image: np.ndarray,
):
    """Handle the face recognition process through NATS"""
    try:
        nats_client = SharedNatsClient.get_instance()
        # Generate face ID if not exists
        if not person.face_id:
            person.face_id = str(uuid.uuid4())

        # Get presigned URL for upload
        response = await nats_client.send_message_with_reply(
            Commands.GET_PRESIGNED_UPLOAD_UNKNOWN_FACE_URL.value,
            {
                "company_id": company_id,
                "face_id": person.face_id,
            },
        )

        if not response or not response.get("success"):
            logger.error("Failed to get presigned URL")
            return

        # Upload the image
        _, img_encoded = cv2.imencode(".jpg", face_image)
        img_bytes = img_encoded.tobytes()

        async with aiohttp.ClientSession() as session:
            async with session.put(response["url"], data=img_bytes) as resp:
                if resp.status != 200:
                    logger.error("Failed to upload face image")
                    return

        # Request recognition
        person.recognition_status = "in_progress"
        recognition_response = await nats_client.send_message_with_reply(
            Commands.EXECUTE_RECOGNITION.value,
            {
                "company_id": company_id,
                "camera_id": camera_id,
                "face_id": person.face_id,
                "track_id": person.track_id,
            },
        )

        if not recognition_response or not recognition_response.get("success"):
            logger.error("Failed to start recognition")
            person.recognition_status = "pending"
            return

    except Exception as e:
        logger.error(f"Error in face recognition: {e}")
        person.recognition_status = "pending"


def process_frames(
    frame_queue: mp.Queue,
    output_queue: mp.Queue,
    company_id: str,
    camera_id: str,
    stop_event: mp.Event,
):
    """Process function to run YOLO detection on frames"""
    loop = None
    try:
        # Import YOLO here inside the process
        from ultralytics import YOLO
        import torch

        # Check available devices
        if torch.cuda.is_available():
            device = "cuda"
            device_name = torch.cuda.get_device_name(0)
            logger.info(f"Using GPU: {device_name}")
        elif torch.backends.mps.is_available():
            device = "mps"
            logger.info("Using Apple M1/M2 GPU")
        else:
            device = "cpu"
            logger.info("No GPU available, using CPU")

        # Load model and move to device
        model = YOLO("yolov8n.pt")
        model.to(device)

        # Add some debug info about model device
        logger.info(f"Model is on device: {next(model.parameters()).device}")

        person_tracker = PersonTracker()

        nats_client = SharedNatsClient.get_instance()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        while not stop_event.is_set():
            try:
                frame = frame_queue.get(timeout=1)
            except:
                continue

            # Run YOLO detection with explicit device
            results = model(
                frame,
                device=device,
                verbose=False,
            )

            boxes = []
            scores = []
            for result in results:
                for box in result.boxes:
                    if box.cls.cpu().numpy()[0] == 0:  # Person class
                        boxes.append(box.xyxy[0].cpu().numpy())  # [x1, y1, x2, y2]
                        scores.append(
                            float(box.conf.cpu().numpy()[0])
                        )  # confidence score

            if boxes:  # Only update if we have detections
                boxes = np.array(boxes)
                scores = np.array(scores)

                # Update tracker
                current_time = time.time()
                updated_tracks, new_tracks = person_tracker.update(
                    frame, boxes, scores, current_time
                )

                # Handle face recognition for new and updated tracks
                for track in new_tracks + updated_tracks:
                    if track.recognition_status == "pending":
                        loop.create_task(
                            handle_face_recognition(
                                nats_client,
                                company_id,
                                camera_id,
                                track,
                                track.face_image,
                            )
                        )

            # Draw results on frame
            for person in person_tracker.tracked_persons.values():
                bbox = person.bbox
                # Draw bounding box
                cv2.rectangle(
                    frame,
                    (int(bbox[0]), int(bbox[1])),
                    (int(bbox[2]), int(bbox[3])),
                    (0, 255, 0),
                    2,
                )

                # Draw ID and status
                status_color = {
                    "pending": (0, 255, 255),  # Yellow
                    "in_progress": (0, 165, 255),  # Orange
                    "recognized": (0, 255, 0),  # Green
                    "unknown": (0, 0, 255),  # Red
                }.get(person.recognition_status, (255, 255, 255))

                cv2.putText(
                    frame,
                    f"ID: {person.track_id} ({person.recognition_status})",
                    (int(bbox[0]), int(bbox[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    status_color,
                    2,
                )

            # Convert to bytes for WebSocket transmission
            _, buffer = cv2.imencode(".jpg", frame)
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

            # Process any pending tasks
            loop.run_until_complete(asyncio.sleep(0))

    except Exception as e:
        logger.error(f"Error in frame processor: {e}")
    finally:
        if loop:
            loop.close()
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
