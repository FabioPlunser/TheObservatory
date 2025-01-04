import multiprocessing as mp
import cv2
import numpy as np
import logging
import time
import torch

logger = logging.getLogger(__name__)


def read_frames(url: str, frame_queue: mp.Queue, stop_event: mp.Event):
    """Process function to read frames from RTSP stream"""
    logger.info(f"Starting frame reader for {url}")
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


def process_frames(frame_queue: mp.Queue, output_queue: mp.Queue, stop_event: mp.Event):
    """Process function to run YOLO detection on frames"""
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

        while not stop_event.is_set():
            try:
                frame = frame_queue.get(timeout=1)
            except:
                continue

            try:
                # Run YOLO detection with explicit device
                results = model(frame, device=device, verbose=False)

                # Draw boxes for detected persons
                for result in results:
                    for box in result.boxes:
                        if box.cls.cpu().numpy()[0] == 0:  # Person class
                            coords = box.xyxy[0].cpu().numpy()
                            conf = float(box.conf.cpu().numpy()[0])

                            cv2.rectangle(
                                frame,
                                (int(coords[0]), int(coords[1])),
                                (int(coords[2]), int(coords[3])),
                                (0, 255, 0),
                                2,
                            )

                            cv2.putText(
                                frame,
                                f"Person {conf:.2f}",
                                (int(coords[0]), int(coords[1] - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 0),
                                2,
                            )

                # Convert to bytes for WebSocket transmission
                _, buffer = cv2.imencode(".jpg", frame)
                frame_bytes = buffer.tobytes()

                # Update output queue
                if output_queue.full():
                    try:
                        output_queue.get_nowait()
                    except:
                        pass
                try:
                    output_queue.put_nowait(frame_bytes)
                except:
                    pass

            except Exception as e:
                logger.error(f"Error processing frame: {e}")
                continue

    except Exception as e:
        logger.error(f"Error in YOLO processor: {e}")
    finally:
        logger.info("YOLO processor stopped")


class VideoProcessor:
    def __init__(self, rtsp_url: str):
        self.rtsp_url = rtsp_url
        # Use multiprocessing.Queue() instead of Queue(maxsize=10)
        self.frame_queue = mp.Queue(maxsize=10)
        self.output_queue = mp.Queue(maxsize=10)
        self.stop_event = mp.Event()
        self.reader_process = None
        self.yolo_process = None

    def start(self):
        """Start both reader and processor processes"""
        try:
            # Create processes with only serializable arguments
            self.reader_process = mp.Process(
                target=read_frames,
                args=(self.rtsp_url, self.frame_queue, self.stop_event),
            )

            self.yolo_process = mp.Process(
                target=process_frames,
                args=(self.frame_queue, self.output_queue, self.stop_event),
            )

            self.reader_process.start()
            self.yolo_process.start()

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
        """Stop all processes"""
        logger.info("Stopping video processor")

        try:
            # Signal processes to stop
            self.stop_event.set()

            # Clean up processes with timeout
            for process_name, process in [
                ("reader", self.reader_process),
                ("YOLO", self.yolo_process),
            ]:

                if process and process.is_alive():
                    process.terminate()
                    logger.info(f"Stopping {process_name} process...")
                    process.kill()

            logger.info("Video processor stopped successfully")

        except Exception as e:
            logger.error(f"Error stopping video processor: {e}")
            # Emergency cleanup - make sure processes are really dead
            for process in [self.reader_process, self.yolo_process]:
                if process and process.is_alive():
                    try:
                        process.kill()
                    except:
                        pass
