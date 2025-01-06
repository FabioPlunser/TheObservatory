import json
from urllib.parse import urlparse
import boto3
import sqlite3
import logging
from datetime import datetime
import nats
import asyncio
from typing import Dict, Any
import sys

# Configure logger with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Default NATS URL
NATS_URL = "nats://54.226.169.138:4222"

# Override NATS URL if provided as a command-line argument
if len(sys.argv) > 1:
    NATS_URL = sys.argv[1]
    logger.info(f"Received NATS URL from CLI: {NATS_URL}")
else:
    logger.warning("No NATS URL provided as a command-line argument.")

SIMILARITY_THRESHOLD = 90.0
KNOWN_FACES_BUCKET = "the-observatory-known-faces"
FACES_TO_CHECK_BUCKET = "the-observatory-faces-to-check"


class FaceRecognitionService:
    def __init__(self, nats_client: nats.NATS):
        logger.info("Initializing FaceRecognitionService")
        try:
            self.s3_client = boto3.client("s3")
            self.rekognition_client = boto3.client(
                "rekognition", region_name="us-east-1"
            )
            self.db_conn = sqlite3.connect("database.db")
            self.cursor = self.db_conn.cursor()
            self.nc = nats_client
            self._initialize_database()
        except Exception as e:
            logger.critical(f"Failed to initialize FaceRecognitionService: {e}")
            raise

    def _initialize_database(self) -> None:
        logger.debug("Starting database initialization")
        try:
            self.cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS cameras (
                    camera_id INTEGER PRIMARY KEY,
                    name TEXT,
                    alarm_active INTEGER DEFAULT 0,
                    last_triggered TEXT
                )
            """
            )
            self.cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    camera_id INTEGER,
                    image_key TEXT,
                    is_known INTEGER,
                    detected_at TEXT,
                    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id)
                )
            """
            )
            self.db_conn.commit()
            logger.info("Database initialization completed successfully")
        except sqlite3.Error as e:
            logger.error(f"Database initialization failed: {e}")
            raise

    async def process_image(self, camera_id: int, image_url: str) -> Dict[str, Any]:
        logger.info(f"Processing image from camera {camera_id}")
        logger.debug(f"Image URL: {image_url}")

        try:
            parsed_url = urlparse(image_url)
            image_key = parsed_url.path.lstrip("/")
            logger.debug(f"Extracted image key: {image_key}")

            known_faces = self.s3_client.list_objects_v2(Bucket=KNOWN_FACES_BUCKET)

            if "Contents" not in known_faces:
                logger.warning(f"No known faces found in bucket {KNOWN_FACES_BUCKET}")
                return {
                    "status": "error",
                    "message": "No known faces available for comparison",
                }

            for known_face in known_faces.get("Contents", []):
                logger.debug(f"Comparing against known face: {known_face['Key']}")

                response = self.rekognition_client.compare_faces(
                    SourceImage={
                        "S3Object": {
                            "Bucket": KNOWN_FACES_BUCKET,
                            "Name": known_face["Key"],
                        }
                    },
                    TargetImage={
                        "S3Object": {"Bucket": FACES_TO_CHECK_BUCKET, "Name": image_key}
                    },
                    SimilarityThreshold=SIMILARITY_THRESHOLD,
                )

                if response["FaceMatches"]:
                    logger.info(f"Known face detected for camera {camera_id}")
                    self.log_detection(camera_id, image_key, True)
                    # TODO remove image from bucket
                    return {"status": "known_face_detected", "camera_id": camera_id}

            logger.warning(f"Unknown face detected for camera {camera_id}")
            await self.trigger_alarm(camera_id, image_url)
            self.log_detection(camera_id, image_key, False)
            return {"status": "unknown_face_detected", "camera_id": camera_id}

        except Exception as e:
            logger.error(f"Error processing image: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    async def trigger_alarm(self, camera_id: int, image_url: str) -> None:
        logger.info(f"Triggering alarm for camera {camera_id}")
        try:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.cursor.execute(
                "UPDATE cameras SET alarm_active = 1, last_triggered = ? WHERE camera_id = ?",
                (current_time, camera_id),
            )
            self.db_conn.commit()
            await self.nc.send_alarm_message(camera_id, image_url)
            logger.info(f"Alarm triggered successfully for camera {camera_id}")
        except Exception as e:
            logger.error(f"Failed to trigger alarm: {e}", exc_info=True)
            raise

    def log_detection(self, camera_id: int, image_key: str, is_known: bool) -> None:
        logger.debug(f"Logging detection - Camera: {camera_id}, Known: {is_known}")
        try:
            self.cursor.execute(
                "INSERT INTO detections (camera_id, image_key, is_known, detected_at) VALUES (?, ?, ?, ?)",
                (
                    camera_id,
                    image_key,
                    int(is_known),
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                ),
            )
            self.db_conn.commit()
            logger.debug("Detection logged successfully")
        except sqlite3.Error as e:
            logger.error(f"Failed to log detection: {e}")
            raise


class NATS:
    def __init__(self):
        self.nc = nats.NATS()

    async def connect(self) -> None:
        logger.info(f"Connecting to NATS server at {NATS_URL}")
        try:
            await self.nc.connect(NATS_URL)
            logger.info("Successfully connected to NATS server")
        except Exception as e:
            logger.error(f"Failed to connect to NATS server: {e}", exc_info=True)
            raise

    async def get_upload_message(self, face_service: FaceRecognitionService) -> None:
        async def compare_faces(msg):
            logger.debug("Received upload message")
            try:
                data = json.loads(msg.data.decode())
                image_url = data["presignedImageUrl"]
                camera_id = data["cameraId"]
                logger.info(f"Processing upload message for camera {camera_id}")
                await face_service.process_image(camera_id, image_url)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode message JSON: {e}")
            except KeyError as e:
                logger.error(f"Missing required field in message: {e}")
            except Exception as e:
                logger.error(f"Unexpected error processing message: {e}", exc_info=True)

        await self.nc.subscribe("upload.finished", cb=compare_faces)
        logger.info("Subscribed to 'upload.finished' topic")
        await asyncio.Event().wait()

    async def send_alarm_message(
        self, camera_id: int, presigned_image_url: str
    ) -> None:
        logger.debug(f"Preparing to send alarm message for camera {camera_id}")
        try:
            alarm_data = {
                "cameraId": camera_id,
                "presignedImageUrl": presigned_image_url,
            }
            await self.nc.publish("alarm.trigger", json.dumps(alarm_data).encode())
            logger.info(f"Alarm message sent successfully for camera {camera_id}")
        except Exception as e:
            logger.error(f"Failed to send alarm message: {e}", exc_info=True)
            raise


async def main() -> None:
    logger.info("Starting Face Recognition Service")
    try:
        nc = NATS()
        await nc.connect()

        service = FaceRecognitionService(nc)
        logger.info("Service initialized successfully")

        await nc.get_upload_message(service)
    except Exception as e:
        logger.critical(f"Critical error in main: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())
