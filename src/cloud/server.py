from bucket_handler import BucketHandler
from nats_client import NatsClient, Commands
from dotenv import load_dotenv

import asyncio
import json
import logging
import boto3
import signal
import os

load_dotenv()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class Server:
    def __init__(self):
        self.bucket_handler = BucketHandler(
            bucket_name=os.getenv("BUCKET_NAME"),
            region_name=os.getenv("REGION"),
        )
        self.nats_client = NatsClient()
        self.loop = asyncio.get_event_loop()
        self.recognition_client = boto3.client("rekognition", region_name="us-east-1")
        self.running = False
        self.tasks = []

    async def setup_subscriptions(self):
        await self.nats_client.add_subscription(
            Commands.INIT_BUCKET.value, self.handle_company_bucket_creation_request
        )
        await self.nats_client.add_subscription(
            Commands.GET_PRESIGNED_UPLOAD_KNOWN_FACE_URL.value,
            self.handle_presigned_upload_known_faces_request,
        )
        await self.nats_client.add_subscription(
            Commands.GET_PRESIGNED_UPLOAD_UNKNOWN_FACE_URL.value,
            self.handle_presigned_upload_unkonwn_faces_request,
        )
        await self.nats_client.add_subscription(
            Commands.GET_PRESIGNED_DOWNLOAD_ALL_KNOWN_FACES.value,
            self.handle_presigned_download_all_known_faces_request,
        )

        await self.nats_client.add_subscription(
            Commands.DELETE_KNOWN_FACE.value, self.handle_delete_known_face_request
        )

        await self.nats_client.add_subscription(
            Commands.EXECUTE_RECOGNITION.value, self.execute_rekognition
        )

    async def handle_company_bucket_creation_request(self, msg):
        """Handle request to create a company folder in the bucket"""
        try:
            logger.info("Handling company bucket creation request")
            data = json.loads(msg.data.decode())

            company_id = data.get("company_id")

            if not company_id:
                raise Exception("Missing required company_id")

            self.bucket_handler.create_company_folder(company_id)

            response = {
                "success": True,
                "company_id": company_id,
            }
            await msg.respond(json.dumps(response).encode())

        except Exception as e:
            logger.error(f"Failed to handle company bucket creation request: {e}")
            await msg.respond(json.dumps({"success": False, "error": str(e)}).encode())

    async def handle_presigned_upload_known_faces_request(self, msg):
        """Handle request for presigned URL for known face upload"""
        try:
            logger.info("Handling presigned upload known face request")
            data = json.loads(msg.data.decode())

            company_id = data.get("company_id")
            face_id = data.get("face_id")

            if not all([company_id, face_id]):
                raise Exception("Missing required fields")

            object_key = f"known_faces/{face_id}.jpg"

            url = self.bucket_handler.generate_presigned_upload_url(
                company_id, object_key, expires_in=3600
            )
            if not url:
                raise Exception("Failed to generate upload URL")

            await msg.respond(
                json.dumps(
                    {
                        "success": True,
                        "url": url,
                        "object_key": f"{company_id}/{object_key}",
                    }
                ).encode()
            )
        except Exception as e:
            logger.error(f"Failed to handle presigned upload request: {e}")
            await msg.respond(json.dumps({"success": False, "error": str(e)}).encode())

    async def handle_presigned_upload_unkonwn_faces_request(self, msg):
        """Handle request for presigned URL for unknown face upload"""
        try:
            logger.info("Handling presigned upload unknown face request")
            data = json.loads(msg.data.decode())
            company_id = data.get("company_id")
            face_id = data.get("face_id")

            if not all([company_id, face_id]):
                raise Exception("Missing required fields")

            object_key = f"unknown_faces/{face_id}.jpg"

            url = self.bucket_handler.generate_presigned_upload_url(
                company_id, object_key, expires_in=3600
            )

            if not url:
                raise Exception("Failed to generate upload URL")

            await msg.respond(
                json.dumps(
                    {
                        "success": True,
                        "url": url,
                        "object_key": f"{company_id}/{object_key}",
                    }
                ).encode()
            )
        except Exception as e:
            logger.error(f"Failed to handle presigned upload request: {e}")
            await msg.respond(json.dumps({"success": False, "error": str(e)}).encode())

    async def handle_presigned_download_all_known_faces_request(self, msg):
        """Handle request for presigned URL for all known faces download"""
        try:
            logger.info("Handling presigned download all known faces request")
            data = json.loads(msg.data.decode())
            company_id = data.get("company_id")
            if not company_id:
                raise Exception("Missing required company_id")

            object_key = f"known_faces/"
            url = self.bucket_handler.generate_list_presigned_url_of_key(
                company_id, object_key, expires_in=3600
            )

            if not url:
                raise Exception("Failed to generate download URL")

            await msg.respond(
                json.dumps(
                    {
                        "success": True,
                        "faces": url,
                    }
                ).encode()
            )
        except Exception as e:
            logger.error(f"Failed to handle presigned download request: {e}")
            await msg.respond(json.dumps({"success": False, "error": str(e)}).encode())

    async def handle_delete_known_face_request(self, msg):
        """Handle request to delete a known face from the bucket"""
        try:
            logger.info("Handling delete known face request")
            data = json.loads(msg.data.decode())
            key = data.get("key")

            if not key:
                raise Exception("Missing required key")

            if not self.bucket_handler.delete_object(key):
                raise Exception("Failed to delete object")

            await msg.respond(
                json.dumps(
                    {
                        "success": True,
                    }
                ).encode()
            )

        except Exception as e:
            logger.error(f"Failed to handle delete known face request: {e}")
            await msg.respond(json.dumps({"success": False, "error": str(e)}).encode())

    async def compare_faces(self, company_id, camera_id, face_id, track_id):
        """Separate method to handle face comparison process"""
        try:
            logger.info("Handling recognition request...")
            # Get list of known faces
            known_faces = self.bucket_handler.get_list_of_objects(
                company_id, "known_faces"
            )

            unknown_face_key = f"{company_id}/unknown_faces/{face_id}.jpg"

            matches = []

            for known_face in known_faces:
                try:
                    logger.info(f"Comparing faces: {known_face} vs {unknown_face_key}")
                    response = self.recognition_client.compare_faces(
                        SourceImage={
                            "S3Object": {
                                "Bucket": self.bucket_handler.bucket_name,
                                "Name": known_face,
                            }
                        },
                        TargetImage={
                            "S3Object": {
                                "Bucket": self.bucket_handler.bucket_name,
                                "Name": unknown_face_key,
                            }
                        },
                        SimilarityThreshold=80.0,
                    )
                    if response["FaceMatches"]:
                        matches.append(
                            {
                                "unknown_face_key": unknown_face_key,
                                "similarity": response["FaceMatches"][0]["Similarity"],
                            }
                        )
                except Exception as e:
                    logger.error(f"Failed to compare faces: {e}")
                    continue

            unknown_face_url = self.bucket_handler.generate_presigned_download_url(
                company_id, unknown_face_key, expires_in=3600
            )

            if not matches:
                # No matches found - send alert
                logger.error("Unknown face detected")
                await self.nats_client.send_message(
                    f"{Commands.ALARM.value}.{company_id}",
                    json.dumps(
                        {
                            "company_id": company_id,
                            "camera_id": camera_id,
                            "unknown_face_url": unknown_face_url,
                            "track_id": track_id,
                            "face_id": face_id,
                        }
                    ).encode(),
                )

            logger.info("Recognition request completed")
            return
        except Exception as e:
            logger.error(f"Failed to handle recognition request: {e}")

    async def execute_rekognition(self, msg):
        """Compare one face against multiple faces"""
        try:
            data = json.loads(msg.data.decode())
            company_id = data.get("company_id")
            camera_id = data.get("camera_id")
            face_id = data.get("face_id")
            track_id = data.get("track_id")

            if not all([company_id, camera_id, face_id]):
                raise Exception("Missing required fields")

            # Create a new task for face comparison
            task = asyncio.create_task(
                self.compare_faces(company_id, camera_id, face_id, track_id)
            )
            self.tasks.append(task)

            # Clean up completed tasks
            self.tasks = [t for t in self.tasks if not t.done()]
            logger.info(f"Active tasks: {len(self.tasks)}")

        except Exception as e:
            logger.error(f"Failed to start recognition request: {e}")

    async def start(self):
        """Start the server"""
        self.running = True
        logger.info("Starting server..")
        await self.nats_client.connect(os.getenv("NATS_URL"))

        await self.setup_subscriptions()

        try:
            while self.running:
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            logger.info("Server stopped")
        finally:
            await self.cleanup()

    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up server resources...")
        self.running = False

        # Cancel all subscription tasks
        for task in self.tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Close NATS connection
        await self.nats_client.close()
        logger.info("Server shutdown complete")

    def stop(self):
        """Stop the server"""
        self.running = False


async def shutdown(sig):
    """Cleanup tasks tied to the service's shutdown."""
    logger.info(f"Received exit signal {sig.name}...")
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    [task.cancel() for task in tasks]
    logger.info(f"Cancelling {len(tasks)} outstanding tasks")
    await asyncio.gather(*tasks, return_exceptions=True)
    asyncio.get_event_loop().stop()


async def main():
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(shutdown(s)))

    server = Server()

    try:
        await server.start()
    except Exception as e:
        logger.error(f"Server error: {e}")
    finally:
        await server.cleanup()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Process interrupted by keyboard")
