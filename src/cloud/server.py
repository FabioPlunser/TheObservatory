from bucket_handler import BucketHandler
from typing import Dict
from botocore.exceptions import ClientError
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
    def __init__(
        self,
        bucket_handler: BucketHandler,
        nats_client: NatsClient,
    ):
        self.bucket_handler = bucket_handler
        self.nats_client = nats_client
        self.rekognition_client = boto3.client("rekognition", region_name="us-east-1")
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

            # Get list of known faces
            known_faces = await self.bucket_handler.get_list_of_objects("known_face")

            unknown_face_key = f"{company_id}/unknown_faces/{face_id}.jpg"

            matches = []

            for known_face in known_faces:
                try:
                    response = self.rekognition_client.compare_faces(
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
                        SimilarityThreshold=80,
                    )
                    if response["FaceMatches"]:
                        matches.append(
                            {
                                "unknown_face_key": unknown_face_key,
                                "similarity": response["FaceMatches"][0]["Similarity"],
                            }
                        )
                except self.rekognition.exceptions.InvalidParameterException:
                    logger.warning(f"No face detected in {known_face}")
                except Exception as e:
                    logger.error(f"Failed to compare faces: {e}")
                    continue

            if not matches:
                # No matches found - send alert
                await self.nats_client.send_message(
                    f"{Commands.ALARM.value}.{company_id}",
                    json.dumps(
                        {
                            "type": "unknown_face",
                            "company_id": company_id,
                            "camera_id": camera_id,
                            "face_id": face_id,
                            "track_id": track_id,
                        }
                    ).encode(),
                )

            await msg.respond(
                json.dumps({"success": True, "matches": matches}).encode()
            )

        except Exception as e:
            logger.error(f"Failed to handle recognition request: {e}")
            await msg.respond(json.dumps({"success": False, "error": str(e)}).encode())

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

    nats_client = NatsClient()
    bucket_handler = BucketHandler(
        os.getenv("BUCKET_NAME"),
        os.getenv("REGION"),
    )

    server = Server(bucket_handler, nats_client)

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
