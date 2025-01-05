import boto3
import os
import logging
import fastAPI

from botocore.config import Config
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class BucketHandler:
    def __init__(
        self,
        bucket_name: str,
        aws_access_key_id: str = None,
        aws_secret_access_key: str = None,
        region_name: str = None,
    ):
        self.bucket_name = bucket_name
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.region_name = region_name
        self.s3 = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name,
        )

        config = Config(connect_timeout=5, read_timeout=5, retries={"max_attempts": 3})

        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name,
            config=config,
        )

    def create_company_folder(self, company_id: str) -> None:
        """Create a folder for a company in the bucket"""
        self.s3.put_object(Bucket=self.bucket_name, Key=f"{company_id}/")
        self.s3.put_object(Bucket=self.bucket_name, Key=f"{company_id}/known_faces/")
        self.s3.put_object(
            Bucket=self.bucket_name, Key=f"{company_id}/recognition_faces/"
        )

    def generate_presigned_upload_url(
        self, company_id: str, object_key: str, expires_in: int = 3600
    ) -> str:
        """Generate a presigned URL for uploading an object to the bucket"""
        try:
            url = self.s3_client.generate_presigned_url(
                "put_object",
                Params={
                    "Bucket": self.bucket_name,
                    "Key": company_id + "/" + object_key,
                    "ContentType": "image/jpeg",
                },
                Expires_in=expires_in,
            )
            return url

        except Exception as e:
            logger.error(f"Error generating presigned URL: {e}")
            return None

    def generate_presigned_download_url(
        self, company_id: str, object_key: str, expires_in: int = 3600
    ) -> str:
        """Generate a presigned URL for downloading an object from the bucket"""
        try:
            url = self.s3_client.generate_presigned_url(
                "get_object",
                Params={
                    "Bucket": self.bucket_name,
                    "Key": company_id + "/" + object_key,
                },
                Expires_in=expires_in,
            )
            return url

        except Exception as e:
            logger.error(f"Error generating presigned URL: {e}")
            return None

    def generate_presigned_download_all_known_faces_url(
        self, company_id: str, expires_in: int = 3600
    ) -> str:
        """Generate a presigned URL for downloading all known faces from the bucket"""
        try:
            url = self.s3_client.generate_presigned_url(
                "get_object",
                Params={
                    "Bucket": self.bucket_name,
                    "Key": company_id + "/known_faces/",
                },
                Expires_in=expires_in,
            )
            return url

        except Exception as e:
            logger.error(f"Error generating presigned URL: {e}")
            return None

    def generate_presigned_download_all_recognition_faces_url(
        self, company_id: str, expires_in: int = 3600
    ) -> str:
        """Generate a presigned URL for downloading all recognition faces from the bucket"""
        try:
            url = self.s3_client.generate_presigned_url(
                "get_object",
                Params={
                    "Bucket": self.bucket_name,
                    "Key": company_id + "/recognition_faces/",
                },
                Expires_in=expires_in,
            )
            return url

        except Exception as e:
            logger.error(f"Error generating presigned URL: {e}")
            return None

    def delete_face(self, company_id: str, face_id: str) -> None:
        """Delete a face from the bucket"""
        self.s3.delete_object(
            Bucket=self.bucket_name, Key=f"{company_id}/known_faces/{face_id}"
        )
