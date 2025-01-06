import boto3
import os
import logging

from botocore.config import Config
from datetime import datetime, timedelta
from typing import List

logger = logging.getLogger(__name__)


class BucketHandler:
    def __init__(
        self,
        bucket_name: str,
        region_name: str = None,
    ):
        self.bucket_name = bucket_name
        self.region_name = region_name
        self.s3 = boto3.client(
            "s3",
            region_name=region_name,
        )

        config = Config(connect_timeout=5, read_timeout=5, retries={"max_attempts": 3})

        self.s3_client = boto3.client(
            "s3",
            region_name=region_name,
            config=config,
        )

    def create_company_folder(self, company_id: str) -> None:
        """Create a folder for a company in the bucket"""
        self.s3.put_object(Bucket=self.bucket_name, Key=f"{company_id}/")
        self.s3.put_object(Bucket=self.bucket_name, Key=f"{company_id}/known_faces/")
        self.s3.put_object(Bucket=self.bucket_name, Key=f"{company_id}/unknown_faces/")

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
                    "ContentType": "*",
                },
                ExpiresIn=expires_in,
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
                ExpiresIn=expires_in,
            )
            return url

        except Exception as e:
            logger.error(f"Error generating presigned URL: {e}")
            return None

    def generate_list_presigned_url_of_key(
        self, company_id: str, folder_name: str, expires_in: int = 3600
    ) -> List[str]:
        """Genereate a list of presigned ULRs for all items in a folder"""
        try:
            if not folder_name.endswith("/"):
                folder_name += "/"

            prefix = f"{company_id}/{folder_name}"
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix,
            )

            if "Contents" not in response:
                logger.error(f"No objects found in folder: {response}")
                logger.error(f"No objects found in folder: {prefix}")
                return []

            urls = []
            for obj in response["Contents"]:
                key = obj["Key"]
                if key == f"{company_id}/{folder_name}":
                    continue
                if key == folder_name:
                    continue
                url = self.s3_client.generate_presigned_url(
                    "get_object",
                    Params={
                        "Bucket": self.bucket_name,
                        "Key": key,
                    },
                    ExpiresIn=expires_in,
                )
                urls.append(
                    {
                        "url": url,
                        "key": key,
                    }
                )
            return urls
        except Exception as e:
            logger.error(f"Error generating presigned URL: {e}")
            return None

    def delete_object(self, s3_key: str) -> None:
        """Delete a face from the bucket"""
        try:
            self.s3.delete_object(Bucket=self.bucket_name, Key=f"{s3_key}")
        except Exception as e:
            logger.error(f"Error deleting object: {e}")
            return False
        
    def get_list_of_objects(self, company_id: str, folder_name: str) -> List[str]:
        """Get a list of objects in a folder"""
        try:
            if not folder_name.endswith("/"):
                folder_name += "/"

            prefix = f"{company_id}/{folder_name}"
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix,
            )

            if "Contents" not in response:
                logger.error(f"No objects found in folder: {response}")
                logger.error(f"No objects found in folder: {prefix}")
                return []

            objects = []
            for obj in response["Contents"]:
                key = obj["Key"]
                if key == f"{company_id}/{folder_name}":
                    continue
                if key == folder_name:
                    continue
                objects.append(key)
            return objects
        except Exception as e:
            logger.error(f"Error getting list of objects: {e}")
            return None
