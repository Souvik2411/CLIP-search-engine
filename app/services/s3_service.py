import boto3
from botocore.exceptions import ClientError
from PIL import Image
from io import BytesIO
from typing import Optional
import logging

from app.config import get_settings

logger = logging.getLogger(__name__)


class S3Service:
    """Service for S3 operations - image retrieval and URL generation."""

    def __init__(self):
        self.settings = get_settings()
        self._client: Optional[boto3.client] = None

    @property
    def client(self):
        """Lazy initialization of S3 client."""
        if self._client is None:
            self._client = boto3.client(
                's3',
                aws_access_key_id=self.settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=self.settings.AWS_SECRET_ACCESS_KEY,
                region_name=self.settings.AWS_REGION
            )
        return self._client

    def get_presigned_url(
        self,
        s3_key: str,
        expiration: Optional[int] = None
    ) -> str:
        """
        Generate a presigned URL for an S3 object.

        Args:
            s3_key: S3 object key
            expiration: URL expiration time in seconds

        Returns:
            Presigned URL string
        """
        if expiration is None:
            expiration = self.settings.S3_URL_EXPIRATION

        try:
            url = self.client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': self.settings.S3_BUCKET_NAME,
                    'Key': s3_key
                },
                ExpiresIn=expiration
            )
            return url
        except ClientError as e:
            logger.error(f"Error generating presigned URL for {s3_key}: {e}")
            raise

    def get_presigned_urls(
        self,
        s3_keys: list[str],
        expiration: Optional[int] = None
    ) -> dict[str, str]:
        """
        Generate presigned URLs for multiple S3 objects.

        Args:
            s3_keys: List of S3 object keys
            expiration: URL expiration time in seconds

        Returns:
            Dict mapping s3_key to presigned URL
        """
        urls = {}
        for key in s3_keys:
            try:
                urls[key] = self.get_presigned_url(key, expiration)
            except Exception as e:
                logger.error(f"Failed to generate URL for {key}: {e}")
                urls[key] = ""
        return urls

    async def get_presigned_urls_async(
        self,
        s3_keys: list[str],
        expiration: Optional[int] = None
    ) -> dict[str, str]:
        """
        Generate presigned URLs for multiple S3 objects in parallel.
        This provides faster performance for large result sets.

        Args:
            s3_keys: List of S3 object keys
            expiration: URL expiration time in seconds

        Returns:
            Dict mapping s3_key to presigned URL
        """
        import asyncio

        async def get_url_async(key: str) -> tuple[str, str]:
            """Helper to get URL in async context."""
            loop = asyncio.get_event_loop()
            try:
                url = await loop.run_in_executor(
                    None,
                    lambda: self.get_presigned_url(key, expiration)
                )
                return (key, url)
            except Exception as e:
                logger.error(f"Failed to generate URL for {key}: {e}")
                return (key, "")

        # Run all URL generations in parallel
        results = await asyncio.gather(*[get_url_async(key) for key in s3_keys])
        return dict(results)

    def download_image(self, s3_key: str) -> Image.Image:
        """
        Download an image from S3 and return as PIL Image.

        Args:
            s3_key: S3 object key

        Returns:
            PIL Image object
        """
        try:
            response = self.client.get_object(
                Bucket=self.settings.S3_BUCKET_NAME,
                Key=s3_key
            )
            image_bytes = response['Body'].read()
            return Image.open(BytesIO(image_bytes)).convert("RGB")
        except ClientError as e:
            logger.error(f"Error downloading image {s3_key}: {e}")
            raise

    def upload_image(
        self,
        image: Image.Image,
        s3_key: str,
        format: str = "JPEG"
    ) -> str:
        """
        Upload a PIL Image to S3.

        Args:
            image: PIL Image object
            s3_key: S3 object key
            format: Image format

        Returns:
            S3 key of uploaded image
        """
        buffer = BytesIO()
        image.save(buffer, format=format)
        buffer.seek(0)

        content_type = f"image/{format.lower()}"

        try:
            self.client.put_object(
                Bucket=self.settings.S3_BUCKET_NAME,
                Key=s3_key,
                Body=buffer,
                ContentType=content_type
            )
            logger.info(f"Uploaded image to {s3_key}")
            return s3_key
        except ClientError as e:
            logger.error(f"Error uploading image to {s3_key}: {e}")
            raise

    def list_images(
        self,
        prefix: str = "",
        max_keys: int = 1000
    ) -> list[str]:
        """
        List image keys in S3 bucket with pagination support.

        Args:
            prefix: S3 key prefix to filter
            max_keys: Maximum number of keys to return (None for all)

        Returns:
            List of S3 keys
        """
        try:
            keys = []
            continuation_token = None

            # Paginate through all results if max_keys is None or > 1000
            while True:
                # Build request parameters
                params = {
                    'Bucket': self.settings.S3_BUCKET_NAME,
                    'Prefix': prefix,
                    'MaxKeys': min(1000, max_keys) if max_keys else 1000
                }

                if continuation_token:
                    params['ContinuationToken'] = continuation_token

                response = self.client.list_objects_v2(**params)

                # Collect image keys
                for obj in response.get('Contents', []):
                    key = obj['Key']
                    # Filter for common image extensions
                    if key.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.gif')):
                        keys.append(key)

                        # Stop if we've reached max_keys
                        if max_keys and len(keys) >= max_keys:
                            return keys

                # Check if there are more results
                if response.get('IsTruncated'):
                    continuation_token = response.get('NextContinuationToken')
                else:
                    break

            return keys
        except ClientError as e:
            logger.error(f"Error listing images: {e}")
            raise

    def check_object_exists(self, s3_key: str) -> bool:
        """
        Check if an object exists in S3.

        Args:
            s3_key: S3 object key

        Returns:
            True if object exists
        """
        try:
            self.client.head_object(
                Bucket=self.settings.S3_BUCKET_NAME,
                Key=s3_key
            )
            return True
        except ClientError:
            return False


# Create singleton instance
s3_service = S3Service()