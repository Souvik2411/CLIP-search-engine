#!/usr/bin/env python3
"""
Upload images from local directory to S3 bucket.

Usage:
    python scripts/upload_images_to_s3.py --local-dir /path/to/images
    python scripts/upload_images_to_s3.py --local-dir /path/to/images --s3-prefix images/batch1/
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List
import boto3
from botocore.exceptions import ClientError
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_settings

# Supported image formats
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.webp', '.JPG', '.JPEG', '.PNG', '.WEBP'}


def get_image_files(directory: Path) -> List[Path]:
    """
    Get all image files from directory.

    Args:
        directory: Path to local directory

    Returns:
        List of image file paths
    """
    image_files = []
    for file_path in directory.rglob('*'):
        if file_path.is_file() and file_path.suffix in SUPPORTED_FORMATS:
            image_files.append(file_path)

    return sorted(image_files)


def upload_to_s3(
    local_dir: Path,
    bucket_name: str,
    s3_prefix: str = "images/",
    aws_access_key: str = None,
    aws_secret_key: str = None,
    region: str = None
):
    """
    Upload images from local directory to S3.

    Args:
        local_dir: Local directory containing images
        bucket_name: S3 bucket name
        s3_prefix: S3 prefix/folder (default: images/)
        aws_access_key: AWS access key (optional, uses .env if not provided)
        aws_secret_key: AWS secret key (optional, uses .env if not provided)
        region: AWS region (optional, uses .env if not provided)
    """
    # Get image files
    print(f"Scanning directory: {local_dir}")
    image_files = get_image_files(local_dir)

    if not image_files:
        print("[ERROR] No image files found in the specified directory.")
        print(f"   Supported formats: {', '.join(SUPPORTED_FORMATS)}")
        return

    print(f"[OK] Found {len(image_files)} images to upload")

    # Initialize S3 client
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=region
        )
    except Exception as e:
        print(f"[ERROR] Failed to initialize S3 client: {e}")
        return

    # Ensure s3_prefix ends with /
    if s3_prefix and not s3_prefix.endswith('/'):
        s3_prefix += '/'

    # Upload files
    uploaded = 0
    failed = 0
    failed_files = []

    print(f"\nUploading to s3://{bucket_name}/{s3_prefix}")
    print("-" * 60)

    for image_path in tqdm(image_files, desc="Uploading"):
        try:
            # Generate S3 key
            # Use relative path from local_dir to preserve folder structure
            relative_path = image_path.relative_to(local_dir)
            s3_key = f"{s3_prefix}{relative_path.as_posix()}"

            # Upload file
            s3_client.upload_file(
                str(image_path),
                bucket_name,
                s3_key,
                ExtraArgs={'ContentType': 'image/jpeg'}  # Set appropriate content type
            )

            uploaded += 1

        except ClientError as e:
            failed += 1
            failed_files.append((str(image_path), str(e)))
            tqdm.write(f"[FAILED] {image_path.name} - {e}")
        except Exception as e:
            failed += 1
            failed_files.append((str(image_path), str(e)))
            tqdm.write(f"[ERROR] {image_path.name} - {e}")

    # Summary
    print("\n" + "=" * 60)
    print("Upload Summary:")
    print("=" * 60)
    print(f"[OK] Successfully uploaded: {uploaded}")
    print(f"[FAILED] Failed: {failed}")
    print(f"[TOTAL] Total processed: {len(image_files)}")

    if failed_files:
        print("\nFailed files:")
        for file_path, error in failed_files[:10]:  # Show first 10 failures
            print(f"  - {Path(file_path).name}: {error}")
        if len(failed_files) > 10:
            print(f"  ... and {len(failed_files) - 10} more")

    print("\n" + "=" * 60)
    print("Next steps:")
    print("=" * 60)
    print("1. Index the uploaded images:")
    print(f"   python scripts/index_images.py --prefix \"{s3_prefix}\"")
    print("\n2. Or reset index and re-index all:")
    print(f"   python scripts/index_images.py --reset-index --prefix \"{s3_prefix.split('/')[0]}/\"")


def main():
    parser = argparse.ArgumentParser(
        description="Upload images from local directory to S3 bucket"
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        required=True,
        help="Local directory containing images"
    )
    parser.add_argument(
        "--s3-prefix",
        type=str,
        default="images/",
        help="S3 prefix/folder (default: images/)"
    )
    parser.add_argument(
        "--bucket-name",
        type=str,
        default=None,
        help="S3 bucket name (default: from .env)"
    )
    parser.add_argument(
        "--aws-access-key",
        type=str,
        default=None,
        help="AWS Access Key ID (default: from .env)"
    )
    parser.add_argument(
        "--aws-secret-key",
        type=str,
        default=None,
        help="AWS Secret Access Key (default: from .env)"
    )
    parser.add_argument(
        "--region",
        type=str,
        default=None,
        help="AWS Region (default: from .env)"
    )

    args = parser.parse_args()

    # Get settings
    settings = get_settings()

    # Validate local directory
    local_dir = Path(args.local_dir)
    if not local_dir.exists():
        print(f"[ERROR] Directory does not exist: {local_dir}")
        sys.exit(1)

    if not local_dir.is_dir():
        print(f"[ERROR] Not a directory: {local_dir}")
        sys.exit(1)

    # Get S3 configuration
    bucket_name = args.bucket_name or settings.S3_BUCKET_NAME
    region = args.region or settings.AWS_REGION
    aws_access_key = args.aws_access_key or settings.AWS_ACCESS_KEY_ID
    aws_secret_key = args.aws_secret_key or settings.AWS_SECRET_ACCESS_KEY

    if not bucket_name:
        print("[ERROR] S3 bucket name not provided and not found in .env")
        sys.exit(1)

    if not aws_access_key or not aws_secret_key:
        print("[ERROR] AWS credentials not provided and not found in .env")
        sys.exit(1)

    # Upload
    upload_to_s3(
        local_dir=local_dir,
        bucket_name=bucket_name,
        s3_prefix=args.s3_prefix,
        aws_access_key=aws_access_key,
        aws_secret_key=aws_secret_key,
        region=region
    )


if __name__ == "__main__":
    main()
