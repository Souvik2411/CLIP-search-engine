#!/usr/bin/env python3
"""
Script to index images from S3 into FAISS.

Usage:
    python scripts/index_images.py --prefix "images/" --batch-size 32
    python scripts/index_images.py --s3-keys key1.jpg key2.jpg
"""

import argparse
import sys
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services import CLIPService, FAISSService, S3Service
from app.config import get_settings
from app.utils.helpers import batch_list


def index_from_s3_prefix(
    s3_service: S3Service,
    clip_service: CLIPService,
    faiss_service: FAISSService,
    prefix: str,
    batch_size: int = 32,
    max_images: int = None
):
    """Index all images from an S3 prefix."""
    print(f"Listing images from S3 with prefix: {prefix}")
    s3_keys = s3_service.list_images(prefix=prefix, max_keys=max_images or 10000)

    if not s3_keys:
        print("No images found!")
        return 0, 0

    print(f"Found {len(s3_keys)} images to index")

    if max_images:
        s3_keys = s3_keys[:max_images]
        print(f"Limited to {len(s3_keys)} images")

    return index_images(s3_service, clip_service, faiss_service, s3_keys, batch_size)


def index_images(
    s3_service: S3Service,
    clip_service: CLIPService,
    faiss_service: FAISSService,
    s3_keys: list[str],
    batch_size: int = 32
):
    """Index a list of S3 keys."""
    indexed_count = 0
    failed_count = 0
    failed_keys = []

    # Process in batches
    batches = list(batch_list(s3_keys, batch_size))

    for batch in tqdm(batches, desc="Processing batches"):
        batch_embeddings = []
        batch_metadata = []

        for s3_key in tqdm(batch, desc="Processing images", leave=False):
            try:
                # Download image
                pil_image = s3_service.download_image(s3_key)

                # Get embedding, architectural labels, and object labels
                embedding, arch_results, object_results = clip_service.get_image_embedding_and_dual_labels(pil_image)
                labels = [label for label, _ in arch_results]
                objects = [obj for obj, _ in object_results]

                # Create metadata
                image_id = s3_key.split('/')[-1].rsplit('.', 1)[0]
                metadata = {
                    'image_id': image_id,
                    's3_key': s3_key,
                    'labels': labels,
                    'objects': objects
                }

                batch_embeddings.append(embedding)
                batch_metadata.append(metadata)
                indexed_count += 1

            except Exception as e:
                print(f"\nFailed to process {s3_key}: {e}")
                failed_count += 1
                failed_keys.append(s3_key)

        # Add batch to index
        if batch_embeddings:
            embeddings_array = np.array(batch_embeddings)
            faiss_service.add_embeddings(embeddings_array, batch_metadata)

    # Save index
    print("\nSaving index...")
    faiss_service.save_index()

    return indexed_count, failed_count


def main():
    parser = argparse.ArgumentParser(description="Index images from S3 into FAISS")
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="S3 prefix to list images from"
    )
    parser.add_argument(
        "--s3-keys",
        nargs="+",
        help="Specific S3 keys to index"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum number of images to index"
    )
    parser.add_argument(
        "--reset-index",
        action="store_true",
        help="Reset the index before indexing"
    )

    args = parser.parse_args()

    # Initialize services
    print("Initializing services...")
    settings = get_settings()

    clip_service = CLIPService()
    faiss_service = FAISSService()
    s3_service = S3Service()

    # Load CLIP model
    print("Loading CLIP model...")
    clip_service.load_model()

    # Load or create index
    if args.reset_index:
        print("Creating new index...")
        faiss_service.create_index(settings.EMBEDDING_DIM)
    else:
        print("Loading existing index...")
        faiss_service.load_index()

    print(f"Current index size: {faiss_service.size}")

    # Index images
    if args.s3_keys:
        indexed, failed = index_images(
            s3_service, clip_service, faiss_service,
            args.s3_keys, args.batch_size
        )
    else:
        indexed, failed = index_from_s3_prefix(
            s3_service, clip_service, faiss_service,
            args.prefix, args.batch_size, args.max_images
        )

    # Print results
    print("\n" + "=" * 50)
    print(f"Indexing complete!")
    print(f"Successfully indexed: {indexed}")
    print(f"Failed: {failed}")
    print(f"Total index size: {faiss_service.size}")
    print("=" * 50)


if __name__ == "__main__":
    main()