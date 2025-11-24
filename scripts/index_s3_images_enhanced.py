#!/usr/bin/env python3
"""
Index all images from S3 with enhanced metadata pipeline.

This script:
1. Lists all images from S3 bucket with specified prefix
2. Downloads each image
3. Processes through enhanced pipeline (CLIP, color, texture, material, style, LLM)
4. Saves to FAISS + SQLite database

Usage:
    python scripts/index_s3_images_enhanced.py --prefix pinterest-interior-design/ --generate-summaries
"""

import argparse
import sys
import time
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.clip_service import CLIPService
from app.services.faiss_service import FAISSService
from app.services.s3_service import s3_service
from app.services.indexing_service import IndexingService
from app.services.metadata_db_service import metadata_db_service
from app.config import get_settings


def main():
    parser = argparse.ArgumentParser(
        description="Index S3 images with enhanced metadata pipeline"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        required=True,
        help="S3 prefix for images (e.g., pinterest-interior-design/)"
    )
    parser.add_argument(
        "--generate-summaries",
        action="store_true",
        help="Generate LLM summaries (costs ~$0.00015 per image)"
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=0,
        help="Start from this index (useful for resuming)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of images to process (for testing)"
    )

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("  ENHANCED S3 IMAGE INDEXING PIPELINE")
    print("=" * 80)

    # List images from S3
    print(f"\n[STATUS] Listing images from S3 with prefix: {args.prefix}")
    try:
        all_s3_keys = s3_service.list_images(prefix=args.prefix, max_keys=10000)
        print(f"[OK] Found {len(all_s3_keys)} images in S3")
    except Exception as e:
        print(f"[ERROR] Failed to list S3 images: {e}")
        sys.exit(1)

    # Check which images are already indexed
    print(f"\n[STATUS] Checking for already-indexed images...")
    indexed_image_ids = set()
    try:
        import sqlite3
        db_path = Path(__file__).parent.parent / "data" / "index" / "metadata.db"
        if db_path.exists():
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT image_id FROM images")
            indexed_image_ids = {row[0] for row in cursor.fetchall()}
            conn.close()
            print(f"[OK] Found {len(indexed_image_ids)} already-indexed images")
    except Exception as e:
        print(f"[WARNING] Could not check indexed images: {e}")
        print(f"[INFO] Will process all images")

    # Filter out already-indexed images
    s3_keys = []
    for s3_key in all_s3_keys:
        image_id = s3_key.split('/')[-1].rsplit('.', 1)[0]
        if image_id not in indexed_image_ids:
            s3_keys.append(s3_key)

    print(f"[OK] {len(s3_keys)} new images to index")

    if len(s3_keys) == 0:
        print(f"[INFO] All images are already indexed!")
        sys.exit(0)

    if not s3_keys:
        print("[ERROR] No images found with the specified prefix")
        sys.exit(1)

    # Apply start_from and limit
    if args.start_from > 0:
        s3_keys = s3_keys[args.start_from:]
        print(f"[INFO] Starting from index {args.start_from}")

    if args.limit:
        s3_keys = s3_keys[:args.limit]
        print(f"[INFO] Limited to {args.limit} images")

    total_images = len(s3_keys)

    # Initialize services
    print("\n[STATUS] Initializing services...")
    print("  - Loading CLIP model...")
    clip_service = CLIPService()
    clip_service.load_model()

    print("  - Loading FAISS index...")
    faiss_service = FAISSService()
    faiss_service.load_index()

    print("  - Initializing SQLite database...")
    metadata_db_service._init_database()

    print("  - Creating indexing service...")
    indexing_service = IndexingService(clip_service, faiss_service)

    print("\n[OK] Services initialized successfully!")

    # Confirm with user
    print("\n" + "=" * 80)
    print(f"[PLAN] INDEXING PLAN:")
    print(f"  - Total images: {total_images}")
    print(f"  - Generate LLM summaries: {'Yes' if args.generate_summaries else 'No'}")
    print(f"  - Estimated time: {total_images * 4 / 60:.1f} minutes")
    if args.generate_summaries:
        print(f"  - Estimated cost: ${total_images * 0.00015:.2f}")
    print("=" * 80)

    response = input("\n[PROMPT] Ready to start indexing? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("[CANCELLED] Indexing cancelled.")
        return

    # Process images
    print("\n" + "=" * 80)
    print("[STATUS] STARTING INDEXING...")
    print("=" * 80 + "\n")

    results = []
    successful = 0
    failed = 0
    total_time = 0

    start_time = time.time()

    for i, s3_key in enumerate(tqdm(s3_keys, desc="Processing images"), 1):
        # Generate image ID from S3 key
        image_id = s3_key.split('/')[-1].rsplit('.', 1)[0]

        print(f"\n[{i}/{total_images}] Processing: {image_id}")

        try:
            # Download image from S3
            print(f"  [STATUS] Downloading from S3...")
            image = s3_service.download_image(s3_key)

            # Process through enhanced pipeline
            print(f"  [STATUS] Processing with enhanced pipeline...")
            metadata = indexing_service.process_image(
                image=image,
                image_id=image_id,
                s3_key=s3_key,
                generate_summary=args.generate_summaries
            )

            successful += 1
            proc_time = metadata["processing_time"]
            total_time += proc_time
            avg_time = total_time / successful

            print(f"  [OK] Success in {proc_time:.2f}s")
            print(f"  [TIMING] CLIP={metadata['timing']['clip']:.2f}s, "
                  f"Color={metadata['timing']['color']:.2f}s, "
                  f"Material={metadata['timing']['material']:.2f}s, "
                  f"Style={metadata['timing']['style']:.2f}s")
            if args.generate_summaries:
                print(f"  [TIMING] LLM={metadata['timing']['llm']:.2f}s")
            print(f"  [PROGRESS] Average: {avg_time:.2f}s/image | "
                  f"ETA: {(total_images - i) * avg_time / 60:.1f} min")

            results.append({
                "success": True,
                "image_id": image_id,
                "processing_time": proc_time
            })

        except Exception as e:
            failed += 1
            print(f"  [ERROR] Failed: {e}")
            results.append({
                "success": False,
                "image_id": image_id,
                "error": str(e)
            })

    # Final statistics
    elapsed = time.time() - start_time

    print("\n" + "=" * 80)
    print("  INDEXING COMPLETE!")
    print("=" * 80)
    print(f"\n[STATS] STATISTICS:")
    print(f"  - Total images: {total_images}")
    print(f"  - Successfully processed: {successful}")
    print(f"  - Failed: {failed}")
    print(f"  - Success rate: {successful/total_images*100:.1f}%")
    print(f"  - Total time: {elapsed/60:.1f} minutes")
    print(f"  - Average time: {total_time/successful:.2f}s per image")

    # Check database
    print(f"\n[STATS] DATABASE STATUS:")
    stats = metadata_db_service.get_stats()
    print(f"  - Images in database: {stats['total_images']}")
    print(f"  - FAISS index size: {faiss_service.size}")

    if stats['top_styles']:
        print(f"\n[STATS] TOP STYLES:")
        for style_info in stats['top_styles'][:5]:
            print(f"  - {style_info['style']}: {style_info['count']} images")

    if stats['top_scenes']:
        print(f"\n[STATS] TOP SCENES:")
        for scene_info in stats['top_scenes'][:5]:
            print(f"  - {scene_info['scene']}: {scene_info['count']} images")

    # Failed images
    if failed > 0:
        print(f"\n[ERROR] FAILED IMAGES:")
        for result in results:
            if not result["success"]:
                print(f"  - {result['image_id']}: {result['error']}")

    print("\n[COMPLETE] All done! Your images now have enhanced metadata.")
    print("\n[INFO] Next steps:")
    print("  1. Start the API: uvicorn app.main:app --reload")
    print("  2. Start Streamlit: streamlit run streamlit_app_dual.py")
    print("  3. Try searching with filters")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
