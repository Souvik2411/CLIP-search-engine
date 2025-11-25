#!/usr/bin/env python3
"""
Script to reindex all existing images with the enhanced metadata pipeline.

This script:
1. Reads existing metadata.json to get S3 keys
2. Downloads each image from S3
3. Processes through enhanced pipeline (color, texture, material, style)
4. Saves to FAISS + SQLite database

Usage:
    python scripts/reindex_with_enhanced_pipeline.py --generate-summaries
    python scripts/reindex_with_enhanced_pipeline.py --no-summaries --batch-size 10
"""

import argparse
import sys
import json
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


def load_existing_metadata(metadata_path: str):
    """Load existing metadata.json to get list of images."""
    print(f"\n📂 Loading existing metadata from: {metadata_path}")

    with open(metadata_path, 'r') as f:
        metadata_list = json.load(f)

    print(f"✅ Found {len(metadata_list)} images to reindex")

    return metadata_list


def reindex_image(indexing_service, s3_key: str, image_id: str, generate_summary: bool):
    """
    Reindex a single image with enhanced pipeline.

    Args:
        indexing_service: IndexingService instance
        s3_key: S3 key for the image
        image_id: Image ID
        generate_summary: Whether to generate LLM summary

    Returns:
        Dictionary with processing info or None if failed
    """
    try:
        # Download image from S3
        print(f"  📥 Downloading {s3_key}...")
        image = s3_service.download_image(s3_key)

        # Process through enhanced pipeline
        print(f"  ⚙️  Processing with enhanced pipeline...")
        metadata = indexing_service.process_image(
            image=image,
            image_id=image_id,
            s3_key=s3_key,
            generate_summary=generate_summary
        )

        return {
            "success": True,
            "image_id": image_id,
            "processing_time": metadata["processing_time"],
            "timing": metadata["timing"]
        }

    except Exception as e:
        print(f"  ❌ Error processing {image_id}: {e}")
        return {
            "success": False,
            "image_id": image_id,
            "error": str(e)
        }


def main():
    parser = argparse.ArgumentParser(
        description="Reindex all images with enhanced metadata pipeline"
    )
    parser.add_argument(
        "--metadata-path",
        type=str,
        default="data/index/metadata.json",
        help="Path to existing metadata.json"
    )
    parser.add_argument(
        "--generate-summaries",
        action="store_true",
        help="Generate LLM summaries (costs ~$0.00015 per image)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of images to process before saving progress"
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
    print("  ENHANCED METADATA REINDEXING PIPELINE")
    print("=" * 80)

    # Load existing metadata
    existing_metadata = load_existing_metadata(args.metadata_path)

    # Apply start_from and limit
    if args.start_from > 0:
        existing_metadata = existing_metadata[args.start_from:]
        print(f"⏭️  Starting from index {args.start_from}")

    if args.limit:
        existing_metadata = existing_metadata[:args.limit]
        print(f"🔢 Limited to {args.limit} images")

    total_images = len(existing_metadata)

    # Initialize services
    print("\n⚙️  Initializing services...")
    print("  • Loading CLIP model...")
    clip_service = CLIPService()
    clip_service.load_model()

    print("  • Loading FAISS index...")
    faiss_service = FAISSService()
    faiss_service.load_index()

    print("  • Initializing SQLite database...")
    metadata_db_service._init_database()

    print("  • Creating indexing service...")
    indexing_service = IndexingService(clip_service, faiss_service)

    print("\n✅ Services initialized successfully!")

    # Confirm with user
    print("\n" + "=" * 80)
    print(f"📊 REINDEXING PLAN:")
    print(f"  • Total images: {total_images}")
    print(f"  • Generate LLM summaries: {'Yes' if args.generate_summaries else 'No'}")
    print(f"  • Estimated time: {total_images * 3.5 / 60:.1f} minutes")
    if args.generate_summaries:
        print(f"  • Estimated cost: ${total_images * 0.00015:.2f}")
    print("=" * 80)

    response = input("\n🚀 Ready to start reindexing? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("❌ Reindexing cancelled.")
        return

    # Process images
    print("\n" + "=" * 80)
    print("🚀 STARTING REINDEXING...")
    print("=" * 80 + "\n")

    results = []
    successful = 0
    failed = 0
    total_time = 0

    start_time = time.time()

    for i, item in enumerate(tqdm(existing_metadata, desc="Processing images"), 1):
        image_id = item["image_id"]
        s3_key = item["s3_key"]

        print(f"\n[{i}/{total_images}] Processing: {image_id}")

        result = reindex_image(
            indexing_service=indexing_service,
            s3_key=s3_key,
            image_id=image_id,
            generate_summary=args.generate_summaries
        )

        results.append(result)

        if result["success"]:
            successful += 1
            proc_time = result["processing_time"]
            total_time += proc_time
            avg_time = total_time / successful

            print(f"  ✅ Success in {proc_time:.2f}s")
            print(f"  📊 Timing: CLIP={result['timing']['clip']:.2f}s, "
                  f"Color={result['timing']['color']:.2f}s, "
                  f"Material={result['timing']['material']:.2f}s, "
                  f"Style={result['timing']['style']:.2f}s")
            print(f"  📈 Average: {avg_time:.2f}s/image | "
                  f"ETA: {(total_images - i) * avg_time / 60:.1f} min")
        else:
            failed += 1
            print(f"  ❌ Failed: {result['error']}")

    # Final statistics
    elapsed = time.time() - start_time

    print("\n" + "=" * 80)
    print("  REINDEXING COMPLETE!")
    print("=" * 80)
    print(f"\n📊 STATISTICS:")
    print(f"  • Total images: {total_images}")
    print(f"  • Successfully processed: {successful}")
    print(f"  • Failed: {failed}")
    print(f"  • Success rate: {successful/total_images*100:.1f}%")
    print(f"  • Total time: {elapsed/60:.1f} minutes")
    print(f"  • Average time: {total_time/successful:.2f}s per image")

    # Check database
    print(f"\n📁 DATABASE STATUS:")
    stats = metadata_db_service.get_stats()
    print(f"  • Images in database: {stats['total_images']}")
    print(f"  • FAISS index size: {faiss_service.size}")

    if stats['top_styles']:
        print(f"\n🎨 TOP STYLES:")
        for style_info in stats['top_styles'][:5]:
            print(f"  • {style_info['style']}: {style_info['count']} images")

    if stats['top_scenes']:
        print(f"\n🏠 TOP SCENES:")
        for scene_info in stats['top_scenes'][:5]:
            print(f"  • {scene_info['scene']}: {scene_info['count']} images")

    # Failed images
    if failed > 0:
        print(f"\n❌ FAILED IMAGES:")
        for result in results:
            if not result["success"]:
                print(f"  • {result['image_id']}: {result['error']}")

    print("\n✅ All done! Your images now have enhanced metadata.")
    print("\n💡 Next steps:")
    print("  1. Start the API: uvicorn app.main:app --reload")
    print("  2. Start Streamlit: streamlit run streamlit_app_dual.py")
    print("  3. Try searching with filters like 'warm modern living room'")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()