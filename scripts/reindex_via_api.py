#!/usr/bin/env python3
"""
Script to reindex all images via the FastAPI API endpoints.
This script makes HTTP requests to the running API server.

Prerequisites:
    - FastAPI server must be running: uvicorn app.main:app --reload

Usage:
    python scripts/reindex_via_api.py --generate-summaries
    python scripts/reindex_via_api.py --no-summaries --batch-size 10
"""

import argparse
import json
import time
import requests
from pathlib import Path


API_BASE_URL = "http://localhost:8000/api/v1"


def check_api_health():
    """Check if the API server is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def load_existing_metadata(metadata_path: str):
    """Load existing metadata.json to get list of images."""
    print(f"\n📂 Loading existing metadata from: {metadata_path}")

    with open(metadata_path, 'r') as f:
        metadata_list = json.load(f)

    print(f"✅ Found {len(metadata_list)} images to reindex")

    return metadata_list


def reindex_image_via_api(s3_key: str, generate_summary: bool):
    """
    Reindex a single image via API.

    Args:
        s3_key: S3 key for the image
        generate_summary: Whether to generate LLM summary

    Returns:
        Response dictionary or None if failed
    """
    try:
        # Call the batch indexing endpoint with single S3 key
        response = requests.post(
            f"{API_BASE_URL}/index/enhanced/batch",
            json={
                "s3_keys": [s3_key],
                "generate_summaries": generate_summary
            },
            timeout=60
        )

        if response.status_code == 200:
            result = response.json()
            return {
                "success": True,
                "data": result
            }
        else:
            return {
                "success": False,
                "error": f"API returned {response.status_code}: {response.text}"
            }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def main():
    parser = argparse.ArgumentParser(
        description="Reindex all images via FastAPI endpoints"
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
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8000",
        help="FastAPI server URL"
    )

    args = parser.parse_args()

    # Update API base URL
    global API_BASE_URL
    API_BASE_URL = f"{args.api_url}/api/v1"

    print("\n" + "=" * 80)
    print("  ENHANCED METADATA REINDEXING VIA API")
    print("=" * 80)

    # Check API server
    print("\n🔍 Checking API server...")
    if not check_api_health():
        print("❌ API server is not running!")
        print("\nPlease start the server first:")
        print("  uvicorn app.main:app --reload --port 8000")
        return

    print("✅ API server is running")

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

    # Confirm with user
    print("\n" + "=" * 80)
    print(f"📊 REINDEXING PLAN:")
    print(f"  • Total images: {total_images}")
    print(f"  • Generate LLM summaries: {'Yes' if args.generate_summaries else 'No'}")
    print(f"  • Estimated time: {total_images * 5 / 60:.1f} minutes")
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

    successful = 0
    failed = 0
    failed_images = []

    start_time = time.time()

    for i, item in enumerate(existing_metadata, 1):
        image_id = item["image_id"]
        s3_key = item["s3_key"]

        print(f"[{i}/{total_images}] Processing: {image_id}")
        print(f"  S3 Key: {s3_key}")

        image_start = time.time()

        result = reindex_image_via_api(
            s3_key=s3_key,
            generate_summary=args.generate_summaries
        )

        image_time = time.time() - image_start

        if result["success"]:
            successful += 1
            print(f"  ✅ Success in {image_time:.2f}s")

            # Show some metadata if available
            if "data" in result and "results" in result["data"]:
                for res in result["data"]["results"]:
                    if "metadata" in res:
                        meta = res["metadata"]
                        print(f"  📊 Style: {meta.get('primary_style', 'N/A')}")
                        print(f"  🏠 Scene: {meta.get('primary_scene', 'N/A')}")
                        print(f"  🎨 Color: {meta.get('color', {}).get('dominant_color', 'N/A')}")
        else:
            failed += 1
            failed_images.append((image_id, result["error"]))
            print(f"  ❌ Failed: {result['error']}")

        # Progress update
        if successful > 0:
            avg_time = (time.time() - start_time) / (successful + failed)
            eta_minutes = (total_images - i) * avg_time / 60
            print(f"  📈 Progress: {i}/{total_images} | ETA: {eta_minutes:.1f} min\n")

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
    if successful > 0:
        print(f"  • Average time: {elapsed/successful:.2f}s per image")

    # Check database stats
    try:
        print(f"\n📁 DATABASE STATUS:")
        stats_response = requests.get(f"{API_BASE_URL}/metadata/stats", timeout=5)
        if stats_response.status_code == 200:
            stats = stats_response.json()
            print(f"  • Images in database: {stats.get('total_images', 'N/A')}")

            if stats.get('top_styles'):
                print(f"\n🎨 TOP STYLES:")
                for style_info in stats['top_styles'][:5]:
                    print(f"  • {style_info['style']}: {style_info['count']} images")

            if stats.get('top_scenes'):
                print(f"\n🏠 TOP SCENES:")
                for scene_info in stats['top_scenes'][:5]:
                    print(f"  • {scene_info['scene']}: {scene_info['count']} images")
    except:
        pass

    # Failed images
    if failed > 0:
        print(f"\n❌ FAILED IMAGES:")
        for img_id, error in failed_images:
            print(f"  • {img_id}: {error}")

    print("\n✅ All done! Your images now have enhanced metadata.")
    print("\n💡 Next steps:")
    print("  1. Try searching with filters: 'warm modern living room'")
    print("  2. Start Streamlit: streamlit run streamlit_app_dual.py")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()