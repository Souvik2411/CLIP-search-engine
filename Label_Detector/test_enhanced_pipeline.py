"""
Test script for the enhanced image metadata pipeline.

This script demonstrates:
1. Loading an image
2. Extracting all features (CLIP, color, texture, material, style)
3. Indexing the image
4. Searching with filters
5. Viewing extracted metadata
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image
import json
import time

# Import services
from app.services.clip_service import CLIPService
from app.services.color_service import color_service
from app.services.texture_service import texture_service
from app.services.material_service import MaterialService
from app.services.style_service import StyleService
from app.services.faiss_service import FAISSService
from app.services.indexing_service import IndexingService
from app.services.enhanced_search_service import EnhancedSearchService
from app.services.metadata_db_service import metadata_db_service


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def test_feature_extraction(image_path: str):
    """Test individual feature extractors."""
    print_section("TESTING FEATURE EXTRACTION")

    # Load image
    print(f"Loading image: {image_path}")
    image = Image.open(image_path)
    print(f"Image size: {image.size}")

    # Initialize services
    print("\nInitializing services...")
    clip_service = CLIPService()
    clip_service.load_model()
    material_service = MaterialService(clip_service)
    style_service = StyleService(clip_service)

    # Test 1: CLIP features
    print_section("1. CLIP Features")
    start = time.time()
    embedding, arch_labels, object_labels = clip_service.get_image_embedding_and_dual_labels(image)
    elapsed = time.time() - start
    print(f"⏱️  Processing time: {elapsed:.2f}s")
    print(f"\n📊 Architectural Labels:")
    for label, score in arch_labels:
        print(f"  • {label}: {score:.3f}")
    print(f"\n🪑 Object Labels:")
    for label, score in object_labels[:5]:
        print(f"  • {label}: {score:.3f}")

    # Test 2: Color features
    print_section("2. Color Features")
    start = time.time()
    color_features = color_service.extract_all_color_features(image)
    elapsed = time.time() - start
    print(f"⏱️  Processing time: {elapsed:.2f}s")
    print(f"\n🎨 Color Palette:")
    for i, color in enumerate(color_features["palette"][:5], 1):
        print(f"  {i}. {color}")
    print(f"\n📊 Color Statistics:")
    print(f"  • Dominant Color: {color_features['dominant_color']}")
    print(f"  • Temperature: {color_features['color_temperature']}")
    print(f"  • Brightness: {color_features['brightness_category']} ({color_features['brightness']:.2f})")
    print(f"  • Saturation: {color_features['saturation_category']} ({color_features['saturation']:.2f})")

    # Test 3: Texture features
    print_section("3. Texture Features")
    start = time.time()
    texture_features = texture_service.extract_texture_features(image)
    elapsed = time.time() - start
    print(f"⏱️  Processing time: {elapsed:.2f}s")
    print(f"\n📊 Texture Classification:")
    for key, value in texture_features["classification"].items():
        print(f"  • {key}: {value}")
    print(f"\n🔢 GLCM Features:")
    for key, value in texture_features["glcm"].items():
        print(f"  • {key}: {value:.3f}")

    # Test 4: Material detection
    print_section("4. Material Detection")
    start = time.time()
    material_features = material_service.extract_material_features(image)
    elapsed = time.time() - start
    print(f"⏱️  Processing time: {elapsed:.2f}s")
    print(f"\n🪵 Top Materials:")
    for mat in material_features["materials"][:5]:
        print(f"  • {mat['name']}: {mat['confidence']:.3f}")
    print(f"\n📊 Material Categories:")
    for cat, score in material_features["categories"].items():
        print(f"  • {cat}: {score:.3f}")

    # Test 5: Style classification
    print_section("5. Style Classification")
    start = time.time()
    style_features = style_service.extract_style_features(image)
    elapsed = time.time() - start
    print(f"⏱️  Processing time: {elapsed:.2f}s")
    print(f"\n🎨 Styles:")
    for style in style_features["styles"]:
        print(f"  • {style['name']}: {style['confidence']:.3f}")
    print(f"\n🏠 Scenes:")
    for scene in style_features["scenes"]:
        print(f"  • {scene['name']}: {scene['confidence']:.3f}")
    print(f"\n✨ Ambiance:")
    for amb in style_features["ambiance"]:
        print(f"  • {amb['name']}: {amb['confidence']:.3f}")


def test_indexing(image_path: str):
    """Test the complete indexing pipeline."""
    print_section("TESTING INDEXING PIPELINE")

    # Load image
    image = Image.open(image_path)

    # Initialize services
    print("Initializing services...")
    clip_service = CLIPService()
    clip_service.load_model()
    faiss_service = FAISSService()
    faiss_service.load_index()

    indexing_service = IndexingService(clip_service, faiss_service)

    # Process image
    print(f"\n📸 Processing image: {image_path}")
    start = time.time()
    metadata = indexing_service.process_image(
        image=image,
        image_id="test_image_001",
        generate_summary=False  # Skip LLM to avoid API cost
    )
    elapsed = time.time() - start

    print(f"\n✅ Processing complete in {elapsed:.2f}s")
    print(f"\n📊 Timing Breakdown:")
    for component, time_spent in metadata["timing"].items():
        if time_spent > 0:
            print(f"  • {component}: {time_spent:.2f}s")

    print(f"\n🔍 Extracted Metadata Summary:")
    print(f"  • Image ID: {metadata['image_id']}")
    print(f"  • Primary Label: {metadata['primary_label']}")
    print(f"  • Primary Style: {metadata['style']['primary_style']}")
    print(f"  • Primary Scene: {metadata['style']['primary_scene']}")
    print(f"  • Dominant Color: {metadata['color']['dominant_color']}")
    print(f"  • Primary Material: {metadata['materials']['primary_category']}")
    print(f"  • Texture: {metadata['texture']['classification']['roughness']}, {metadata['texture']['classification']['pattern']}")

    return metadata


def test_search(query: str = "modern living room"):
    """Test enhanced search with filters."""
    print_section("TESTING ENHANCED SEARCH")

    # Initialize services
    print("Initializing services...")
    clip_service = CLIPService()
    clip_service.load_model()
    faiss_service = FAISSService()
    faiss_service.load_index()

    search_service = EnhancedSearchService(clip_service, faiss_service)

    # Search
    print(f"\n🔍 Searching for: '{query}'")
    start = time.time()
    results = search_service.search_with_text(query, top_k=5, use_filters=True)
    elapsed = time.time() - start

    print(f"\n✅ Search complete in {elapsed*1000:.0f}ms")
    print(f"\n📊 Found {len(results)} results:")

    for i, result in enumerate(results, 1):
        print(f"\n  Result {i}:")
        print(f"    • Image ID: {result['image_id']}")
        print(f"    • Similarity: {result['similarity_score']:.3f}")
        print(f"    • Style: {result['metadata']['style']['primary']}")
        print(f"    • Scene: {result['metadata']['style']['scene']}")
        print(f"    • Color: {result['metadata']['color']['dominant_color']} ({result['metadata']['color']['temperature']})")


def test_metadata_filters():
    """Test metadata filtering."""
    print_section("TESTING METADATA FILTERS")

    # Test various filter combinations
    filters = [
        {"color_temp": "warm", "scene": "living room"},
        {"material_category": "wood", "style": "modern"},
        {"brightness": "bright", "texture_roughness": "smooth"},
    ]

    for i, filter_set in enumerate(filters, 1):
        print(f"\n🔍 Filter Set {i}: {filter_set}")
        image_ids = metadata_db_service.search_by_filters(**filter_set, limit=10)
        print(f"  ✅ Found {len(image_ids)} matching images")
        if image_ids:
            print(f"  📋 Sample IDs: {image_ids[:3]}")


def test_database_stats():
    """Test database statistics."""
    print_section("DATABASE STATISTICS")

    stats = metadata_db_service.get_stats()

    print(f"📊 Total Images: {stats['total_images']}")

    print(f"\n🎨 Top Styles:")
    for item in stats['top_styles'][:5]:
        print(f"  • {item['style']}: {item['count']} images")

    print(f"\n🏠 Top Scenes:")
    for item in stats['top_scenes'][:5]:
        print(f"  • {item['scene']}: {item['count']} images")


def main():
    """Main test runner."""
    print("\n" + "█" * 80)
    print("  ENHANCED IMAGE METADATA PIPELINE - TEST SUITE")
    print("█" * 80)

    # Check if image path is provided
    if len(sys.argv) < 2:
        print("\n⚠️  Usage: python test_enhanced_pipeline.py <image_path>")
        print("\nExample:")
        print("  python test_enhanced_pipeline.py path/to/your/image.jpg")
        return

    image_path = sys.argv[1]

    # Verify image exists
    if not Path(image_path).exists():
        print(f"\n❌ Error: Image not found at {image_path}")
        return

    try:
        # Run tests
        print("\n🚀 Starting tests...\n")

        # Test 1: Feature extraction
        test_feature_extraction(image_path)

        # Test 2: Complete indexing pipeline
        metadata = test_indexing(image_path)

        # Test 3: Enhanced search
        test_search("modern minimalist living room")

        # Test 4: Metadata filters
        test_metadata_filters()

        # Test 5: Database stats
        test_database_stats()

        print_section("ALL TESTS COMPLETED SUCCESSFULLY! ✅")

    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
