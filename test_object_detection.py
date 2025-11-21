"""Test script for object detection functionality."""
import sys
from pathlib import Path
from PIL import Image
import numpy as np

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from app.services.clip_service import CLIPService
from app.config import get_settings

def test_object_detection():
    """Test the dual label detection (architectural + objects)."""
    print("Testing Object Detection Functionality")
    print("=" * 60)

    # Initialize service
    print("\n1. Initializing CLIP service...")
    clip_service = CLIPService()
    clip_service.load_model()
    print("[OK] CLIP model loaded")

    # Check configuration
    settings = get_settings()
    print(f"\n2. Configuration:")
    print(f"   Architecture labels: {len(settings.ARCHITECTURE_LABELS)} labels")
    print(f"   Object labels: {len(settings.FURNITURE_OBJECT_LABELS)} labels")
    print(f"   Sample architecture labels: {settings.ARCHITECTURE_LABELS[:3]}")
    print(f"   Sample object labels: {settings.FURNITURE_OBJECT_LABELS[:5]}")

    # Create a test image (random image for testing)
    print("\n3. Creating test image...")
    # Create a simple test image (white background with some patterns)
    test_image = Image.new('RGB', (512, 512), color='white')
    print("[OK] Test image created (512x512)")

    # Test the new dual label detection method
    print("\n4. Testing get_image_embedding_and_dual_labels()...")
    try:
        embedding, arch_results, object_results = clip_service.get_image_embedding_and_dual_labels(
            test_image,
            top_k_arch=5,
            top_k_objects=7
        )

        print("[OK] Dual label detection completed")
        print(f"\n   Embedding shape: {embedding.shape}")
        print(f"   Embedding type: {type(embedding)}")
        print(f"   Embedding normalized: {np.linalg.norm(embedding):.4f} (should be ~1.0)")

        print(f"\n   Architectural Labels (Top 5):")
        for i, (label, prob) in enumerate(arch_results, 1):
            print(f"      {i}. {label}: {prob:.4f}")

        print(f"\n   Detected Objects (Top 7):")
        for i, (obj, prob) in enumerate(object_results, 1):
            print(f"      {i}. {obj}: {prob:.4f}")

    except Exception as e:
        print(f"[FAIL] Error during dual label detection: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test that the method returns correct types
    print("\n5. Verifying return types...")
    assert isinstance(embedding, np.ndarray), "Embedding should be numpy array"
    assert embedding.shape == (512,), "Embedding should be 512-dimensional"
    assert isinstance(arch_results, list), "Arch results should be a list"
    assert isinstance(object_results, list), "Object results should be a list"
    assert len(arch_results) == 5, "Should return 5 architectural labels"
    assert len(object_results) == 7, "Should return 7 object labels"

    for label, prob in arch_results:
        assert isinstance(label, str), "Label should be string"
        assert isinstance(prob, float), "Probability should be float"
        assert 0 <= prob <= 1, "Probability should be between 0 and 1"

    for obj, prob in object_results:
        assert isinstance(obj, str), "Object should be string"
        assert isinstance(prob, float), "Probability should be float"
        assert 0 <= prob <= 1, "Probability should be between 0 and 1"

    print("[OK] All type checks passed")

    # Test with different top_k values
    print("\n6. Testing with different top_k values...")
    _, arch_3, obj_10 = clip_service.get_image_embedding_and_dual_labels(
        test_image,
        top_k_arch=3,
        top_k_objects=10
    )
    assert len(arch_3) == 3, "Should return 3 architectural labels"
    assert len(obj_10) == 10, "Should return 10 object labels"
    print("[OK] Different top_k values work correctly")

    print("\n" + "=" * 60)
    print("All object detection tests passed successfully!")
    print("=" * 60)
    print("\nNote: To test with real images, place an interior design image")
    print("in the project directory and update this test script.")

if __name__ == "__main__":
    test_object_detection()
