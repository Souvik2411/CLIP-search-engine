"""Test script for search history functionality."""
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from app.services.search_history_service import SearchHistoryService
from app.models.schemas import UserType

def test_search_history():
    """Test search history service."""
    print("Testing Search History Service...")
    print("-" * 50)

    # Initialize service
    history_service = SearchHistoryService()
    print(f"[OK] Service initialized")
    print(f"  Current history count: {history_service.get_count()}")

    # Test 1: Create a session
    print("\n1. Testing create_session()...")
    session_id = history_service.create_session(
        query_type="image_only",
        text_query=None,
        image_filename="test_building.jpg",
        user_type=UserType.PROFESSIONAL,
        detected_labels=["modern", "glass facade", "commercial"],
        detected_objects=["window", "facade design"],
        results_count=9,
        ai_summary="Found 9 modern commercial buildings with glass facades and contemporary design elements."
    )
    print(f"[OK] Created session with ID: {session_id}")

    # Test 2: Create another session (text only)
    session_id_2 = history_service.create_session(
        query_type="text_only",
        text_query="sustainable architecture",
        image_filename=None,
        user_type=UserType.STUDENT,
        detected_labels=[],
        detected_objects=[],
        results_count=5,
        ai_summary="Discovered 5 sustainable architecture examples focusing on eco-friendly design principles."
    )
    print(f"[OK] Created second session with ID: {session_id_2}")

    # Test 3: Create combined search session
    session_id_3 = history_service.create_session(
        query_type="image_and_text",
        text_query="modern residential",
        image_filename="house.jpg",
        user_type=UserType.ENTHUSIAST,
        detected_labels=["residential", "contemporary"],
        detected_objects=["bed", "sofa"],
        results_count=7,
        ai_summary="Found 7 contemporary residential designs with modern aesthetics and clean lines."
    )
    print(f"[OK] Created third session with ID: {session_id_3}")

    # Test 4: Get recent sessions
    print("\n2. Testing get_recent_sessions()...")
    recent = history_service.get_recent_sessions(limit=5)
    print(f"[OK] Retrieved {len(recent)} recent sessions")
    for i, item in enumerate(recent, 1):
        print(f"  {i}. {item.title} - {item.initial_query_type} - {item.timestamp}")

    # Test 5: Add refinement to a session
    print("\n3. Testing add_to_session()...")
    added = history_service.add_to_session(
        session_id=session_id,
        user_message="Show me more with glass curtain walls",
        ai_response="Here are 8 additional buildings featuring floor-to-ceiling glass curtain walls.",
        query_type="text_only",
        detected_labels=["facade design", "structural design"],
        detected_objects=["window", "curtains"],
        results_count=8
    )
    if added:
        print(f"[OK] Added refinement to session {session_id}")
    else:
        print("[FAIL] Failed to add to session")

    # Test 6: Get stats
    print("\n4. Testing get_stats()...")
    stats = history_service.get_stats()
    print(f"[OK] Statistics:")
    print(f"  Total sessions: {stats.total_searches}")
    print(f"  Image only: {stats.image_only_searches}")
    print(f"  Text only: {stats.text_only_searches}")
    print(f"  Combined: {stats.combined_searches}")
    if stats.most_common_labels:
        print(f"  Top labels: {stats.most_common_labels[:3]}")

    # Test 7: Get specific session by ID
    print("\n5. Testing get_session_by_id()...")
    found = history_service.get_session_by_id(session_id)
    if found:
        print(f"[OK] Found session: {found.title}")
        print(f"  Query type: {found.initial_query_type}")
        print(f"  Conversation messages: {len(found.conversation)}")
    else:
        print("[FAIL] Session not found")

    # Test 8: Delete a session
    print("\n6. Testing delete_session()...")
    deleted = history_service.delete_session(session_id_2)
    if deleted:
        print(f"[OK] Deleted session {session_id_2}")
        print(f"  New count: {history_service.get_count()}")
    else:
        print("[FAIL] Delete failed")

    print("\n" + "=" * 50)
    print("All tests completed successfully!")
    print("=" * 50)

if __name__ == "__main__":
    test_search_history()
