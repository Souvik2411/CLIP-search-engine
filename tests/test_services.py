"""Unit tests for service modules."""
import pytest
import numpy as np
from PIL import Image
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from app.utils.helpers import (
    normalize_embedding,
    fuse_embeddings,
    load_image_from_bytes,
    cosine_similarity,
    batch_list
)


class TestHelpers:
    """Test utility helper functions."""

    def test_normalize_embedding(self):
        """Test embedding normalization."""
        embedding = np.array([3.0, 4.0])
        normalized = normalize_embedding(embedding)

        # Check unit length
        assert np.isclose(np.linalg.norm(normalized), 1.0)

        # Check direction preserved
        assert np.allclose(normalized, np.array([0.6, 0.8]))

    def test_normalize_zero_embedding(self):
        """Test normalizing zero vector."""
        embedding = np.array([0.0, 0.0, 0.0])
        normalized = normalize_embedding(embedding)
        assert np.allclose(normalized, embedding)

    def test_fuse_embeddings(self):
        """Test embedding fusion."""
        img_emb = normalize_embedding(np.array([1.0, 0.0]))
        txt_emb = normalize_embedding(np.array([0.0, 1.0]))

        # Default weight 0.7 for image
        fused = fuse_embeddings(img_emb, txt_emb, image_weight=0.7)

        # Check it's normalized
        assert np.isclose(np.linalg.norm(fused), 1.0)

        # Check weighted average
        expected = 0.7 * img_emb + 0.3 * txt_emb
        expected = expected / np.linalg.norm(expected)
        assert np.allclose(fused, expected)

    def test_fuse_embeddings_equal_weight(self):
        """Test fusion with equal weights."""
        img_emb = normalize_embedding(np.array([1.0, 0.0]))
        txt_emb = normalize_embedding(np.array([0.0, 1.0]))

        fused = fuse_embeddings(img_emb, txt_emb, image_weight=0.5)

        # Should be diagonal direction
        assert np.allclose(fused, normalize_embedding(np.array([1.0, 1.0])))

    def test_load_image_from_bytes(self):
        """Test loading image from bytes."""
        # Create test image
        img = Image.new('RGB', (100, 100), color='red')
        from io import BytesIO
        img_bytes = BytesIO()
        img.save(img_bytes, format='JPEG')

        # Load from bytes
        loaded = load_image_from_bytes(img_bytes.getvalue())
        assert isinstance(loaded, Image.Image)
        assert loaded.mode == 'RGB'
        assert loaded.size == (100, 100)

    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])

        sim = cosine_similarity(a, b)
        assert np.isclose(sim, 1.0)

        # Orthogonal vectors
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        sim = cosine_similarity(a, b)
        assert np.isclose(sim, 0.0)

    def test_batch_list(self):
        """Test list batching."""
        items = list(range(10))
        batches = list(batch_list(items, 3))

        assert len(batches) == 4
        assert batches[0] == [0, 1, 2]
        assert batches[1] == [3, 4, 5]
        assert batches[2] == [6, 7, 8]
        assert batches[3] == [9]


class TestFAISSService:
    """Test FAISS service functionality."""

    @pytest.fixture
    def faiss_service(self):
        """Create a FAISS service instance."""
        from app.services.faiss_service import FAISSService
        service = FAISSService()
        service.create_index(dimension=512)
        return service

    def test_create_index(self, faiss_service):
        """Test index creation."""
        assert faiss_service.is_loaded
        assert faiss_service.size == 0

    def test_add_embeddings(self, faiss_service):
        """Test adding embeddings to index."""
        embeddings = np.random.rand(5, 512).astype(np.float32)
        metadata = [
            {"image_id": f"img{i}", "s3_key": f"images/img{i}.jpg"}
            for i in range(5)
        ]

        faiss_service.add_embeddings(embeddings, metadata)
        assert faiss_service.size == 5

    def test_search(self, faiss_service):
        """Test searching the index."""
        # Add some vectors
        embeddings = np.random.rand(10, 512).astype(np.float32)
        metadata = [
            {"image_id": f"img{i}", "s3_key": f"images/img{i}.jpg"}
            for i in range(10)
        ]
        faiss_service.add_embeddings(embeddings, metadata)

        # Search
        query = embeddings[0]  # Use first vector
        results = faiss_service.search(query, top_k=3)

        assert len(results) == 3
        assert all(isinstance(r, tuple) for r in results)
        # First result should be the query itself with high score
        assert results[0][0]["image_id"] == "img0"

    def test_search_empty_index(self, faiss_service):
        """Test searching empty index."""
        query = np.random.rand(512).astype(np.float32)
        results = faiss_service.search(query, top_k=5)
        assert len(results) == 0


class TestSearchHistoryService:
    """Test search history service."""

    @pytest.fixture
    def history_service(self):
        """Create a history service instance."""
        from app.services.search_history_service import SearchHistoryService
        from app.models.schemas import UserType

        # Create temporary history file
        service = SearchHistoryService()
        service._sessions = []  # Clear existing sessions
        return service

    def test_create_session(self, history_service):
        """Test creating a search session."""
        from app.models.schemas import UserType

        session_id = history_service.create_session(
            query_type="text_only",
            text_query="modern architecture",
            image_filename=None,
            user_type=UserType.GENERAL,
            detected_labels=["modern", "contemporary"],
            detected_objects=[],
            results_count=10,
            ai_summary="Found 10 modern architectural images."
        )

        assert session_id is not None
        assert history_service.get_count() == 1

    def test_add_to_session(self, history_service):
        """Test adding refinement to session."""
        from app.models.schemas import UserType

        # Create session
        session_id = history_service.create_session(
            query_type="text_only",
            text_query="architecture",
            image_filename=None,
            user_type=UserType.GENERAL,
            detected_labels=[],
            detected_objects=[],
            results_count=5,
            ai_summary="Found 5 images."
        )

        # Add refinement
        success = history_service.add_to_session(
            session_id=session_id,
            user_message="Show me more modern styles",
            ai_response="Here are 7 more modern designs.",
            query_type="text_only",
            detected_labels=["modern"],
            detected_objects=[],
            results_count=7
        )

        assert success
        session = history_service.get_session_by_id(session_id)
        assert len(session.conversation) == 3  # Initial + user + assistant

    def test_get_stats(self, history_service):
        """Test getting history statistics."""
        from app.models.schemas import UserType

        # Create various sessions
        history_service.create_session(
            "text_only", "query1", None, UserType.GENERAL,
            [], [], 5, "summary1"
        )
        history_service.create_session(
            "image_only", None, "img.jpg", UserType.PROFESSIONAL,
            ["modern"], [], 8, "summary2"
        )
        history_service.create_session(
            "image_and_text", "query2", "img2.jpg", UserType.STUDENT,
            ["contemporary"], [], 3, "summary3"
        )

        stats = history_service.get_stats()
        assert stats.total_searches == 3
        assert stats.text_only_searches == 1
        assert stats.image_only_searches == 1
        assert stats.combined_searches == 1

    def test_delete_session(self, history_service):
        """Test deleting a session."""
        from app.models.schemas import UserType

        session_id = history_service.create_session(
            "text_only", "test", None, UserType.GENERAL,
            [], [], 5, "summary"
        )

        assert history_service.get_count() == 1
        success = history_service.delete_session(session_id)
        assert success
        assert history_service.get_count() == 0


class TestFavoritesService:
    """Test favorites service."""

    @pytest.fixture
    def favorites_service(self):
        """Create a favorites service instance."""
        from app.services.favorites_service import FavoritesService
        service = FavoritesService()
        service._favorites = []  # Clear existing
        return service

    def test_add_favorite(self, favorites_service):
        """Test adding a favorite."""
        fav_id = favorites_service.add_favorite(
            image_id="img1",
            s3_key="images/img1.jpg",
            url="https://example.com/img1.jpg",
            labels=["modern"],
            note="Beautiful design"
        )

        assert fav_id is not None
        assert favorites_service.get_count() == 1

    def test_is_favorited(self, favorites_service):
        """Test checking favorite status."""
        favorites_service.add_favorite(
            image_id="img1",
            s3_key="images/img1.jpg",
            url="https://example.com/img1.jpg"
        )

        assert favorites_service.is_favorited("img1")
        assert not favorites_service.is_favorited("img2")

    def test_batch_check_favorited(self, favorites_service):
        """Test batch checking favorites."""
        favorites_service.add_favorite(
            image_id="img1",
            s3_key="images/img1.jpg",
            url="https://example.com/img1.jpg"
        )
        favorites_service.add_favorite(
            image_id="img3",
            s3_key="images/img3.jpg",
            url="https://example.com/img3.jpg"
        )

        status = favorites_service.batch_check_favorited(["img1", "img2", "img3"])
        assert status["img1"] == True
        assert status["img2"] == False
        assert status["img3"] == True

    def test_remove_favorite(self, favorites_service):
        """Test removing a favorite."""
        favorites_service.add_favorite(
            image_id="img1",
            s3_key="images/img1.jpg",
            url="https://example.com/img1.jpg"
        )

        assert favorites_service.get_count() == 1
        success = favorites_service.remove_favorite("img1")
        assert success
        assert favorites_service.get_count() == 0
