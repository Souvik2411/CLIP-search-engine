"""Comprehensive API tests for ARCHINZA Search Pipeline."""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
from io import BytesIO
from PIL import Image
import numpy as np

from app.main import app
from app.models.schemas import UserType


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def mock_image():
    """Create a mock image file."""
    image = Image.new('RGB', (224, 224), color='red')
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    return img_byte_arr


@pytest.fixture
def mock_services():
    """Mock all services to avoid loading models."""
    with patch('app.api.routes.clip_service') as mock_clip, \
         patch('app.api.routes.faiss_service') as mock_faiss, \
         patch('app.api.routes.s3_service') as mock_s3, \
         patch('app.api.routes.llm_service') as mock_llm:

        # Mock CLIP service
        mock_clip.is_loaded = True
        mock_clip.get_image_embedding_and_dual_labels.return_value = (
            np.random.rand(512).astype(np.float32),
            [("modern architecture", 0.85), ("contemporary design", 0.75)],
            [("sofa", 0.80), ("coffee table", 0.70)]
        )
        mock_clip.get_text_embedding.return_value = np.random.rand(512).astype(np.float32)

        # Mock FAISS service
        mock_faiss.is_loaded = True
        mock_faiss.size = 75
        mock_faiss.search.return_value = [
            ({"image_id": "img1", "s3_key": "images/img1.jpg", "labels": ["modern"]}, 0.95),
            ({"image_id": "img2", "s3_key": "images/img2.jpg", "labels": ["contemporary"]}, 0.90)
        ]

        # Mock S3 service (async method)
        import asyncio
        async def mock_get_urls(keys, expiration=None):
            return {
                "images/img1.jpg": "https://example.com/img1.jpg",
                "images/img2.jpg": "https://example.com/img2.jpg"
            }
        mock_s3.get_presigned_urls_async = mock_get_urls

        # Mock LLM service (async method)
        async def mock_generate_summary(query_labels, result_count, user_type, text_query=None):
            return (
                "Found 2 modern architectural images with contemporary design elements.",
                ["Explore minimalist designs", "Search for sustainable architecture", "Compare with classical styles"]
            )
        mock_llm.generate_search_summary = mock_generate_summary

        yield {
            'clip': mock_clip,
            'faiss': mock_faiss,
            's3': mock_s3,
            'llm': mock_llm
        }


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_check_success(self, client, mock_services):
        """Test health check returns successful status."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "index_size" in data


class TestSearchEndpoint:
    """Test search functionality."""

    def test_search_text_only(self, client, mock_services):
        """Test text-only search."""
        response = client.post(
            "/api/v1/search",
            data={
                "text_query": "modern kitchen design",
                "user_type": "general"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "detected_labels" in data
        assert "summary" in data
        assert data["query_type"] == "text_only"

    def test_search_image_only(self, client, mock_services, mock_image):
        """Test image-only search."""
        response = client.post(
            "/api/v1/search",
            files={"image": ("test.jpg", mock_image, "image/jpeg")},
            data={"user_type": "professional"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "detected_labels" in data
        assert "detected_objects" in data
        assert data["query_type"] == "image_only"
        assert len(data["detected_labels"]) > 0

    def test_search_image_and_text(self, client, mock_services, mock_image):
        """Test combined image and text search."""
        response = client.post(
            "/api/v1/search",
            files={"image": ("test.jpg", mock_image, "image/jpeg")},
            data={
                "text_query": "minimalist bedroom",
                "user_type": "student"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert data["query_type"] == "image_and_text"

    def test_search_no_input_fails(self, client):
        """Test search with no input returns error."""
        response = client.post("/api/v1/search")
        assert response.status_code == 400  # Validation error (now HTTPException)

    def test_search_user_types(self, client, mock_services):
        """Test all user types work correctly."""
        for user_type in ["professional", "student", "enthusiast", "general"]:
            response = client.post(
                "/api/v1/search",
                data={
                    "text_query": "architecture",
                    "user_type": user_type
                }
            )
            assert response.status_code == 200

    def test_search_returns_correct_structure(self, client, mock_services):
        """Test response structure is correct."""
        response = client.post(
            "/api/v1/search",
            data={"text_query": "modern architecture"}
        )
        data = response.json()

        # Check top-level fields
        assert "results" in data
        assert "detected_labels" in data
        assert "detected_objects" in data
        assert "summary" in data
        assert "follow_up_suggestions" in data
        assert "query_type" in data
        assert "total_results" in data

        # Check result structure
        if len(data["results"]) > 0:
            result = data["results"][0]
            assert "image_id" in result
            assert "s3_key" in result
            assert "url" in result
            assert "score" in result


class TestIndexEndpoint:
    """Test indexing functionality."""

    def test_index_stats(self, client, mock_services):
        """Test getting index statistics."""
        response = client.get("/api/v1/index/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_vectors" in data
        assert "embedding_dimension" in data
        assert "index_loaded" in data


class TestHistoryEndpoint:
    """Test search history functionality."""

    def test_get_history(self, client):
        """Test retrieving search history."""
        response = client.get("/api/v1/history")
        assert response.status_code == 200
        data = response.json()
        assert "history" in data
        assert "total_count" in data

    def test_get_history_with_limit(self, client):
        """Test history pagination."""
        response = client.get("/api/v1/history?limit=5")
        assert response.status_code == 200
        data = response.json()
        assert len(data["history"]) <= 5

    def test_get_history_stats(self, client):
        """Test history statistics."""
        response = client.get("/api/v1/history/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_searches" in data
        assert "image_only_searches" in data
        assert "text_only_searches" in data
        assert "combined_searches" in data

    def test_create_session(self, client):
        """Test creating a new search session."""
        response = client.post(
            "/api/v1/history/session",
            json={
                "query_type": "text_only",
                "text_query": "modern architecture",
                "user_type": "general",
                "detected_labels": ["modern", "contemporary"],
                "detected_objects": [],
                "results_count": 10,
                "ai_summary": "Found 10 modern architectural images."
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data


class TestFavoritesEndpoint:
    """Test favorites functionality."""

    def test_get_favorites(self, client):
        """Test retrieving favorites."""
        response = client.get("/api/v1/favorites")
        assert response.status_code == 200
        data = response.json()
        assert "favorites" in data
        assert "total_count" in data

    def test_add_favorite(self, client):
        """Test adding a favorite."""
        response = client.post(
            "/api/v1/favorites",
            json={
                "image_id": "test_image_1",
                "s3_key": "images/test.jpg",
                "url": "https://example.com/test.jpg",
                "labels": ["modern", "minimalist"],
                "note": "Beautiful design"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "favorite_id" in data

    def test_check_favorite_status(self, client):
        """Test checking if image is favorited."""
        response = client.get("/api/v1/favorites/check/test_image_1")
        assert response.status_code == 200
        data = response.json()
        assert "is_favorited" in data

    def test_batch_check_favorites(self, client):
        """Test batch checking favorite status."""
        response = client.post(
            "/api/v1/favorites/check-batch",
            json={"image_ids": ["img1", "img2", "img3"]}
        )
        assert response.status_code == 200
        data = response.json()
        assert "favorites" in data


class TestInputValidation:
    """Test input validation."""

    def test_invalid_user_type(self, client):
        """Test invalid user type is rejected."""
        response = client.post(
            "/api/v1/search",
            data={
                "text_query": "architecture",
                "user_type": "invalid_type"
            }
        )
        assert response.status_code == 422

    def test_empty_text_query(self, client, mock_services):
        """Test empty text query."""
        response = client.post(
            "/api/v1/search",
            data={
                "text_query": "",
                "user_type": "general"
            }
        )
        # Empty string is technically provided, so it processes (returns 200)
        # This could be enhanced to reject empty strings in future
        assert response.status_code == 200


class TestErrorHandling:
    """Test error handling."""

    def test_search_with_faiss_error(self, client, mock_services):
        """Test graceful handling of FAISS errors."""
        mock_services['faiss'].search.side_effect = Exception("FAISS error")

        response = client.post(
            "/api/v1/search",
            data={"text_query": "test"}
        )
        assert response.status_code == 500

    def test_delete_nonexistent_session(self, client):
        """Test deleting non-existent session."""
        response = client.delete("/api/v1/history/nonexistent-id")
        assert response.status_code == 404

    def test_delete_nonexistent_favorite(self, client):
        """Test removing non-existent favorite."""
        response = client.delete("/api/v1/favorites/nonexistent-id")
        assert response.status_code == 404
