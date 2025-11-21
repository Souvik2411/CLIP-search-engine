"""Comprehensive error handling utilities."""
import logging
from typing import Any
from fastapi import HTTPException, status
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class SearchError(Exception):
    """Base exception for search-related errors."""
    pass


class ModelLoadError(SearchError):
    """Error loading ML models."""
    pass


class IndexError(SearchError):
    """Error with FAISS index operations."""
    pass


class S3Error(SearchError):
    """Error with S3 operations."""
    pass


class ValidationError(SearchError):
    """Input validation error."""
    pass


def handle_service_error(error: Exception, context: str) -> HTTPException:
    """
    Convert service errors to appropriate HTTP exceptions.

    Args:
        error: The exception that occurred
        context: Context about where the error occurred

    Returns:
        HTTPException with appropriate status code and message
    """
    logger.error(f"{context}: {str(error)}", exc_info=True)

    # Map specific errors to HTTP status codes
    if isinstance(error, ValidationError):
        return HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error)
        )
    elif isinstance(error, ModelLoadError):
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model service temporarily unavailable"
        )
    elif isinstance(error, IndexError):
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Search index temporarily unavailable"
        )
    elif isinstance(error, S3Error):
        return HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Storage service error"
        )
    else:
        # Generic server error
        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred"
        )


async def handle_global_exception(request, exc):
    """
    Global exception handler for uncaught exceptions.

    Args:
        request: The FastAPI request
        exc: The exception

    Returns:
        JSONResponse with error details
    """
    logger.error(f"Uncaught exception: {exc}", exc_info=True)

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred. Please try again later.",
            "path": str(request.url)
        }
    )


def validate_search_input(
    image: Any,
    text_query: str | None,
    max_text_length: int = 500,
    max_image_size: int = 10 * 1024 * 1024
) -> None:
    """
    Validate search endpoint inputs.

    Args:
        image: Uploaded image file
        text_query: Text query string
        max_text_length: Maximum allowed text length
        max_image_size: Maximum allowed image size in bytes

    Raises:
        ValidationError: If validation fails
    """
    if image is None and text_query is None:
        raise ValidationError("Must provide either image, text_query, or both")

    if text_query is not None:
        if not text_query.strip():
            raise ValidationError("Text query cannot be empty")
        if len(text_query) > max_text_length:
            raise ValidationError(
                f"Text query too long. Maximum {max_text_length} characters allowed."
            )

    if image is not None:
        # Check content type
        allowed_types = ["image/jpeg", "image/jpg", "image/png", "image/webp"]
        if image.content_type not in allowed_types:
            raise ValidationError(
                f"Invalid image format. Allowed: {', '.join(allowed_types)}"
            )
