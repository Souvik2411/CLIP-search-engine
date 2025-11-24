from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request
from typing import Optional
from slowapi import Limiter
from slowapi.util import get_remote_address
import logging

from app.models.schemas import (
    SearchResponse,
    ImageMetadata,
    UserType,
    HealthResponse,
    IndexRequest,
    IndexResponse,
    SearchHistoryResponse,
    SearchHistoryStats,
    FavoritesResponse
)
from app.services import CLIPService, CLIPONNXService, FAISSService, S3Service, LLMService
from app.services.search_history_service import SearchHistoryService
from app.services.favorites_service import FavoritesService
from app.services.indexing_service import IndexingService
from app.services.enhanced_search_service import EnhancedSearchService
from app.services.metadata_db_service import metadata_db_service
from app.utils.helpers import load_image_from_bytes, fuse_embeddings
from app.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Initialize services
settings = get_settings()

# Use ONNX service if enabled, otherwise use standard PyTorch
if settings.USE_ONNX:
    logger.info("Using ONNX-optimized CLIP service (2-3x faster)")
    clip_service = CLIPONNXService()
else:
    logger.info("Using standard PyTorch CLIP service")
    clip_service = CLIPService()

faiss_service = FAISSService()
s3_service = S3Service()
llm_service = LLMService()
history_service = SearchHistoryService()
favorites_service = FavoritesService()

# Initialize enhanced services
indexing_service = IndexingService(clip_service, faiss_service)
enhanced_search_service = EnhancedSearchService(clip_service, faiss_service)


# Startup will be handled by lifespan in main.py (deprecated @router.on_event)
# Model loading is now handled via lifecycle in main.py


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=clip_service.is_loaded,
        index_size=faiss_service.size
    )


@router.post("/search", response_model=SearchResponse)
@limiter.limit("20/minute")
async def search(
    request: Request,
    image: Optional[UploadFile] = File(None),
    text_query: Optional[str] = Form(None),
    user_type: UserType = Form(UserType.GENERAL)
):
    """
    Search for similar architectural images.

    Supports three input cases:
    - Image only: Upload an image to find similar images
    - Text only: Provide text description to search
    - Image + Text: Combine both for refined search

    Rate limit: 20 requests per minute per IP
    """
    # Validate input
    if image is None and text_query is None:
        raise HTTPException(
            status_code=400,
            detail="Must provide either image, text_query, or both"
        )

    # Validate text query length
    if text_query and len(text_query) > 500:
        raise HTTPException(
            status_code=400,
            detail="Text query too long. Maximum 500 characters allowed."
        )

    # Validate image if provided
    if image:
        # Check file size (max 10MB)
        content = await image.read()
        if len(content) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail="Image file too large. Maximum 10MB allowed."
            )

        # Check content type
        allowed_types = ["image/jpeg", "image/jpg", "image/png", "image/webp"]
        if image.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image format. Allowed: {', '.join(allowed_types)}"
            )

        # Reset file pointer for later reading
        await image.seek(0)

    query_embedding = None
    detected_labels = []
    detected_objects = []
    query_type = ""

    try:
        # Case 1: Image only
        if image is not None and text_query is None:
            query_type = "image_only"
            image_bytes = await image.read()
            pil_image = load_image_from_bytes(image_bytes)

            # Get embedding, architectural labels, and object labels in one pass
            query_embedding, arch_results, object_results = clip_service.get_image_embedding_and_dual_labels(pil_image)
            detected_labels = [label for label, _ in arch_results]
            detected_objects = [obj for obj, _ in object_results]

        # Case 2: Text only
        elif image is None and text_query is not None:
            query_type = "text_only"
            query_embedding = clip_service.get_text_embedding(text_query)

        # Case 3: Image + Text
        else:
            query_type = "image_and_text"
            image_bytes = await image.read()
            pil_image = load_image_from_bytes(image_bytes)

            # Get image embedding, architectural labels, and object labels
            image_embedding, arch_results, object_results = clip_service.get_image_embedding_and_dual_labels(pil_image)
            detected_labels = [label for label, _ in arch_results]
            detected_objects = [obj for obj, _ in object_results]

            # Get text embedding
            text_embedding = clip_service.get_text_embedding(text_query)

            # Fuse embeddings
            query_embedding = fuse_embeddings(
                image_embedding,
                text_embedding,
                settings.IMAGE_TEXT_FUSION_WEIGHT
            )

        # Search FAISS index
        search_results = faiss_service.search(
            query_embedding,
            top_k=settings.TOP_K_RESULTS
        )

        if not search_results:
            return SearchResponse(
                results=[],
                detected_labels=detected_labels,
                detected_objects=detected_objects,
                summary="No similar images found in the database.",
                follow_up_suggestions=["Try uploading a different image", "Use broader search terms"],
                query_type=query_type,
                total_results=0
            )

        # Get presigned URLs for results (async for better performance)
        # Handle both old format (s3_url) and new format (s3_key)
        s3_keys = []
        for meta, _ in search_results:
            if 's3_key' in meta:
                s3_key = meta['s3_key']
                # Skip local images
                if not s3_key.startswith('local/'):
                    s3_keys.append(s3_key)
            elif 's3_url' in meta:
                # Extract s3_key from presigned URL (backward compatibility)
                s3_url = meta['s3_url']
                if 's3.amazonaws.com/' in s3_url:
                    # Parse: https://bucket.s3.amazonaws.com/key?params -> key
                    s3_key = s3_url.split('s3.amazonaws.com/')[-1].split('?')[0]
                    s3_keys.append(s3_key)

        urls = await s3_service.get_presigned_urls_async(s3_keys) if s3_keys else {}

        # Build result objects
        results = []
        for meta, score in search_results:
            # Get s3_key from metadata (handle both formats)
            s3_key = meta.get('s3_key')
            if not s3_key and 's3_url' in meta:
                s3_url = meta['s3_url']
                if 's3.amazonaws.com/' in s3_url:
                    s3_key = s3_url.split('s3.amazonaws.com/')[-1].split('?')[0]
                else:
                    s3_key = s3_url  # Fallback

            # Get URL (either from generated presigned URLs or use stored s3_url)
            url = urls.get(s3_key, '') if s3_key else ''
            if not url and 's3_url' in meta:
                url = meta['s3_url']  # Use stored URL as fallback

            results.append(ImageMetadata(
                image_id=meta['image_id'],
                s3_key=s3_key or '',
                url=url,
                score=score,
                labels=meta.get('labels')
            ))

        # Generate summary using LLM
        summary, suggestions = await llm_service.generate_search_summary(
            query_labels=detected_labels,
            result_count=len(results),
            user_type=user_type,
            text_query=text_query
        )

        # Note: Session creation/updates are now handled by the frontend
        # The frontend calls /history/session endpoints to manage sessions

        return SearchResponse(
            results=results,
            detected_labels=detected_labels,
            detected_objects=detected_objects,
            summary=summary,
            follow_up_suggestions=suggestions,
            query_type=query_type,
            total_results=len(results)
        )

    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search/similar/{image_id}", response_model=SearchResponse)
async def search_similar_by_id(
    image_id: str,
    top_k: int = 5
):
    """
    Fast search for similar images using an existing image's embedding.
    This is optimized for the "More Like This" feature - no image download or re-processing needed.

    Args:
        image_id: The image ID to find similar images for
        top_k: Number of similar images to return (default 5)

    Returns:
        SearchResponse with similar images (excluding the query image itself)
    """
    try:
        # Use fast search by image ID (bypasses image download and CLIP processing)
        search_results = faiss_service.search_by_image_id(
            image_id=image_id,
            top_k=top_k,
            exclude_self=True
        )

        if not search_results:
            return SearchResponse(
                results=[],
                detected_labels=[],
                detected_objects=[],
                summary="No similar images found.",
                follow_up_suggestions=[],
                query_type="similar_search",
                total_results=0
            )

        # Get presigned URLs for results (async for better performance)
        # Handle both old format (s3_url) and new format (s3_key)
        s3_keys = []
        for meta, _ in search_results:
            if 's3_key' in meta:
                s3_key = meta['s3_key']
                # Skip local images
                if not s3_key.startswith('local/'):
                    s3_keys.append(s3_key)
            elif 's3_url' in meta:
                # Extract s3_key from presigned URL (backward compatibility)
                s3_url = meta['s3_url']
                if 's3.amazonaws.com/' in s3_url:
                    # Parse: https://bucket.s3.amazonaws.com/key?params -> key
                    s3_key = s3_url.split('s3.amazonaws.com/')[-1].split('?')[0]
                    s3_keys.append(s3_key)

        urls = await s3_service.get_presigned_urls_async(s3_keys) if s3_keys else {}

        # Build result objects
        results = []
        for meta, score in search_results:
            # Get s3_key from metadata (handle both formats)
            s3_key = meta.get('s3_key')
            if not s3_key and 's3_url' in meta:
                s3_url = meta['s3_url']
                if 's3.amazonaws.com/' in s3_url:
                    s3_key = s3_url.split('s3.amazonaws.com/')[-1].split('?')[0]
                else:
                    s3_key = s3_url  # Fallback

            # Get URL (either from generated presigned URLs or use stored s3_url)
            url = urls.get(s3_key, '') if s3_key else ''
            if not url and 's3_url' in meta:
                url = meta['s3_url']  # Use stored URL as fallback

            results.append(ImageMetadata(
                image_id=meta['image_id'],
                s3_key=s3_key or '',
                url=url,
                score=score,
                labels=meta.get('labels')
            ))

        return SearchResponse(
            results=results,
            detected_labels=[],
            detected_objects=[],
            summary=f"Found {len(results)} similar images",
            follow_up_suggestions=[],
            query_type="similar_search",
            total_results=len(results)
        )

    except Exception as e:
        logger.error(f"Similar search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/index", response_model=IndexResponse)
async def index_images(request: IndexRequest):
    """
    Index images from S3 into FAISS.

    This endpoint downloads images from S3, generates embeddings,
    and adds them to the FAISS index.
    """
    indexed_count = 0
    failed_count = 0
    failed_keys = []

    for s3_key in request.s3_keys:
        try:
            # Download image from S3
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

            # Add to index
            import numpy as np
            faiss_service.add_embeddings(
                np.array([embedding]),
                [metadata]
            )

            indexed_count += 1

        except Exception as e:
            logger.error(f"Failed to index {s3_key}: {e}")
            failed_count += 1
            failed_keys.append(s3_key)

    # Save index after batch processing
    if indexed_count > 0:
        faiss_service.save_index()

    return IndexResponse(
        indexed_count=indexed_count,
        failed_count=failed_count,
        failed_keys=failed_keys,
        message=f"Successfully indexed {indexed_count} images"
    )


@router.delete("/index/{image_id}")
async def remove_from_index(image_id: str):
    """Remove an image from the FAISS index."""
    try:
        faiss_service.remove_by_ids([image_id])
        faiss_service.save_index()
        return {"message": f"Removed {image_id} from index"}
    except Exception as e:
        logger.error(f"Error removing {image_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/index/stats")
async def get_index_stats():
    """Get statistics about the FAISS index."""
    return {
        "total_vectors": faiss_service.size,
        "embedding_dimension": settings.EMBEDDING_DIM,
        "index_loaded": faiss_service.is_loaded
    }


@router.get("/history", response_model=SearchHistoryResponse)
async def get_search_history(limit: int = 20):
    """
    Get recent search sessions.

    Args:
        limit: Maximum number of sessions to return (default 20, max 100)
    """
    # Ensure limit is reasonable
    limit = min(limit, 100)

    history_items = history_service.get_recent_sessions(limit=limit)
    total_count = history_service.get_count()

    return SearchHistoryResponse(
        history=history_items,
        total_count=total_count
    )


@router.get("/history/stats", response_model=SearchHistoryStats)
async def get_history_stats():
    """Get statistics about search history."""
    return history_service.get_stats()


@router.delete("/history/{session_id}")
async def delete_search_history(session_id: str):
    """
    Delete a specific session from history.

    Args:
        session_id: The ID of the session to delete
    """
    success = history_service.delete_session(session_id)

    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Session with ID {session_id} not found"
        )

    return {"message": f"Session {session_id} deleted successfully"}


@router.delete("/history")
async def clear_search_history():
    """Clear all search history."""
    count = history_service.clear_history()
    return {
        "message": f"Cleared {count} items from search history",
        "cleared_count": count
    }


@router.post("/history/session")
async def create_session(request: dict):
    """
    Create a new search session.
    """
    from app.models.schemas import UserType

    # Convert user_type string to enum
    user_type = request.get("user_type", "general")
    user_type_enum = UserType(user_type) if user_type in [ut.value for ut in UserType] else UserType.GENERAL

    session_id = history_service.create_session(
        query_type=request.get("query_type"),
        text_query=request.get("text_query"),
        image_filename=request.get("image_filename"),
        user_type=user_type_enum,
        detected_labels=request.get("detected_labels", []),
        detected_objects=request.get("detected_objects", []),
        results_count=request.get("results_count", 0),
        ai_summary=request.get("ai_summary", "")
    )

    return {"session_id": session_id}


@router.post("/history/session/{session_id}/add")
async def add_to_session(session_id: str, request: dict):
    """
    Add a refinement to an existing session.
    """
    success = history_service.add_to_session(
        session_id=session_id,
        user_message=request.get("user_message"),
        ai_response=request.get("ai_response"),
        query_type=request.get("query_type"),
        detected_labels=request.get("detected_labels", []),
        detected_objects=request.get("detected_objects", []),
        results_count=request.get("results_count", 0)
    )

    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Session with ID {session_id} not found"
        )

    return {"message": "Added to session successfully"}


@router.get("/favorites", response_model=FavoritesResponse)
async def get_favorites(limit: Optional[int] = None):
    """
    Get all favorited images.

    Args:
        limit: Maximum number of favorites to return
    """
    favorites = favorites_service.get_favorites(limit=limit)
    total_count = favorites_service.get_count()

    # Update URLs with fresh presigned URLs (async for better performance)
    s3_keys = [fav.s3_key for fav in favorites]
    if s3_keys:
        urls = await s3_service.get_presigned_urls_async(s3_keys)
        for fav in favorites:
            fav.url = urls.get(fav.s3_key, fav.url)

    return FavoritesResponse(
        favorites=favorites,
        total_count=total_count
    )


@router.post("/favorites")
async def add_favorite(request: dict):
    """
    Add an image to favorites.

    Request body should include:
        - image_id: ID of the image
        - s3_key: S3 key of the image
        - url: Presigned URL for the image
        - labels: Optional list of labels
        - objects: Optional list of objects
        - note: Optional user note
    """
    favorite_id = favorites_service.add_favorite(
        image_id=request.get("image_id"),
        s3_key=request.get("s3_key"),
        url=request.get("url"),
        labels=request.get("labels"),
        objects=request.get("objects"),
        note=request.get("note")
    )

    return {"favorite_id": favorite_id, "message": "Added to favorites"}


@router.delete("/favorites/{image_id}")
async def remove_favorite(image_id: str):
    """
    Remove an image from favorites.

    Args:
        image_id: The ID of the image to remove
    """
    success = favorites_service.remove_favorite(image_id)

    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Favorite with image ID {image_id} not found"
        )

    return {"message": f"Removed from favorites"}


@router.get("/favorites/check/{image_id}")
async def check_favorite(image_id: str):
    """
    Check if an image is favorited.

    Args:
        image_id: The ID of the image to check
    """
    is_favorited = favorites_service.is_favorited(image_id)
    return {"is_favorited": is_favorited}


@router.post("/favorites/check-batch")
async def check_favorites_batch(request: dict):
    """
    Check multiple images for favorite status in one request.
    This endpoint significantly improves performance by reducing
    multiple API calls to a single batch operation.

    Request body should include:
        - image_ids: List of image IDs to check

    Returns:
        Dictionary mapping each image_id to its favorite status (True/False)
    """
    image_ids = request.get("image_ids", [])
    if not image_ids:
        return {}

    favorites_status = favorites_service.batch_check_favorited(image_ids)
    return {"favorites": favorites_status}


@router.delete("/favorites")
async def clear_favorites():
    """Clear all favorites."""
    count = favorites_service.clear_favorites()
    return {
        "message": f"Cleared {count} favorites",
        "cleared_count": count
    }


# ===== ENHANCED INDEXING ENDPOINTS =====

@router.post("/index/enhanced")
@limiter.limit("10/hour")
async def index_image_enhanced(
    request: Request,
    image: UploadFile = File(...),
    s3_key: Optional[str] = Form(None),
    generate_summary: bool = Form(True)
):
    """
    Enhanced indexing endpoint that extracts comprehensive features.

    This extracts:
    - CLIP embeddings and labels
    - Color palette and statistics
    - Texture features (GLCM, LBP)
    - Material detection
    - Style classification
    - Scene detection
    - LLM summary

    Processing time: 3-10 seconds per image (offline, acceptable)
    Rate limit: 10 requests per hour per IP
    """
    try:
        # Validate file size
        content = await image.read()
        if len(content) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail="Image file too large. Maximum 10MB allowed."
            )

        # Load image
        await image.seek(0)
        image_bytes = await image.read()
        pil_image = load_image_from_bytes(image_bytes)

        # Process image through enhanced pipeline
        metadata = indexing_service.process_image(
            image=pil_image,
            s3_key=s3_key,
            generate_summary=generate_summary
        )

        return {
            "success": True,
            "image_id": metadata["image_id"],
            "processing_time": metadata["processing_time"],
            "metadata": {
                "primary_label": metadata["primary_label"],
                "primary_style": metadata["style"]["primary_style"],
                "primary_scene": metadata["style"]["primary_scene"],
                "dominant_color": metadata["color"]["dominant_color"],
                "primary_material": metadata["materials"]["primary_category"],
                "summary": metadata.get("summary"),
            },
            "timing_breakdown": metadata["timing"]
        }

    except Exception as e:
        logger.error(f"Enhanced indexing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/index/enhanced/batch")
async def index_batch_enhanced(request: dict):
    """
    Batch index images from S3 with enhanced features.

    Request body:
        - s3_keys: List of S3 keys to index
        - generate_summaries: Whether to generate LLM summaries (default: True)
    """
    s3_keys = request.get("s3_keys", [])
    generate_summaries = request.get("generate_summaries", True)

    if not s3_keys:
        raise HTTPException(status_code=400, detail="No s3_keys provided")

    if len(s3_keys) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 images per batch")

    try:
        # Download and process images
        images = []
        for s3_key in s3_keys:
            try:
                pil_image = s3_service.download_image(s3_key)
                images.append((pil_image, s3_key))
            except Exception as e:
                logger.error(f"Error downloading {s3_key}: {e}")
                continue

        # Process batch
        results = indexing_service.process_batch(
            images=images,
            generate_summaries=generate_summaries
        )

        return {
            "success": True,
            "processed_count": len(results),
            "failed_count": len(s3_keys) - len(results),
            "results": [
                {
                    "image_id": r["image_id"],
                    "processing_time": r["processing_time"]
                }
                for r in results
            ]
        }

    except Exception as e:
        logger.error(f"Batch indexing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ===== ENHANCED SEARCH ENDPOINTS =====

@router.post("/search/enhanced")
@limiter.limit("30/minute")
async def search_enhanced(
    request: Request,
    image: Optional[UploadFile] = File(None),
    text_query: Optional[str] = Form(None),
    use_filters: bool = Form(True),
    top_k: int = Form(10)
):
    """
    Enhanced search with hybrid vector + metadata filtering.

    Features:
    - Automatic filter extraction from text queries
      (e.g., "warm modern living room" → filters: warm color, modern style, living room scene)
    - Fast metadata pre-filtering in SQLite
    - FAISS vector search on filtered candidates only
    - Enriched results with full metadata

    Response time: ~100-200ms (much faster than full FAISS scan)
    Rate limit: 30 requests per minute per IP
    """
    if image is None and text_query is None:
        raise HTTPException(
            status_code=400,
            detail="Must provide either image, text_query, or both"
        )

    try:
        # Case 1: Text only
        if image is None and text_query is not None:
            results = enhanced_search_service.search_with_text(
                query=text_query,
                top_k=top_k,
                use_filters=use_filters
            )

        # Case 2: Image only
        elif image is not None and text_query is None:
            image_bytes = await image.read()
            pil_image = load_image_from_bytes(image_bytes)

            results = enhanced_search_service.search_with_image(
                image=pil_image,
                top_k=top_k,
                filters=None
            )

        # Case 3: Image + Text
        else:
            image_bytes = await image.read()
            pil_image = load_image_from_bytes(image_bytes)

            results = enhanced_search_service.search_with_image_and_text(
                image=pil_image,
                text=text_query,
                top_k=top_k,
                use_filters=use_filters
            )

        return {
            "success": True,
            "results": results,
            "total_results": len(results)
        }

    except Exception as e:
        logger.error(f"Enhanced search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ===== METADATA ENDPOINTS =====

@router.get("/metadata/{image_id}")
async def get_image_metadata_endpoint(image_id: str):
    """
    Get comprehensive metadata for an image.

    Returns all extracted features:
    - CLIP labels
    - Color palette and statistics
    - Texture features
    - Materials
    - Style and scene
    - Summary
    """
    metadata = metadata_db_service.get_image_metadata(image_id)

    if not metadata:
        raise HTTPException(status_code=404, detail=f"Image {image_id} not found")

    return {"image_id": image_id, "metadata": metadata}


@router.get("/metadata/stats")
async def get_metadata_stats():
    """
    Get database statistics.

    Returns:
    - Total images indexed
    - Top styles distribution
    - Top scenes distribution
    """
    stats = metadata_db_service.get_stats()
    return stats


@router.get("/metadata/search/filters")
async def search_by_filters_endpoint(
    color_temp: Optional[str] = None,
    brightness: Optional[str] = None,
    material_category: Optional[str] = None,
    texture_roughness: Optional[str] = None,
    style: Optional[str] = None,
    scene: Optional[str] = None,
    limit: int = 50
):
    """
    Search images by metadata filters only (no vector search).

    Useful for exploring the dataset by specific attributes.

    Query parameters:
    - color_temp: warm, cool, neutral
    - brightness: bright, medium, dark
    - material_category: wood, metal, fabric, leather, stone, glass, ceramic, plastic
    - texture_roughness: smooth, moderate, rough
    - style: modern, contemporary, minimalist, industrial, rustic, vintage, etc.
    - scene: living room, bedroom, kitchen, bathroom, etc.
    - limit: max results (default 50, max 200)
    """
    limit = min(limit, 200)

    image_ids = metadata_db_service.search_by_filters(
        color_temp=color_temp,
        brightness=brightness,
        material_category=material_category,
        texture_roughness=texture_roughness,
        style=style,
        scene=scene,
        limit=limit
    )

    # Get full metadata for each image
    results = []
    for image_id in image_ids:
        meta = metadata_db_service.get_image_metadata(image_id)
        if meta:
            results.append({
                "image_id": image_id,
                "s3_url": meta.get("s3_url"),
                "primary_style": meta.get("primary_style"),
                "primary_scene": meta.get("primary_scene"),
                "dominant_color": meta.get("dominant_color"),
                "primary_material": meta.get("primary_material_category"),
            })

    return {
        "results": results,
        "total_results": len(results)
    }