import numpy as np
from PIL import Image
from typing import Optional, List, Dict, Any
import logging
import re

from app.services.clip_service import CLIPService
from app.services.faiss_service import FAISSService
from app.services.metadata_db_service import metadata_db_service

logger = logging.getLogger(__name__)


class EnhancedSearchService:
    """
    Enhanced search service with hybrid vector + metadata filtering.

    Search Strategy:
    1. Parse query for filter keywords (color, material, style, etc.)
    2. Filter by metadata in SQLite (fast) -> get candidate image IDs
    3. Perform FAISS vector search on candidates only
    4. Return top-k results with enriched metadata
    """

    def __init__(self, clip_service: CLIPService, faiss_service: FAISSService):
        self.clip_service = clip_service
        self.faiss_service = faiss_service

        # Filter keyword mappings
        self.COLOR_KEYWORDS = {
            "warm": "warm",
            "cool": "cool",
            "neutral": "neutral",
            "bright": "bright",
            "dark": "dark",
            "vibrant": "vibrant",
            "muted": "muted",
        }

        self.MATERIAL_KEYWORDS = {
            "wood": "wood",
            "wooden": "wood",
            "metal": "metal",
            "metallic": "metal",
            "fabric": "fabric",
            "leather": "leather",
            "stone": "stone",
            "marble": "stone",
            "glass": "glass",
            "ceramic": "ceramic",
            "plastic": "plastic",
        }

        self.TEXTURE_KEYWORDS = {
            "smooth": "smooth",
            "rough": "rough",
            "textured": "moderate",
        }

        self.STYLE_KEYWORDS = {
            "modern": "modern",
            "contemporary": "contemporary",
            "minimalist": "minimalist",
            "industrial": "industrial",
            "rustic": "rustic",
            "vintage": "vintage",
            "scandinavian": "scandinavian",
            "bohemian": "bohemian",
        }

        self.SCENE_KEYWORDS = {
            "living room": "living room",
            "bedroom": "bedroom",
            "kitchen": "kitchen",
            "bathroom": "bathroom",
            "dining room": "dining room",
            "office": "office",
        }

    def parse_query_filters(self, query: str) -> Dict[str, Optional[str]]:
        """
        Extract filter keywords from natural language query.

        Args:
            query: User query string

        Returns:
            Dictionary with extracted filters
        """
        query_lower = query.lower()

        filters = {
            "color_temp": None,
            "brightness": None,
            "material_category": None,
            "texture_roughness": None,
            "style": None,
            "scene": None,
        }

        # Check for color keywords
        for keyword, value in self.COLOR_KEYWORDS.items():
            if keyword in query_lower:
                if value in ["warm", "cool", "neutral"]:
                    filters["color_temp"] = value
                elif value in ["bright", "dark"]:
                    filters["brightness"] = value
                break

        # Check for material keywords
        for keyword, value in self.MATERIAL_KEYWORDS.items():
            if keyword in query_lower:
                filters["material_category"] = value
                break

        # Check for texture keywords
        for keyword, value in self.TEXTURE_KEYWORDS.items():
            if keyword in query_lower:
                filters["texture_roughness"] = value
                break

        # Check for style keywords
        for keyword, value in self.STYLE_KEYWORDS.items():
            if keyword in query_lower:
                filters["style"] = value
                break

        # Check for scene keywords
        for keyword, value in self.SCENE_KEYWORDS.items():
            if keyword in query_lower:
                filters["scene"] = value
                break

        # Log extracted filters
        active_filters = {k: v for k, v in filters.items() if v is not None}
        if active_filters:
            logger.info(f"Extracted filters: {active_filters}")

        return filters

    def search_with_text(
        self,
        query: str,
        top_k: int = 10,
        use_filters: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search with text query using hybrid approach.

        Args:
            query: Text query
            top_k: Number of results to return
            use_filters: Whether to apply metadata filters

        Returns:
            List of results with metadata
        """
        logger.info(f"Searching for: '{query}' (top_k={top_k}, filters={use_filters})")

        # Step 1: Parse query for filters
        filters = self.parse_query_filters(query) if use_filters else {}

        # Step 2: Get candidate image IDs from metadata filters
        candidate_ids = None
        if use_filters and any(filters.values()):
            candidate_ids = metadata_db_service.search_by_filters(
                color_temp=filters.get("color_temp"),
                brightness=filters.get("brightness"),
                material_category=filters.get("material_category"),
                texture_roughness=filters.get("texture_roughness"),
                style=filters.get("style"),
                scene=filters.get("scene"),
                limit=500  # Get more candidates for better results
            )

            logger.info(f"Found {len(candidate_ids)} candidates matching filters")

            if len(candidate_ids) == 0:
                logger.warning("No images match the filters")
                return []

        # Step 3: Get query embedding
        query_embedding = self.clip_service.get_text_embedding(query)

        # Step 4: Search FAISS
        if candidate_ids:
            # Filter FAISS search to candidates
            results = self._search_with_candidates(
                query_embedding,
                candidate_ids,
                top_k
            )
        else:
            # Full FAISS search (no filters)
            results = self.faiss_service.search(query_embedding, top_k=top_k)

        # Step 5: Enrich results with metadata
        enriched_results = self._enrich_results(results)

        logger.info(f"Returning {len(enriched_results)} results")

        return enriched_results

    def search_with_image(
        self,
        image: Image.Image,
        top_k: int = 10,
        filters: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search with image query.

        Args:
            image: PIL Image object
            top_k: Number of results to return
            filters: Optional manual filters

        Returns:
            List of results with metadata
        """
        logger.info(f"Searching with image (top_k={top_k})")

        # Step 1: Get candidate image IDs from filters (if provided)
        candidate_ids = None
        if filters and any(filters.values()):
            candidate_ids = metadata_db_service.search_by_filters(
                color_temp=filters.get("color_temp"),
                brightness=filters.get("brightness"),
                material_category=filters.get("material_category"),
                texture_roughness=filters.get("texture_roughness"),
                style=filters.get("style"),
                scene=filters.get("scene"),
                limit=500
            )

            logger.info(f"Found {len(candidate_ids)} candidates matching filters")

            if len(candidate_ids) == 0:
                return []

        # Step 2: Get image embedding
        image_embedding = self.clip_service.get_image_embedding(image)

        # Step 3: Search FAISS
        if candidate_ids:
            results = self._search_with_candidates(
                image_embedding,
                candidate_ids,
                top_k
            )
        else:
            results = self.faiss_service.search(image_embedding, top_k=top_k)

        # Step 4: Enrich results
        enriched_results = self._enrich_results(results)

        return enriched_results

    def search_with_image_and_text(
        self,
        image: Image.Image,
        text: str,
        top_k: int = 10,
        image_weight: float = 0.7,
        use_filters: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search with both image and text (fusion).

        Args:
            image: PIL Image object
            text: Text query
            top_k: Number of results
            image_weight: Weight for image embedding (0-1)
            use_filters: Whether to parse text for filters

        Returns:
            List of results with metadata
        """
        logger.info(f"Searching with image + text: '{text}' (weight={image_weight})")

        # Step 1: Parse text for filters
        filters = self.parse_query_filters(text) if use_filters else {}

        # Step 2: Get candidates
        candidate_ids = None
        if use_filters and any(filters.values()):
            candidate_ids = metadata_db_service.search_by_filters(
                color_temp=filters.get("color_temp"),
                brightness=filters.get("brightness"),
                material_category=filters.get("material_category"),
                texture_roughness=filters.get("texture_roughness"),
                style=filters.get("style"),
                scene=filters.get("scene"),
                limit=500
            )

            if len(candidate_ids) == 0:
                return []

        # Step 3: Get embeddings
        image_embedding = self.clip_service.get_image_embedding(image)
        text_embedding = self.clip_service.get_text_embedding(text)

        # Step 4: Fuse embeddings
        fused_embedding = (
            image_weight * image_embedding +
            (1 - image_weight) * text_embedding
        )

        # Normalize
        fused_embedding = fused_embedding / np.linalg.norm(fused_embedding)

        # Step 5: Search
        if candidate_ids:
            results = self._search_with_candidates(
                fused_embedding,
                candidate_ids,
                top_k
            )
        else:
            results = self.faiss_service.search(fused_embedding, top_k=top_k)

        # Step 6: Enrich
        enriched_results = self._enrich_results(results)

        return enriched_results

    def _search_with_candidates(
        self,
        query_embedding: np.ndarray,
        candidate_ids: List[str],
        top_k: int
    ) -> List[tuple[Dict[str, Any], float]]:
        """
        Search FAISS but only consider candidate image IDs.

        Args:
            query_embedding: Query vector
            candidate_ids: List of candidate image IDs
            top_k: Number of results

        Returns:
            List of (metadata, score) tuples
        """
        # Search FAISS with more results
        all_results = self.faiss_service.search(
            query_embedding,
            top_k=min(100, self.faiss_service.size)
        )

        # Filter to candidates only
        candidate_set = set(candidate_ids)
        filtered_results = [
            (meta, score) for meta, score in all_results
            if meta.get("image_id") in candidate_set
        ]

        # Return top-k
        return filtered_results[:top_k]

    def _enrich_results(
        self,
        results: List[tuple[Dict[str, Any], float]]
    ) -> List[Dict[str, Any]]:
        """
        Enrich FAISS results with full metadata from SQLite.

        Args:
            results: List of (faiss_metadata, score) tuples

        Returns:
            List of enriched result dictionaries
        """
        enriched = []

        for faiss_meta, score in results:
            image_id = faiss_meta.get("image_id")

            # Get full metadata from SQLite
            full_meta = metadata_db_service.get_image_metadata(image_id)

            if full_meta:
                enriched.append({
                    "image_id": image_id,
                    "s3_url": faiss_meta.get("s3_url"),
                    "similarity_score": float(score),
                    "metadata": {
                        "primary_label": full_meta.get("primary_label"),
                        "summary": full_meta.get("summary"),
                        "color": {
                            "dominant_color": full_meta.get("dominant_color"),
                            "temperature": full_meta.get("color_temperature"),
                            "brightness": full_meta.get("brightness_category"),
                        },
                        "materials": {
                            "primary": full_meta.get("primary_material"),
                            "category": full_meta.get("primary_material_category"),
                        },
                        "style": {
                            "primary": full_meta.get("primary_style"),
                            "scene": full_meta.get("primary_scene"),
                        },
                        "texture": {
                            "roughness": full_meta.get("texture_roughness"),
                            "pattern": full_meta.get("texture_pattern"),
                        }
                    }
                })

        return enriched


# Note: Will be initialized in routes with clip_service and faiss_service
# enhanced_search_service = EnhancedSearchService(clip_service, faiss_service)
