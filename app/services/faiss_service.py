import faiss
import numpy as np
import json
from pathlib import Path
from typing import Optional
import logging

from app.config import get_settings

logger = logging.getLogger(__name__)


class FAISSService:
    """Service for FAISS vector index operations."""

    def __init__(self):
        self.settings = get_settings()
        self.index: Optional[faiss.IndexFlatIP] = None
        self.metadata: list[dict] = []
        self._loaded = False

    def create_index(self, dimension: int = 512):
        """
        Create a new FAISS index.

        Args:
            dimension: Embedding dimension (default 512 for CLIP ViT-B/32)
        """
        # Using IndexFlatIP for inner product (cosine similarity with normalized vectors)
        self.index = faiss.IndexFlatIP(dimension)
        self.metadata = []
        self._loaded = True
        logger.info(f"Created new FAISS index with dimension {dimension}")

    def load_index(self):
        """Load FAISS index and metadata from disk."""
        index_path = Path(self.settings.FAISS_INDEX_PATH)
        metadata_path = Path(self.settings.FAISS_METADATA_PATH)

        if not index_path.exists():
            logger.warning(f"Index file not found at {index_path}, creating new index")
            self.create_index(self.settings.EMBEDDING_DIM)
            return

        self.index = faiss.read_index(str(index_path))
        logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")

        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            logger.info(f"Loaded metadata for {len(self.metadata)} items")
        else:
            self.metadata = []
            logger.warning("Metadata file not found")

        self._loaded = True

    def save_index(self):
        """Save FAISS index and metadata to disk."""
        if self.index is None:
            logger.error("No index to save")
            return

        index_path = Path(self.settings.FAISS_INDEX_PATH)
        metadata_path = Path(self.settings.FAISS_METADATA_PATH)

        # Ensure directory exists
        index_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(index_path))
        logger.info(f"Saved FAISS index to {index_path}")

        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f)
        logger.info(f"Saved metadata to {metadata_path}")

    def add_embeddings(
        self,
        embeddings: np.ndarray,
        metadata_list: list[dict]
    ):
        """
        Add embeddings to the index.

        Args:
            embeddings: Array of shape (n, dimension)
            metadata_list: List of metadata dicts for each embedding
        """
        if not self._loaded:
            self.load_index()

        if self.index is None:
            self.create_index(embeddings.shape[1])

        # Ensure embeddings are float32 and contiguous
        embeddings = np.ascontiguousarray(embeddings.astype(np.float32))

        self.index.add(embeddings)
        self.metadata.extend(metadata_list)

        logger.info(f"Added {len(metadata_list)} embeddings to index")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10
    ) -> list[tuple[dict, float]]:
        """
        Search for similar embeddings.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return

        Returns:
            List of (metadata, score) tuples
        """
        if not self._loaded:
            self.load_index()

        if self.index is None or self.index.ntotal == 0:
            logger.warning("Index is empty")
            return []

        # Ensure query is proper shape and type
        query = np.ascontiguousarray(
            query_embedding.reshape(1, -1).astype(np.float32)
        )

        # Limit top_k to available vectors
        k = min(top_k, self.index.ntotal)

        # Search
        scores, indices = self.index.search(query, k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0 and idx < len(self.metadata):
                results.append((self.metadata[idx], float(scores[0][i])))

        return results

    def get_embedding_by_image_id(self, image_id: str) -> Optional[np.ndarray]:
        """
        Get the embedding for a specific image_id.

        Args:
            image_id: The image ID to lookup

        Returns:
            The embedding vector if found, None otherwise
        """
        if not self._loaded:
            self.load_index()

        if self.index is None or self.index.ntotal == 0:
            return None

        # Find the index of this image_id in metadata
        for idx, meta in enumerate(self.metadata):
            if meta.get('image_id') == image_id:
                # Reconstruct the embedding from FAISS index
                embedding = self.index.reconstruct(idx)
                return embedding

        logger.warning(f"Image ID {image_id} not found in index")
        return None

    def search_by_image_id(
        self,
        image_id: str,
        top_k: int = 10,
        exclude_self: bool = True
    ) -> list[tuple[dict, float]]:
        """
        Search for similar images using an existing image's embedding.
        This is much faster than re-uploading and processing an image.

        Args:
            image_id: The image ID to use as query
            top_k: Number of results to return
            exclude_self: Whether to exclude the query image from results

        Returns:
            List of (metadata, score) tuples
        """
        # Get the embedding for this image
        embedding = self.get_embedding_by_image_id(image_id)

        if embedding is None:
            logger.error(f"Could not find embedding for image_id: {image_id}")
            return []

        # Search with this embedding
        # Request one extra result if we're excluding self
        k = top_k + 1 if exclude_self else top_k
        results = self.search(embedding, top_k=k)

        # Exclude the query image itself if requested
        if exclude_self and results:
            results = [r for r in results if r[0].get('image_id') != image_id]
            results = results[:top_k]  # Ensure we return exactly top_k

        return results

    def remove_by_ids(self, image_ids: list[str]):
        """
        Remove embeddings by image IDs.
        Note: This requires rebuilding the index.

        Args:
            image_ids: List of image IDs to remove
        """
        if not self._loaded:
            self.load_index()

        ids_to_remove = set(image_ids)

        # Filter metadata and get indices to keep
        new_metadata = []
        indices_to_keep = []

        for i, meta in enumerate(self.metadata):
            if meta.get('image_id') not in ids_to_remove:
                new_metadata.append(meta)
                indices_to_keep.append(i)

        if len(indices_to_keep) == len(self.metadata):
            logger.warning("No matching IDs found to remove")
            return

        # Rebuild index with remaining vectors
        if indices_to_keep:
            # Get vectors to keep
            vectors = np.array([
                self.index.reconstruct(i) for i in indices_to_keep
            ]).astype(np.float32)

            # Create new index
            self.create_index(self.settings.EMBEDDING_DIM)
            self.index.add(vectors)
            self.metadata = new_metadata

            logger.info(f"Removed {len(ids_to_remove)} items, {len(new_metadata)} remaining")
        else:
            self.create_index(self.settings.EMBEDDING_DIM)
            self.metadata = []
            logger.info("All items removed, index is now empty")

    @property
    def is_loaded(self) -> bool:
        """Check if index is loaded."""
        return self._loaded

    @property
    def size(self) -> int:
        """Get number of vectors in index."""
        if self.index is None:
            return 0
        return self.index.ntotal