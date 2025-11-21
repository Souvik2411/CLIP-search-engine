import json
from pathlib import Path
from typing import Optional, List
from datetime import datetime
import logging
import uuid

from app.config import get_settings
from app.models.schemas import FavoriteItem

logger = logging.getLogger(__name__)


class FavoritesService:
    """Service for managing user's favorite images."""

    def __init__(self):
        self.settings = get_settings()
        self.favorites_file = Path("data/favorites/favorites.json")
        self.favorites_file.parent.mkdir(parents=True, exist_ok=True)
        self._favorites: List[dict] = []
        self._load_favorites()

    def _load_favorites(self):
        """Load favorites from disk."""
        if self.favorites_file.exists():
            try:
                with open(self.favorites_file, 'r') as f:
                    self._favorites = json.load(f)
                logger.info(f"Loaded {len(self._favorites)} favorites")
            except Exception as e:
                logger.error(f"Error loading favorites: {e}")
                self._favorites = []
        else:
            self._favorites = []
            logger.info("No existing favorites found, starting fresh")

    def _save_favorites(self):
        """Save favorites to disk."""
        try:
            with open(self.favorites_file, 'w') as f:
                json.dump(self._favorites, f, indent=2, default=str)
            logger.info(f"Saved {len(self._favorites)} favorites")
        except Exception as e:
            logger.error(f"Error saving favorites: {e}")

    def add_favorite(
        self,
        image_id: str,
        s3_key: str,
        url: str,
        labels: Optional[List[str]] = None,
        objects: Optional[List[str]] = None,
        note: Optional[str] = None
    ) -> str:
        """
        Add an image to favorites.

        Args:
            image_id: ID of the image
            s3_key: S3 key of the image
            url: Presigned URL for the image
            labels: Architectural labels for the image
            objects: Objects detected in the image
            note: Optional user note

        Returns:
            Favorite ID
        """
        # Check if already favorited
        for fav in self._favorites:
            if fav["image_id"] == image_id:
                logger.info(f"Image {image_id} already favorited")
                return fav["id"]

        favorite_id = str(uuid.uuid4())
        now = datetime.now()

        favorite = {
            "id": favorite_id,
            "image_id": image_id,
            "s3_key": s3_key,
            "url": url,
            "timestamp": now.isoformat(),
            "labels": labels or [],
            "objects": objects or [],
            "note": note
        }

        # Add to beginning of list (most recent first)
        self._favorites.insert(0, favorite)

        self._save_favorites()
        logger.info(f"Added favorite: {favorite_id} for image {image_id}")

        return favorite_id

    def remove_favorite(self, image_id: str) -> bool:
        """
        Remove an image from favorites.

        Args:
            image_id: The image ID to remove

        Returns:
            True if removed, False if not found
        """
        for i, fav in enumerate(self._favorites):
            if fav["image_id"] == image_id:
                self._favorites.pop(i)
                self._save_favorites()
                logger.info(f"Removed favorite for image: {image_id}")
                return True
        return False

    def get_favorites(self, limit: Optional[int] = None) -> List[FavoriteItem]:
        """
        Get all favorites or a limited number.

        Args:
            limit: Maximum number of favorites to return

        Returns:
            List of FavoriteItem objects
        """
        favorites = self._favorites[:limit] if limit else self._favorites
        return [FavoriteItem(**fav) for fav in favorites]

    def is_favorited(self, image_id: str) -> bool:
        """
        Check if an image is favorited.

        Args:
            image_id: The image ID to check

        Returns:
            True if favorited, False otherwise
        """
        return any(fav["image_id"] == image_id for fav in self._favorites)

    def batch_check_favorited(self, image_ids: List[str]) -> dict[str, bool]:
        """
        Check multiple images for favorite status in one operation.

        Args:
            image_ids: List of image IDs to check

        Returns:
            Dictionary mapping image_id to favorite status (True/False)
        """
        # Create a set of favorited image IDs for O(1) lookup
        favorited_set = {fav["image_id"] for fav in self._favorites}

        # Return status for each requested image_id
        return {image_id: (image_id in favorited_set) for image_id in image_ids}

    def get_count(self) -> int:
        """Get total number of favorites."""
        return len(self._favorites)

    def clear_favorites(self) -> int:
        """
        Clear all favorites.

        Returns:
            Number of favorites cleared
        """
        count = len(self._favorites)
        self._favorites = []
        self._save_favorites()
        logger.info(f"Cleared {count} favorites")
        return count

    def update_note(self, image_id: str, note: str) -> bool:
        """
        Update the note for a favorite.

        Args:
            image_id: The image ID
            note: New note text

        Returns:
            True if updated, False if not found
        """
        for fav in self._favorites:
            if fav["image_id"] == image_id:
                fav["note"] = note
                self._save_favorites()
                logger.info(f"Updated note for favorite: {image_id}")
                return True
        return False
