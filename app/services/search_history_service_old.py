import json
from pathlib import Path
from typing import Optional, List
from datetime import datetime
import logging
import uuid

from app.config import get_settings
from app.models.schemas import SearchHistoryItem, SearchHistoryStats, UserType, ConversationMessage

logger = logging.getLogger(__name__)


class SearchHistoryService:
    """Service for managing search history."""

    def __init__(self):
        self.settings = get_settings()
        self.history_file = Path("data/history/search_history.json")
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        self._history: List[dict] = []
        self._load_history()

    def _load_history(self):
        """Load search history from disk."""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    self._history = json.load(f)
                logger.info(f"Loaded {len(self._history)} search history items")
            except Exception as e:
                logger.error(f"Error loading search history: {e}")
                self._history = []
        else:
            self._history = []
            logger.info("No existing search history found, starting fresh")

    def _save_history(self):
        """Save search history to disk."""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self._history, f, indent=2, default=str)
            logger.info(f"Saved {len(self._history)} search history items")
        except Exception as e:
            logger.error(f"Error saving search history: {e}")

    def add_search(
        self,
        query_type: str,
        text_query: Optional[str],
        image_filename: Optional[str],
        user_type: UserType,
        detected_labels: List[str],
        detected_objects: List[str],
        results_count: int,
        top_result_url: Optional[str] = None
    ) -> str:
        """
        Add a new search to history.

        Args:
            query_type: Type of query (image_only, text_only, image_and_text)
            text_query: Text query if provided
            image_filename: Original filename if image was uploaded
            user_type: User type for the search
            detected_labels: Architectural labels detected from input
            detected_objects: Objects detected from input
            results_count: Number of results returned
            top_result_url: URL of the top result

        Returns:
            ID of the created history item
        """
        search_id = str(uuid.uuid4())

        history_item = {
            "id": search_id,
            "timestamp": datetime.now().isoformat(),
            "query_type": query_type,
            "text_query": text_query,
            "image_filename": image_filename,
            "user_type": user_type.value if isinstance(user_type, UserType) else user_type,
            "detected_labels": detected_labels,
            "detected_objects": detected_objects,
            "results_count": results_count,
            "top_result_url": top_result_url
        }

        # Add to beginning of list (most recent first)
        self._history.insert(0, history_item)

        # Keep only last 100 searches to prevent unlimited growth
        if len(self._history) > 100:
            self._history = self._history[:100]

        self._save_history()
        logger.info(f"Added search to history: {search_id}")

        return search_id

    def get_recent_searches(self, limit: int = 20) -> List[SearchHistoryItem]:
        """
        Get recent searches.

        Args:
            limit: Maximum number of items to return (default 20)

        Returns:
            List of SearchHistoryItem objects
        """
        items = self._history[:limit]
        return [SearchHistoryItem(**item) for item in items]

    def get_all_searches(self) -> List[SearchHistoryItem]:
        """
        Get all search history.

        Returns:
            List of all SearchHistoryItem objects
        """
        return [SearchHistoryItem(**item) for item in self._history]

    def get_search_by_id(self, search_id: str) -> Optional[SearchHistoryItem]:
        """
        Get a specific search by ID.

        Args:
            search_id: The search ID to retrieve

        Returns:
            SearchHistoryItem if found, None otherwise
        """
        for item in self._history:
            if item["id"] == search_id:
                return SearchHistoryItem(**item)
        return None

    def delete_search(self, search_id: str) -> bool:
        """
        Delete a specific search from history.

        Args:
            search_id: The search ID to delete

        Returns:
            True if deleted, False if not found
        """
        for i, item in enumerate(self._history):
            if item["id"] == search_id:
                self._history.pop(i)
                self._save_history()
                logger.info(f"Deleted search from history: {search_id}")
                return True
        return False

    def clear_history(self) -> int:
        """
        Clear all search history.

        Returns:
            Number of items cleared
        """
        count = len(self._history)
        self._history = []
        self._save_history()
        logger.info(f"Cleared {count} items from search history")
        return count

    def get_stats(self) -> SearchHistoryStats:
        """
        Get statistics about search history.

        Returns:
            SearchHistoryStats object with aggregated data
        """
        total = len(self._history)
        image_only = sum(1 for item in self._history if item["query_type"] == "image_only")
        text_only = sum(1 for item in self._history if item["query_type"] == "text_only")
        combined = sum(1 for item in self._history if item["query_type"] == "image_and_text")

        # Count label frequencies
        label_counts = {}
        for item in self._history:
            for label in item.get("detected_labels", []):
                label_counts[label] = label_counts.get(label, 0) + 1

        # Get top 10 most common labels
        most_common = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        return SearchHistoryStats(
            total_searches=total,
            image_only_searches=image_only,
            text_only_searches=text_only,
            combined_searches=combined,
            most_common_labels=most_common
        )

    def get_count(self) -> int:
        """Get total number of history items."""
        return len(self._history)
