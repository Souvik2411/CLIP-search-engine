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
    """Service for managing search history as conversation sessions (like ChatGPT)."""

    def __init__(self):
        self.settings = get_settings()
        self.history_file = Path("data/history/search_sessions.json")
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        self._sessions: List[dict] = []
        self._load_history()

    def _load_history(self):
        """Load search sessions from disk."""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    self._sessions = json.load(f)
                logger.info(f"Loaded {len(self._sessions)} search sessions")
            except Exception as e:
                logger.error(f"Error loading search history: {e}")
                self._sessions = []
        else:
            self._sessions = []
            logger.info("No existing search history found, starting fresh")

    def _save_history(self):
        """Save search sessions to disk."""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self._sessions, f, indent=2, default=str)
            logger.info(f"Saved {len(self._sessions)} search sessions")
        except Exception as e:
            logger.error(f"Error saving search history: {e}")

    def _generate_title(
        self,
        text_query: Optional[str],
        detected_labels: List[str],
        detected_objects: List[str]
    ) -> str:
        """Generate a descriptive title for the session."""
        if text_query:
            # Use first 50 chars of text query
            return text_query[:50] + ("..." if len(text_query) > 50 else "")

        # Use detected objects or labels
        if detected_objects:
            if len(detected_objects) == 1:
                return f"Search for {detected_objects[0]}"
            else:
                return f"Search for {detected_objects[0]} and more"

        if detected_labels:
            if len(detected_labels) == 1:
                return f"{detected_labels[0]} search"
            else:
                return f"{detected_labels[0]} and more"

        return "Image search"

    def create_session(
        self,
        query_type: str,
        text_query: Optional[str],
        image_filename: Optional[str],
        user_type: UserType,
        detected_labels: List[str],
        detected_objects: List[str],
        results_count: int,
        ai_summary: str
    ) -> str:
        """
        Create a new search session.

        Args:
            query_type: Type of query (image_only, text_only, image_and_text)
            text_query: Text query if provided
            image_filename: Original filename if image was uploaded
            user_type: User type for the session
            detected_labels: Architectural labels detected
            detected_objects: Objects detected
            results_count: Number of results returned
            ai_summary: AI-generated summary

        Returns:
            Session ID
        """
        session_id = str(uuid.uuid4())
        now = datetime.now()

        # Generate title
        title = self._generate_title(text_query, detected_labels, detected_objects)

        # Create initial conversation with assistant summary
        conversation = [
            {
                "role": "assistant",
                "content": ai_summary,
                "timestamp": now.isoformat(),
                "query_type": None,
                "detected_labels": detected_labels,
                "detected_objects": detected_objects,
                "results_count": results_count
            }
        ]

        session = {
            "id": session_id,
            "title": title,
            "timestamp": now.isoformat(),
            "last_updated": now.isoformat(),
            "initial_query_type": query_type,
            "initial_text_query": text_query,
            "initial_image_filename": image_filename,
            "user_type": user_type.value if isinstance(user_type, UserType) else user_type,
            "conversation": conversation,
            "total_results_returned": results_count
        }

        # Add to beginning of list (most recent first)
        self._sessions.insert(0, session)

        # Keep only last 50 sessions
        if len(self._sessions) > 50:
            self._sessions = self._sessions[:50]

        self._save_history()
        logger.info(f"Created new search session: {session_id} - {title}")

        return session_id

    def add_to_session(
        self,
        session_id: str,
        user_message: str,
        ai_response: str,
        query_type: str,
        detected_labels: List[str],
        detected_objects: List[str],
        results_count: int
    ) -> bool:
        """
        Add a refinement to an existing session.

        Args:
            session_id: The session to add to
            user_message: User's refinement query
            ai_response: AI's response summary
            query_type: Type of refinement query
            detected_labels: Labels detected in refinement
            detected_objects: Objects detected in refinement
            results_count: Number of results from refinement

        Returns:
            True if successful, False if session not found
        """
        for session in self._sessions:
            if session["id"] == session_id:
                now = datetime.now()

                # Add user message
                session["conversation"].append({
                    "role": "user",
                    "content": user_message,
                    "timestamp": now.isoformat(),
                    "query_type": query_type,
                    "detected_labels": [],
                    "detected_objects": [],
                    "results_count": None
                })

                # Add AI response
                session["conversation"].append({
                    "role": "assistant",
                    "content": ai_response,
                    "timestamp": now.isoformat(),
                    "query_type": None,
                    "detected_labels": detected_labels,
                    "detected_objects": detected_objects,
                    "results_count": results_count
                })

                # Update metadata
                session["last_updated"] = now.isoformat()
                session["total_results_returned"] += results_count

                # Move session to top (most recent)
                self._sessions.remove(session)
                self._sessions.insert(0, session)

                self._save_history()
                logger.info(f"Added refinement to session: {session_id}")
                return True

        logger.warning(f"Session not found: {session_id}")
        return False

    def get_recent_sessions(self, limit: int = 20) -> List[SearchHistoryItem]:
        """
        Get recent search sessions.

        Args:
            limit: Maximum number of sessions to return

        Returns:
            List of SearchHistoryItem objects
        """
        sessions = self._sessions[:limit]
        return [SearchHistoryItem(**session) for session in sessions]

    def get_session_by_id(self, session_id: str) -> Optional[SearchHistoryItem]:
        """
        Get a specific session by ID.

        Args:
            session_id: The session ID to retrieve

        Returns:
            SearchHistoryItem if found, None otherwise
        """
        for session in self._sessions:
            if session["id"] == session_id:
                return SearchHistoryItem(**session)
        return None

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a specific session.

        Args:
            session_id: The session ID to delete

        Returns:
            True if deleted, False if not found
        """
        for i, session in enumerate(self._sessions):
            if session["id"] == session_id:
                self._sessions.pop(i)
                self._save_history()
                logger.info(f"Deleted session: {session_id}")
                return True
        return False

    def clear_history(self) -> int:
        """
        Clear all search history.

        Returns:
            Number of sessions cleared
        """
        count = len(self._sessions)
        self._sessions = []
        self._save_history()
        logger.info(f"Cleared {count} sessions from history")
        return count

    def get_stats(self) -> SearchHistoryStats:
        """
        Get statistics about search history.

        Returns:
            SearchHistoryStats object with aggregated data
        """
        total = len(self._sessions)
        image_only = sum(1 for s in self._sessions if s["initial_query_type"] == "image_only")
        text_only = sum(1 for s in self._sessions if s["initial_query_type"] == "text_only")
        combined = sum(1 for s in self._sessions if s["initial_query_type"] == "image_and_text")

        # Count label frequencies across all sessions
        label_counts = {}
        for session in self._sessions:
            for msg in session.get("conversation", []):
                for label in msg.get("detected_labels", []):
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
        """Get total number of sessions."""
        return len(self._sessions)
