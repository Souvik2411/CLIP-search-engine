from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum
from datetime import datetime


class UserType(str, Enum):
    """User type for context-aware responses."""
    PROFESSIONAL = "professional"
    STUDENT = "student"
    ENTHUSIAST = "enthusiast"
    GENERAL = "general"


class SearchRequest(BaseModel):
    """Request model for search endpoint."""
    text_query: Optional[str] = Field(None, description="Text description to search for")
    user_type: UserType = Field(default=UserType.GENERAL, description="Type of user for context-aware responses")


class ImageMetadata(BaseModel):
    """Metadata for a single image result."""
    image_id: str
    s3_key: str
    url: str
    score: float
    labels: Optional[list[str]] = None


class SearchResponse(BaseModel):
    """Response model for search endpoint."""
    results: list[ImageMetadata]
    detected_labels: list[str] = Field(default_factory=list, description="Architectural style labels detected from input image")
    detected_objects: list[str] = Field(default_factory=list, description="Furniture and objects detected from input image")
    summary: str = Field(description="AI-generated summary of results")
    follow_up_suggestions: list[str] = Field(default_factory=list, description="Suggested follow-up queries")
    query_type: str = Field(description="Type of query: image_only, text_only, or image_and_text")
    total_results: int


class IndexRequest(BaseModel):
    """Request model for indexing images."""
    s3_keys: list[str] = Field(description="List of S3 keys to index")
    batch_size: int = Field(default=32, description="Batch size for processing")


class IndexResponse(BaseModel):
    """Response model for indexing operation."""
    indexed_count: int
    failed_count: int
    failed_keys: list[str] = Field(default_factory=list)
    message: str


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    model_loaded: bool
    index_size: int
    version: str = "1.0.0"


class EmbeddingRequest(BaseModel):
    """Request for generating embeddings (internal use)."""
    text: Optional[str] = None


class EmbeddingResponse(BaseModel):
    """Response with embedding vector."""
    embedding: list[float]
    labels: list[str] = Field(default_factory=list)


class ConversationMessage(BaseModel):
    """Individual message in a search conversation."""
    role: str = Field(description="Role: 'user' or 'assistant'")
    content: str = Field(description="Message content")
    timestamp: datetime = Field(description="When the message was sent")
    query_type: Optional[str] = Field(None, description="Type of query for user messages")
    detected_labels: list[str] = Field(default_factory=list, description="Labels detected (if applicable)")
    detected_objects: list[str] = Field(default_factory=list, description="Objects detected (if applicable)")
    results_count: Optional[int] = Field(None, description="Number of results (if applicable)")


class SearchHistoryItem(BaseModel):
    """Search session with full conversation thread (like ChatGPT)."""
    id: str = Field(description="Unique session identifier")
    title: str = Field(description="Session title (auto-generated from first query)")
    timestamp: datetime = Field(description="When the session started")
    last_updated: datetime = Field(description="When the session was last updated")
    initial_query_type: str = Field(description="Type of initial query: image_only, text_only, or image_and_text")
    initial_text_query: Optional[str] = Field(None, description="Initial text query if provided")
    initial_image_filename: Optional[str] = Field(None, description="Initial image filename if uploaded")
    user_type: UserType = Field(description="User type for the session")
    conversation: list[ConversationMessage] = Field(default_factory=list, description="Full conversation thread")
    total_results_returned: int = Field(description="Total results across all queries in session")


class SearchHistoryResponse(BaseModel):
    """Response model for search history."""
    history: list[SearchHistoryItem]
    total_count: int = Field(description="Total number of history items")


class SearchHistoryStats(BaseModel):
    """Statistics about search history."""
    total_searches: int
    image_only_searches: int
    text_only_searches: int
    combined_searches: int
    most_common_labels: list[tuple[str, int]] = Field(default_factory=list, description="Most common labels and their counts")


class FavoriteItem(BaseModel):
    """A favorited image."""
    id: str = Field(description="Unique favorite identifier")
    image_id: str = Field(description="ID of the favorited image")
    s3_key: str = Field(description="S3 key of the image")
    url: str = Field(description="Presigned URL for the image")
    timestamp: datetime = Field(description="When the image was favorited")
    labels: Optional[list[str]] = Field(None, description="Architectural labels for the image")
    objects: Optional[list[str]] = Field(None, description="Objects detected in the image")
    note: Optional[str] = Field(None, description="User's note about this favorite")


class FavoritesResponse(BaseModel):
    """Response model for favorites list."""
    favorites: list[FavoriteItem]
    total_count: int = Field(description="Total number of favorites")