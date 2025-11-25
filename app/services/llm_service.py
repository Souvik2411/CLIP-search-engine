from openai import AsyncOpenAI
from typing import Optional
import logging

from app.config import get_settings
from app.models.schemas import UserType


logger = logging.getLogger(__name__)


class LLMService:
    """Service for GPT-4o-mini integration - summaries and follow-ups."""

    def __init__(self):
        self.settings = get_settings()
        self._client: Optional[AsyncOpenAI] = None

    @property
    def client(self) -> AsyncOpenAI:
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            self._client = AsyncOpenAI(api_key=self.settings.OPENAI_API_KEY)
        return self._client

    async def generate_search_summary(
        self,
        query_labels: list[str],
        result_count: int,
        user_type: UserType,
        text_query: Optional[str] = None
    ) -> tuple[str, list[str]]:
        """
        Generate a summary and follow-up suggestions for search results.

        Args:
            query_labels: Labels detected from input image
            result_count: Number of results returned
            user_type: Type of user for context-aware response
            text_query: Optional text query

        Returns:
            Tuple of (summary, follow_up_suggestions)
        """
        # Build context based on user type
        user_context = self._get_user_context(user_type)

        # Build the prompt
        prompt = self._build_summary_prompt(
            query_labels,
            result_count,
            user_context,
            text_query
        )

        try:
            response = await self.client.chat.completions.create(
                model=self.settings.LLM_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": """You are a friendly, warm architecture and design assistant. Your responses should be conversational and engaging.

When responding:
- Vary your opening approach - NEVER repeat the same greeting. Use different approaches like:
  * Direct acknowledgment: "Perfect! I've found..."
  * Enthusiasm: "Excellent choice searching for..."
  * Discovery tone: "Here's what I discovered..."
  * Observational: "These results showcase..."
  * Educational: "Looking at these designs..."
  * Appreciative: "Beautiful selection here..."
- AVOID cliché phrases like "I'm so glad you're diving into the world of architecture and design"
- Keep responses fresh and varied
- Explain what you found and why it's relevant to their query
- Be helpful and encouraging without being overly enthusiastic
- Provide 3 actionable follow-up suggestions

Example responses:
- "Perfect! I've found 9 contemporary living rooms featuring sectional sofas with clean lines and neutral palettes."
- "These results showcase modern kitchen designs with marble countertops and industrial accents."
- "Excellent choice! Here are minimalist bedrooms that emphasize natural light and simple forms."
"""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=self.settings.LLM_MAX_TOKENS,
                temperature=0.7
            )

            content = response.choices[0].message.content

            # Parse the response
            summary, suggestions = self._parse_response(content)

            return summary, suggestions

        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            # Return fallback response
            return self._get_fallback_response(result_count, query_labels)

    def _get_user_context(self, user_type: UserType) -> str:
        """Get context string based on user type."""
        contexts = {
            UserType.PROFESSIONAL: "an architecture professional who needs technical details and industry-specific terminology",
            UserType.STUDENT: "an architecture student who benefits from educational context and learning-oriented suggestions",
            UserType.ENTHUSIAST: "an architecture enthusiast interested in design concepts and visual aesthetics",
            UserType.GENERAL: "a general user interested in architecture and design"
        }
        return contexts.get(user_type, contexts[UserType.GENERAL])

    def _build_summary_prompt(
        self,
        query_labels: list[str],
        result_count: int,
        user_context: str,
        text_query: Optional[str]
    ) -> str:
        """Build the prompt for summary generation."""
        labels_str = ", ".join(query_labels) if query_labels else "various architectural elements"

        prompt = f"""The user is {user_context}.

Search context:
- Detected architectural elements: {labels_str}
- Number of similar images found: {result_count}
"""

        if text_query:
            prompt += f"- User's text query: \"{text_query}\"\n"

        prompt += """
Please provide:
1. A brief summary (2-3 sentences) describing what the search found and its relevance
2. Three follow-up suggestions the user might want to explore

Format your response as:
SUMMARY: [Your summary here]

SUGGESTIONS:
- [Suggestion 1]
- [Suggestion 2]
- [Suggestion 3]"""

        return prompt

    def _parse_response(self, content: str) -> tuple[str, list[str]]:
        """Parse the LLM response into summary and suggestions."""
        summary = ""
        suggestions = []

        lines = content.strip().split('\n')
        in_suggestions = False

        for line in lines:
            line = line.strip()
            if line.startswith("SUMMARY:"):
                summary = line.replace("SUMMARY:", "").strip()
            elif line.startswith("SUGGESTIONS:"):
                in_suggestions = True
            elif in_suggestions and line.startswith("-"):
                suggestion = line.lstrip("- ").strip()
                if suggestion:
                    suggestions.append(suggestion)
            elif not in_suggestions and summary and not line.startswith("SUGGESTIONS"):
                # Continue summary if it spans multiple lines
                summary += " " + line

        # Fallback if parsing fails
        if not summary:
            summary = content.split('\n')[0] if content else "Search completed successfully."
        if not suggestions:
            suggestions = [
                "Explore similar architectural styles",
                "Search for related design elements",
                "Compare with different time periods"
            ]

        return summary.strip(), suggestions[:3]

    def _get_fallback_response(
        self,
        result_count: int,
        query_labels: list[str]
    ) -> tuple[str, list[str]]:
        """Generate fallback response if LLM call fails."""
        labels_str = ", ".join(query_labels[:3]) if query_labels else "architectural elements"

        summary = f"Found {result_count} similar images featuring {labels_str}."
        suggestions = [
            "Try refining your search with specific architectural terms",
            "Explore different angles or perspectives",
            "Search for related architectural styles"
        ]

        return summary, suggestions

    async def generate_custom_response(
        self,
        prompt: str,
        system_message: Optional[str] = None
    ) -> str:
        """
        Generate a custom response for other use cases.

        Args:
            prompt: User prompt
            system_message: Optional system message

        Returns:
            Generated response
        """
        if system_message is None:
            system_message = "You are a helpful architecture and design assistant."

        try:
            response = await self.client.chat.completions.create(
                model=self.settings.LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.settings.LLM_MAX_TOKENS,
                temperature=0.7
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error generating custom response: {e}")
            raise

    def generate_summary(
        self,
        prompt: str,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate a synchronous summary for indexing pipeline.
        Uses the synchronous OpenAI client for non-async contexts.

        Args:
            prompt: The prompt for summary generation
            max_tokens: Maximum tokens for response

        Returns:
            Generated summary text
        """
        from openai import OpenAI

        if max_tokens is None:
            max_tokens = self.settings.LLM_MAX_TOKENS

        try:
            # Use synchronous client for offline indexing
            sync_client = OpenAI(api_key=self.settings.OPENAI_API_KEY)

            response = sync_client.chat.completions.create(
                model=self.settings.LLM_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful architecture and design assistant. Provide concise, natural descriptions."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=max_tokens,
                temperature=0.7
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            raise


# Create singleton instance
llm_service = LLMService()