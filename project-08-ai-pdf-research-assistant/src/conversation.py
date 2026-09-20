"""Conversation History Module.

Manages conversational state, multi-turn message history,
and formatting for the OpenRouter / OpenAI Chat Completions API.
"""

from typing import List, Dict, Any, Optional


class ConversationManager:
    """Manages the message history across multi-turn interactions."""

    def __init__(self):
        self.messages: List[Dict[str, Any]] = []

    def add_user_message(self, content: str) -> None:
        """Record a user query into the history."""
        self.messages.append({
            "role": "user",
            "content": content.strip(),
        })

    def add_assistant_message(self, content: str, citations: Optional[List[Dict[str, Any]]] = None) -> None:
        """Record an assistant response into the history along with optional chunk citations."""
        msg: Dict[str, Any] = {
            "role": "assistant",
            "content": content.strip(),
        }
        if citations:
            msg["citations"] = citations
        self.messages.append(msg)

    def get_messages_for_llm(self, max_turns: int = 4) -> List[Dict[str, str]]:
        """Retrieve recent conversation turns formatted for the LLM API.

        Args:
            max_turns: Maximum number of recent (user, assistant) exchanges to include.

        Returns:
            List of standard role-content dicts suitable for ChatCompletion payloads.
        """
        # Exclude the very last user message because it will be formatted separately with retrieved context
        recent = self.messages[:-1]
        turn_window = max_turns * 2
        sliced = recent[-turn_window:] if len(recent) > turn_window else recent

        formatted = []
        for msg in sliced:
            formatted.append({
                "role": msg["role"],
                "content": msg["content"],
            })
        return formatted

    def clear(self) -> None:
        """Reset and wipe all conversation turns."""
        self.messages = []

    def get_all_messages(self) -> List[Dict[str, Any]]:
        """Return the full list of recorded message objects."""
        return self.messages

    def has_messages(self) -> bool:
        """Check if any turns have been recorded."""
        return len(self.messages) > 0
