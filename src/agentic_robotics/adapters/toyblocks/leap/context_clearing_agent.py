"""Define an abstract base class for an agent that can clear its conversation context."""

from abc import ABC, abstractmethod

MIN_CLEARABLE_CONTEXT_PERCENT = 90
"""The context must be at least this percent full (e.g., 40 = 40%) before it can be cleared."""

MAX_CONTEXT_SIZE_TOKENS = 150000
"""Maximum size of the context during experimentation, in tokens."""


class ContextClearingAgent(ABC):
    """An agent that can clear its conversation context."""

    @abstractmethod
    async def clear_context(self) -> None:
        """Clear the conversation context early.

        When the conversation history reaches its size limit, it will be
        automatically cleared. You may use this tool to clear the context
        early, but only after it has reached at least 90% of its maximum size.
        """
        ...
