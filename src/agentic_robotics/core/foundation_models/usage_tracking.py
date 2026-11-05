"""Define classes and functions to represent and validate model token usage."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenUsageSnapshot:
    """A snapshot of the tokens read from and written to cache during a model call."""

    input_tokens: int
    tokens_read: int
    tokens_written: int
    cost_usd: float | None
    started_at: str | None = None
    """ISO 8601-formatted local date and time when the model was called."""

    duration_s: float | None = None
