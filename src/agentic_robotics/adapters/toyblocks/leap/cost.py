"""A dollar ledger shared by every model client that spends against one budget."""

from __future__ import annotations

import threading


class UnpricedModelCallError(RuntimeError):
    """A model call reported no cost, so the budget cannot be enforced."""


class CostLedger:
    """Running total of model API cost, in dollars, against a fixed cap.

    OpenRouter reports each call's dollar cost in its usage block, which `OpenAIModelClient`
    records as `TokenUsageSnapshot.cost_usd`. Every loop spending against the same budget (the
    top-level agent and each of its subagents) adds to one ledger, so the cap covers all of them.
    A call with no cost, or a zero cost, is an error rather than an under-count: with a pinned
    paid model no call is free, and a total that silently omits calls is not a budget.
    """

    def __init__(self, cap_usd: float) -> None:
        self.cap_usd = cap_usd
        self._total_usd = 0.0
        self._lock = threading.Lock()

    @property
    def total_usd(self) -> float:
        """Dollars spent so far, as reported by the model provider."""
        return self._total_usd

    @property
    def exceeded(self) -> bool:
        """Whether the total has reached the cap."""
        return self._total_usd >= self.cap_usd

    def add(self, cost_usd: float | None) -> None:
        """Record one model call's cost.

        :raises UnpricedModelCallError: If the cost is missing, zero or negative
        """
        if cost_usd is None or cost_usd <= 0:
            raise UnpricedModelCallError(
                f"A model call reported a cost of {cost_usd!r}; the provider did not price it, "
                "so the budget cannot be enforced. Check that the endpoint returns `cost` in its "
                "usage block."
            )
        with self._lock:
            self._total_usd += cost_usd

    def status(self) -> str:
        """The total against the cap, for the agent's status line."""
        return f"${self._total_usd:.2f} of ${self.cap_usd:.2f}"
