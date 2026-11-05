"""Fill the markers in the developer's experimentation prompt: live prices, and the loop's own
settings.

The developer's `experimentation.md` tells the agent what its tokens cost, so that it can
spend its budget deliberately: a docstring edit invalidates the cached prefix and the next call
rewrites the whole history at the cache-write price. The prompt carries markers rather than
numbers, because the numbers belong to the pinned model and OpenRouter changes them without
notice:

    $[[CACHE_WRITE_COST_USD]]   $[[CACHE_READ_COST_USD]]   $[[OUTPUT_COST_USD]]

`fill_prompt_markers` replaces each with the price per million tokens that OpenRouter lists for
the run's model, formatted to cents, and refuses to start the run if the model is not listed or a
marker would survive, so the agent never reads a literal marker. The listing needs no API key.
Prices are read once per launch and logged, so the transcript's prompt records what the agent was
told.

Other markers name settings the loop enforces, passed in by the caller so the prompt can never
disagree with the code: `[[MIN_CLEARABLE_CONTEXT_PERCENT]]` is the context-clear gate
(`ContextPolicy.self_wipe_fraction`, from `MIN_CLEARABLE_CONTEXT_PERCENT` in the developer's
`context_clearing_agent.py`).
"""

from __future__ import annotations

import json
import logging
import re
import urllib.request
from typing import Any

logger = logging.getLogger("agentic_robotics.examples.leap.prices")

MODELS_URL = "https://openrouter.ai/api/v1/models"
TIMEOUT_S = 15.0

MARKER_FIELDS: dict[str, str] = {
    "CACHE_WRITE_COST_USD": "input_cache_write",
    "CACHE_READ_COST_USD": "input_cache_read",
    "OUTPUT_COST_USD": "completion",
    "INPUT_COST_USD": "prompt",
}
"""Marker name -> field of OpenRouter's per-token `pricing` object."""

_MARKER = re.compile(r"\[\[([A-Za-z0-9_]+)\]\]")


def openrouter_prices(model_id: str) -> dict[str, float]:
    """Return the model's prices in USD per million tokens, keyed by OpenRouter's field names.

    :param model_id: OpenRouter slug, e.g. `anthropic/claude-opus-5.5`
    :raises SystemExit: If OpenRouter cannot be reached or does not list the model
    """
    try:
        with urllib.request.urlopen(MODELS_URL, timeout=TIMEOUT_S) as response:
            models: list[dict[str, Any]] = json.load(response)["data"]
    except Exception as error:
        raise SystemExit(
            f"Could not read OpenRouter's model listing for prices: {error}"
        ) from error
    for model in models:
        if model.get("id") == model_id:
            pricing = model.get("pricing", {})
            return {
                field: float(pricing[field]) * 1_000_000
                for field in MARKER_FIELDS.values()
                if field in pricing
            }
    raise SystemExit(f"OpenRouter does not list {model_id!r}, so its prices cannot be filled in.")


def fill_prompt_markers(prompt: str, model_id: str, values: dict[str, str] | None = None) -> str:
    """Replace every `[[MARKER]]` in `prompt`: from `values` first, then the model's live prices.

    A prompt with no markers is returned unchanged without contacting OpenRouter.

    :param values: Non-price markers and their text, e.g. the loop's context-clear gate
    :raises SystemExit: If a marker is neither in `values` nor a price OpenRouter lists
    """
    markers = set(_MARKER.findall(prompt))
    if not markers:
        return prompt
    filled = prompt
    for marker, text in (values or {}).items():
        filled = filled.replace(f"[[{marker}]]", text)
    price_markers = sorted(set(_MARKER.findall(filled)))
    if price_markers:
        prices = openrouter_prices(model_id)
        for marker in price_markers:
            field = MARKER_FIELDS.get(marker)
            if field is None or field not in prices:
                raise SystemExit(
                    f"The prompt's [[{marker}]] marker has no price for {model_id!r} and no "
                    f"value was passed; price markers: {', '.join(sorted(MARKER_FIELDS))}."
                )
            filled = filled.replace(f"[[{marker}]]", f"{prices[field]:.2f}")
        logger.info(
            "Prompt prices for %s, USD per million tokens: %s",
            model_id,
            ", ".join(f"{MARKER_FIELDS[m]}=${prices[MARKER_FIELDS[m]]:.2f}" for m in price_markers),
        )
    if values:
        logger.info("Prompt settings: %s", ", ".join(f"{k}={v}" for k, v in values.items()))
    leftover = _MARKER.findall(filled)
    if leftover:
        raise SystemExit(f"Unfilled markers remain in the prompt: {leftover}")
    return filled
