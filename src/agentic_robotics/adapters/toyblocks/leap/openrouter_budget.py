"""Ask OpenRouter what the key and the account have left, and refuse a run that would not fit.

Before any paid run, and at session start, the spend is re-queried, with a warning when a budget
constraint is likely. Two ceilings apply. The key has its own limit (`limit`, reset per
`limit_reset`; today's spend is `usage_daily`). The account has prepaid credits
(`total_credits - total_usage`) that only IT, which holds the account, refills. A run that hits
either ceiling dies mid-episode with HTTP 402, so the check runs before the first model call.

    from agentic_robotics.adapters.toyblocks.leap.openrouter_budget import require_budget
    require_budget(planned_usd=50.0)   # raises SystemExit with the numbers when it does not fit

`scripts/openrouter_spend.sh` prints the same numbers for a person. Both read the key from
`OPENROUTER_API_KEY` or `~/.openrouter_key`, never print it, and only warn when OpenRouter cannot
be reached, so a network blip does not block a run the budget allows.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

logger = logging.getLogger("agentic_robotics.examples.leap.openrouter_budget")

RESERVE_USD = 20.0
"""Credit that must remain after the planned spend: episodes overshoot their caps slightly."""

TIMEOUT_S = 15.0


def api_key() -> str | None:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if key:
        return key
    path = Path.home() / ".openrouter_key"
    return path.read_text(encoding="utf-8").strip() if path.is_file() else None


def _get(url: str, key: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
        return json.load(response)["data"]


def credit_status() -> dict[str, Any] | None:
    """Spend and remaining amounts, or None (with a warning) when they cannot be fetched."""
    key = api_key()
    if key is None:
        logger.warning("No OpenRouter key found; cannot check the budget.")
        return None
    try:
        auth = _get("https://openrouter.ai/api/v1/auth/key", key)
        credits = _get("https://openrouter.ai/api/v1/credits", key)
    except (urllib.error.URLError, OSError, KeyError, ValueError) as exc:
        logger.warning(f"Could not query the OpenRouter budget ({type(exc).__name__}: {exc}).")
        return None
    limit = auth.get("limit")
    return {
        "usage_daily": float(auth.get("usage_daily") or 0.0),
        "usage_monthly": float(auth.get("usage_monthly") or 0.0),
        "key_limit": float(limit) if limit is not None else None,
        "key_limit_reset": auth.get("limit_reset"),
        "key_limit_remaining": float(auth["limit_remaining"]) if limit is not None else None,
        "account_credit_remaining": float(credits["total_credits"]) - float(credits["total_usage"]),
    }


def describe(status: dict[str, Any]) -> str:
    key_text = (
        "no key limit"
        if status["key_limit"] is None
        else f"key limit ${status['key_limit']:.2f} per {status['key_limit_reset'] or 'key'}, "
        f"${status['key_limit_remaining']:.2f} left"
    )
    return (
        f"OpenRouter: ${status['usage_daily']:.2f} spent today, "
        f"${status['usage_monthly']:.2f} this month; "
        f"{key_text}; account credit ${status['account_credit_remaining']:.2f} left."
    )


def require_budget(
    planned_usd: float, reserve_usd: float = RESERVE_USD, ignore: bool = False
) -> None:
    """Log the budget and refuse to proceed when `planned_usd` plus the reserve does not fit.

    :param planned_usd: What the run may spend (its cap, or episodes times a per-episode estimate)
    :param reserve_usd: Credit that must remain afterwards
    :param ignore: Log the shortfall and continue anyway (a `--ignore-budget` flag)
    :raises SystemExit: When the run does not fit and `ignore` is false
    """
    status = credit_status()
    if status is None:
        return
    logger.info(describe(status))
    needed = planned_usd + reserve_usd
    ceilings = {"account credit": status["account_credit_remaining"]}
    if status["key_limit_remaining"] is not None:
        ceilings[f"key limit ({status['key_limit_reset'] or 'key'})"] = status[
            "key_limit_remaining"
        ]
    short = {name: left for name, left in ceilings.items() if left < needed}
    if not short:
        return
    detail = "; ".join(f"{name} has ${left:.2f}" for name, left in short.items())
    message = (
        f"This run may spend ${planned_usd:.2f} and ${reserve_usd:.2f} must remain, but {detail}. "
        "Top up the OpenRouter account (IT) or raise the key limit before starting."
    )
    if ignore:
        logger.warning(message + " Continuing because --ignore-budget was passed.")
        return
    raise SystemExit(message)
