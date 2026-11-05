"""Time spent in tool calls, read back from a transcript.

The core loop stamps every `function_call_output` with `duration_s` (and `started_at`) in its
transcript metadata, so robot execution time is recoverable from any transcript. Model calls
carry no timing yet (`TokenUsageSnapshot` holds tokens and cost only), so reasoning time is not
recoverable the same way.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def tool_call_seconds(transcript_path: Path) -> dict[str, float]:
    """Seconds spent inside tool calls, by category, summed from the transcript's `duration_s`.

    :param transcript_path: A loop's `transcript.jsonl`
    :return: `total`, `motor`, `perception` and `other` seconds, each rounded to milliseconds
    """
    from agentic_robotics.adapters.toyblocks.leap.motor_skills_agent import MotorSkillsAgent
    from agentic_robotics.adapters.toyblocks.leap.perception_tools_agent import (
        PerceptionToolsAgent,
    )

    motor = set(MotorSkillsAgent.__abstractmethods__)
    perception = set(PerceptionToolsAgent.__abstractmethods__)
    name_by_call_id: dict[str, str] = {}
    seconds = {"total": 0.0, "motor": 0.0, "perception": 0.0, "other": 0.0}
    for line in transcript_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if item.get("type") == "function_call":
            name_by_call_id[item["call_id"]] = item["name"]
        elif item.get("type") == "function_call_output":
            duration = (item.get("metadata") or {}).get("duration_s")
            if duration is None:
                continue
            name = name_by_call_id.get(item.get("call_id", ""), "")
            category = "motor" if name in motor else "perception" if name in perception else "other"
            seconds[category] += duration
            seconds["total"] += duration
    return {key: round(value, 3) for key, value in seconds.items()}
