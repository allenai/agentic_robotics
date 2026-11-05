"""The Claude Code model host against a scripted SDK transport: no network, no CLI, no model.

The transport answers the SDK's initialize handshake, emits stream-json frames the way the
bundled CLI does, and issues the hook `control_request`s for each tool call, choosing what to
emit next from the loop's hook reply. So what runs is the real SDK (its parser, control
protocol and hook dispatch) and the loop under test; what is faked is the CLI and the model.

Unverified here, and what the first pilot has to check: the real CLI's frame order, the shape of
`tool_response` for an MCP tool, how a denied call is reported in the tool result, and that an
`updatedToolOutput` replacement is what the model sees.
"""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from claude_agent_sdk._internal.transport import Transport

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "examples" / "agent") not in sys.path:
    sys.path.insert(0, str(REPO / "examples" / "agent"))

from agentic_robotics.adapters.toyblocks.leap.evaluate import (
    BATCHED_IMAGE_NOTE,
    claude_code_model_id,
    claude_code_summary,
    final_assistant_message,
    make_claude_code_evaluation_loop_class,
    tool_call_counts,
)
from agentic_robotics.adapters.toyblocks.leap.messages import (
    evaluation_final_update,
    evaluation_status_line,
)
from agentic_robotics.adapters.toyblocks.leap.timing import tool_call_seconds
from agentic_robotics.adapters.toyblocks.release.search_spell_scoring import (
    declared_outcome,
)
from agentic_robotics.core.claude_code_loop import (
    ClaudeCodeConfig,
    strip_images,
    tool_result_to_openai_output,
)
from agentic_robotics.core.foundation_models import Transcript

KEY = "toyblocks"
TASK = "Spell the word."
IMAGE = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "aGk="}}
USAGE = {
    "input_tokens": 100,
    "cache_read_input_tokens": 1000,
    "cache_creation_input_tokens": 10,
    "output_tokens": 5,
}


@dataclass
class HookStep:
    """Issue one hook callback; `then` maps the loop's reply to the frames that follow."""

    event: str
    hook_input: dict[str, Any]
    then: Callable[[dict[str, Any]], list[Any]] = lambda reply: []
    tool_use_id: str | None = None


@dataclass
class ScriptedTransport(Transport):
    script: list[Any]
    replies: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    """Every hook reply the loop sent, as (tool_use_id or event, hookSpecificOutput)."""

    def __post_init__(self) -> None:
        self._out: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._control_replies: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._prompted = asyncio.Event()
        self._hooks_config: dict[str, Any] = {}
        self._ready = False
        self._producer: asyncio.Task[None] | None = None
        self._counter = 0

    async def connect(self) -> None:
        self._ready = True
        self._producer = asyncio.create_task(self._produce())

    async def write(self, data: str) -> None:
        message = json.loads(data)
        if message["type"] == "control_request":
            request = message["request"]
            if request["subtype"] == "initialize":
                self._hooks_config = request.get("hooks") or {}
                await self._out.put(
                    {
                        "type": "control_response",
                        "response": {
                            "subtype": "success",
                            "request_id": message["request_id"],
                            "response": {"commands": [], "output_style": "default"},
                        },
                    }
                )
            else:
                await self._out.put(
                    {
                        "type": "control_response",
                        "response": {
                            "subtype": "success",
                            "request_id": message["request_id"],
                            "response": {},
                        },
                    }
                )
        elif message["type"] == "control_response":
            await self._control_replies.put(message)
        elif message["type"] == "user":
            self._prompted.set()

    async def read_messages(self) -> AsyncIterator[dict[str, Any]]:
        while True:
            item = await self._out.get()
            if item.get("type") == "__end__":
                return
            yield item

    async def close(self) -> None:
        self._ready = False
        if self._producer is not None:
            self._producer.cancel()

    def is_ready(self) -> bool:
        return self._ready

    async def end_input(self) -> None:
        pass

    async def _produce(self) -> None:
        await self._prompted.wait()
        await self._out.put(
            {
                "type": "system",
                "subtype": "init",
                "session_id": "s",
                "model": "claude-opus-5-5",
                "tools": [],
                "mcp_servers": [{"name": KEY, "status": "connected"}],
                "permissionMode": "dontAsk",
                "claude_code_version": "fake",
            }
        )
        await self._play(self.script)
        await self._out.put({"type": "__end__"})

    async def _play(self, frames: list[Any]) -> None:
        for frame in frames:
            if isinstance(frame, HookStep):
                callback_id = self._hooks_config[frame.event][0]["hookCallbackIds"][0]
                self._counter += 1
                await self._out.put(
                    {
                        "type": "control_request",
                        "request_id": f"cr_{self._counter}",
                        "request": {
                            "subtype": "hook_callback",
                            "callback_id": callback_id,
                            "input": frame.hook_input,
                            "tool_use_id": frame.tool_use_id,
                        },
                    }
                )
                reply = await self._control_replies.get()
                assert reply["response"]["subtype"] == "success", reply
                output = reply["response"].get("response") or {}
                self.replies.append(
                    (frame.tool_use_id or frame.event, output.get("hookSpecificOutput") or {})
                )
                await self._play(frame.then(output.get("hookSpecificOutput") or {}))
            else:
                await self._out.put(frame)


def assistant(
    message_id: str, blocks: list[dict[str, Any]], stop_reason: str = "tool_use"
) -> dict[str, Any]:
    return {
        "type": "assistant",
        "message": {
            "id": message_id,
            "model": "claude-opus-5-5",
            "content": blocks,
            "usage": dict(USAGE),
            "stop_reason": stop_reason,
        },
        "session_id": "s",
        "parent_tool_use_id": None,
    }


def tool_use(
    tool_use_id: str, name: str, arguments: dict[str, Any] | None = None
) -> dict[str, Any]:
    return {
        "type": "tool_use",
        "id": tool_use_id,
        "name": f"mcp__{KEY}__{name}",
        "input": arguments or {},
    }


def tool_result(tool_use_id: str, content: Any, is_error: bool = False) -> dict[str, Any]:
    return {
        "type": "user",
        "message": {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": content,
                    "is_error": is_error,
                }
            ],
        },
        "session_id": "s",
        "parent_tool_use_id": None,
    }


def result_frame(text: str, cost: float = 0.42) -> dict[str, Any]:
    return {
        "type": "result",
        "subtype": "success",
        "duration_ms": 5000,
        "duration_api_ms": 4000,
        "is_error": False,
        "num_turns": 3,
        "session_id": "s",
        "total_cost_usd": cost,
        "usage": {"input_tokens": 300, "output_tokens": 50},
        "modelUsage": {
            "claude-opus-5-5": {"inputTokens": 300, "outputTokens": 50, "costUSD": cost}
        },
        "result": text,
    }


def pre(tool_use_id: str, name: str, then: Callable[[dict[str, Any]], list[Any]]) -> HookStep:
    return HookStep(
        "PreToolUse",
        {
            "session_id": "s",
            "transcript_path": "/dev/null",
            "cwd": "/",
            "permission_mode": "dontAsk",
            "hook_event_name": "PreToolUse",
            "tool_name": f"mcp__{KEY}__{name}",
            "tool_input": {},
            "tool_use_id": tool_use_id,
        },
        then,
        tool_use_id,
    )


def post(
    tool_use_id: str,
    name: str,
    response: Any,
    duration_ms: int,
    then: Callable[[dict[str, Any]], list[Any]],
) -> HookStep:
    return HookStep(
        "PostToolUse",
        {
            "session_id": "s",
            "transcript_path": "/dev/null",
            "cwd": "/",
            "permission_mode": "dontAsk",
            "hook_event_name": "PostToolUse",
            "tool_name": f"mcp__{KEY}__{name}",
            "tool_input": {},
            "tool_response": response,
            "tool_use_id": tool_use_id,
            "duration_ms": duration_ms,
        },
        then,
        tool_use_id,
    )


def prompt_submit() -> HookStep:
    return HookStep(
        "UserPromptSubmit",
        {
            "session_id": "s",
            "transcript_path": "/dev/null",
            "cwd": "/",
            "permission_mode": "dontAsk",
            "hook_event_name": "UserPromptSubmit",
            "prompt": TASK,
        },
    )


def seen_result(output: dict[str, Any], original: Any) -> Any:
    """What the model sees: the hook's replacement if it gave one, else the original."""
    return output.get("updatedToolOutput", original)


def run_episode(
    tmp_path: Path, script: list[Any], motor_cutoff: int, ceiling: int, batch_images: bool = True
) -> tuple[Any, ScriptedTransport, list[tuple[int, str]]]:
    snapshots: list[tuple[int, str]] = []

    async def after_motor_call(count: int, name: str) -> None:
        snapshots.append((count, name))

    async def main() -> Any:
        transport = ScriptedTransport(script)
        loop_class = make_claude_code_evaluation_loop_class(
            motor_cutoff, ceiling, after_motor_call, batch_images=batch_images
        )
        agent = loop_class(
            Transcript(out_dir=tmp_path),
            ClaudeCodeConfig(model="claude-opus-5-5", mcp_server_key=KEY),
            "http://127.0.0.1:1/mcp",
            ["pick", "return_home", "get_image_observation"],
            metadata_callback=lambda: {"frame_count": 7},
            transport=transport,
        )
        await agent.run(task=TASK, prompt=TASK)
        return agent, transport

    agent, transport = asyncio.run(main())
    return agent, transport, snapshots


def test_cutoff_snapshots_images_and_usage(tmp_path: Path) -> None:
    """A step that overruns the cutoff runs every call it issued; the next step is denied with
    the final-answer request; snapshots are per motor call; an image is dropped only when a
    motor call follows in the same step; usage is counted once per message id."""
    script = [
        prompt_submit(),
        # One step, emitted as three frames sharing the message id: a thinking block first
        # (writes no row), then text with a call, then a parallel call.
        assistant("msg_1", [{"type": "thinking", "thinking": "Plan.", "signature": "sig"}]),
        assistant("msg_1", [{"type": "text", "text": "Picking."}, tool_use("t1", "pick")]),
        assistant("msg_1", [tool_use("t2", "return_home")]),
        pre(
            "t1",
            "pick",
            lambda _: [
                post(
                    "t1",
                    "pick",
                    [{"type": "text", "text": "picked"}, IMAGE],
                    1500,
                    lambda out: [
                        tool_result(
                            "t1", seen_result(out, [{"type": "text", "text": "picked"}, IMAGE])
                        ),
                    ],
                ),
            ],
        ),
        pre(
            "t2",
            "return_home",
            lambda _: [
                post(
                    "t2",
                    "return_home",
                    [{"type": "text", "text": "home"}, IMAGE],
                    2500,
                    lambda out: [
                        tool_result(
                            "t2", seen_result(out, [{"type": "text", "text": "home"}, IMAGE])
                        ),
                    ],
                ),
            ],
        ),
        # The budget is spent at the start of this step, so its call is denied.
        assistant("msg_2", [tool_use("t3", "get_image_observation")]),
        pre(
            "t3",
            "get_image_observation",
            lambda out: [
                tool_result(
                    "t3",
                    out.get("permissionDecisionReason", "ran"),
                    is_error="permissionDecision" in out,
                ),
            ],
        ),
        assistant("msg_3", [{"type": "text", "text": "Done. [[SUCCESS]]"}], stop_reason="end_turn"),
        result_frame("Done. [[SUCCESS]]"),
    ]
    agent, transport, snapshots = run_episode(tmp_path, script, motor_cutoff=2, ceiling=10)

    assert agent.motor_calls_made == 2  # every motor skill counts, return_home included
    assert agent.skill_calls_made == 2
    assert agent.total_calls_made == 2
    assert agent.hit_cutoff
    assert agent.denials == 1
    assert agent.ordering_fallbacks == 0
    assert snapshots == [(1, "pick"), (2, "return_home")]

    replies = dict(transport.replies)
    assert replies["UserPromptSubmit"]["additionalContext"] == evaluation_status_line(2, 10)
    assert replies["t1"]["updatedToolOutput"] == [
        {"type": "text", "text": "picked"},
        {"type": "text", "text": BATCHED_IMAGE_NOTE},
    ]
    assert replies["t1"]["additionalContext"] == evaluation_status_line(1, 9)
    assert "updatedToolOutput" not in replies["t2"]
    assert replies["t2"]["additionalContext"] == evaluation_status_line(0, 8)
    assert replies["t3"]["permissionDecision"] == "deny"
    assert replies["t3"]["permissionDecisionReason"] == evaluation_final_update(TASK)

    assert [u.input_tokens for u in agent.usage_history] == [1110, 1110, 1110]
    assert agent.usage_history[0].tokens_read == 1000
    assert agent.usage_history[0].tokens_written == 10
    assert all(u.cost_usd is None for u in agent.usage_history)

    transcript = tmp_path / "transcript.jsonl"
    # Every step's usage reaches the transcript exactly once, on the step's first written row,
    # even when the step's first frame was thinking only.
    rows = [json.loads(line) for line in transcript.read_text().splitlines()]
    with_usage = [r for r in rows if (r.get("metadata") or {}).get("usage")]
    assert [r["metadata"]["message_id"] for r in with_usage] == ["msg_1", "msg_2", "msg_3"]
    assert (
        with_usage[0]["type"] == "message"
        and with_usage[0]["metadata"]["usage"]["input_tokens"] == 1110
    )
    counts = tool_call_counts(transcript)
    assert (
        counts["total"] == 3
        and counts["motor"] == 2
        and counts["by_name"]["get_image_observation"] == 1
    )
    seconds = tool_call_seconds(transcript)
    assert seconds["motor"] == 4.0 and seconds["total"] == 4.0
    assert final_assistant_message(transcript) == "Done. [[SUCCESS]]"
    assert (
        declared_outcome(
            final_assistant_message(transcript),
            {"[[SUCCESS]]": "success", "[[IMPOSSIBLE]]": "impossible"},
        )
        == "success"
    )
    records = [json.loads(line) for line in transcript.read_text().splitlines()]
    outputs = {r["call_id"]: r for r in records if r.get("type") == "function_call_output"}
    assert outputs["t1"]["metadata"]["outcome"] == "Tool Success"
    assert outputs["t1"]["metadata"]["pre"] == {"frame_count": 7}
    assert outputs["t3"]["metadata"]["outcome"] == "Denied"
    assert "duration_s" not in outputs["t3"]["metadata"]
    assert sorted(p.name for p in (tmp_path / "images").iterdir()) == ["image_0000.png"]

    summary = claude_code_summary(agent)
    assert summary["cost_usd"] == 0.42
    assert summary["cost_source"] == "sdk_estimate"
    assert summary["model_calls"] == 3
    assert summary["model_seconds"] == 4.0
    assert summary["result"]["num_turns"] == 3
    assert summary["init"]["mcp_servers"] == [{"name": KEY, "status": "connected"}]


def test_total_ceiling_denies_the_next_step(tmp_path: Path) -> None:
    script = [
        prompt_submit(),
        assistant("msg_1", [tool_use("t1", "get_image_observation")]),
        pre(
            "t1",
            "get_image_observation",
            lambda _: [
                post(
                    "t1",
                    "get_image_observation",
                    [IMAGE],
                    100,
                    lambda out: [tool_result("t1", seen_result(out, [IMAGE]))],
                ),
            ],
        ),
        assistant("msg_2", [tool_use("t2", "pick")]),
        pre(
            "t2",
            "pick",
            lambda out: [
                tool_result("t2", out.get("permissionDecisionReason", "ran"), is_error=True)
            ],
        ),
        assistant(
            "msg_3",
            [{"type": "text", "text": "Nothing to do. [[IMPOSSIBLE]]"}],
            stop_reason="end_turn",
        ),
        result_frame("Nothing to do. [[IMPOSSIBLE]]"),
    ]
    agent, transport, snapshots = run_episode(tmp_path, script, motor_cutoff=50, ceiling=1)
    assert agent.total_calls_made == 1 and agent.motor_calls_made == 0
    assert agent.hit_cutoff and agent.denials == 1
    assert snapshots == []
    assert dict(transport.replies)["t2"]["permissionDecision"] == "deny"
    # A perception call's image is never dropped, whatever follows it.
    assert "updatedToolOutput" not in dict(transport.replies)["t1"]
    assert (
        declared_outcome(
            final_assistant_message(tmp_path / "transcript.jsonl"),
            {"[[SUCCESS]]": "success", "[[IMPOSSIBLE]]": "impossible"},
        )
        == "impossible"
    )


def test_no_cutoff_means_no_denial(tmp_path: Path) -> None:
    script = [
        prompt_submit(),
        assistant("msg_1", [tool_use("t1", "pick")]),
        pre(
            "t1",
            "pick",
            lambda _: [post("t1", "pick", "ok", 100, lambda out: [tool_result("t1", "ok")])],
        ),
        assistant("msg_2", [{"type": "text", "text": "Done. [[SUCCESS]]"}], stop_reason="end_turn"),
        result_frame("Done. [[SUCCESS]]"),
    ]
    agent, transport, _ = run_episode(tmp_path, script, motor_cutoff=5, ceiling=10)
    assert not agent.hit_cutoff and agent.denials == 0 and agent.motor_calls_made == 1
    assert dict(transport.replies)["t1"].get("permissionDecision") is None


def test_strip_images_and_output_conversion() -> None:
    assert strip_images([{"type": "text", "text": "a"}], "note") is None
    assert strip_images({"content": [IMAGE]}, "note") == {
        "content": [{"type": "text", "text": "note"}]
    }
    assert strip_images("plain", "note") is None
    assert tool_result_to_openai_output("x") == [{"type": "input_text", "text": "x"}]
    assert tool_result_to_openai_output([IMAGE])[0]["image_url"].startswith(
        "data:image/png;base64,"
    )


def test_claude_code_model_id() -> None:
    assert claude_code_model_id("anthropic/claude-opus-5.5") == "claude-opus-5-5"
    assert claude_code_model_id("anthropic/claude-opus-5-5") == "claude-opus-5-5"
    with pytest.raises(SystemExit):
        claude_code_model_id("openai/gpt-6.1-sol-pro")
