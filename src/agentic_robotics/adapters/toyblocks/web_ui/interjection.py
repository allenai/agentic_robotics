"""The channel a watching human interjects through: a message queue, a pause latch, a budget grant.

This is the mechanism behind the `interjection` thread, and it is deliberately the only piece of
that thread with no dependency on the loop, the model client, or any adapter. `AgenticLoop` asks
it three questions between turns -- has anyone typed, am I allowed to start another turn, may I
have more tool calls -- and a UI answers them. `server.py` is one such UI; a terminal reader would
be another.

Everything here assumes one event loop. The UI side and the agentic loop side run as coroutines on
the same loop (see `agentic_web_thread` in `server.py`), so the plain list needs no lock, and
`take()` is atomic with respect to `submit()` for the same reason.

**Stop halts the motion itself**: the motion layer asks :attr:`stop_requested` before every
simulated step, so the arm holds where it is, the skill unwinds, the rest of that turn's calls come
back skipped, and the loop is held until :meth:`resume`. The episode survives it.

**Pause stops the loop starting a *new* turn**; it never interrupts a turn already in flight, and
the calls already queued in that turn still run. That is the cheap seam the project already relies
on: a skill is atomic, and a model call has already been paid for by the time a pause could land on
it. A human who pauses mid-skill gets the hold they asked for as soon as that skill returns, and
can type without racing the loop meanwhile. Pausing to *look* therefore costs the agent nothing,
which is what separates it from stopping.

The intended home for this file is `core/`, once its shape has survived a run.
"""

from __future__ import annotations

import asyncio


class InterjectionChannel:
    """Carries a watching human's messages, pause requests, and budget grants into a loop."""

    def __init__(self) -> None:
        """Initialize an empty channel that is running, not paused."""
        self._pending: list[str] = []
        self._resumed = asyncio.Event()
        self._resumed.set()

        self._grants: asyncio.Queue[int] = asyncio.Queue()
        self._reset_requested = False
        self._task_requested: str | None = None
        self._stop_requested = False
        self._activity = asyncio.Event()

    # -- The UI side ---------------------------------------------------------------------------

    def submit(self, text: str) -> None:
        """Queue one message from the human, to be delivered before the loop's next turn.

        Several messages may queue up while a skill runs; they are delivered in order, as
        separate conversation items, so nothing is lost by typing twice.

        :param text: The message the human typed
        """
        self._pending.append(text)
        self._activity.set()

    def request_stop(self) -> None:
        """Halt the robot now, drop whatever calls were still queued, and hold the loop.

        Distinct from :meth:`pause`, which lets the skill in flight finish. This one is read by
        the motion layer before every simulated step, so the arm stops where it is. The loop is
        paused as well, because a stop that let the next turn start immediately would be a stutter
        rather than a stop; :meth:`resume` releases it. The episode is not ended -- that is what
        `request_reset` is for.
        """
        self._stop_requested = True
        self._resumed.clear()
        self._activity.set()

    @property
    def stop_requested(self) -> bool:
        """Whether a stop is in force. Read from the simulator's thread between steps."""
        return self._stop_requested

    def pause(self) -> None:
        """Stop the loop starting another turn, until :meth:`resume`."""
        self._resumed.clear()
        self._activity.set()

    def resume(self) -> None:
        """Let the loop start turns again, and let the robot move again."""
        self._stop_requested = False
        self._resumed.set()
        self._activity.set()

    @property
    def paused(self) -> bool:
        """Whether the loop is currently held between turns."""
        return not self._resumed.is_set()

    def request_reset(self) -> None:
        """Ask for the episode to be reset -- scene and conversation history together."""
        self._reset_requested = True
        self._activity.set()

    def request_task(self, name: str) -> None:
        """Ask for a different task's scene to be loaded.

        With no episode running it is applied at once. During an episode it ends that episode:
        the motion layer reads :attr:`task_request_pending` beside :attr:`stop_requested`, the
        rest of the turn's calls come back skipped, and the loop raises out of its next turn.
        """
        self._task_requested = name
        self._activity.set()

    @property
    def task_request_pending(self) -> bool:
        """Whether a task switch is waiting to be applied. Read from the simulator's thread too."""
        return self._task_requested is not None

    def grant_tool_calls(self, count: int) -> None:
        """Answer the loop's request for more budget; a count of zero ends the episode.

        :param count: Additional tool calls the human is granting
        """
        self._grants.put_nowait(count)

    # -- The loop side ------------------------------------------------------------------------

    def has_pending(self) -> bool:
        """Whether anything has been typed since :meth:`take` was last called."""
        return bool(self._pending)

    def take(self) -> list[str]:
        """Return every message received since the last call, clearing the queue."""
        taken, self._pending = self._pending, []
        return taken

    def take_task_request(self) -> str | None:
        """The task most recently asked for since this was last called, clearing the request."""
        requested, self._task_requested = self._task_requested, None
        return requested

    def take_reset_request(self) -> bool:
        """Whether a reset was asked for since this was last called, clearing the request."""
        requested, self._reset_requested = self._reset_requested, False
        return requested

    async def wait_for_activity(self) -> None:
        """Block until the human does anything: types, pauses, resumes, or asks for a reset.

        For a server with no episode running, which has no turn boundary to poll at. Re-check
        the state you care about after waking; several things may have happened at once.
        """
        await self._activity.wait()
        self._activity.clear()

    async def wait_until_resumed(self) -> None:
        """Block until the loop is allowed to start a turn. Returns at once when not paused."""
        await self._resumed.wait()

    async def next_grant(self) -> int:
        """Block until the human answers a request for more tool calls.

        :return: Additional tool calls granted; zero to end the episode
        """
        return await self._grants.get()
