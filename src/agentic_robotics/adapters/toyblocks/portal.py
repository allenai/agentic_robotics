"""Render the agentic run directories as browsable HTML, one page per episode.

A run directory holds `transcript.jsonl` (one Responses API item per line), the
images those items carried spilled into `images/`, and a video. Read as text, the line
that decides an episode is a `pick` with a normalized point and a filename,
which says nothing about whether the point was on a block. So this script draws the
point on the image and puts the frame before beside the frame after.

Manual-driver runs are also rendered: they hold `calls.jsonl` and `frame_NNNN.png`
instead of a transcript, and their per-call records carry the same point.

Only stdlib is used, so this runs under either venv.

`agentic-robotics portal` serves it, re-reading the logs on every request so a run in progress
can be watched by refreshing; `agentic-robotics portal --static` instead writes
`<runs root>/portal/`, self-contained, so the whole runs tree can be copied off the session.

Port 8082, because 8000 is the policy server, 8080 Viser, and 8081 the manual driver.
Forward it to your laptop the same way you reach the Viser viewer.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from itertools import pairwise
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

DEFAULT_RUNS_ROOT = Path(
    os.environ.get("AGENTIC_ROBOTICS_RUNS_DIR", Path(__file__).resolve().parents[4] / "runs")
)
PORTAL_DIR_NAME = "portal"
FACT_SEPARATOR = " \u00b7 "
"""Hoisted out of the f-string that joins on it, which 3.11 cannot parse inline."""

TRANSCRIPT_NAME = "transcript.jsonl"
MANIFEST_NAME = "run.json"
CALLS_NAME = "calls.jsonl"
IMAGE_DIR_NAME = "images"

REASONING_TEXT_TYPES = frozenset({"reasoning_text", "summary_text", "text"})
MESSAGE_TEXT_TYPES = frozenset({"output_text", "input_text", "text"})

TOOL_FAILURE_PREFIX = "The tool call failed"
"""What core.mcp_conversion prepends to an errored result. All a run logged before the loop
recorded an outcome says about failure, and it marks only the is_error case."""

TOOL_FAILED_OUTCOME = "Tool Failure"
"""The loop's outcome for a call the server ran and reported an error for."""

NEVER_EXECUTED_OUTCOMES = frozenset({"Invalid JSON", "Tool Raised Exception"})
"""Outcomes for a call whose arguments never parsed, or whose dispatch raised client-side."""

BLINK_MS = 700
"""How long each frame is held when blinking between the pre- and post-state."""


# --------------------------------------------------------------------------------------
# Reading the logs
# --------------------------------------------------------------------------------------


@dataclass
class Call:
    """One tool call, with the thinking that led to it and the frames around it."""

    index: int
    tool: str
    point: tuple[int, int] | None
    """Exactly the numbers the model typed. Never a converted copy: a readout showing a point
    the model did not send is the same lie whether the reader is the model or a person."""

    arguments: dict[str, Any]

    point_label: str = "x,y"
    """Which coordinate space `point` is in, named the way the tool's own parameter names it."""

    point_pct: tuple[float, float] | None = None
    """Where to place the crosshair, as percentages of the frame."""

    thoughts: list[str] = field(default_factory=list)
    """Reasoning text the model emitted before this call. Its signature is never read."""

    says: list[str] = field(default_factory=list)
    """Assistant messages the model emitted before this call."""

    interjections: list[str] = field(default_factory=list)
    """What a watching human typed before this call. The opening prompt is not one of these:
    the first user message is the task, and every later one is somebody correcting the run."""

    pre_image: str | None = None
    post_image: str | None = None
    """Paths relative to the runs root, so both link modes can reach them."""

    result_text: list[str] = field(default_factory=list)

    orphaned: bool = False
    """True when the call never executed: it raised, or no output was ever logged for it."""

    failed: bool = False
    """True when the call executed and reported an error, which is a different thing."""

    seconds: float | None = None
    approx_seconds: bool = False

    frames: tuple[int, int] | None = None
    """Video frames this call wrote, as [first, past-last). Equal bounds mean it wrote none."""

    episode: int = 0
    """Which episode of the transcript this call belongs to; a run without resets has only 0."""

    started_at: str = ""
    """Wall clock the call was dispatched, as the loop recorded it."""

    think_seconds: float | None = None
    """How long the model took to choose this call: the gap since the previous call finished.
    Nothing times a model turn directly, and this is the whole of what the transcript can say."""


@dataclass
class Run:
    """One episode's directory, parsed into something renderable."""

    name: str
    source: str
    """Which log the calls came from, or why there are none."""

    status: str
    status_class: str

    prompt: list[str] = field(default_factory=list)
    calls: list[Call] = field(default_factory=list)
    final_says: list[str] = field(default_factory=list)
    final_thoughts: list[str] = field(default_factory=list)
    final_interjections: list[str] = field(default_factory=list)
    """Anything a human typed after the last call, which no call can carry."""

    video: str | None = None
    first_image: str | None = None
    last_image: str | None = None

    skipped_lines: int = 0

    @property
    def tool_counts(self) -> dict[str, int]:
        """How many times each tool was called, in first-call order."""
        counts: dict[str, int] = {}
        for call in self.calls:
            counts[call.tool] = counts.get(call.tool, 0) + 1
        return counts

    @property
    def num_failed(self) -> int:
        """How many calls either failed or never executed."""
        return sum(1 for call in self.calls if call.failed or call.orphaned)


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    """Read a JSONL file, tolerating lines that are blank or were never finished.

    A transcript ends with a blank line, and a run killed mid-write leaves a partial
    one, so an unreadable line is counted rather than allowed to end the parse.

    :param path: Path to the JSONL file
    :return: The parsed objects, and how many lines could not be read
    """
    items: list[dict[str, Any]] = []
    skipped = 0

    for line in path.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except ValueError:
            skipped += 1
            continue

        if isinstance(parsed, dict):
            items.append(parsed)
        else:
            skipped += 1

    return items, skipped


def _texts(item: dict[str, Any], allowed: frozenset[str]) -> list[str]:
    """Pull the plain text out of a conversation item's content blocks.

    :param item: Conversation item
    :param allowed: Block types whose text is collected
    :return: The item's text, in order
    """
    content = item.get("content")
    if isinstance(content, str):
        return [content] if content.strip() else []
    if not isinstance(content, list):
        return []

    texts = []
    for block in content:
        if not isinstance(block, dict) or block.get("type") not in allowed:
            continue
        text = block.get("text")
        if isinstance(text, str) and text.strip():
            texts.append(text)

    return texts


def _reasoning_texts(item: dict[str, Any]) -> list[str]:
    """Pull a reasoning item's text, ignoring its multi-KB signature entirely."""
    summary = item.get("summary")
    summarized = _texts({"content": summary}, REASONING_TEXT_TYPES) if summary else []
    return summarized + _texts(item, REASONING_TEXT_TYPES)


def _output_blocks(item: dict[str, Any]) -> tuple[list[str], str | None]:
    """Separate a function call output into its text and its image filename.

    :param item: A `function_call_output` item, whose images the transcript replaced
        with the filename it wrote them to
    :return: The output's text blocks, and the last image's filename if it had one
    """
    output = item.get("output")
    if isinstance(output, str):
        return ([output] if output.strip() else []), None
    if not isinstance(output, list):
        return [], None

    texts: list[str] = []
    image: str | None = None

    for block in output:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "input_image":
            url = block.get("image_url")
            if isinstance(url, str) and url and ";base64," not in url:
                image = url
        else:
            text = block.get("text")
            if isinstance(text, str) and text.strip():
                texts.append(text)

    return texts, image


def _mtime(path: Path) -> float | None:
    """Return a file's modification time, or None if it is not there."""
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _find_video(run_dir: Path) -> str | None:
    """Return the run's video, relative to the runs root, if it wrote one."""
    videos = sorted(run_dir.glob("*.mp4"))
    return f"{run_dir.name}/{videos[0].name}" if videos else None


def _observation_resolution(run_dir: Path) -> tuple[int, int] | None:
    """The resolution a run's raw pixel arguments are expressed in, per its manifest.

    :param run_dir: The run's directory
    :return: Width and height, or None when the run wrote no manifest or an older schema
    """
    try:
        manifest = json.loads((run_dir / MANIFEST_NAME).read_text())
        resolution = manifest["observation"]["observation_resolution"]
        return int(resolution["width"]), int(resolution["height"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _load_transcript_run(run_dir: Path) -> Run:
    """Parse an agentic run, whose calls come from the transcript the host wrote.

    :param run_dir: The run's directory
    :return: The parsed run
    """
    items, skipped = read_jsonl(run_dir / TRANSCRIPT_NAME)
    obs_resolution = _observation_resolution(run_dir)
    run = Run(name=run_dir.name, source=TRANSCRIPT_NAME, status="", status_class="warn")
    run.skipped_lines = skipped
    run.video = _find_video(run_dir)

    outputs = {
        item.get("call_id"): item for item in items if item.get("type") == "function_call_output"
    }

    thoughts: list[str] = []
    says: list[str] = []
    interjections: list[str] = []

    for item in items:
        item_type = item.get("type")
        role = item.get("role")

        if role == "user" and item_type in (None, "message"):
            # The first user message is the task. A later one is a human interrupting a run in
            # progress, and folding it into the prompt would show the agent as having been told
            # up front what somebody actually had to catch and correct mid-episode.
            (run.prompt if not run.prompt else interjections).extend(
                _texts(item, MESSAGE_TEXT_TYPES)
            )
        elif item_type == "reasoning":
            thoughts.extend(_reasoning_texts(item))
        elif item_type == "message" or (item_type is None and role == "assistant"):
            says.extend(_texts(item, MESSAGE_TEXT_TYPES))
        elif item_type == "function_call":
            call = _build_call(item, len(run.calls), outputs, obs_resolution)
            call.thoughts, call.says = thoughts, says
            call.interjections = interjections
            thoughts, says, interjections = [], [], []
            run.calls.append(call)

    run.final_thoughts, run.final_says = thoughts, says
    run.final_interjections = interjections

    _think_times(run.calls)
    _pair_images(run.calls, run_dir, f"{run_dir.name}/{IMAGE_DIR_NAME}")
    _time_calls(run.calls, run_dir)
    _describe_status(run, items)

    return run


def _build_call(
    item: dict[str, Any],
    index: int,
    outputs: dict[Any, dict[str, Any]],
    obs_resolution: tuple[int, int] | None = None,
) -> Call:
    """Turn one `function_call` item and its matching output into a Call.

    :param item: The `function_call` item
    :param index: Position of this call within the episode
    :param outputs: Every `function_call_output` in the transcript, keyed by call id
    :return: The call, without its images paired in yet
    """
    raw_arguments = item.get("arguments")
    try:
        arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else {}
    except ValueError:
        arguments = {"unparsed": raw_arguments}

    # Two conventions reach this: `point` is an ImagePoint normalized to [0,1000), and `pixel`
    # is a PixelCoord in the observation's own resolution. The oracle skills take the second,
    # so reading only the first would draw nothing for any run they produce.
    point = point_pct = None
    point_label = "x,y"
    arguments = arguments if isinstance(arguments, dict) else {}

    normalized = arguments.get("point")
    raw = arguments.get("pixel")

    if isinstance(normalized, dict) and "x" in normalized and "y" in normalized:
        point = (int(normalized["x"]), int(normalized["y"]))
        point_pct = (point[0] / 10, point[1] / 10)
    elif isinstance(raw, dict) and "u" in raw and "v" in raw:
        point = (int(raw["u"]), int(raw["v"]))
        point_label = "u,v"
        if obs_resolution is not None:
            width, height = obs_resolution
            point_pct = (point[0] / width * 100, point[1] / height * 100)

    call = Call(
        index=index,
        tool=str(item.get("name", "?")),
        point=point,
        arguments=arguments,
        point_label=point_label,
        point_pct=point_pct,
    )

    output = outputs.get(item.get("call_id"))
    if output is None:
        call.orphaned = True
    else:
        call.result_text, call.post_image = _output_blocks(output)

        # The loop nests everything it records about a call under one key, so the rest of the
        # line stays exactly the Responses API item it was.
        metadata = output.get("metadata") or {}
        call.seconds = metadata.get("duration_s")
        call.episode = metadata.get("episode_idx", 0)
        call.started_at = metadata.get("started_at", "")

        # The loop samples an opaque payload from the adapter around each call; ToyBlocks's
        # carries the video's frame counter, which is what turns a call into a clip.
        first = (metadata.get("pre") or {}).get("frame_count")
        last = (metadata.get("post") or {}).get("frame_count")
        if isinstance(first, int) and isinstance(last, int):
            call.frames = (first, last)

        outcome = metadata.get("outcome")
        if outcome is None:
            call.failed = any(text.startswith(TOOL_FAILURE_PREFIX) for text in call.result_text)
        else:
            # Four outcomes, of which the prepended sentence above marks one. A call whose
            # arguments never parsed, or whose dispatch raised, did not reach the server at all.
            call.failed = outcome == TOOL_FAILED_OUTCOME
            call.orphaned = outcome in NEVER_EXECUTED_OUTCOMES

    return call


def _load_calls_run(run_dir: Path) -> Run:
    """Parse a manual-driver run, whose calls come from the per-call log it wrote.

    :param run_dir: The run's directory
    :return: The parsed run
    """
    records, skipped = read_jsonl(run_dir / CALLS_NAME)
    run = Run(name=run_dir.name, source=CALLS_NAME, status="", status_class="ok")
    run.skipped_lines = skipped
    run.video = _find_video(run_dir)
    run.prompt = ["Driven by hand from the manual skill driver; no model was in the loop."]

    frames = sorted(path.name for path in run_dir.iterdir() if path.name.startswith("frame_"))

    for index, record in enumerate(records):
        point = record.get("image_point") or {}
        call = Call(
            index=index,
            tool=str(record.get("tool", "?")),
            point=(int(point["x"]), int(point["y"])) if "x" in point and "y" in point else None,
            arguments={key: record[key] for key in ("pixel_coord",) if key in record},
        )
        call.seconds = record.get("duration_s")
        call.result_text = [str(text) for text in record.get("text") or []]
        call.failed = bool(record.get("is_error"))

        raised = record.get("raised")
        if raised:
            call.orphaned = True
            call.result_text.append(str(raised))

        frame_id = record.get("frame_id")
        if isinstance(frame_id, int):
            # A run cut short can name a frame it never wrote, while the frame the call
            # acted on is already on disk, so the two are resolved independently.
            if 0 <= frame_id < len(frames):
                call.post_image = f"{run_dir.name}/{frames[frame_id]}"
            if 0 < frame_id <= len(frames):
                call.pre_image = f"{run_dir.name}/{frames[frame_id - 1]}"

        run.calls.append(call)

    # A call that errored names no frame of its own, but the frame it acted on is known,
    # and seeing where a failed call was pointing is the whole reason to look at it.
    previous = f"{run_dir.name}/{frames[0]}" if frames else None
    for call in run.calls:
        if call.pre_image is None:
            call.pre_image = previous
        previous = call.post_image or call.pre_image or previous

    if frames:
        run.first_image = f"{run_dir.name}/{frames[0]}"
        run.last_image = f"{run_dir.name}/{frames[-1]}"

    calls = len(run.calls)
    failures = sum(1 for call in run.calls if call.failed or call.orphaned)
    run.status = f"{calls} hand-issued call{'' if calls == 1 else 's'}"
    if failures:
        run.status += f", {failures} failed"
        run.status_class = "bad"

    return run


def _pair_images(calls: list[Call], run_dir: Path, prefix: str) -> None:
    """Make each call's pre-state the previous frame the episode is known to have shown.

    A transcript records only the frame a call returned, so the frame a call acted on
    is the one before it. The first call has none unless the episode asked for an image.

    :param calls: The episode's calls, modified in place
    :param run_dir: The run's directory, used to drop images that are not on disk
    :param prefix: Path prefix, relative to the runs root, that the filenames sit under
    """
    runs_root = run_dir.parent
    previous: str | None = None

    for call in calls:
        if call.post_image is not None:
            relative = f"{prefix}/{call.post_image}"
            call.post_image = relative if (runs_root / relative).is_file() else None

        call.pre_image = previous
        if call.post_image is not None:
            previous = call.post_image


def _think_times(calls: list[Call]) -> None:
    """Fill in how long the model took before each call, where the run recorded enough to say.

    The gap between one call finishing and the next being dispatched. That is the model turn plus
    whatever the loop did around it, and no finer measurement exists in the transcript.

    :param calls: The episode's calls, modified in place
    """
    for previous, call in pairwise(calls):
        if not (previous.started_at and call.started_at and previous.seconds is not None):
            continue
        try:
            before = datetime.fromisoformat(previous.started_at).timestamp() + previous.seconds
            after = datetime.fromisoformat(call.started_at).timestamp()
        except ValueError:
            continue
        if after >= before:
            call.think_seconds = round(after - before, 1)


def _time_calls(calls: list[Call], run_dir: Path) -> None:
    """Approximate each call's wall-clock from the modification times of its images.

    Only for the calls that logged no duration of their own: a run from before the loop
    recorded one, or a call whose output never arrived. It is labelled as approximate
    wherever it is shown.

    :param calls: The episode's calls, modified in place
    :param run_dir: The run's directory
    """
    runs_root = run_dir.parent
    previous = _mtime(run_dir / TRANSCRIPT_NAME)

    for call in calls:
        stamp = _mtime(runs_root / call.post_image) if call.post_image else None

        if (
            call.seconds is None
            and stamp is not None
            and previous is not None
            and stamp >= previous
        ):
            call.seconds = round(stamp - previous, 1)
            call.approx_seconds = True

        # Advanced even for a call that logged its own duration, so the next inferred one
        # measures from the right place.
        if stamp is not None:
            previous = stamp


def _describe_status(run: Run, items: list[dict[str, Any]]) -> None:
    """Infer how the episode ended, which is the one thing the transcript's tail tells us.

    :param run: The run, whose status is set
    :param items: Every item read from its transcript
    """
    if not items:
        run.status, run.status_class = "empty transcript", "warn"
        return

    orphaned = [call.index for call in run.calls if call.orphaned]
    if orphaned:
        listed = ", ".join(f"#{index}" for index in orphaned)
        run.status = f"call {listed} left orphaned"
        run.status_class = "bad"
        return

    last = items[-1].get("type")
    if last == "message" or (last is None and items[-1].get("role") == "assistant"):
        run.status, run.status_class = "ended with a final answer", "ok"
    else:
        run.status, run.status_class = f"ended mid-stream on a {last} item", "warn"

    if run.calls:
        first = run.calls[0]
        run.first_image = first.pre_image or first.post_image
    for call in reversed(run.calls):
        if call.post_image is not None:
            run.last_image = call.post_image
            break


def load_run(run_dir: Path) -> Run:
    """Parse whichever log a run directory turns out to hold.

    :param run_dir: The run's directory
    :return: The parsed run, which may carry no calls at all
    """
    if (run_dir / TRANSCRIPT_NAME).is_file():
        return _load_transcript_run(run_dir)
    if (run_dir / CALLS_NAME).is_file():
        return _load_calls_run(run_dir)

    run = Run(
        name=run_dir.name,
        source="no log found",
        status=f"no {TRANSCRIPT_NAME} and no {CALLS_NAME}",
        status_class="bad",
    )
    run.video = _find_video(run_dir)
    return run


def discover_runs(runs_root: Path, skip: Path | None = None) -> list[Path]:
    """List the non-empty run directories under the runs root, most recently written first.

    Ordering is by directory mtime, not by name. A run's name is a caller-chosen prefix plus a
    timestamp, so the prefixes sort against each other rather than the dates: a `tamp-inline-`
    run would lead the index over far newer `menu-` runs purely because `t` follows `m`. An mtime is
    the last write into the directory, which for a finished run lands a couple of minutes after
    the timestamp in its own name, and it orders the runs whose names carry no date at all.
    Name breaks ties, so the order is stable.

    Empty directories are skipped. Every `agentic-robotics menu` launch creates a boot run directory
    it never writes to, and listing those would bury the runs that hold something under rows with
    nothing behind them. Deleting them is a treadmill, since the next menu launch makes another; not
    listing them is not. A run that starts and dies still leaves its directory on disk as a record
    either way.

    :param runs_root: Directory holding one subdirectory per run
    :param skip: Directory to leave out, so a generated portal is not itself a run
    :return: The run directories that hold at least one entry, newest first
    """
    return sorted(
        (
            path
            for path in runs_root.iterdir()
            if path.is_dir() and any(path.iterdir()) and (skip is None or path != skip)
        ),
        key=lambda path: (path.stat().st_mtime, path.name),
        reverse=True,
    )


# --------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Urls:
    """Where the pages link, which differs between static files and the served app."""

    served: bool

    def index(self) -> str:
        """Link to the run index."""
        return "/" if self.served else "index.html"

    def calls(self) -> str:
        """Link to the cross-run call table."""
        return "/calls" if self.served else "calls.html"

    def run(self, name: str) -> str:
        """Link to one run's page."""
        return f"/run/{name}" if self.served else f"run-{name}.html"

    def asset(self, relative: str) -> str:
        """Link to a file inside the runs root, named relative to it."""
        return f"/file/{relative}" if self.served else f"../{relative}"


STYLE = """
:root {
  color-scheme: dark;
  /* Strata primitives, under their own names so a value can be checked against
     strata/DESIGN.md. Kept identical to web_ui/index.html, which is the other page
     a developer sees during a run. */
  --extra-dark-teal: #032629; --dark-teal: #0a3235; --teal-60: #6e9291;
  --cream: #faf2e9; --pink: #f0529c; --green: #0fcb8c;
  --error-red: #fd4645; --warning-orange: #ffa31c;
  --cream-10: #faf2e91a; --cream-20: #faf2e933; --cream-50: #faf2e980; --cream-70: #faf2e9b3;

  --bg: var(--extra-dark-teal);
  --surface: var(--dark-teal);
  --edge: var(--cream-20);
  --text: var(--cream);
  --dim: var(--cream-70);
  --faint: var(--cream-50);
  --accent: var(--pink);
  --interactive: var(--teal-60);
  --ok: var(--green);
  --err: var(--error-red);
  --warn: var(--warning-orange);

  --ui: "Manrope", Arial, sans-serif;
  --mono: "Roboto Mono", ui-monospace, "SF Mono", Menlo, Consolas, monospace;

  --space-2xs: 8px; --space-xs: 12px; --space-md: 16px; --space-xl: 20px;
  --radius: 4px;
  --stage: 640px;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text); font: 400 14px/1.5 var(--ui); }
a { color: var(--interactive); text-decoration: none; }
a:hover { text-decoration: underline; }
header { display: flex; align-items: baseline; gap: var(--space-md); flex-wrap: wrap;
         padding: var(--space-xs) var(--space-xl); border-bottom: 1px solid var(--edge);
         position: sticky; top: 0; background: var(--bg); z-index: 5; }
header h1 { font-size: 15px; font-weight: 600; margin: 0; letter-spacing: .01em; }
header nav { display: flex; gap: var(--space-xs); font-size: 13px; }
header .controls { margin-left: auto; display: flex; gap: var(--space-md); align-items: center;
                   font-size: 12px; color: var(--dim); }
main { padding: var(--space-xl); display: flex; flex-direction: column; gap: var(--space-xl); }
.badge { font-size: 12px; padding: 2px 10px; border-radius: 999px; background: var(--cream-10);
         border: 1px solid var(--edge); color: var(--dim); white-space: nowrap; }
.badge.ok { border-color: var(--ok); color: var(--ok); }
.badge.warn { border-color: var(--warn); color: var(--warn); }
.badge.bad { border-color: var(--err); color: var(--err); }
.card { border: 1px solid var(--edge); border-radius: var(--radius); padding: var(--space-md);
        background: var(--surface); }
.card h2 { font-size: 12px; text-transform: uppercase; letter-spacing: .06em; color: var(--faint);
           margin: 0 0 10px; font-weight: 600; }
.muted { color: var(--faint); }
.mono { font: 400 12px/1.5 var(--mono); }
.nums { font-variant-numeric: tabular-nums; }
.summary { display: flex; gap: var(--space-xl); align-items: flex-start; flex-wrap: wrap; }
.summary .card { flex: 1 1 320px; }
.summary video { width: 480px; max-width: 100%; border: 1px solid var(--edge);
                 border-radius: var(--radius); }
.strip { display: flex; gap: var(--space-2xs); overflow-x: auto; padding-bottom: 6px; }
.strip a { flex: 0 0 auto; text-align: center; font-size: 11px; color: var(--faint);
           line-height: 1.3; }
.strip img { display: block; width: 128px; border: 1px solid var(--edge);
             border-radius: var(--radius); }
.strip a.bad img { border-color: var(--err); }
.episode { font-size: 12px; text-transform: uppercase; letter-spacing: .06em; font-weight: 600;
           color: var(--accent); margin: var(--space-2xs) 0 0;
           border-top: 1px solid var(--edge); padding-top: var(--space-md); }
.call { border: 1px solid var(--edge); border-radius: var(--radius); background: var(--surface);
        padding: var(--space-md); scroll-margin-top: 64px; }
.call h2 { display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
           margin: 0 0 var(--space-xs); font-size: 14px; text-transform: none; letter-spacing: 0;
           color: var(--text); }
.call h2 .idx { color: var(--faint); font-variant-numeric: tabular-nums; }
.call h2 .tool { font-weight: 600; }
.pair { display: flex; gap: var(--space-md); flex-wrap: wrap; }
.stage { position: relative; width: var(--stage); max-width: 100%; margin: 0; line-height: 0; }
.stage img { display: block; width: 100%; height: auto; image-rendering: pixelated;
             border: 1px solid var(--edge); border-radius: var(--radius); background: #000; }
.stage figcaption { line-height: 1.4; padding-top: 5px; font-size: 12px; color: var(--faint); }
.stage.missing { display: flex; align-items: center; justify-content: center; height: 200px;
                 border: 1px dashed var(--edge); border-radius: var(--radius); color: var(--faint);
                 font-size: 12px; line-height: 1.4; text-align: center; padding: var(--space-2xs); }
.mark { position: absolute; width: 0; height: 0; pointer-events: none; }
.mark::before, .mark::after { content: ""; position: absolute; background: var(--accent); }
.mark::before { left: -10px; top: 0; width: 21px; height: 1px; }
.mark::after { top: -10px; left: 0; height: 21px; width: 1px; }
.mark i { position: absolute; left: -6px; top: -6px; width: 13px; height: 13px;
          border: 1px solid var(--accent); border-radius: 50%; }
body.blinking .pair[data-pre] .stage.post { display: none; }
details { margin-top: var(--space-xs); }
summary { cursor: pointer; font-size: 12px; color: var(--faint); }
details p { white-space: pre-wrap; margin: var(--space-2xs) 0 0; color: var(--dim);
            max-width: 90ch; }
.said { white-space: pre-wrap; margin: 0 0 10px; max-width: 90ch; }
.interjection { white-space: pre-wrap; margin: 0 0 var(--space-xs); max-width: 90ch;
                border-left: 3px solid var(--green); padding: var(--space-2xs) var(--space-xs);
                background: var(--cream-10); color: var(--text); }
.interjection::before { content: "a human interjected"; display: block; font-size: 11px;
                        text-transform: uppercase; letter-spacing: .06em; color: var(--green);
                        margin-bottom: 4px; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { text-align: left; padding: var(--space-2xs) 10px; border-bottom: 1px solid var(--edge);
         vertical-align: middle; }
th { font-size: 11px; text-transform: uppercase; letter-spacing: .06em; color: var(--faint); }
tbody tr:hover { background: var(--cream-10); }
td img { display: block; width: 112px; border: 1px solid var(--edge);
         border-radius: var(--radius); }
input[type=search], select { font: inherit; font-size: 13px; padding: 5px 9px;
       border-radius: var(--radius); border: 1px solid var(--edge); background: var(--bg);
       color: var(--text); }
input[type=search]:focus, select:focus { outline: 2px solid var(--interactive);
                                         outline-offset: 1px; }
footer { color: var(--faint); font-size: 12px; padding: 6px var(--space-xl) 24px; max-width: 100ch;
         line-height: 1.6; }
"""

SCRIPT = """
const one = (sel, root) => (root || document).querySelector(sel);
const all = (sel, root) => [...(root || document).querySelectorAll(sel)];

function applyZoom() {
  const zoom = one("#zoom");
  if (zoom) document.documentElement.style.setProperty("--stage", zoom.value + "px");
}

function labelPixels() {
  // The transcript stores a normalized point, so the raw pixel needs the image itself.
  for (const stage of all(".stage")) {
    const img = one("img", stage), out = one(".uv", stage);
    if (!img || !out || !stage.dataset.x) continue;
    const show = () => {
      if (!img.naturalWidth) return;
      const u = Math.floor(+stage.dataset.x / 1000 * img.naturalWidth);
      const v = Math.floor(+stage.dataset.y / 1000 * img.naturalHeight);
      out.textContent = `u,v = ${u},${v} in ${img.naturalWidth}x${img.naturalHeight}`;
    };
    img.complete ? show() : img.addEventListener("load", show);
  }
}

let blinkTimer = null, blinkPost = true;

function paintBlink() {
  for (const pair of all(".pair[data-pre]")) {
    const img = one(".stage.pre img", pair);
    if (img) img.src = blinkPost ? pair.dataset.post : pair.dataset.pre;
  }
}

function applyBlink() {
  const on = one("#blink") && one("#blink").checked;
  document.body.classList.toggle("blinking", !!on);
  if (blinkTimer) { clearInterval(blinkTimer); blinkTimer = null; }
  blinkPost = false;
  paintBlink();
  if (!on) return;
  blinkTimer = setInterval(() => { blinkPost = !blinkPost; paintBlink(); }, BLINK_MS);
}

function filterRows() {
  const needle = (one("#filter") ? one("#filter").value : "").toLowerCase().trim();
  let shown = 0;
  for (const row of all("[data-row]")) {
    const visible = !needle || row.dataset.row.toLowerCase().includes(needle);
    row.style.display = visible ? "" : "none";
    if (visible) shown += 1;
  }
  const count = one("#shown");
  if (count) count.textContent = shown + " shown";
}

for (const [id, handler] of [["zoom", applyZoom], ["blink", applyBlink],
                             ["filter", filterRows]]) {
  const node = one("#" + id);
  if (node) node.addEventListener(id === "filter" ? "input" : "change", handler);
}

applyZoom(); labelPixels(); filterRows();
"""


def _page(title: str, header: str, body: str, urls: Urls) -> str:
    """Wrap rendered body markup in the shared page shell.

    :param title: Document title
    :param header: Markup for the sticky header's leading content
    :param body: Markup for the page's main content
    :param urls: Link scheme in use
    :return: A complete HTML document
    """
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(title)}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700&family=Roboto+Mono:wght@400;500&display=swap">
<style>{STYLE}</style>
</head>
<body>
<header>
  <h1>agentic run portal</h1>
  <nav><a href="{urls.index()}">runs</a><a href="{urls.calls()}">all calls</a></nav>
  {header}
</header>
{body}
<script>const BLINK_MS = {BLINK_MS};{SCRIPT}</script>
</body>
</html>
"""


def _badge(text: str, kind: str = "") -> str:
    """Render a pill-shaped label."""
    return f'<span class="badge {kind}">{escape(text)}</span>'


def _stage(
    image: str | None,
    caption: str,
    call: Call,
    urls: Urls,
    role: str,
) -> str:
    """Render one frame with the requested point drawn on it.

    The point is normalized to [0,1000), so placing the crosshair is percentage math
    and needs no image decode. The raw pixel is filled in by the page once the image
    has loaded and its natural size is known.

    :param image: Path to the frame relative to the runs root, or None if there is none
    :param caption: Text shown under the frame
    :param call: The call this frame brackets
    :param urls: Link scheme in use
    :param role: Either `pre` or `post`, which the blink mode keys off
    :return: Markup for the frame
    """
    if image is None:
        return f'<figure class="stage missing {role}">{escape(caption)}</figure>'

    classes = f"stage {role}"
    data = ""
    mark = ""
    readout = ""

    if call.point is not None:
        x, y = call.point
        readout = f' <span class="uv nums muted">{call.point_label} = {x},{y}</span>'
        if call.point_label == "x,y":
            # The page fills in the raw pixel once it knows the image's natural size.
            data = f' data-x="{x}" data-y="{y}"'
        if call.point_pct is not None:
            left, top = call.point_pct
            mark = f'<div class="mark" style="left:{left:.2f}%;top:{top:.2f}%"><i></i></div>'

    return (
        f'<figure class="{classes}"{data}>'
        f'<img src="{urls.asset(image)}" alt="{escape(caption)}" loading="lazy">'
        f"{mark}"
        f"<figcaption>{escape(caption)}{readout}</figcaption>"
        f"</figure>"
    )


def _render_calls(calls: list[Call], urls: Urls) -> str:
    """Render every call, with a divider wherever the transcript started a new episode.

    A run that was never reset has one episode and no dividers, which is every run from before
    the loop recorded an episode index.

    :param calls: The run's calls, in order
    :param urls: Link scheme in use
    :return: Markup for the whole sequence
    """
    multi_episode = len({call.episode for call in calls}) > 1
    out: list[str] = []
    previous: int | None = None

    for call in calls:
        if multi_episode and call.episode != previous:
            out.append(f'<h2 class="episode">Episode {call.episode}</h2>')
        previous = call.episode
        out.append(_render_call(call, urls))

    return "".join(out)


def _render_call(call: Call, urls: Urls) -> str:
    """Render one call as a before-and-after block.

    :param call: The call to render
    :param urls: Link scheme in use
    :return: Markup for the call
    """
    interjected = "".join(
        f'<p class="interjection">{escape(text)}</p>' for text in call.interjections
    )

    flags = ""
    if call.failed:
        flags += _badge("executed and reported an error", "bad")
    if call.orphaned:
        flags += _badge("never executed", "bad")
    if call.seconds is not None:
        prefix = "~" if call.approx_seconds else ""
        flags += _badge(f"{prefix}{call.seconds:g}s")
    if call.frames is not None:
        first, last = call.frames
        # A call that wrote no frames is not a gap in the video: the simulator does not step
        # while the model is thinking, so the next call resumes at this same frame.
        flags += _badge(
            f"frames {first}\u2013{last} ({last - first})"
            if last > first
            else f"frame {first}, no video"
        )

    extra = {key: value for key, value in call.arguments.items() if key != "point"}
    if extra:
        flags += f'<span class="mono muted">{escape(json.dumps(extra))}</span>'

    pair_data = ""
    if call.pre_image is not None and call.post_image is not None:
        pair_data = (
            f' data-pre="{urls.asset(call.pre_image)}" data-post="{urls.asset(call.post_image)}"'
        )

    before = _stage(
        call.pre_image,
        "before" if call.pre_image else "no frame before this call was logged",
        call,
        urls,
        "pre",
    )
    after = _stage(
        call.post_image,
        "after" if call.post_image else "no frame returned by this call",
        call,
        urls,
        "post",
    )

    said = "".join(f'<p class="said">{escape(text)}</p>' for text in call.says)
    thoughts = ""
    if call.thoughts:
        joined = escape("\n\n".join(call.thoughts))
        thoughts = f"<details><summary>reasoning</summary><p>{joined}</p></details>"

    result = ""
    if call.result_text:
        joined = escape("\n".join(call.result_text))
        result = f'<p class="mono" style="color:#ff9d9d">{joined}</p>'

    return (
        f'<article class="call" id="call-{call.index}">'
        f"{interjected}"
        f'<h2><span class="idx">#{call.index}</span>'
        f'<span class="tool">{escape(call.tool)}</span>{flags}'
        f'<a class="muted" style="margin-left:auto;font-size:12px" '
        f'href="#call-{call.index}">permalink</a></h2>'
        f"{said}{result}"
        f'<div class="pair"{pair_data}>{before}{after}</div>'
        f"{thoughts}"
        f"</article>"
    )


def render_run_page(run: Run, urls: Urls) -> str:
    """Render one episode's page.

    :param run: The parsed run
    :param urls: Link scheme in use
    :return: A complete HTML document
    """
    header = (
        f"{_badge(run.name)}{_badge(run.status, run.status_class)}"
        '<div class="controls">'
        '<label>width <select id="zoom">'
        '<option value="320">320</option><option value="480">480</option>'
        '<option value="640" selected>640</option><option value="960">960</option>'
        "</select></label>"
        '<label><input type="checkbox" id="blink"> blink before/after</label>'
        "</div>"
    )

    counts = ", ".join(f"{count}x {tool}" for tool, count in run.tool_counts.items())
    facts = [
        f"{len(run.calls)} tool calls" + (f" ({counts})" if counts else ""),
        f"read from {run.source}",
    ]
    if run.num_failed:
        facts.append(f"{run.num_failed} failed or never executed")
    if run.skipped_lines:
        facts.append(f"{run.skipped_lines} unreadable log lines skipped")

    prompt = escape("\n\n".join(run.prompt)) if run.prompt else "no prompt logged"
    video = (
        f'<div class="card"><h2>Video</h2>'
        f'<video controls preload="metadata" src="{urls.asset(run.video)}"></video></div>'
        if run.video
        else ""
    )

    summary = (
        '<section class="summary">'
        f'<div class="card"><h2>Task</h2><p class="said">{prompt}</p>'
        f'<p class="muted mono">{escape(FACT_SEPARATOR.join(facts))}</p></div>'
        f"{video}"
        "</section>"
    )

    thumbs = []
    for call in run.calls:
        if call.post_image is None:
            continue
        kind = "bad" if call.orphaned or call.failed else ""
        thumbs.append(
            f'<a class="{kind}" href="#call-{call.index}">'
            f'<img src="{urls.asset(call.post_image)}" alt="call {call.index}" '
            f'loading="lazy"><br>#{call.index} {escape(call.tool)}</a>'
        )

    strip = ""
    if thumbs:
        strip = (
            f'<div class="card"><h2>Episode</h2><div class="strip">{"".join(thumbs)}</div></div>'
        )

    calls = _render_calls(run.calls, urls)
    if not run.calls:
        calls = f'<div class="card"><h2>Calls</h2><p class="muted">{escape(run.status)}</p></div>'

    final = ""
    if run.final_says or run.final_thoughts:
        said = "".join(f'<p class="said">{escape(text)}</p>' for text in run.final_says)
        thoughts = ""
        if run.final_thoughts:
            joined = escape("\n\n".join(run.final_thoughts))
            thoughts = f"<details><summary>reasoning</summary><p>{joined}</p></details>"

        evidence = ""
        if run.last_image:
            evidence = (
                f'<div class="card" style="flex:0 0 auto"><h2>What happened</h2>'
                f'<img style="width:480px;max-width:100%;border-radius:4px;display:block" '
                f'src="{urls.asset(run.last_image)}" alt="final frame"></div>'
            )

        final = (
            '<section class="summary">'
            f'<div class="card"><h2>What the agent says happened</h2>{said}{thoughts}</div>'
            f"{evidence}</section>"
        )

    footer = (
        "<footer>The transcript records no model id, seeds, checkpoint, dataset episode, "
        "steps executed or gripper state, so none of those appear above. Frame ranges are "
        "half-open and name frames in <code>video.mp4</code>; a call with no range wrote no "
        "video, and the next call resumes at the same frame. Durations are as logged, except "
        "where marked <code>~</code> and inferred from image file timestamps.</footer>"
    )

    body = f"<main>{summary}{strip}{calls}{final}</main>{footer}"
    return _page(f"{run.name} - run portal", header, body, urls)


def render_index_page(runs: list[Run], urls: Urls) -> str:
    """Render the index over every run directory.

    :param runs: Every parsed run, newest first
    :param urls: Link scheme in use
    :return: A complete HTML document
    """
    header = (
        f"{_badge(f'{len(runs)} runs')}"
        '<div class="controls">'
        '<input type="search" id="filter" placeholder="filter runs">'
        '<span id="shown" class="nums"></span></div>'
    )

    rows = []
    for run in runs:
        counts = ", ".join(f"{count}x {tool}" for tool, count in run.tool_counts.items())
        thumb = (
            f'<img src="{urls.asset(run.last_image)}" alt="final frame" loading="lazy">'
            if run.last_image
            else '<span class="muted mono">none</span>'
        )
        video = (
            f'<a href="{urls.asset(run.video)}">video</a>'
            if run.video
            else '<span class="muted">none</span>'
        )
        notes = []
        if run.num_failed:
            notes.append(f"{run.num_failed} failed")
        if run.skipped_lines:
            notes.append(f"{run.skipped_lines} bad lines")

        rows.append(
            f'<tr data-row="{escape(run.name)} {escape(run.status)} {escape(counts)}">'
            f"<td>{thumb}</td>"
            f'<td><a href="{urls.run(run.name)}"><b>{escape(run.name)}</b></a><br>'
            f'<span class="muted mono">{escape(run.source)}</span></td>'
            f"<td>{_badge(run.status, run.status_class)}</td>"
            f'<td class="nums">{len(run.calls)}<br>'
            f'<span class="muted mono">{escape(counts)}</span></td>'
            f'<td class="mono muted">{escape(", ".join(notes))}</td>'
            f"<td>{video}</td></tr>"
        )

    body = (
        '<main><div class="card"><table><thead><tr>'
        "<th>final frame</th><th>run</th><th>ending</th><th>calls</th><th>notes</th><th>video</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div></main>"
    )
    return _page("run portal", header, body, urls)


def render_calls_page(runs: list[Run], urls: Urls) -> str:
    """Render every call across every run as one filterable table.

    :param runs: Every parsed run
    :param urls: Link scheme in use
    :return: A complete HTML document
    """
    total = sum(len(run.calls) for run in runs)
    header = (
        f"{_badge(f'{total} calls')}"
        '<div class="controls">'
        '<input type="search" id="filter" placeholder="filter by run or tool">'
        '<span id="shown" class="nums"></span></div>'
    )

    rows = []
    for run in runs:
        for call in run.calls:
            point = f"{call.point[0]},{call.point[1]}" if call.point else ""
            thumb = (
                f'<img src="{urls.asset(call.post_image)}" alt="after" loading="lazy">'
                if call.post_image
                else '<span class="muted mono">none</span>'
            )
            flags = []
            if call.failed:
                flags.append(_badge("error", "bad"))
            if call.orphaned:
                flags.append(_badge("never ran", "bad"))

            rows.append(
                f'<tr data-row="{escape(run.name)} {escape(call.tool)}">'
                f"<td>{thumb}</td>"
                f'<td><a href="{urls.run(run.name)}#call-{call.index}">'
                f"{escape(run.name)}</a></td>"
                f'<td class="nums">#{call.index}</td>'
                f"<td><b>{escape(call.tool)}</b></td>"
                f'<td class="nums mono">{point}</td>'
                f"<td>{''.join(flags)}</td></tr>"
            )

    body = (
        '<main><div class="card"><table><thead><tr>'
        "<th>after</th><th>run</th><th>#</th><th>tool</th><th>x,y</th><th>flags</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div></main>"
    )
    return _page("all calls - run portal", header, body, urls)


# --------------------------------------------------------------------------------------
# The two modes
# --------------------------------------------------------------------------------------


def build_static(runs_root: Path, out_dir: Path) -> list[Path]:
    """Write the whole portal as files next to the runs.

    :param runs_root: Directory holding one subdirectory per run
    :param out_dir: Directory the pages are written to
    :return: Every path written
    """
    urls = Urls(served=False)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = [load_run(path) for path in discover_runs(runs_root, skip=out_dir)]
    written = []

    for run in runs:
        path = out_dir / f"run-{run.name}.html"
        path.write_text(render_run_page(run, urls))
        written.append(path)

    for name, page in (
        ("index.html", render_index_page(runs, urls)),
        ("calls.html", render_calls_page(runs, urls)),
    ):
        path = out_dir / name
        path.write_text(page)
        written.append(path)

    return written


class _RequestHandler(BaseHTTPRequestHandler):
    """Serves the portal, re-reading the logs on every request."""

    runs_root: Path
    """Bound onto a subclass by `serve`, since the server constructs handlers itself."""

    def log_message(self, format: str, *args: object) -> None:
        """Silence the default request logging, which is noise for a viewer.

        :param format: Format string supplied by the base handler
        :param args: Format arguments supplied by the base handler
        """

    def do_GET(self) -> None:
        """Serve the index, the call table, one run's page, or a file under the runs root."""
        route = unquote(urlparse(self.path).path)
        urls = Urls(served=True)

        if route in ("/", "/index.html"):
            runs = [load_run(path) for path in discover_runs(self.runs_root)]
            self._respond_html(render_index_page(runs, urls))
        elif route in ("/calls", "/calls.html"):
            runs = [load_run(path) for path in discover_runs(self.runs_root)]
            self._respond_html(render_calls_page(runs, urls))
        elif route.startswith("/run/"):
            self._serve_run(route.removeprefix("/run/"), urls)
        elif route.startswith("/file/"):
            self._serve_file(route.removeprefix("/file/"))
        else:
            self._respond(404, "text/plain", b"not found")

    def _serve_run(self, name: str, urls: Urls) -> None:
        """Serve one run's page, refusing a name that is not a run directory.

        :param name: Name of the run directory
        :param urls: Link scheme in use
        """
        run_dir = self._resolve(name)
        if run_dir is None or not run_dir.is_dir():
            self._respond(404, "text/plain", b"no such run")
            return

        self._respond_html(render_run_page(load_run(run_dir), urls))

    def _serve_file(self, relative: str) -> None:
        """Serve an image or video from inside the runs root.

        :param relative: Path to the file, relative to the runs root
        """
        path = self._resolve(relative)
        if path is None or not path.is_file():
            self._respond(404, "text/plain", b"not found")
            return

        types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".mp4": "video/mp4",
        }
        self._respond(
            200, types.get(path.suffix.lower(), "application/octet-stream"), path.read_bytes()
        )

    def _resolve(self, relative: str) -> Path | None:
        """Resolve a request path under the runs root, or None if it escapes it.

        :param relative: Path relative to the runs root
        :return: The resolved path, or None if it lies outside the runs root
        """
        root = self.runs_root.resolve()
        path = (root / relative).resolve()
        return path if path == root or root in path.parents else None

    def _respond_html(self, page: str) -> None:
        """Write a complete HTML response.

        :param page: The document to send
        """
        self._respond(200, "text/html; charset=utf-8", page.encode())

    def _respond(self, status: int, content_type: str, payload: bytes) -> None:
        """Write a complete response.

        :param status: HTTP status code
        :param content_type: Value for the Content-Type header
        :param payload: Response body
        """
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)


def serve(runs_root: Path, host: str, port: int) -> None:
    """Serve the portal until interrupted, rebuilding each page on request.

    :param runs_root: Directory holding one subdirectory per run
    :param host: Interface to bind
    :param port: Port to bind
    """

    class BoundHandler(_RequestHandler):
        """A handler class carrying this server's runs root."""

    BoundHandler.runs_root = runs_root

    server = ThreadingHTTPServer((host, port), BoundHandler)
    print(f"[portal] reading runs from {runs_root}")
    print(f"[portal] serving at http://localhost:{port}/ (forward port {port})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[portal] stopped")
    finally:
        server.server_close()
