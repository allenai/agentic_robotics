"""Build a `timeline.json` from a finished run: what a video of it should show, and when.

The run's clocks disagree. A twenty-minute run has ten minutes of footage, since frames exist only
while a skill executes, and carries more reasoning than anyone will read. So a video of a run is
an edit, and this file is where the edit is decided. It draws nothing: `video_render.py` draws
whatever this emits, so a disagreement about pacing or selection is a change here, and the same
timeline could drive a player page.

The edit follows four rules:

**Nothing on screen is authored.** Every word the video shows is a substring of the run's own
transcript or manifest: the task sentence is cut from the real prompt between its fixed markers,
the model id is the manifest's, reasoning is quoted verbatim with an ellipsis wherever text was
cut, and the model's final message is shown as it was written. The only text we write is labels
for mechanism -- a tool chip, "took 18.4 s", "words in full". No phase labels, no paraphrase of
what the model was doing. The interesting output of this project is how the loop reasons and
fails, and a narration would be a second author's opinion of that.

**Selection is explicit and reviewable.** Which span of a thought is quoted, which words are
bold, and which calls are highlighted live in an `edit.json` beside the run, keyed by call index.
`--draft-edit` writes one from the mechanical defaults below so a developer edits rather than
starts from nothing; `build` refuses a quote that is not a substring of the thought, so an edit
cannot put words in the model's mouth by mistake. With no `edit.json` the defaults apply. A
`quote` may be a list of spans, each verbatim and in the order the model wrote them; they are shown
with an ellipsis between them, the same mark the renderer uses for a cut at either end.

**Show real durations, do not play them.** A skill clip is sped up towards `ACT_TARGET_SECONDS`
with its last `SLOW_TAIL_FRAMES` at 1x, and the caption says the factor. The footer carries the
real elapsed clock and the real per-call time, so a viewer reading eight seconds of a rotate knows
it took thirty-nine.

**Motion never stops and never slows.** A skill clip is never played below real time, and a
segment with footage lasts exactly as long as its clip, never holding the last frame for the
quote to be read. A quote that needs longer than the target
lowers the speed-up towards 1x instead (the clip lasts up to its real duration); a quote that needs
longer than the clip is real is cut short, and the viewer pauses. Reading times are scaled by
`READING_SCALE`, so the panel dwells less than the words strictly need. The only held frames
are ones with nothing to play: the final card, a `segments`-mode look, an interjection, and a skill
that wrote no footage.

Perception has two treatments, chosen with `--looks`. `folded` (the default) shows the zooms the
model requested as a thumbnail row inside the skill segment they led to, so looking costs no screen
time. `segments` gives each batch of consecutive looks a short segment of its own: the held frame
shows where each zoom was cropped from and the panel shows what came back. The first makes a
video about the robot; the second makes one about the loop.

Default quote: the tail of the thought, whole sentences from the end up to `EXCERPT_WORDS`,
because in every transcript read so far the reasoning ends with the decision. Default bold: its
last sentence. Default highlight: none.

`--cut short` is the whole run, every skill's footage intact, in `SHORT_TARGET_SECONDS`, with
reading time given only to the two or three calls the edit highlights. Every other skill joins a montage: one segment playing its calls back to back
at one speed, the chip and the quote changing as the footage passes. A montage call carries the
same quote a skill segment would, shown plain with no bold span so that emphasis marks only the
highlights, but the montage is paced by its footage alone, so the quote is on screen only as long
as the clip lasts: a viewer who wants it pauses, and the cut does not wait. The montage speed
is whatever spends the seconds the highlights and final card leave over, clamped to
`MONTAGE_SPEED_RANGE`. Highlighted skills are shown exactly as in the full cut, so a highlight
there reads the same as its slide still.

    python examples/agent/video_timeline.py <run-dir> [--cut full|short] [--looks folded|segments] [--out PATH] [--edit PATH]
    python examples/agent/video_timeline.py <run-dir> --draft-edit [--force]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agentic_robotics.adapters.toyblocks import portal as run_portal

FPS = 25
"""Frames per second `IncrementalVideoWriter` encodes at. Not recorded in the manifest."""

WORDS_PER_MINUTE = 200
"""Reading speed a quote's dwell is computed from."""

EXCERPT_WORDS = 36
"""Word budget for a default quote: about eight seconds of reading after `READING_SCALE`."""

SETTLE_SECONDS = 1.5
"""Added to every reading time, so a segment does not cut the instant the last word is read."""

READING_SCALE = 0.75
"""Every reading time is multiplied by this. Below 1 because the cut should move on before a
slow reader finishes, since a viewer who wants the words can pause."""

MIN_SEGMENT_SECONDS = 4.0
LOOK_SECONDS_EACH = 1.5
"""Per zoom in a `segments`-mode look segment; three zooms hold for 4.5 s."""

LOOKS_PER_SEGMENT = 3
"""Zooms shown at once. A batch of more than this splits into several look segments, and a
folded row shows the last three with a count of the rest."""

FINAL_SECONDS = 8.0

ACT_TARGET_SECONDS = 8.0
"""What a skill clip is ramped towards when its quote needs no longer. Longer clips get a bigger
speed-up; a longer quote raises the target, so the speed-up is smaller and the clip lasts longer,
but never longer than it really did."""

MAX_SPEED = 4.0
SLOW_TAIL_FRAMES = 40
"""The end of a skill -- contact, release -- plays at 1x however fast the approach ran."""

SHORT_TARGET_SECONDS = 80.0
"""What `--cut short` aims for in total, under a ceiling of ninety. Highlights and the final
card take what their quotes need, and the montage speed is set to spend the rest."""

MONTAGE_SPEED_RANGE = (3.0, 10.0)
"""A montage slower than this is not a montage; faster and a pick is two frames."""

LOOK_TOOLS = {"get_image_observation", "get_zoomed_image", "get_detailed_image"}

TASK_START = "Complete the following task using the available MCP tools:"
TASK_END = "Every pixel you give to a tool"
"""The task sentence sits between these in every prompt `config.py` builds. Outside them is
boilerplate about the tool surface, which the video does not need."""

TASK_DROP = ("There are various letters and symbols on the faces of the blocks on the table.",)
"""Lead-in sentences removed from the front of the task when present. This one describes the
scene, and the scene is on screen."""

TOOL_SHORT = {
    "place_on_table": "place",
    "get_zoomed_image": "zoom",
    "get_image_observation": "image",
    "get_detailed_image": "detailed image",
}


@dataclass
class Look:
    """One perception call, as a segment shows it."""

    call_index: int
    image: str
    """The returned image, relative to the run directory."""

    label: str
    """`zoom 7× at (205, 230)`, from the call's own arguments."""

    rect: tuple[int, int, int, int] | None = None
    """Where the zoom was cropped from, in camera pixels: left, top, width, height."""

    seconds: float | None = None


@dataclass
class Segment:
    """One stretch of the finished video."""

    kind: str
    """One of skill, look, interject, final."""

    seconds: float
    frames: tuple[int, int] | None = None
    """Half-open range of `video.mp4` this segment plays, then holds its last frame."""

    hold_frame: int | None = None
    speed: float = 1.0
    slow_tail_frames: int = 0

    call_index: int | None = None
    call_span: tuple[int, int] | None = None
    """First and last call index a look segment covers."""

    tool: str = ""
    tool_label: str = ""
    outcome: str = ""
    point: tuple[int, int] | None = None
    """Observation pixel the model typed, exactly."""

    real_seconds: float | None = None
    think_seconds: float | None = None
    elapsed_seconds: float | None = None
    """Wall clock from the run's first call to this one's dispatch."""

    quote: str = ""
    """What the panel shows. Several spans are joined with `SPAN_JOIN`, so this is verbatim
    between ellipses rather than as a whole."""

    bold: str = ""
    leading_cut: bool = False
    trailing_cut: bool = False
    inner_cut: bool = False
    """True when the quote is more than one span, so text was cut between them."""
    thought_words: int = 0
    """Length of the full thought, so the renderer can say how much is not shown."""

    quote_before_looks: bool = False
    """True when the quote is the thought before the zooms this segment folds in, not the thought
    before the skill itself. Said on screen, since attribution is part of quoting."""

    looks: list[Look] = field(default_factory=list)
    looks_hidden: int = 0
    """Zooms in the batch beyond the ones shown."""

    highlight: bool = False

    calls: list[dict[str, Any]] = field(default_factory=list)
    """For a montage: the skill calls it plays through, in order, each with its frames, chip,
    outcome, timing and quote fields, so the panel can follow the footage call by call."""

    still_image: str = ""
    """The last full view the model had before this call -- a returned observation or a skill's
    post-state image, never a zoom crop -- relative to the run directory. Runs recorded before
    2026-09-15 have no frame ranges, so a still of one is drawn over this."""


class EditError(ValueError):
    """An `edit.json` entry that would show text the model did not write."""


SPAN_JOIN = " … "
"""Between the spans of a multi-span quote: the same ellipsis the renderer draws at a cut end."""


# -- text ---------------------------------------------------------------------------------------


def sentences(text: str) -> list[str]:
    """Split on sentence ends, keeping paragraph breaks as sentence boundaries."""
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [part for part in parts if part.strip()]


def thought_text(call: Any) -> str:
    """Everything the model wrote before a call: reasoning, then any visible message."""
    return "\n\n".join(part.strip() for part in call.thoughts + call.says if part.strip())


def default_quote(text: str, budget: int = EXCERPT_WORDS) -> tuple[str, str]:
    """The tail of a thought within the word budget, and its last sentence as the bold span."""
    if not text:
        return "", ""
    parts = sentences(text)
    picked: list[str] = []
    words = 0
    for sentence in reversed(parts):
        count = len(sentence.split())
        if picked and words + count > budget:
            break
        picked.insert(0, sentence)
        words += count
    quote = " ".join(picked)
    # The quote has to be a literal substring: join back through the original text.
    start = text.find(picked[0])
    end = text.find(picked[-1], start) + len(picked[-1])
    quote = text[start:end]
    # Emphasis needs a lead-in to stand against; a single-sentence quote is shown plain.
    return quote, picked[-1] if len(picked) > 1 else ""


def reading_seconds(text: str) -> float:
    return (len(text.split()) / WORDS_PER_MINUTE * 60 + SETTLE_SECONDS) * READING_SCALE


def task_sentence(prompt: list[str]) -> str:
    """The task, cut from the prompt between its fixed markers; the whole prompt if they are absent."""
    text = "\n\n".join(prompt).strip()
    if TASK_START in text:
        text = text.split(TASK_START, 1)[1]
    if TASK_END in text:
        text = text.split(TASK_END, 1)[0]
    text = text.strip()
    for lead in TASK_DROP:
        if text.startswith(lead):
            text = text[len(lead) :].strip()
    return text


# -- geometry -----------------------------------------------------------------------------------


def ramp(frame_count: int, target: float = ACT_TARGET_SECONDS) -> tuple[float, float]:
    """Playback speed for a skill clip, and the seconds it then lasts.

    The clip is sped up to finish in about `target` seconds, never below 1x and never above
    `MAX_SPEED`, with the tail at 1x. A clip shorter than the target plays at 1x for its real
    duration: the segment is never padded to reach the target.
    """
    real = frame_count / FPS
    if real <= target:
        return 1.0, real
    tail = min(SLOW_TAIL_FRAMES, frame_count)
    body = frame_count - tail
    speed = min(MAX_SPEED, max(1.0, (body / FPS) / max(target - tail / FPS, 0.1)))
    return speed, body / FPS / speed + tail / FPS


def crop_rect(
    point: tuple[int, int], zoom: float, obs: tuple[int, int], camera: tuple[int, int]
) -> tuple[int, int, int, int]:
    """Where `get_zoomed_image` cropped from, in camera pixels; mirrors `crop_around`."""
    u = round(point[0] / obs[0] * camera[0])
    v = round(point[1] / obs[1] * camera[1])
    window_w = min(max(round(camera[0] / zoom), 1), camera[0])
    window_h = min(max(round(camera[1] / zoom), 1), camera[1])
    left = min(max(u - window_w // 2, 0), camera[0] - window_w)
    top = min(max(v - window_h // 2, 0), camera[1] - window_h)
    return left, top, window_w, window_h


def tool_label(call: Any) -> str:
    """`pick (200, 210)`, `rotate cw (368, 252)`, `zoom 7× at (205, 230)`: the call's own words."""
    name = TOOL_SHORT.get(call.tool, call.tool)
    parts = [name]
    zoom = call.arguments.get("zoom")
    if call.tool == "get_zoomed_image" and isinstance(zoom, (int, float)):
        parts = [f"zoom {zoom:g}×"]
    direction = call.arguments.get("direction")
    if isinstance(direction, str):
        parts.append(direction)
    if call.point is not None:
        parts.append(
            ("at " if call.tool in LOOK_TOOLS else "") + f"({call.point[0]}, {call.point[1]})"
        )
    return " ".join(parts)


def elapsed_since(start: str, now: str) -> float | None:
    if not start or not now:
        return None
    return round(
        datetime.fromisoformat(now).timestamp() - datetime.fromisoformat(start).timestamp(), 1
    )


# -- the edit file -------------------------------------------------------------------------------


def load_edit(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.is_file():
        return {}
    data = json.loads(path.read_text())
    return {str(key): value for key, value in data.get("calls", {}).items()}


def apply_quote(segment: Segment, text: str, entry: dict[str, Any], call_index: int) -> None:
    """Fill the quote fields from an edit entry, or from the defaults, and verify them.

    `quote` in the entry is a string or a list of strings. Every span has to be a substring of
    the thought, the spans have to occur in the order the model wrote them, and `bold` has to sit
    inside one span; anything else is refused rather than shown.
    """
    quote, bold = default_quote(text)
    spans = [quote] if quote else []
    if "quote" in entry:
        raw = entry["quote"]
        spans = [raw] if isinstance(raw, str) else list(raw)
        spans = [span for span in spans if span]
        bold = entry.get("bold", "")
    elif "bold" in entry:
        bold = entry["bold"]
    cursor = 0
    for span in spans:
        found = text.find(span, cursor)
        if found < 0:
            what = (
                "is not a substring of what the model wrote"
                if span not in text
                else "is out of order with the span before it"
            )
            raise EditError(f"call {call_index}: a quote span {what}:\n  {span!r}")
        cursor = found + len(span)
    if bold and not any(bold in span for span in spans):
        raise EditError(
            f"call {call_index}: the bold span is not inside one quote span:\n  {bold!r}"
        )
    segment.quote, segment.bold = SPAN_JOIN.join(spans), bold
    segment.thought_words = len(text.split())
    if spans:
        segment.leading_cut = text.find(spans[0]) > 0
        segment.trailing_cut = cursor < len(text)
        segment.inner_cut = len(spans) > 1


def is_curated(call: Any, entry: dict[str, Any]) -> bool:
    """Whether an edit entry says something the mechanical draft would not have.

    `--draft-edit` writes every call's default quote out, so the presence of a `quote` key means
    nothing; a quote that differs from the default, or a highlight, is a developer's decision.
    """
    if entry.get("highlight"):
        return True
    if "quote" in entry:
        return entry["quote"] != default_quote(thought_text(call))[0]
    return False


def featured(calls: list[Any], edit: dict[str, dict[str, Any]]) -> Any:
    """The call in a batch whose thought is quoted: one with a curated entry, else the first.

    A thought before a zoom is where the model reads a face and changes its mind, so an edit that
    highlights or quotes such a call must survive the zooms being folded into a skill segment.
    """
    for call in calls:
        if is_curated(call, edit.get(str(call.index), {})):
            return call
    return calls[0]


def draft_edit(run_dir: Path) -> dict[str, Any]:
    """An `edit.json` holding the defaults for every call that has a thought, for a developer to review."""
    run = run_portal._load_transcript_run(run_dir)
    calls: dict[str, Any] = {}
    for call in run.calls:
        text = thought_text(call)
        if not text:
            continue
        quote, bold = default_quote(text)
        calls[str(call.index)] = {
            "tool": tool_label(call),
            "include": True,
            "quote": quote,
            "bold": bold,
            "highlight": False,
        }
    return {
        "run": run.name,
        "how": (
            "Each entry quotes the model verbatim. `quote` is a substring of the thought before that "
            "call, or a list of such substrings in the order the model wrote them (shown with an "
            "ellipsis between); `bold` must sit inside one of them. The build refuses anything else. "
            "`include: false` drops the call's segment, `highlight: true` rings its footer dot, "
            '`looks: "segment"|"folded"` overrides --looks for the zooms before that call.'
        ),
        "calls": calls,
    }


# -- the build ----------------------------------------------------------------------------------


def build(
    run_dir: Path, *, looks_mode: str = "folded", edit_path: Path | None = None, cut: str = "full"
) -> dict[str, Any]:
    run = run_portal._load_transcript_run(run_dir)
    manifest = json.loads((run_dir / "run.json").read_text())
    observation = manifest["observation"]
    obs = (
        observation["observation_resolution"]["width"],
        observation["observation_resolution"]["height"],
    )
    camera = (observation["camera_resolution"]["width"], observation["camera_resolution"]["height"])
    edit = load_edit(edit_path if edit_path is not None else run_dir / "edit.json")

    run_start = next((call.started_at for call in run.calls if call.started_at), "")
    skill_calls = [call for call in run.calls if call.tool not in LOOK_TOOLS]
    track = [
        {
            "call_index": call.index,
            "kind": "look" if call.tool in LOOK_TOOLS else "skill",
            "failed": call.failed,
            "highlight": bool(edit.get(str(call.index), {}).get("highlight")),
        }
        for call in run.calls
    ]

    segments: list[Segment] = []
    pending: list[Look] = []
    pending_calls: list[Any] = []
    pending_first_call: Any = None
    previous_end = 0  # Last frame index the video has shown
    skill_number = 0
    last_full_image = ""  # Most recent non-zoom image returned to the model

    def look_of(call: Any) -> Look:
        rect = None
        zoom = call.arguments.get("zoom")
        if call.point is not None and call.point_label == "u,v" and isinstance(zoom, (int, float)):
            rect = crop_rect(call.point, float(zoom), obs, camera)
        image = f"images/{Path(call.post_image).name}" if call.post_image else ""
        return Look(
            call_index=call.index,
            image=image,
            label=tool_label(call),
            rect=rect,
            seconds=call.seconds,
        )

    def voice_for(call: Any, entry: dict[str, Any]) -> Any:
        """Whose thought a skill's panel quotes: a curated zoom before it when the skill's own thought
        was not curated, else the first zoom with a thought when the skill has none, else the skill."""
        if not pending_calls:
            return call
        curated_look = next(
            (c for c in pending_calls if is_curated(c, edit.get(str(c.index), {}))), None
        )
        if curated_look is not None and not is_curated(call, entry):
            return curated_look
        if not thought_text(call):
            return next((c for c in pending_calls if thought_text(c)), call)
        return call

    def flush_looks_as_segments(entry: dict[str, Any]) -> None:
        nonlocal pending, pending_calls, pending_first_call
        if not pending:
            return
        speaker = featured(pending_calls, edit)  # One chunk carries the batch's thought
        for start in range(0, len(pending), LOOKS_PER_SEGMENT):
            chunk = pending[start : start + LOOKS_PER_SEGMENT]
            chunk_calls = pending_calls[start : start + LOOKS_PER_SEGMENT]
            voice = speaker if speaker in chunk_calls else None
            segment = Segment(
                kind="look",
                seconds=max(MIN_SEGMENT_SECONDS, LOOK_SECONDS_EACH * len(chunk)),
                hold_frame=previous_end,
                call_index=chunk[0].call_index,
                call_span=(chunk[0].call_index, chunk[-1].call_index),
                tool="get_zoomed_image" if all("zoom" in l.label for l in chunk) else "look",
                tool_label=(
                    chunk[0].label
                    if len(chunk) == 1
                    else f"zoom ×{len(chunk)}"
                    if all("zoom" in l.label for l in chunk)
                    else f"looks ×{len(chunk)}"
                ),
                looks=chunk,
                elapsed_seconds=elapsed_since(run_start, chunk_calls[0].started_at),
                think_seconds=voice.think_seconds if voice is not None else None,
                highlight=any(edit.get(str(c.index), {}).get("highlight") for c in chunk_calls),
            )
            if voice is not None and thought_text(voice):
                apply_quote(
                    segment, thought_text(voice), edit.get(str(voice.index), {}), voice.index
                )
                segment.seconds = max(
                    segment.seconds, reading_seconds(segment.quote) if segment.quote else 0
                )
            segments.append(segment)
        pending, pending_calls, pending_first_call = [], [], None

    for call in run.calls:
        entry = edit.get(str(call.index), {})
        full_view_before = last_full_image
        if call.tool != "get_zoomed_image" and call.post_image:
            last_full_image = f"images/{Path(call.post_image).name}"

        for text in call.interjections:
            segments.append(
                Segment(
                    kind="interject",
                    seconds=max(MIN_SEGMENT_SECONDS, reading_seconds(text)),
                    hold_frame=previous_end,
                    call_index=call.index,
                    quote=text,
                    elapsed_seconds=elapsed_since(run_start, call.started_at),
                )
            )

        if call.tool in LOOK_TOOLS:
            if not pending:
                pending_first_call = call
            pending.append(look_of(call))
            pending_calls.append(call)
            continue

        skill_number += 1
        if entry.get("include", True) is False:
            pending, pending_calls, pending_first_call = [], [], None
            if call.frames and call.frames[1] > call.frames[0]:
                previous_end = call.frames[1] - 1
            continue

        featured = bool(entry.get("highlight")) or any(
            edit.get(str(c.index), {}).get("highlight") for c in pending_calls
        )
        wrote = call.frames is not None and call.frames[1] > call.frames[0]
        if cut == "short" and not featured:
            # Into the montage: footage kept, the quote shown for as long as the clip lasts and no
            # longer, zooms counted but not shown.
            if not segments or segments[-1].kind != "montage":
                segments.append(Segment(kind="montage", seconds=0.0, hold_frame=previous_end))
            montage = segments[-1]
            voice = voice_for(call, entry)
            spoken = Segment(kind="skill", seconds=0.0)
            apply_quote(spoken, thought_text(voice), edit.get(str(voice.index), {}), voice.index)
            montage.calls.append(
                {
                    "call_index": call.index,
                    "frames": list(call.frames) if wrote else None,
                    "tool_label": tool_label(call),
                    "outcome": "failed"
                    if call.failed
                    else ("never ran" if call.orphaned else "ok"),
                    "point": list(call.point)
                    if call.point is not None and call.point_label == "u,v"
                    else None,
                    "real_seconds": call.seconds,
                    "think_seconds": voice.think_seconds,
                    "elapsed_seconds": elapsed_since(run_start, call.started_at),
                    "thought_words": spoken.thought_words,
                    "looks": len(pending),
                    "quote": spoken.quote,
                    "bold": "",  # Emphasis is reserved for the highlights; a montage quote is shown plain.
                    "leading_cut": spoken.leading_cut,
                    "trailing_cut": spoken.trailing_cut,
                    "inner_cut": spoken.inner_cut,
                    "quote_before_looks": voice is not call,
                }
            )
            if wrote:
                first = montage.frames[0] if montage.frames else call.frames[0]
                montage.frames = (first, call.frames[1])
                previous_end = call.frames[1] - 1
            pending, pending_calls, pending_first_call = [], [], None
            continue

        mode = entry.get("looks", looks_mode)
        if mode == "segments":
            flush_looks_as_segments(entry)

        segment = Segment(
            kind="skill",
            seconds=0.0,
            call_index=call.index,
            tool=call.tool,
            tool_label=tool_label(call),
            outcome="failed" if call.failed else ("never ran" if call.orphaned else "ok"),
            point=call.point if call.point_label == "u,v" else None,
            real_seconds=call.seconds,
            think_seconds=call.think_seconds,
            elapsed_seconds=elapsed_since(run_start, call.started_at),
            highlight=bool(entry.get("highlight")),
        )
        segment.still_image = full_view_before

        # Folded zooms bring their thought with them when the skill has none of its own, or when
        # a zoom's thought was curated and the skill's was not; the segment says which it shows.
        voice = voice_for(call, entry) if mode != "skip" else call
        apply_quote(segment, thought_text(voice), edit.get(str(voice.index), {}), voice.index)
        if voice is not call:
            segment.quote_before_looks = True
            segment.think_seconds = voice.think_seconds
            segment.highlight = segment.highlight or bool(
                edit.get(str(voice.index), {}).get("highlight")
            )
        if pending and mode != "skip":
            shown = pending[-LOOKS_PER_SEGMENT:]
            segment.looks = shown
            segment.looks_hidden = len(pending) - len(shown)
        pending, pending_calls, pending_first_call = [], [], None

        # The quote sets the target the clip is ramped towards, not a hold after it: the segment is
        # exactly as long as its footage plays, and footage never plays below real time.
        reading = reading_seconds(segment.quote) if segment.quote else 0.0
        if wrote:
            frame_count = call.frames[1] - call.frames[0]
            segment.speed, segment.seconds = ramp(frame_count, max(ACT_TARGET_SECONDS, reading))
            segment.frames = call.frames
            segment.slow_tail_frames = min(SLOW_TAIL_FRAMES, frame_count)
            previous_end = call.frames[1] - 1
        else:
            segment.hold_frame = previous_end
            segment.seconds = max(MIN_SEGMENT_SECONDS, reading)
        segments.append(segment)

    final = "\n\n".join(part.strip() for part in run.final_says if part.strip())
    if final or pending:
        segment = Segment(
            kind="final",
            seconds=max(FINAL_SECONDS, reading_seconds(final) if final else 0),
            hold_frame=previous_end,
            quote=final,
        )
        if pending:
            if looks_mode == "segments":
                flush_looks_as_segments({})
            else:
                segment.looks = pending[-LOOKS_PER_SEGMENT:]
                segment.looks_hidden = len(pending) - len(segment.looks)
        segments.append(segment)

    montages = [segment for segment in segments if segment.kind == "montage"]
    if montages:
        footage = (
            sum(segment.frames[1] - segment.frames[0] for segment in montages if segment.frames)
            / FPS
        )
        spent = sum(segment.seconds for segment in segments if segment.kind != "montage")
        speed = footage / max(SHORT_TARGET_SECONDS - spent, 1.0)
        speed = min(max(speed, MONTAGE_SPEED_RANGE[0]), MONTAGE_SPEED_RANGE[1])
        for segment in montages:
            segment.speed = speed
            segment.seconds = (
                (segment.frames[1] - segment.frames[0]) / FPS / speed
                if segment.frames
                else MIN_SEGMENT_SECONDS
            )
            segment.call_index = segment.calls[0]["call_index"]
            segment.call_span = (segment.calls[0]["call_index"], segment.calls[-1]["call_index"])

    return {
        "run": run.name,
        "video": "video.mp4",
        "fps": FPS,
        "cut": cut,
        "looks_mode": looks_mode,
        "observation_resolution": obs,
        "camera_resolution": camera,
        "model": manifest["config"]["model_config"]["model_id"],
        "task": task_sentence(run.prompt),
        "task_name": manifest["config"]["task_config"]["name"],
        "skill_backend": manifest["config"].get("skill_backend", ""),
        "status": run.status,
        "total_calls": len(run.calls),
        "total_skills": len(skill_calls),
        "track": track,
        "segments": [asdict(segment) for segment in segments],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--cut",
        choices=("full", "short"),
        default="full",
        help=f"short: whole run in about {SHORT_TARGET_SECONDS:.0f} s, reasoning only at highlighted calls",
    )
    parser.add_argument(
        "--looks",
        choices=("folded", "segments"),
        default="folded",
        help="How perception calls are shown (default: folded into the skill they led to)",
    )
    parser.add_argument(
        "--edit",
        type=Path,
        default=None,
        help="edit.json to apply (default: <run>/edit.json if present)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Where to write the timeline (default: <run>/timeline.json)",
    )
    parser.add_argument(
        "--draft-edit", action="store_true", help="Write <run>/edit.json from the defaults and exit"
    )
    parser.add_argument("--force", action="store_true", help="Overwrite an existing edit.json")
    args = parser.parse_args()

    if args.draft_edit:
        out = args.edit or args.run_dir / "edit.json"
        if out.exists() and not args.force:
            raise SystemExit(f"{out} exists; pass --force to overwrite it.")
        draft = draft_edit(args.run_dir)
        out.write_text(json.dumps(draft, indent=2, ensure_ascii=False))
        print(f"{len(draft['calls'])} calls with a thought -> {out}")
        return

    timeline = build(args.run_dir, looks_mode=args.looks, edit_path=args.edit, cut=args.cut)
    out = args.out or args.run_dir / "timeline.json"
    out.write_text(json.dumps(timeline, indent=2, ensure_ascii=False))
    total = sum(segment["seconds"] for segment in timeline["segments"])
    kinds = {}
    for segment in timeline["segments"]:
        kinds[segment["kind"]] = kinds.get(segment["kind"], 0) + 1
    montage = next((s["speed"] for s in timeline["segments"] if s["kind"] == "montage"), None)
    speed_note = f", montage at {montage:.1f}×" if montage else ""
    print(
        f"{len(timeline['segments'])} segments {kinds}, {total / 60:.1f} min ({args.cut}, {args.looks}{speed_note}) -> {out}"
    )


if __name__ == "__main__":
    main()
