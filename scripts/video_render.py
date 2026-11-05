"""Render a run's `timeline.json` into an mp4: the frame on the left, the transcript on the right.

`video_timeline.py` decides what is on screen and for how long, and what is quoted. This draws
it and holds no opinion about pacing or selection: every duration and every word comes from the
timeline, so a disagreement about the edit is a change to that file rather than to this one.

The layout is side by side rather than an overlay -- 1000 px of frame beside a 600 px panel at
1600x720 -- so nothing is ever drawn over the scene except the point the model typed and, in a
look segment, the rectangles a zoom was cropped from. A caption above the frame says which
camera it is, what resolution the model actually saw, and the playback speed whenever footage is
sped up; the footer under the panel carries the real elapsed clock, the real per-call time, and
one dot per skill call with a tick for every perception call between them.

Text on screen is the timeline's, verbatim, and the renderer draws an ellipsis at whichever end
of a quote was cut. Markdown emphasis markers in the model's final message are the one thing
stripped, as formatting rather than words.

A montage segment (from `--cut short`) plays several skills' footage at one speed; the panel
follows it call by call with the chip, the quote and the footer clock. The quote is drawn exactly
as a skill segment's would be, but is on screen only while that call's footage plays, since the
montage takes its length from the footage and not from the reading time: it is there to pause on.

    python scripts/video_render.py <run-dir> [--timeline PATH] [--out PATH] [--seconds N] [--crf N]
    python scripts/video_render.py <run-dir> --stills DIR [--timeline PATH]

`--stills` writes one PNG per highlighted skill segment instead of a video -- the composite as the
video shows it, at the first frame of the skill (the scene the model reasoned about, with the point
it typed) and at the last -- for dropping into slides.

Run it with the heavy venv: decoding and encoding go through `imageio-ffmpeg`, which the light venv
does not carry. A 5-minute video renders in about 4 minutes; `--seconds 60` previews the opening.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import imageio as iio
import numpy
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
FONT = FONT_DIR / "DejaVuSans.ttf"
FONT_BOLD = FONT_DIR / "DejaVuSans-Bold.ttf"
FONT_MONO = FONT_DIR / "DejaVuSansMono.ttf"

# Strata palette, as run_portal and web_ui/index.html use it.
BG = (3, 38, 41)
SURFACE = (10, 50, 53)
EDGE = (60, 85, 88)
TEXT = (250, 242, 233)
SOFT = (210, 215, 215)
DIM = (160, 175, 175)
ACCENT = (240, 82, 156)
OK = (15, 203, 140)
ERR = (253, 70, 69)
WARN = (255, 163, 28)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

WIDTH, HEIGHT = 1600, 720
FRAME_W = 1000
FRAME_H = FRAME_W * 9 // 16
FRAME_TOP = (HEIGHT - FRAME_H) // 2 - 20
PANEL_X = FRAME_W
PANEL_W = WIDTH - FRAME_W
PAD = 28
COL_W = PANEL_W - 2 * PAD
FOOTER_H = 72

THUMB_W = 170
STACK_W = 208


def font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(str(path), size)
    except OSError:
        return ImageFont.load_default()


FACES = {
    "title": font(FONT_BOLD, 20),
    "meta": font(FONT, 15),
    "task": font(FONT, 16),
    "label": font(FONT_BOLD, 15),
    "chip": font(FONT_MONO, 17),
    "quote": font(FONT, 19),
    "quote_bold": font(FONT_BOLD, 19),
    "quote_small": font(FONT, 17),
    "quote_small_bold": font(FONT_BOLD, 17),
    "note": font(FONT, 14),
    "caption": font(FONT, 17),
    "thumb": font(FONT_MONO, 13),
    "foot": font(FONT, 15),
    "q19": font(FONT, 19),
    "q19b": font(FONT_BOLD, 19),
    "q17": font(FONT, 17),
    "q17b": font(FONT_BOLD, 17),
    "q15": font(FONT, 15),
    "q15b": font(FONT_BOLD, 15),
    "q13": font(FONT, 13),
    "q13b": font(FONT_BOLD, 13),
}


# -- text ---------------------------------------------------------------------------------------


def wrap_measured(
    draw: ImageDraw.ImageDraw, text: str, face: ImageFont.FreeTypeFont, width: int
) -> list[str]:
    """Greedy wrap by measured width, keeping paragraph breaks."""
    lines: list[str] = []
    for paragraph in text.split("\n"):
        line = ""
        for word in paragraph.split():
            trial = f"{line} {word}".strip()
            if draw.textlength(trial, font=face) <= width or not line:
                line = trial
            else:
                lines.append(line)
                line = word
        lines.append(line)
    return lines


def rich_tokens(
    quote: str, bold: str, leading_cut: bool, trailing_cut: bool
) -> list[tuple[str, bool]]:
    """Words with a bold flag, the bold span located by substring, ellipses at the cut ends.

    A word is bold when any of it falls inside the span, so a span that stops short of a word's
    trailing comma or period takes the punctuation with it instead of leaving it stranded after a
    space; the text is split into words first and weighted second.
    """
    tokens: list[tuple[str, bool]] = []
    start = quote.find(bold) if bold else -1
    end = start + len(bold)
    position = 0
    for paragraph_index, paragraph in enumerate(quote.split("\n")):
        if paragraph_index:
            tokens.append(("\n", False))
            position += 1
        for match in re.finditer(r"\S+", paragraph):
            word_start, word_end = position + match.start(), position + match.end()
            is_bold = start >= 0 and word_start < end and word_end > start
            tokens.append((match.group(), is_bold))
        position += len(paragraph)
    if leading_cut and tokens:
        tokens[0] = ("…" + tokens[0][0], tokens[0][1])
    if trailing_cut and tokens:
        tokens[-1] = (tokens[-1][0] + "…", tokens[-1][1])
    return tokens


def draw_rich(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    tokens: list[tuple[str, bool]],
    face: ImageFont.FreeTypeFont,
    bold: ImageFont.FreeTypeFont,
    width: int,
    max_y: int,
    fill: tuple[int, int, int],
    leading: int = 7,
    measure: bool = False,
) -> tuple[int, bool]:
    """Draw word tokens with wrapping; stop at `max_y` with an ellipsis. Returns (end y, clipped).

    With `measure` nothing is drawn and `max_y` is ignored, so the caller can pick a face that fits.
    """
    x0, y = xy
    x = x0
    step = face.size + leading
    for index, (word, is_bold) in enumerate(tokens):
        if word == "\n":
            x, y = x0, y + step
            continue
        current = bold if is_bold else face
        advance = draw.textlength(word + " ", font=current)
        if x + advance > x0 + width and x > x0:
            x, y = x0, y + step
        if not measure and y + step > max_y:
            draw.text((x, y), "…", font=face, fill=fill)
            return y + step, True
        if not measure:
            draw.text((x, y), word, font=current, fill=fill)
        x += advance
    return y + step, False


QUOTE_SIZES = (19, 17, 15, 13)
"""Quote sizes tried largest first; the first that fits the room above the footer is used."""


# -- frame marks -------------------------------------------------------------------------------


def marker(draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
    """The point the model typed: accent cross on a dark halo, open centre so the target stays visible."""
    for colour, width in ((BLACK, 5), (ACCENT, 2)):
        for dx, dy in ((1, 0), (0, 1)):
            draw.line(
                [(x + dx * 7, y + dy * 7), (x + dx * 24, y + dy * 24)], fill=colour, width=width
            )
            draw.line(
                [(x - dx * 7, y - dy * 7), (x - dx * 24, y - dy * 24)], fill=colour, width=width
            )
        draw.ellipse([x - 15, y - 15, x + 15, y + 15], outline=colour, width=width)


def crop_box(draw: ImageDraw.ImageDraw, rect: tuple[int, int, int, int]) -> None:
    left, top, width, height = rect
    draw.rectangle([left - 2, top - 2, left + width + 2, top + height + 2], outline=BLACK, width=1)
    draw.rectangle([left, top, left + width, top + height], outline=WHITE, width=3)


# -- the composite ---------------------------------------------------------------------------------


class Renderer:
    def __init__(self, timeline: dict[str, Any], run_dir: Path) -> None:
        self.timeline = timeline
        self.run_dir = run_dir
        self.camera = tuple(timeline["camera_resolution"])
        self.obs = tuple(timeline["observation_resolution"])
        self._images: dict[str, Image.Image] = {}
        self.skills = [item for item in timeline["track"] if item["kind"] == "skill"]

    def image(self, name: str, size: tuple[int, int]) -> Image.Image | None:
        key = f"{name}@{size}"
        if key not in self._images:
            path = self.run_dir / name
            if not path.is_file():
                return None
            self._images[key] = Image.open(path).convert("RGB").resize(size)
        return self._images[key]

    def compose(
        self,
        frame: Image.Image,
        segment: dict[str, Any],
        speeding: bool,
        source_index: int | None = None,
        observation: bool = False,
    ) -> Image.Image:
        canvas = Image.new("RGB", (WIDTH, HEIGHT), BG)
        draw = ImageDraw.Draw(canvas)
        kind = segment["kind"]
        current: dict[str, Any] | None = None
        if kind == "montage":
            # The call whose footage this frame belongs to; before any call's frames, the first.
            current = segment["calls"][0]
            for call in segment["calls"]:
                if (
                    call.get("frames")
                    and source_index is not None
                    and call["frames"][0] <= source_index
                ):
                    current = call

        # --- the frame, marked -------------------------------------------------------------
        marked = frame.copy()
        frame_draw = ImageDraw.Draw(marked)
        scale = (self.camera[0] / self.obs[0], self.camera[1] / self.obs[1])
        if kind == "look":
            for look in segment["looks"]:
                if look.get("rect"):
                    crop_box(frame_draw, tuple(look["rect"]))
        elif kind == "montage":
            if current is not None and current.get("point"):
                u, v = current["point"]
                marker(frame_draw, round(u * scale[0]), round(v * scale[1]))
        elif segment.get("point"):
            u, v = segment["point"]
            marker(frame_draw, round(u * scale[0]), round(v * scale[1]))
        canvas.paste(marked.resize((FRAME_W, FRAME_H)), (0, FRAME_TOP))

        caption = (
            f"front camera · Isaac Sim · rendered {self.camera[0]}×{self.camera[1]} "
            f"· the model sees {self.obs[0]}×{self.obs[1]}"
        )
        if observation:
            caption = f"front camera · Isaac Sim · the model's own {self.obs[0]}×{self.obs[1]} observation before this call"
        elif (
            kind in ("skill", "montage")
            and segment.get("frames")
            and speeding
            and segment.get("speed", 1) > 1.01
        ):
            caption += f"  ·  {segment['speed']:.1f}× speed"
        elif not segment.get("frames") or (not speeding and kind != "skill"):
            caption += "  ·  held frame"
        draw.text((16, 16), caption, font=FACES["caption"], fill=DIM)

        # --- panel header: the manifest's model, the prompt's task sentence ------------------
        draw.rectangle([PANEL_X, 0, WIDTH, HEIGHT], fill=SURFACE)
        draw.line([(PANEL_X, 0), (PANEL_X, HEIGHT)], fill=EDGE, width=1)
        x = PANEL_X + PAD
        draw.text((x, 22), self.timeline["model"], font=FACES["title"], fill=TEXT)
        backend = self.timeline.get("skill_backend") or ""
        draw.text(
            (x, 48),
            f"{backend + ' skills · ' if backend else ''}{self.timeline['run']}",
            font=FACES["meta"],
            fill=DIM,
        )
        y = 80
        for line in wrap_measured(draw, self.timeline["task"], FACES["task"], COL_W)[:4]:
            draw.text((x, y), line, font=FACES["task"], fill=SOFT)
            y += 22
        draw.line([(x, y + 8), (WIDTH - PAD, y + 8)], fill=EDGE, width=1)
        y += 24

        # --- body -------------------------------------------------------------------------
        footer_top = HEIGHT - FOOTER_H
        if kind == "skill":
            colour = ERR if segment["outcome"] == "failed" else OK
            y = self.chip(draw, x, y, segment["tool_label"], colour)
            if segment["outcome"] == "failed":
                draw.text((x, y - 8), "the skill reported a failure", font=FACES["note"], fill=ERR)
                y += 14
        elif kind == "montage" and current is not None:
            colour = ERR if current["outcome"] == "failed" else OK
            y = self.chip(draw, x, y, current["tool_label"], colour)
            if current["outcome"] == "failed":
                draw.text((x, y - 8), "the skill reported a failure", font=FACES["note"], fill=ERR)
                y += 14
            if current.get("looks"):
                draw.text(
                    (x, y),
                    f"{current['looks']} zoom{'s' if current['looks'] != 1 else ''} before this, not shown",
                    font=FACES["note"],
                    fill=DIM,
                )
                y += 22
        elif kind == "look":
            y = self.chip(draw, x, y, segment["tool_label"], WHITE)
        elif kind == "interject":
            draw.text((x, y), "A HUMAN INTERJECTED · verbatim", font=FACES["label"], fill=OK)
            y += 26
        elif kind == "final":
            draw.text((x, y), "FINAL ANSWER · verbatim", font=FACES["label"], fill=ACCENT)
            y += 26

        if kind == "look":
            compact = bool(segment.get("quote"))
            y = self.stacked_looks(
                canvas, draw, x, y, segment["looks"], width=120 if compact else STACK_W
            )
        elif segment.get("looks"):
            label = "LOOKED AT BEFORE THIS" if kind == "skill" else "LOOKED AT AFTER"
            if segment.get("looks_hidden"):
                label += f"  ·  and {segment['looks_hidden']} more"
            draw.text((x, y), label, font=FACES["label"], fill=DIM)
            y += 24
            y = self.thumb_row(canvas, draw, x, y, segment["looks"])

        # In a montage the quote, its cuts and its timing are the current call's, not the segment's.
        spoken = current if kind == "montage" and current is not None else segment
        quote = spoken.get("quote", "")
        if kind == "look" and footer_top - 30 - y < 80:
            quote = (
                ""  # Three stacked zooms leave no room; the skill that follows carries the thought.
            )
        if kind in ("skill", "montage") and not quote and not spoken.get("thought_words"):
            draw.text(
                (x, y), "no reasoning recorded before this call", font=FACES["note"], fill=DIM
            )
        if quote:
            if kind == "final":
                quote = quote.replace("**", "")
            elif kind in ("skill", "look", "montage"):
                excerpt = (
                    spoken.get("leading_cut")
                    or spoken.get("trailing_cut")
                    or spoken.get("inner_cut")
                )
                label = "REASONING · excerpt" if excerpt else "REASONING · in full"
                if spoken.get("quote_before_looks"):
                    label += (
                        " · before the zooms above"
                        if segment.get("looks")
                        else " · before the zooms"
                    )
                draw.text((x, y), label, font=FACES["label"], fill=ACCENT)
                y += 26
            tokens = rich_tokens(
                quote,
                spoken.get("bold", ""),
                spoken.get("leading_cut", False),
                spoken.get("trailing_cut", False),
            )
            limit = footer_top - 30
            face, bold = FACES["q13"], FACES["q13b"]
            for size in QUOTE_SIZES:
                candidate, candidate_bold = FACES[f"q{size}"], FACES[f"q{size}b"]
                end, _ = draw_rich(
                    draw,
                    (x, y),
                    tokens,
                    candidate,
                    candidate_bold,
                    COL_W,
                    limit,
                    TEXT,
                    measure=True,
                )
                if end <= limit:
                    face, bold = candidate, candidate_bold
                    break
            y, clipped = draw_rich(draw, (x, y), tokens, face, bold, COL_W, limit, TEXT)
            note = []
            if spoken.get("thought_words") and kind in ("skill", "look", "montage"):
                shown = len(quote.split())
                note.append(
                    f"{spoken['thought_words']} words in full"
                    if spoken["thought_words"] > shown
                    else "shown in full"
                )
            if spoken.get("think_seconds") is not None:
                note.append(f"thought for {spoken['think_seconds']:.1f} s")
            if note and y + 20 < footer_top - 8:
                draw.text((x, y + 2), " · ".join(note), font=FACES["note"], fill=DIM)

        if kind == "montage" and current is not None:
            self.footer(
                draw,
                {
                    **segment,
                    "kind": "skill",
                    "call_index": current["call_index"],
                    "elapsed_seconds": current.get("elapsed_seconds"),
                    "real_seconds": current.get("real_seconds"),
                },
            )
        else:
            self.footer(draw, segment)
        return canvas

    def chip(
        self, draw: ImageDraw.ImageDraw, x: int, y: int, text: str, colour: tuple[int, int, int]
    ) -> int:
        width = draw.textlength(text, font=FACES["chip"])
        draw.rounded_rectangle([x, y, x + width + 22, y + 32], radius=6, outline=colour, width=2)
        draw.text((x + 11, y + 7), text, font=FACES["chip"], fill=colour)
        return y + 46

    def thumb_row(
        self,
        canvas: Image.Image,
        draw: ImageDraw.ImageDraw,
        x: int,
        y: int,
        looks: list[dict[str, Any]],
    ) -> int:
        height = THUMB_W * 9 // 16
        for look in looks:
            image = self.image(look["image"], (THUMB_W, height)) if look.get("image") else None
            if image is not None:
                canvas.paste(image, (x, y))
            draw.rectangle([x - 1, y - 1, x + THUMB_W, y + height], outline=WHITE, width=1)
            draw.text(
                (x, y + height + 4),
                look["label"].replace(" at ", " "),
                font=FACES["thumb"],
                fill=DIM,
            )
            x += THUMB_W + 12
        return y + height + 30

    def stacked_looks(
        self,
        canvas: Image.Image,
        draw: ImageDraw.ImageDraw,
        x: int,
        y: int,
        looks: list[dict[str, Any]],
        width: int = STACK_W,
    ) -> int:
        draw.text((x, y), "WHAT THE MODEL SAW", font=FACES["label"], fill=DIM)
        y += 24
        height = width * 9 // 16
        for look in looks:
            image = self.image(look["image"], (width, height)) if look.get("image") else None
            if image is not None:
                canvas.paste(image, (x, y))
            draw.rectangle([x - 1, y - 1, x + width, y + height], outline=WHITE, width=1)
            draw.text((x + width + 14, y + 8), look["label"], font=FACES["chip"], fill=TEXT)
            if look.get("seconds") is not None:
                draw.text(
                    (x + width + 14, y + 34),
                    f"{look['seconds']:.2f} s",
                    font=FACES["note"],
                    fill=DIM,
                )
            y += height + 10
        return y + 6

    def footer(self, draw: ImageDraw.ImageDraw, segment: dict[str, Any]) -> None:
        x = PANEL_X + PAD
        top = HEIGHT - FOOTER_H
        draw.line([(x, top), (WIDTH - PAD, top)], fill=EDGE, width=1)

        current = segment.get("call_index")
        total_skills = self.timeline["total_skills"]
        total_calls = self.timeline["total_calls"]
        parts: list[str] = []
        if segment["kind"] == "look" and segment.get("call_span"):
            first, last = segment["call_span"]
            parts.append(f"calls {first}–{last} of {total_calls}")
        elif segment["kind"] == "skill" and current is not None:
            number = next(
                (i + 1 for i, item in enumerate(self.skills) if item["call_index"] == current), None
            )
            parts.append(
                f"skill {number} of {total_skills}  (call {current} of {total_calls})"
                if number
                else f"call {current} of {total_calls}"
            )
        elif segment["kind"] == "final":
            parts.append(f"after {total_calls} calls")
        elapsed = segment.get("elapsed_seconds")
        if elapsed is not None:
            parts.append(f"{int(elapsed // 60)}:{int(elapsed % 60):02d} into the run")
        if segment.get("real_seconds") is not None and segment["kind"] == "skill":
            parts.append(f"took {segment['real_seconds']:.1f} s")
        draw.text((x, top + 10), "   ·   ".join(parts), font=FACES["foot"], fill=DIM)

        # One dot per skill call, a tick per perception call; filled up to the current call.
        looks_between = 0
        dots = 0
        for item in self.timeline["track"]:
            if item["kind"] == "look":
                looks_between += 1
            else:
                dots += 1
        pitch = max(8, min(17, (COL_W - 4) // max(1, dots + looks_between * 0.35)))
        tick_pitch = max(3, pitch * 0.35)
        cx = x + 4
        cy = top + 50
        done_index = current if current is not None else 10**9
        for item in self.timeline["track"]:
            reached = item["call_index"] <= done_index
            if item["kind"] == "look":
                colour = (120, 135, 135) if reached else EDGE
                if item["highlight"]:
                    draw.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], outline=WARN, width=2)
                draw.rectangle([cx - 1, cy - 2, cx + 1, cy + 2], fill=colour)
                cx += tick_pitch
                continue
            colour = ERR if item["failed"] else (ACCENT if reached else EDGE)
            if item["highlight"]:
                draw.ellipse([cx - 7, cy - 7, cx + 7, cy + 7], outline=WARN, width=2)
            draw.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=colour)
            cx += pitch


# -- playback ----------------------------------------------------------------------------------


def open_reader(path: Path, attempts: int = 3) -> Any:
    """Open the source video, retrying when imageio gives up on the header.

    imageio launches ffmpeg and kills it if the stream header does not arrive in time. A run's
    `video.mp4` is a few hundred megabytes on a network filesystem with its index at the end, and the first open
    after a while has repeatedly been killed that way ("Could not load meta information") and then
    succeeded unchanged once the file was in cache. So a failed open reads the file through before
    the retry; an immediate retry alone was not enough for a 475 MB run.
    """
    for attempt in range(1, attempts + 1):
        try:
            return iio.get_reader(path)
        except OSError:
            if attempt == attempts:
                raise
            print(
                f"could not open {path.name} (attempt {attempt}); reading it through once, then retrying"
            )
            with open(
                path, "rb"
            ) as handle:  # Pull the whole file into the page cache, index included
                while handle.read(1 << 24):
                    pass


def source_indices(segment: dict[str, Any], fps: int) -> list[tuple[int, bool]]:
    """(source frame, speeding) per output frame. Clips play then hold; everything else holds."""
    count = max(1, round(segment["seconds"] * fps))
    frames = segment.get("frames")
    if not frames:
        return [(segment.get("hold_frame") or 0, False)] * count

    start, stop = frames
    tail = min(segment.get("slow_tail_frames", 0), stop - start)
    speed = max(segment.get("speed", 1.0), 0.01)
    out: list[tuple[int, bool]] = []
    cursor = float(start)
    while cursor < stop - tail:
        out.append((int(cursor), speed > 1.01))
        cursor += speed
    out.extend((index, False) for index in range(stop - tail, stop))
    while len(out) < count:
        out.append((stop - 1, False))
    return out


def render(
    run_dir: Path, timeline_path: Path, out_path: Path, seconds: float | None, crf: int
) -> None:
    timeline = json.loads(timeline_path.read_text())
    fps = timeline["fps"]
    renderer = Renderer(timeline, run_dir)

    plan: list[tuple[int, bool, dict[str, Any]]] = []
    for segment in timeline["segments"]:
        plan.extend((index, speeding, segment) for index, speeding in source_indices(segment, fps))
        if seconds is not None and len(plan) >= seconds * fps:
            plan = plan[: int(seconds * fps)]
            break

    reader = open_reader(run_dir / timeline["video"])
    writer = iio.get_writer(
        out_path,
        fps=fps,
        quality=None,
        macro_block_size=16,
        output_params=[
            "-crf",
            str(crf),
            "-preset",
            "medium",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
        ],
    )

    cursor = -1
    cached: Image.Image | None = None
    exhausted = False
    clamped = 0
    last_key: tuple[int, bool, int] | None = None
    last_output: Any = None
    progress = tqdm(plan, unit="frame", desc=out_path.name, dynamic_ncols=True)
    try:
        for wanted, speeding, segment in progress:
            progress.set_postfix_str(
                f"{segment['kind']} #{segment.get('call_index')}", refresh=False
            )
            while cursor < wanted and not exhausted:
                try:
                    cached = Image.fromarray(reader.get_next_data())
                except (IndexError, StopIteration):
                    # A `menu` session flushes the encoder only on scene switch or quit; until then the
                    # transcript names frames the file does not have yet. Hold the last real frame.
                    exhausted = True
                    break
                cursor += 1
            if exhausted:
                clamped += 1
            assert cached is not None
            key = (cursor, speeding, id(segment))
            if key != last_key:
                last_key = key
                last_output = numpy.asarray(
                    renderer.compose(cached, segment, speeding, source_index=cursor)
                )
            writer.append_data(last_output)
    finally:
        progress.close()
        writer.close()
        reader.close()

    print(f"{len(plan)} frames ({len(plan) / fps / 60:.1f} min) -> {out_path}")
    if clamped:
        print(
            f"  {clamped} output frames held the last decoded frame: the source video ends at "
            f"{cursor + 1}, before the frames the transcript names. Dispose the scene, then re-render."
        )


def stills(run_dir: Path, timeline_path: Path, out_dir: Path) -> None:
    """One PNG per highlighted skill segment, at the skill's first and last frame.

    A run recorded before frame ranges were logged has no footage to index, so its stills are
    drawn over the observation the model saw before the call, upscaled to the camera size, and
    the caption says so.
    """
    timeline = json.loads(timeline_path.read_text())
    renderer = Renderer(timeline, run_dir)
    footage = any(segment.get("frames") for segment in timeline["segments"])
    wanted: dict[int, list[tuple[dict[str, Any], str]]] = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for segment in timeline["segments"]:
        if segment["kind"] != "skill" or not segment.get("highlight"):
            continue
        if not footage and segment.get("still_image"):
            observation = (
                Image.open(run_dir / segment["still_image"])
                .convert("RGB")
                .resize(tuple(renderer.camera))
            )
            image = renderer.compose(observation, segment, False, observation=True)
            path = out_dir / f"call{segment['call_index']:02d}-observation.png"
            image.save(path)
            written.append(path)
        elif segment.get("frames"):
            wanted.setdefault(segment["frames"][0], []).append((segment, "before"))
            wanted.setdefault(segment["frames"][1] - 1, []).append((segment, "after"))
        else:
            wanted.setdefault(segment.get("hold_frame") or 0, []).append((segment, "held"))
    if wanted:
        reader = open_reader(run_dir / timeline["video"])
        try:
            for index, frame in enumerate(reader):
                if index > max(wanted):
                    break
                for segment, when in wanted.get(index, []):
                    image = renderer.compose(
                        Image.fromarray(frame), segment, False, source_index=index
                    )
                    path = out_dir / f"call{segment['call_index']:02d}-{when}.png"
                    image.save(path)
                    written.append(path)
        finally:
            reader.close()
    if not written:
        print("no highlighted skill segment in the timeline; nothing to write")
    for path in written:
        print(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--timeline", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--stills",
        type=Path,
        default=None,
        help="Write PNGs of the highlighted segments here instead of a video",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=None,
        help="Stop after this many seconds of output, for a preview",
    )
    parser.add_argument(
        "--crf", type=int, default=23, help="x264 quality; 23 is the codec default, 28 is smaller"
    )
    args = parser.parse_args()
    if args.stills is not None:
        stills(args.run_dir, args.timeline or args.run_dir / "timeline.json", args.stills)
        return
    render(
        args.run_dir,
        args.timeline or args.run_dir / "timeline.json",
        args.out or args.run_dir / "edited.mp4",
        args.seconds,
        args.crf,
    )


if __name__ == "__main__":
    main()
