"""Tile the final observation of every evaluation episode into a successes grid and a failures grid.

A check on the goal classifier: two image grids per suite, one of every
episode scored a success and one of every failure, so a person can confirm the labels by eye
and see the failure modes at a glance.

    python examples/agent/leap/final_state_grid.py --episodes SUITE/episodes.jsonl --out DIR \\
        [--remap <recorded prefix>=<local prefix>]

The tile is the last frame of the run's `video.mp4`, which the scorer recorded after settling
the scene and opening the gripper, so it is the scene the classifier judged; when ffmpeg is not
available (on PATH or through the `imageio-ffmpeg` package) it falls back to the last image the
agent was shown, which may be a zoomed crop. The caption is the variant without its `stack_`
prefix, the instance, and the five goal conjuncts as bits in the classifier's order: on(top,
mid), on(mid, bottom), ori(top), ori(mid), ori(bottom). Needs Pillow.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

TILE = (384, 234)
CAPTION_H = 26
BORDER = 6
"""Width of the tile border that says the verdict: bright green for a success, bright red for a
failure, so a grid reads at a glance."""
TITLE_H = 40
VERDICT_COLOR = {True: (0, 220, 0), False: (255, 0, 0), None: (160, 160, 160)}
CROP = (0.08, 0.30, 0.92, 0.95)
"""Fraction of the frame kept (left, top, right, bottom): the table area where the blocks are, so
a tile shows the blocks at a readable size rather than the whole room."""


def ffmpeg_binary() -> str | None:
    import shutil

    if shutil.which("ffmpeg"):
        return "ffmpeg"
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


FFMPEG = ffmpeg_binary()


def final_frame(run_dir: Path) -> Image.Image | None:
    """The last frame of the run's video, else the last image the agent saw."""
    import io
    import subprocess

    video = run_dir / "video.mp4"
    if FFMPEG and video.is_file():
        result = subprocess.run(
            [
                FFMPEG,
                "-v",
                "error",
                "-sseof",
                "-0.2",
                "-i",
                str(video),
                "-frames:v",
                "1",
                "-f",
                "image2pipe",
                "-vcodec",
                "png",
                "-",
            ],
            capture_output=True,
            check=False,
        )
        if result.returncode == 0 and result.stdout:
            return Image.open(io.BytesIO(result.stdout)).convert("RGB")
    images = sorted((run_dir / "images").glob("image_*.jpg"))
    return Image.open(images[-1]).convert("RGB") if images else None


def task_of(row: dict[str, Any]) -> str:
    """The LEAP rows say `variant`, the Blocksworld rows `task`."""
    return row.get("variant") or row.get("task") or "?"


def caption(row: dict[str, Any]) -> str:
    summary = row.get("summary") or {}
    conjuncts = summary.get("conjuncts") or {}
    keys = sorted(
        (k for k in conjuncts if k.startswith("g0_conj_")), key=lambda k: int(k.rsplit("_", 1)[1])
    )
    bits = "".join("1" if conjuncts[k] == 1 else "0" for k in keys)
    mode = f" {row['mode']}" if row.get("mode") else ""
    mark = "" if row.get("success") is None else ("ok " if row["success"] else "FAIL ")
    return f"{mark}{task_of(row).removeprefix('stack_')} #{row['instance']}{mode}  {bits}"


def title_font() -> Any:
    """A large font for the title when one is installed; Pillow's default otherwise."""
    from PIL import ImageFont

    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "LiberationSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, 26)
        except OSError:
            continue
    return ImageFont.load_default()


def grid(
    rows: list[dict[str, Any]], remap: tuple[str, str] | None, title: str | None = None
) -> Image.Image:
    # As square as the count allows, wider rather than taller when it does not divide evenly
    #: six tiles are 2 rows of 3, seven are 3 of 3, twelve are 3 of 4.
    columns = max(1, math.ceil(math.sqrt(len(rows))))
    lines = (len(rows) + columns - 1) // columns
    top = TITLE_H if title else 0
    sheet = Image.new(
        "RGB", (columns * TILE[0], top + max(1, lines) * (TILE[1] + CAPTION_H)), "white"
    )
    draw = ImageDraw.Draw(sheet)
    if title:
        draw.text((8, 8), title, fill="black", font=title_font())
    for index, row in enumerate(rows):
        run_dir = row["run_dir"]
        if remap and run_dir.startswith(remap[0]):
            run_dir = remap[1] + run_dir[len(remap[0]) :]
        x = (index % columns) * TILE[0]
        y = top + (index // columns) * (TILE[1] + CAPTION_H)
        frame = final_frame(Path(run_dir))
        if frame is not None:
            w, h = frame.size
            frame = frame.crop(
                (int(CROP[0] * w), int(CROP[1] * h), int(CROP[2] * w), int(CROP[3] * h))
            )
            sheet.paste(frame.resize(TILE), (x, y))
        else:
            draw.text((x + 8, y + 8), "no image", fill="red")
        draw.rectangle(
            [x, y, x + TILE[0] - 1, y + TILE[1] - 1],
            outline=VERDICT_COLOR[row.get("success")],
            width=BORDER,
        )
        draw.text((x + 4, y + TILE[1] + 6), caption(row), fill="black")
    return sheet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--episodes",
        type=Path,
        nargs="+",
        required=True,
        help="episodes.jsonl of one or more suites",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--remap", default=None, help="PREFIX=REPLACEMENT applied to run_dir paths")
    parser.add_argument(
        "--title", default=None, help="Title drawn above the grid (default: the output name)"
    )
    parser.add_argument(
        "--combined",
        default=None,
        metavar="NAME",
        help="One grid NAME.png of every episode from every file, successes and failures together, "
        "captioned ok/FAIL (the view used for the Blocksworld pilot)",
    )
    args = parser.parse_args()
    remap = tuple(args.remap.split("=", 1)) if args.remap else None
    args.out.mkdir(parents=True, exist_ok=True)
    if args.combined:
        rows = []
        for episodes in args.episodes:
            rows += [
                json.loads(line)
                for line in episodes.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        rows = [r for r in rows if r.get("summary")]
        rows.sort(key=lambda r: (task_of(r), r["instance"], r.get("mode", "")))
        target = args.out / f"{args.combined}.png"
        wins = sum(1 for r in rows if r["success"])
        grid(rows, remap, args.title or f"{args.combined}: {wins}/{len(rows)} succeeded").save(
            target
        )
        print(f"{target}: {len(rows)} episodes")
        return 0
    for episodes in args.episodes:
        rows = [
            json.loads(line)
            for line in episodes.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        rows = [r for r in rows if r.get("summary")]
        rows.sort(key=lambda r: (task_of(r), r["instance"]))
        name = episodes.parent.name
        for label, subset in (
            ("successes", [r for r in rows if r["success"]]),
            ("failures", [r for r in rows if not r["success"]]),
        ):
            if not subset:
                continue
            target = args.out / f"{name}_{label}.png"
            grid(subset, remap, args.title or f"{name}: {label} ({len(subset)})").save(target)
            print(f"{target}: {len(subset)} episodes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
