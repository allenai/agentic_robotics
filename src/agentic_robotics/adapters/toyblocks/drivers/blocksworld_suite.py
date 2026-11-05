"""Run the Blocksworld suite for one mode and model: every instance, the chosen reset states, one
`agentic-robotics episode --plan-mode <mode>` subprocess per episode.

    # Inside the container (the subprocesses boot Isaac):
    agentic-robotics-blocksworld-suite run \\
        --mode closed --model opus-5.5 --instances 0-4 --parallel 2 \\
        --suite-dir <suite-dir>

    agentic-robotics-blocksworld-suite summarize --suite-dir <suite-dir>

Five sampled reset states per instance, three goal families at five and six blocks, so 30
episodes per mode and model, the same reset-state indices in every mode so comparisons are
paired on identical scenes. The scheduling is `leap.evaluate_suite`'s
(its launch stagger, its budget stop, its package restore); this driver differs in the command
it builds and the row it records. `summarize` tabulates success, skill calls, model calls,
dollars and wall-clock by instance, which is what the modes are compared on.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from agentic_robotics.adapters.toyblocks.drivers.evaluate_suite import (
    BUDGET_ERROR,
    RUN_DIR_PATTERN,
    restore_packages,
    wait_for_launch_slot,
)
from agentic_robotics.adapters.toyblocks.leap.openrouter_budget import require_budget
from agentic_robotics.adapters.toyblocks.release.blocksworld_modes import MODES
from agentic_robotics.adapters.toyblocks.tasks.blocksworld_problems import all_problems

logger = logging.getLogger("agentic_robotics.examples.blocksworld.run_suite")

EPISODE_COST_ESTIMATE_USD = 0.30
"""Planning figure per episode, the LEAP Oracle suite's $0.17-0.25 with a margin for six blocks."""

_lock = threading.Lock()
_budget_stop = threading.Event()


def parse_instances(text: str) -> list[int]:
    """`0-4` or `0,2,3` to a list of reset-state indices."""
    if "-" in text:
        first, last = text.split("-", 1)
        return list(range(int(first), int(last) + 1))
    return [int(part) for part in text.split(",") if part.strip()]


def recorded(suite_dir: Path) -> dict[tuple[str, int], dict[str, Any]]:
    """Episodes already in `episodes.jsonl`, keyed by (slug, instance)."""
    path = suite_dir / "episodes.jsonl"
    if not path.is_file():
        return {}
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    return {(row["task"], row["instance"]): row for row in rows}


def episode_command(args: argparse.Namespace, slug: str, instance: int) -> list[str]:
    """The `agentic-robotics episode` command for one episode."""
    command = [
        sys.executable,
        "-m",
        "agentic_robotics.adapters.toyblocks.cli",
        "episode",
        "--task",
        slug,
        "--plan-mode",
        args.mode,
        "--instance",
        str(instance),
        "--model",
        args.model,
        "--skills",
        args.skills,
        "--physx-scale",
        str(args.physx_scale),
    ]
    if args.runs_dir is not None:
        command += ["--runs-dir", str(args.runs_dir)]
    return command


def run_one(args: argparse.Namespace, slug: str, instance: int) -> dict[str, Any] | None:
    """Run one episode as a subprocess and record its summary line; None when skipped for money."""
    if _budget_stop.is_set():
        return None
    wait_for_launch_slot(args.stagger)
    if _budget_stop.is_set():
        return None
    command = episode_command(args, slug, instance)
    log_path = args.suite_dir / "logs" / f"{slug}_{instance}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    text = log_path.read_text(encoding="utf-8", errors="replace")
    match = RUN_DIR_PATTERN.search(text)
    run_dir = Path(match.group(1)) if match else None
    summary_path = run_dir / "evaluation_summary.json" if run_dir else None
    summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path and summary_path.is_file()
        else None
    )
    row = {
        "task": slug,
        "instance": instance,
        "mode": args.mode,
        "model": args.model,
        "exit_code": completed.returncode,
        "run_dir": str(run_dir) if run_dir else None,
        "seconds": round(time.time() - started, 1),
        "success": summary.get("success") if summary else None,
        "summary": summary,
    }
    if BUDGET_ERROR.search((summary or {}).get("error") or "") or (
        summary is None and BUDGET_ERROR.search(text)
    ):
        _budget_stop.set()
        logger.error(
            f"{slug} instance {instance}: refused for money; not recorded, no further episodes start."
        )
        return None
    with _lock, (args.suite_dir / "episodes.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")
    status = (
        "ok" if completed.returncode == 0 and summary else f"FAILED exit={completed.returncode}"
    )
    logger.info(
        f"{slug} instance {instance}: {status}, success={row['success']}, {row['seconds']} s"
    )
    return row


def run(args: argparse.Namespace) -> None:
    slugs = args.tasks or [problem.slug for problem in all_problems()]
    instances = parse_instances(args.instances)
    args.suite_dir.mkdir(parents=True, exist_ok=True)
    done = recorded(args.suite_dir)
    todo = [(slug, k) for slug in slugs for k in instances if (slug, k) not in done]
    logger.info(f"{len(done)} episodes recorded, {len(todo)} to run ({args.mode}, {args.model}).")
    if not todo:
        return
    require_budget(len(todo) * EPISODE_COST_ESTIMATE_USD, ignore=args.ignore_budget)
    restore_packages(slugs)
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        list(pool.map(lambda pair: run_one(args, *pair), todo))


def summarize(args: argparse.Namespace) -> None:
    rows = list(recorded(args.suite_dir).values())
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[row["task"]].append(row)
    print(f"{'task':<16}{'n':>3}{'success':>9}{'skills':>8}{'model':>7}{'$':>8}{'sec':>8}")
    for slug in sorted(by_task):
        group = by_task[slug]
        scored = [r for r in group if r["success"] is not None]
        wins = sum(1 for r in scored if r["success"])
        summaries = [r["summary"] or {} for r in group]

        def mean(key: str, summaries: list[dict[str, Any]] = summaries) -> float:
            values = [s.get(key) for s in summaries if isinstance(s.get(key), (int, float))]
            return sum(values) / len(values) if values else float("nan")

        print(
            f"{slug:<16}{len(group):>3}{f'{wins}/{len(scored)}':>9}{mean('motor_calls_made'):>8.1f}"
            f"{mean('model_calls'):>7.1f}{mean('cost_usd'):>8.2f}{mean('seconds'):>8.0f}"
        )
    scored = [r for r in rows if r["success"] is not None]
    print(
        f"total: {sum(1 for r in scored if r['success'])}/{len(scored)} scored of {len(rows)} recorded"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    runner = sub.add_parser("run")
    runner.add_argument("--mode", choices=MODES, required=True)
    runner.add_argument(
        "--model", required=True, help="OPENAI_API_MODEL_CONFIGS name, e.g. opus-5.5"
    )
    runner.add_argument("--suite-dir", type=Path, required=True)
    runner.add_argument("--instances", default="0-4", help="Reset-state indices, `0-4` or `0,2`")
    runner.add_argument(
        "--tasks", nargs="*", default=None, help="Instance slugs (default: all six)"
    )
    runner.add_argument("--skills", default="inline_oracle", help="Skill module (inline_oracle)")
    runner.add_argument(
        "--physx-scale",
        type=int,
        default=1,
        help="agentic-robotics --physx-scale for every episode (16 fits four Isaac processes on the workstation)",
    )
    runner.add_argument(
        "--parallel", type=int, default=1, help="Isaac processes at once (about 7.6 GB each)"
    )
    runner.add_argument(
        "--stagger", type=float, default=45.0, help="Seconds between subprocess starts"
    )
    runner.add_argument("--runs-dir", type=Path, default=None)
    runner.add_argument("--ignore-budget", action="store_true")
    runner.set_defaults(func=run)
    summarizer = sub.add_parser("summarize")
    summarizer.add_argument("--suite-dir", type=Path, required=True)
    summarizer.set_defaults(func=summarize)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
