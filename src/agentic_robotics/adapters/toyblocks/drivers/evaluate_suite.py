"""Run the LEAP evaluation suite for one baseline and model: every variant package, the chosen
reset-state instances, one `agentic-robotics episode --baseline ...` subprocess per episode.

    # Inside the container (the subprocesses boot Isaac):
    agentic-robotics-suite run \\
        --baseline control --model opus-5.5 --instances 0,1 --parallel 2 \\
        --suite-dir <suite-dir>

    agentic-robotics-suite summarize \\
        --suite-dir <suite-dir>

Two task instances per variant, 48 variants, so 96 episodes per baseline and model, with the same instance indices for every baseline so comparisons are paired
on identical scenes. An episode is `agentic-robotics episode --baseline B --task <variant> --instance k
--model M [--docstrings SNAPSHOT]`; the episode mode resets to reset state k before the first
model call and scores the final scene with the package's goal classifier, writing
`evaluation_summary.json` in its run directory. This driver only schedules those subprocesses,
`--parallel` at a time (each is one Isaac process, about 7.6 GB), appends one line per finished
episode to `<suite-dir>/episodes.jsonl`, and skips episodes already recorded there, so a stopped
suite resumes where it left off. `summarize` tabulates success by family and variant from the
recorded summaries.

A full suite costs well over the $3 the agent may spend unasked; the developer starts each one.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from agentic_robotics.adapters.toyblocks import runner_bootstrap as boot
from agentic_robotics.adapters.toyblocks.leap.openrouter_budget import require_budget

logger = logging.getLogger("agentic_robotics.examples.leap.evaluate_suite")

FAMILIES = tuple(
    f"stack_{region}_{proximity}_{skill}"
    for region in ("feasible", "outside")
    for proximity in ("isolated", "adjacent")
    for skill in ("flip", "rotate")
)
FAMILIES = (*FAMILIES, "ai2_stacked")
"""The nine task families, as `prompts/evaluation.leap_tasks` names them: the eight factor
combinations and the long-horizon Ai2 Stack family (ten seeds; `--variants` picks some)."""

RUN_DIR_PATTERN = re.compile(r"Run artifacts in (\S+)")

EPISODE_COST_ESTIMATE_USD = 0.25
"""Planning figure per evaluation episode: the released Control and Oracle suites on Opus 5.5 cost
about $0.21 and $0.17 an episode; shorter local episodes cost $0.03-0.08."""
"""`runner_bootstrap` logs the run directory with this line."""


def variant_slugs(family: str) -> list[str]:
    """Every variant slug of a family, from its problem files."""
    from agentic_robotics.adapters.toyblocks import task_variants

    return sorted(p.stem for p in (task_variants.PROBLEMS_DIR / family).glob("*.pddl"))


def recorded(suite_dir: Path) -> dict[tuple[str, int], dict[str, Any]]:
    """Episodes already in `episodes.jsonl`, keyed by (variant slug, instance)."""
    path = suite_dir / "episodes.jsonl"
    if not path.is_file():
        return {}
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    return {(row["variant"], row["instance"]): row for row in rows}


def episode_command(args: argparse.Namespace, variant: str, instance: int) -> list[str]:
    """The `agentic-robotics episode` command for one episode."""
    command = [
        sys.executable,
        "-m",
        "agentic_robotics.adapters.toyblocks.cli",
        "episode",
        "--task",
        variant,
        "--baseline",
        args.baseline,
        "--instance",
        str(instance),
        "--model",
        args.model,
    ]
    if args.docstrings is not None:
        command += ["--docstrings", str(args.docstrings)]
    if args.runs_dir is not None:
        command += ["--runs-dir", str(args.runs_dir)]
    if args.skip_block_check:
        command.append("--skip-block-check")
    if args.skills:
        command += ["--skills", args.skills]
    if args.backend:
        command += ["--backend", args.backend]
    if args.allow_declaration_mismatch:
        command.append("--allow-declaration-mismatch")
    if args.maniflow:
        command += ["--maniflow", args.maniflow]
    return command


_launch_gate = threading.Lock()
_last_launch = [0.0]


def wait_for_launch_slot(stagger_s: float) -> None:
    """Space subprocess starts at least `stagger_s` apart, so Isaac boots do not peak together."""
    with _launch_gate:
        wait = _last_launch[0] + stagger_s - time.time()
        if wait > 0:
            time.sleep(wait)
        _last_launch[0] = time.time()


BUDGET_ERROR = re.compile(
    r"Error code: 40[23]\b|budget limit exceeded|insufficient credits", re.IGNORECASE
)
"""An OpenRouter refusal for money (402 credits, 403 org member limit), as the episode's error or log shows it."""

_budget_stop = threading.Event()
"""Set by the first episode refused for money; episodes not yet started are then skipped."""


def run_one(
    args: argparse.Namespace, variant: str, instance: int, lock: threading.Lock
) -> dict[str, Any] | None:
    """Run one episode as a subprocess and record its summary line.

    Returns None, writing no row, for an episode skipped or refused for money, which a row would
    record as a failure (as the 2026-10-07 org budget 403 did to 119 episodes): with no row, the
    suite's resume reruns it once the budget is restored.
    """
    if _budget_stop.is_set():
        return None
    wait_for_launch_slot(args.stagger)
    if _budget_stop.is_set():
        return None
    command = episode_command(args, variant, instance)
    log_path = args.suite_dir / "logs" / f"{variant}_{instance}.log"
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
        if summary_path is not None and summary_path.is_file()
        else None
    )
    row = {
        "variant": variant,
        "family": variant.rsplit("_", 1)[0],
        "instance": instance,
        "baseline": args.baseline,
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
            f"{variant} instance {instance}: refused for money; not recorded, and no further episodes start."
        )
        return None
    with lock, (args.suite_dir / "episodes.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")
    status = (
        "ok" if completed.returncode == 0 and summary else f"FAILED exit={completed.returncode}"
    )
    logger.info(
        f"{variant} instance {instance}: {status}, success={row['success']}, {row['seconds']} s"
    )
    return row


def restore_packages(variants: list[str]) -> None:
    """Check that every variant's package is built before any episode starts."""
    restored = 0
    for variant in variants:
        if boot.task_package_is_built(variant):
            continue
        raise SystemExit(
            f"Task package {variant} is not built under {boot.TASKS_DIR}; "
            "build it before running the suite."
        )
    logger.info(
        f"Task packages: {len(variants) - restored} already built, {restored} restored from saved copies."
    )


def run_suite(args: argparse.Namespace) -> int:
    args.suite_dir.mkdir(parents=True, exist_ok=True)
    done = recorded(args.suite_dir)
    # `--variants` narrows only the families it names; a family with no listed variant runs whole.
    # Membership is by the family's own problem files, since a Search & Spell slug carries more
    # underscores than a LEAP one and no suffix rule tells the family apart.
    listed = set(args.variants or [])
    narrowed = {family for family in args.families if listed & set(variant_slugs(family))}
    todo = [
        (variant, instance)
        for family in args.families
        for variant in variant_slugs(family)
        if family not in narrowed or variant in args.variants
        for instance in args.instances
        if (variant, instance) not in done
        or (args.retry_failed and done[(variant, instance)]["success"] is None)
    ]
    logger.info(
        f"Suite {args.baseline}/{args.model}: {len(todo)} episodes to run, {len(done)} already recorded, "
        f"{args.parallel} at a time."
    )
    restore_packages(sorted({variant for variant, _ in todo}))
    require_budget(len(todo) * EPISODE_COST_ESTIMATE_USD, ignore=args.ignore_budget)
    (args.suite_dir / "suite.json").write_text(
        json.dumps(
            {
                "baseline": args.baseline,
                "model": args.model,
                "docstrings": str(args.docstrings) if args.docstrings else None,
                "instances": args.instances,
                "families": list(args.families),
                "variants": list(args.variants or []),
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    lock = threading.Lock()
    failures = skipped = 0
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        for row in pool.map(lambda pair: run_one(args, pair[0], pair[1], lock), todo):
            if row is None:
                skipped += 1
            else:
                failures += row["success"] is None
    if _budget_stop.is_set():
        logger.error(
            f"Suite stopped: OpenRouter refused for money; {skipped} episodes not recorded, to rerun once it is restored."
        )
    logger.info(
        f"Suite finished: {len(todo) - skipped} run, {failures} without a summary, {skipped} not recorded."
    )
    return 1 if failures or skipped else 0


def rescore(args: argparse.Namespace) -> int:
    """Re-score every recorded episode from its `final_state.json` with the classifier corrected
    for letter case.

    Rewrites `episodes.jsonl` in place (keeping a `.bak`), setting `success` to the corrected
    verdict and keeping the uncorrected one as `summary.success_strict`. Episodes without a
    final state (scored before 2026-10-04's scorer) keep their recorded verdict.
    """
    from agentic_robotics.adapters.toyblocks.leap.spelling import spelling_success

    path = args.suite_dir / "episodes.jsonl"
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    changed = 0
    for row in rows:
        summary = row.get("summary")
        run_dir = Path(row["run_dir"]) if row.get("run_dir") else None
        if args.remap and run_dir is not None and str(run_dir).startswith(args.remap[0]):
            run_dir = Path(args.remap[1] + str(run_dir)[len(args.remap[0]) :])
        if not summary or run_dir is None or not (run_dir / "final_state.json").is_file():
            continue
        state = json.loads((run_dir / "final_state.json").read_text(encoding="utf-8"))
        uncorrected = summary.get("success_strict", summary["success"])
        corrected = spelling_success(state, summary.get("conjuncts") or {}, row["variant"])
        if corrected is None:
            continue
        summary["success_strict"] = uncorrected
        summary["success"] = corrected
        changed += corrected != row["success"]
        row["success"] = corrected
    path.with_suffix(".jsonl.bak").write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    logger.info(f"Re-scored {len(rows)} episodes; {changed} verdicts changed.")
    return summarize(args)


def backfill_timing(args: argparse.Namespace) -> int:
    """Add `summary.tool_seconds` to every recorded episode, from its run's transcript.

    Rewrites `episodes.jsonl` in place (keeping a `.bak`). Tool-side seconds were always in the
    transcripts' `duration_s`; model-side seconds were not recorded before 2026-10-05 and cannot
    be backfilled. Episodes whose run directory or transcript is missing are left as they are.
    """
    from agentic_robotics.adapters.toyblocks.leap.timing import tool_call_seconds

    path = args.suite_dir / "episodes.jsonl"
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    filled = 0
    for row in rows:
        summary = row.get("summary")
        run_dir = Path(row["run_dir"]) if row.get("run_dir") else None
        if args.remap and run_dir is not None and str(run_dir).startswith(args.remap[0]):
            run_dir = Path(args.remap[1] + str(run_dir)[len(args.remap[0]) :])
        if not summary or run_dir is None or not (run_dir / "transcript.jsonl").is_file():
            continue
        summary["tool_seconds"] = tool_call_seconds(run_dir / "transcript.jsonl")
        filled += 1
    path.with_suffix(".jsonl.bak").write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    logger.info(f"Backfilled tool seconds for {filled} of {len(rows)} episodes.")
    return 0


def summarize(args: argparse.Namespace) -> int:
    """Tabulate success by family over one suite directory, or over several (`--suite-dir` may
    repeat, for a suite fanned out over one job per family); the table is written to the
    first one."""
    rows = [row for suite_dir in args.suite_dirs for row in recorded(suite_dir).values()]
    args.suite_dir = args.suite_dirs[0]
    if not rows:
        raise SystemExit(f"No episodes recorded under {args.suite_dirs}.")
    by_family: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_family.setdefault(row["family"], []).append(row)
    lines = [f"# {rows[0]['baseline']} / {rows[0]['model']}: {len(rows)} episodes", ""]
    lines += ["| family | episodes | successes | rate | no summary |", "|---|---|---|---|---|"]
    total_ok = total_n = 0
    for family in sorted(by_family):
        at = by_family[family]
        scored = [r for r in at if r["success"] is not None]
        ok = sum(1 for r in scored if r["success"])
        total_ok += ok
        total_n += len(scored)
        rate = f"{ok / len(scored):.2f}" if scored else "-"
        lines.append(f"| {family} | {len(at)} | {ok} | {rate} | {len(at) - len(scored)} |")
    rate = f"{total_ok / total_n:.2f}" if total_n else "-"
    lines += ["", f"Overall: {total_ok} of {total_n} scored episodes succeeded ({rate})."]
    text = "\n".join(lines) + "\n"
    (args.suite_dir / "summary.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="Run the suite's episodes")
    run.add_argument("--baseline", choices=("control", "oracle", "naive", "harness"), required=True)
    run.add_argument("--model", required=True, help="OPENAI_API_MODEL_CONFIGS name")
    run.add_argument(
        "--docstrings", type=Path, default=None, help="Snapshot JSON for naive or harness"
    )
    run.add_argument("--instances", type=lambda s: [int(v) for v in s.split(",")], default=[0, 1])
    run.add_argument("--families", nargs="+", default=list(FAMILIES))
    run.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help="Only these variant slugs, for the families they belong to; other families run whole",
    )
    run.add_argument("--parallel", type=int, default=2, help="Episodes (Isaac processes) at a time")
    run.add_argument(
        "--stagger", type=float, default=30.0, help="Seconds between subprocess launches"
    )
    run.add_argument(
        "--suite-dir", type=Path, required=True, help="Where episodes.jsonl and logs go"
    )
    run.add_argument(
        "--runs-dir", type=Path, default=None, help="Passed to agentic-robotics as --runs-dir"
    )
    run.add_argument(
        "--retry-failed", action="store_true", help="Re-run recorded episodes without a summary"
    )
    run.add_argument(
        "--skip-block-check",
        action="store_true",
        help="Passed to every episode: run even if the checkpoint's block library cannot be read",
    )
    run.add_argument(
        "--skills",
        default=None,
        help="Passed to every episode: the skill module to drive (inline_oracle for the scripted skills; default: the configured backend's)",
    )
    run.add_argument(
        "--backend",
        choices=("oracle", "maniflow"),
        default=None,
        help="Passed to every episode: the skill backend, when --skills names a module agentic-robotics cannot derive it from",
    )
    run.add_argument(
        "--allow-declaration-mismatch",
        action="store_true",
        help="Passed to every episode: record how the served tools differ from the declaration and run anyway",
    )
    run.add_argument(
        "--ignore-budget",
        action="store_true",
        help="Start even if OpenRouter credit cannot cover the estimate",
    )
    run.add_argument(
        "--maniflow",
        default=None,
        help="Passed to every episode: the MANIFLOW_CONFIGS key instead of config.SELECTED_MANIFLOW "
        "(the policy server must serve the same checkpoint)",
    )

    rest = commands.add_parser("restore", help="Restore every variant's saved package, then exit")
    rest.add_argument("--families", nargs="+", default=list(FAMILIES))

    summ = commands.add_parser("summarize", help="Tabulate success by family")
    summ.add_argument(
        "--suite-dir",
        dest="suite_dirs",
        type=Path,
        required=True,
        action="append",
        help="May repeat",
    )

    resc = commands.add_parser(
        "rescore", help="Re-score recorded episodes with the classifier corrected for letter case"
    )
    resc.add_argument("--suite-dir", type=Path, required=True)
    resc.add_argument(
        "--remap",
        type=lambda s: tuple(s.split("=", 1)),
        default=None,
        help="PREFIX=REPLACEMENT for run_dir paths",
    )

    fill = commands.add_parser(
        "backfill-timing", help="Add tool-side seconds to recorded episodes from their transcripts"
    )
    fill.add_argument("--suite-dir", type=Path, required=True)
    fill.add_argument(
        "--remap",
        type=lambda s: tuple(s.split("=", 1)),
        default=None,
        help="PREFIX=REPLACEMENT for run_dir paths",
    )
    return parser.parse_args()


def main() -> int:
    from agentic_robotics.core.logging import configure_logging

    boot.prepare_import_path()
    configure_logging()
    args = parse_args()
    if args.command == "restore":
        restore_packages(
            sorted(variant for family in args.families for variant in variant_slugs(family))
        )
        return 0
    if args.command == "rescore":
        return rescore(args)
    if args.command == "backfill-timing":
        return backfill_timing(args)
    return run_suite(args) if args.command == "run" else summarize(args)


if __name__ == "__main__":
    sys.exit(main())
