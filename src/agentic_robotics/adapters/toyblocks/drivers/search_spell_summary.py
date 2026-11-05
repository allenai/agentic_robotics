"""Tabulate a Search & Spell suite from its `episodes.jsonl` (what `leap/evaluate_suite.py run`
writes, one row per episode with the episode's `evaluation_summary.json` under `summary`).

    agentic-robotics-spell-summary --suite-dir <dir> [--suite-dir <dir> ...]

Three tables, for the three kinds of evidence the release's agentic experiments are for:

- **exhaustiveness**, on infeasible families: how often the agent declared the task impossible
  (correct), declared success (a false success), or declared nothing;
- **efficiency**: motor calls, calls of any kind, wall clock and cost, by family group;
- **pattern exploitation**: patterned against unpatterned, paired on word, inventory,
  feasibility and instance, as the difference in motor calls.

Family groups are (collection, feasibility, letter blocks). `declared` is the summary's reading
of the final message (`[[SUCCESS]]`, `[[IMPOSSIBLE]]`, both or none); `exposure` counts the
target-letter faces the agent ever brought to the front or an uncovered top, against the faces
the family holds (two per present letter in the unpatterned collection, one block per letter in
the patterned one). The table is written to `<first suite dir>/search_spell_summary.md`.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from agentic_robotics.adapters.toyblocks.release.search_spell_families import (
    parse_slug,
)


def load_rows(suite_dirs: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for suite_dir in suite_dirs:
        path = suite_dir / "episodes.jsonl"
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            parts = parse_slug(row["variant"])
            if parts is None or not row.get("summary"):
                continue
            row["parts"] = parts
            rows.append(row)
    return rows


def letters_of(summary: dict[str, Any]) -> int | None:
    return summary.get("letter_blocks")


def group_key(row: dict[str, Any]) -> tuple[str, str, int | None]:
    parts = row["parts"]
    return (parts["collection"], parts["feasibility"], letters_of(row["summary"]))


def fmt_rate(k: int, n: int) -> str:
    return f"{k}/{n} ({100 * k / n:.0f}%)" if n else "-"


def median_range(values: list[float], digits: int = 0) -> str:
    if not values:
        return "-"
    med = statistics.median(values)
    lo, hi = min(values), max(values)
    return f"{med:.{digits}f} ({lo:.{digits}f}-{hi:.{digits}f})"


def exhaustiveness(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "## Exhaustiveness (infeasible families)",
        "",
        "| collection | letter blocks | episodes | declared impossible (correct) | declared success (false) | declared nothing | all present target faces exposed |",
        "|---|---|---|---|---|---|---|",
    ]
    groups: dict[tuple[str, int | None], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["parts"]["feasibility"] == "infeasible":
            groups[(row["parts"]["collection"], letters_of(row["summary"]))].append(row)
    for (collection, letters), members in sorted(
        groups.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0)
    ):
        n = len(members)
        declared = [m["summary"].get("declared") for m in members]
        exposed_all = sum(1 for m in members if _all_present_exposed(m))
        lines.append(
            f"| {collection} | {letters} | {n} | {fmt_rate(declared.count('impossible'), n)} | "
            f"{fmt_rate(declared.count('success') + declared.count('both'), n)} | "
            f"{fmt_rate(declared.count(None), n)} | {fmt_rate(exposed_all, n)} |"
        )
    false_success = sum(
        1
        for r in rows
        if r["parts"]["feasibility"] == "infeasible"
        and r["summary"].get("declared") in ("success", "both")
    )
    lines += ["", f"False success declarations over every infeasible episode: {false_success}.", ""]
    return lines


def _all_present_exposed(row: dict[str, Any]) -> bool:
    """Whether every target-letter face the family holds was exposed at some point.

    Unpatterned families hold one uppercase and one lowercase face per present letter; patterned
    families one block per present letter. The absent letter has no faces."""
    summary = row["summary"]
    parts = row["parts"]
    word = parts["word"].upper()
    present = [c for c in word if c != (summary.get("search_spell") or {}).get("absent_letter")]
    exposed = {e.split("@")[0] for e in summary.get("target_letters_exposed_ever") or []}
    if parts["collection"] == "unpatterned":
        return all(c in exposed and c.lower() in exposed for c in present)
    return all(c in exposed or c.lower() in exposed for c in present)


def efficiency(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "## Efficiency, by family group (median and range)",
        "",
        "| collection | feasibility | letter blocks | episodes | solved | motor calls | all calls | hit cutoff | minutes | cost $ |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    groups: dict[tuple[str, str, int | None], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[group_key(row)].append(row)
    for key, members in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2] or 0)):
        s = [m["summary"] for m in members]
        n = len(members)
        lines.append(
            f"| {key[0]} | {key[1]} | {key[2]} | {n} | {fmt_rate(sum(1 for x in s if x.get('success')), n)} | "
            f"{median_range([x.get('motor_calls_made', 0) for x in s])} | "
            f"{median_range([x.get('total_calls_made', 0) for x in s])} | "
            f"{sum(1 for x in s if x.get('hit_cutoff'))} | "
            f"{median_range([x.get('seconds', 0) / 60 for x in s], 1)} | "
            f"{median_range([x.get('cost_usd', 0) for x in s], 2)} |"
        )
    lines.append("")
    return lines


def exploitation(rows: list[dict[str, Any]]) -> list[str]:
    """Patterned against unpatterned, paired on (word, inventory, feasibility, instance)."""
    by_pair: dict[tuple[str, str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        p = row["parts"]
        by_pair[
            (
                p["word"],
                p["inventory"],
                p["feasibility"],
                str(row.get("instance")) + ":" + (p["instance"] or ""),
            )
        ][p["collection"]] = row["summary"]
    pairs = [(k, v) for k, v in by_pair.items() if "patterned" in v and "unpatterned" in v]
    lines = [
        "## Pattern exploitation (paired: unpatterned minus patterned motor calls)",
        "",
        f"Pairs: {len(pairs)}.",
        "",
    ]
    if not pairs:
        return [*lines, ""]
    diffs = [
        v["unpatterned"].get("motor_calls_made", 0) - v["patterned"].get("motor_calls_made", 0)
        for _, v in pairs
    ]
    pos = sum(1 for d in diffs if d > 0)
    lines += [
        f"Median difference {statistics.median(diffs):.0f} motor calls (range {min(diffs)} to {max(diffs)}); "
        f"unpatterned needed more in {pos} of {len(diffs)} pairs.",
        "",
        "| word | inventory | feasibility | instance | patterned calls | unpatterned calls | difference |",
        "|---|---|---|---|---|---|---|",
    ]
    for (word, inventory, feasibility, instance), v in sorted(pairs):
        a, b = (
            v["patterned"].get("motor_calls_made", 0),
            v["unpatterned"].get("motor_calls_made", 0),
        )
        lines.append(
            f"| {word} | {inventory} | {feasibility} | {instance} | {a} | {b} | {b - a:+d} |"
        )
    lines.append("")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--suite-dir", type=Path, action="append", required=True)
    args = parser.parse_args()
    rows = load_rows(args.suite_dir)
    if not rows:
        print("No Search & Spell episodes recorded.", file=sys.stderr)
        return 1
    lines = [f"# Search & Spell: {len(rows)} episodes", ""]
    lines += exhaustiveness(rows) + efficiency(rows) + exploitation(rows)
    text = "\n".join(lines)
    out = args.suite_dir[0] / "search_spell_summary.md"
    out.write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"\nWritten to {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
