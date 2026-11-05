"""Language-only plan generation on the Rao problems, in PlanBench's one-shot format, against one model.

Every query is `planbench_domain.t1_query` (the domain intro, one solved example, the query ending at
`[PLAN]`), with no image, as PlanBench's interface has none. Each of the six instances is queried
`--samples` times (the thirty Rao task instances are six symbolic problems, so sampling stands in for
the reset states), with the previous instance (cyclic) as the solved example, as `prompt` defaults.
The reply is parsed with PlanBench's parser and replayed with the domain's checker; validity and
optimality are what is scored. Requests go to OpenRouter's chat-completions endpoint with the
provider's default sampling; `OPENROUTER_API_KEY` must be set.

Writes `<out>/queries.jsonl` (one record per query: prompt, reply, parsed plan, verdict, usage, cost)
and `<out>/summary.json`. Re-running skips queries already recorded, so an interrupted run resumes.

    OPENROUTER_API_KEY=... PYTHONPATH=src agentic-robotics-planbench \
        --out <suite-dir> --model anthropic/claude-opus-5.5 --samples 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from dataclasses import asdict
from pathlib import Path

from agentic_robotics.adapters.toyblocks.drivers.planbench_domain import (
    check,
    instances,
    parse_plan,
    t1_query,
)

URL = "https://openrouter.ai/api/v1/chat/completions"


def ask(model: str, prompt: str, max_tokens: int) -> dict:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "usage": {"include": True},
        }
    ).encode()
    req = urllib.request.Request(
        URL,
        data=body,
        headers={
            "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
            "Content-Type": "application/json",
        },
    )
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.load(r)
        except Exception as e:
            if attempt == 3:
                raise
            print(f"  retry {attempt + 1}: {e}", file=sys.stderr)
            time.sleep(5 * (attempt + 1))
    raise AssertionError


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", default="anthropic/claude-opus-5.5")
    p.add_argument("--samples", type=int, default=5)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the queries and exit without calling the model",
    )
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    log = a.out / "queries.jsonl"
    done = (
        {(r["instance"], r["sample"]) for r in map(json.loads, log.read_text().splitlines())}
        if log.exists()
        else set()
    )
    insts = instances()
    records = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    for i, inst in enumerate(insts):
        example = insts[i - 1]
        prompt = t1_query(inst, example)
        for s in range(a.samples):
            if (inst.instance_id, s) in done:
                continue
            if a.dry_run:
                print(
                    f"--- instance {inst.instance_id} ({inst.problem.slug}), example {example.instance_id}, {len(prompt)} chars"
                )
                continue
            t0 = time.time()
            reply = ask(a.model, prompt, a.max_tokens)
            text = reply["choices"][0]["message"]["content"]
            plan = parse_plan(text)
            verdict = check(inst, plan)
            rec = {
                "instance": inst.instance_id,
                "slug": inst.problem.slug,
                "sample": s,
                "example": example.instance_id,
                "model": a.model,
                "prompt": prompt,
                "reply": text,
                "plan": plan,
                **asdict(verdict),
                "optimal": verdict.optimal,
                "usage": reply.get("usage"),
                "cost_usd": (reply.get("usage") or {}).get("cost"),
                "seconds": round(time.time() - t0, 1),
                "finish_reason": reply["choices"][0].get("finish_reason"),
            }
            records.append(rec)
            with log.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            print(
                f"{inst.problem.slug} s{s}: valid={verdict.valid} len={verdict.length}/{verdict.optimal_length} "
                f"first_failure={verdict.first_failure} ${rec['cost_usd']} {rec['seconds']}s"
            )
    if a.dry_run:
        return
    by = {}
    for r in records:
        d = by.setdefault(r["slug"], {"n": 0, "valid": 0, "optimal": 0, "lengths": []})
        d["n"] += 1
        d["valid"] += r["valid"]
        d["optimal"] += r["optimal"]
        d["lengths"].append(r["length"])
    summary = {
        "model": a.model,
        "samples": a.samples,
        "n": len(records),
        "valid": sum(r["valid"] for r in records),
        "optimal": sum(r["optimal"] for r in records),
        "cost_usd": round(sum(r["cost_usd"] or 0 for r in records), 4),
        "by_problem": by,
    }
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "by_problem"}))
    for slug, d in by.items():
        print(
            f"  {slug}: valid {d['valid']}/{d['n']}, optimal {d['optimal']}/{d['n']}, lengths {d['lengths']}"
        )


if __name__ == "__main__":
    main()
