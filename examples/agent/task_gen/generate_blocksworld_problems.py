"""Write the Blocksworld task problems, `data/problems/blocksworld_<family>_<blocks>.pddl`, one per instance.

    python examples/agent/task_gen/generate_blocksworld_problems.py

Each is a single-problem task (the legacy `data/problems/<slug>.pddl` form `task_variants`
reads), whose package samples five reset states: the design's five random states are
`agentic-robotics generate-env --task <slug> --reset-states 5 --generate-env`, and an
episode runs on one of them with `--instance k`. The instances themselves, their goal texts
and optimal plan lengths, come from `blocksworld.problems`; this script only writes the files and
prints what it wrote.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parents[1]
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from agentic_robotics.adapters.toyblocks.tasks.blocksworld_problems import all_problems

PROBLEMS_DIR = Path(__file__).resolve().parents[3] / "data/problems"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--slugs",
        nargs="*",
        default=None,
        help="Instances to write (default: all six), e.g. blocksworld_tower_5",
    )
    args = parser.parse_args()
    PROBLEMS_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    for problem in all_problems():
        if args.slugs is not None and problem.slug not in args.slugs:
            continue
        path = PROBLEMS_DIR / f"{problem.slug}.pddl"
        path.write_text(problem.pddl())
        written += 1
        print(
            f"{path.name}: {problem.num_blocks} blocks, optimal {problem.optimal_plan_length} skills"
        )
        print(f"  {problem.goal_text}")
    if not written:
        raise SystemExit("No instance matched --slugs.")


if __name__ == "__main__":
    main()
