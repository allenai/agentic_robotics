#!/usr/bin/env bash
# Run one evaluation suite on this machine: the policy server if the skills need one, then
# `agentic-robotics-suite run` and `agentic-robotics-suite summarize`. The released suites ran these same
# steps inside cluster batch jobs; this is the portable part. Needs one NVIDIA GPU, an
# interpreter that imports ToyBlocks (`python` on PATH, or `$AGENTIC_ROBOTICS_PYTHON`), and
# `~/.openrouter_key` or `$OPENROUTER_API_KEY` for the model.
#
# Usage:
#   scripts/run_suite_local.sh --baseline <control|oracle|naive|harness> --model <config key> \
#       --suite-dir <dir> [--instances 0,1] [--parallel 2] [--maniflow <key>] [--policy-port 8000] \
#       [-- <further agentic-robotics-suite run arguments>]
#
# Without `--maniflow`, the episodes run the scripted skills (`--backend oracle`), which plan with
# cuRobo and need no policy server. `--maniflow <key>` runs the ManiFlow skills instead: it starts
# `agentic-robotics serve` for that checkpoint first, waits for its port, and passes the key on
# to every episode; the server is stopped when the suite ends. `--baseline` is the agent's
# condition (which prompt and declarations it gets), independent of which skills execute.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${AGENTIC_ROBOTICS_PYTHON:-python}"
BASELINE=""; MODEL=""; SUITE_DIR=""; INSTANCES="0,1"; PARALLEL="2"; MANIFLOW=""; POLICY_PORT="8000"
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --baseline) BASELINE="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --suite-dir) SUITE_DIR="$2"; shift 2 ;;
        --instances) INSTANCES="$2"; shift 2 ;;
        --parallel) PARALLEL="$2"; shift 2 ;;
        --maniflow) MANIFLOW="$2"; shift 2 ;;
        --policy-port) POLICY_PORT="$2"; shift 2 ;;
        --) shift; EXTRA=("$@"); break ;;
        -h|--help) sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown argument: $1 (arguments for the suite driver go after --)" >&2; exit 2 ;;
    esac
done
[[ -n "$BASELINE" && -n "$MODEL" && -n "$SUITE_DIR" ]] || { echo "--baseline, --model and --suite-dir are required" >&2; exit 2; }

export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | sed 's/^/GPU: /'
"$PYTHON" -c "import toyblocks" 2>/dev/null || { echo "$PYTHON cannot import toyblocks; set AGENTIC_ROBOTICS_PYTHON" >&2; exit 2; }

cd "$REPO_ROOT"
mkdir -p "$SUITE_DIR"
SKILLS_ARG=(--backend oracle)
POLICY_PID=""
if [[ -n "$MANIFLOW" ]]; then
    SKILLS_ARG=(--maniflow "$MANIFLOW")
    "$PYTHON" -m agentic_robotics.adapters.toyblocks.cli serve --maniflow "$MANIFLOW" \
        --policy-port "$POLICY_PORT" > "$SUITE_DIR/policy.log" 2>&1 &
    POLICY_PID=$!
    trap '[[ -n "$POLICY_PID" ]] && kill "$POLICY_PID" 2>/dev/null || true' EXIT
    for _ in $(seq 1 120); do
        (echo > "/dev/tcp/127.0.0.1/$POLICY_PORT") 2>/dev/null && break
        kill -0 "$POLICY_PID" 2>/dev/null || { echo "policy server exited; see $SUITE_DIR/policy.log" >&2; exit 1; }
        sleep 5
    done
    (echo > "/dev/tcp/127.0.0.1/$POLICY_PORT") 2>/dev/null || { echo "policy server never opened port $POLICY_PORT" >&2; exit 1; }
fi

"$PYTHON" -m agentic_robotics.adapters.toyblocks.drivers.evaluate_suite run \
    --baseline "$BASELINE" --model "$MODEL" --instances "$INSTANCES" --parallel "$PARALLEL" \
    --suite-dir "$SUITE_DIR" "${SKILLS_ARG[@]}" "${EXTRA[@]}"
"$PYTHON" -m agentic_robotics.adapters.toyblocks.drivers.evaluate_suite summarize --suite-dir "$SUITE_DIR"
