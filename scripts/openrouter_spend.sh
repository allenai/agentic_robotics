#!/usr/bin/env bash
#
# Print what the OpenRouter key has spent and what is left, outside any experiment's path.
#
#   ./scripts/openrouter_spend.sh            # key from $OPENROUTER_API_KEY or ~/.openrouter_key
#   ./scripts/openrouter_spend.sh --no-probe # skip the one-token guardrail probe
#
# Three ceilings apply and all are shown: the key's own limit (`limit`, with `limit_reset`
# saying how often it resets; `usage_daily` counts against it when the reset is daily), the
# account's prepaid credits (`total_credits` minus `total_usage`), which no reset refills, and
# the org member's guardrail budget, which no endpoint reports, so the first two can show money
# left while every run dies on its 403 ("Org member budget limit exceeded (monthly limit)"). So
# the script also makes one one-token Haiku call (about $0.00001) and prints the guardrail's
# refusal if there is one. A run stops with HTTP 402
# when the first two are exhausted and 403 on the third. Reference:
# openrouter.ai/docs/api-reference (GET /api/v1/auth/key, GET /api/v1/credits). The key is
# never printed.

set -euo pipefail

key="${OPENROUTER_API_KEY:-}"
if [[ -z "$key" && -f "$HOME/.openrouter_key" ]]; then
    key=$(tr -d '[:space:]' < "$HOME/.openrouter_key")
fi
if [[ -z "$key" ]]; then
    echo "No OpenRouter key: set OPENROUTER_API_KEY or write ~/.openrouter_key." >&2
    exit 1
fi

auth=$(curl -sf https://openrouter.ai/api/v1/auth/key -H "Authorization: Bearer $key")
credits=$(curl -sf https://openrouter.ai/api/v1/credits -H "Authorization: Bearer $key")
python3 - "$auth" "$credits" <<'PY'
import json, sys
key = json.loads(sys.argv[1])["data"]
acct = json.loads(sys.argv[2])["data"]
print(f"Key spend:      today ${key.get('usage_daily', 0):.2f}, this week ${key.get('usage_weekly', 0):.2f}, "
      f"this month ${key.get('usage_monthly', 0):.2f}, lifetime ${key.get('usage', 0):.2f}")
limit = key.get("limit")
if limit is None:
    print("Key limit:      none")
else:
    print(f"Key limit:      ${limit:.2f} per {key.get('limit_reset') or 'key lifetime'}, "
          f"${key.get('limit_remaining', 0):.2f} remaining")
remaining = acct["total_credits"] - acct["total_usage"]
print(f"Account credit: ${remaining:.2f} remaining of ${acct['total_credits']:.2f} bought "
      f"(${acct['total_usage']:.2f} used)")
PY
if [[ "${1:-}" != "--no-probe" ]]; then
    probe=$(curl -s https://openrouter.ai/api/v1/chat/completions -H "Authorization: Bearer $key" \
        -H "Content-Type: application/json" \
        -d '{"model":"anthropic/claude-haiku-4.5","max_tokens":1,"messages":[{"role":"user","content":"hi"}]}')
    python3 - "$probe" <<'PY'
import json, sys
try:
    reply = json.loads(sys.argv[1])
except json.JSONDecodeError:
    print("Guardrail:      probe returned no JSON; the endpoint may be down")
    sys.exit(0)
error = reply.get("error")
if error:
    print(f"Guardrail:      REFUSED ({error.get('code')}): {error.get('message')}")
else:
    print(f"Guardrail:      ok (one-token probe cost ${reply.get('usage', {}).get('cost', 0):.5f})")
PY
fi
