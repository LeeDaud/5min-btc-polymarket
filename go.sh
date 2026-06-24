#!/usr/bin/env bash
# 快捷实盘:  ./go.sh           (保守)
#           ./go.sh agg       (激进)
set -euo pipefail
PROFILE="${1:-conservative}"
[[ "$PROFILE" == "agg" ]] && PROFILE="aggressive"
cd "$(dirname "$0")"
while true; do
    echo "=== BTC 5m Live ($PROFILE) starting at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
    .venv/Scripts/python -u scripts/test_btc_5m_session_exit_sl.py --profile "$PROFILE" --execute || true
    echo "=== Exited at $(date -u +%Y-%m-%dT%H:%M:%SZ), restarting in 3s... ==="
    sleep 3
done
