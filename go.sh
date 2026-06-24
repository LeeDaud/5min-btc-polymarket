#!/usr/bin/env bash
# 快捷实盘:  ./go.sh           (保守)
#           ./go.sh agg       (激进)
set -euo pipefail
PROFILE="${1:-conservative}"
[[ "$PROFILE" == "agg" ]] && PROFILE="aggressive"
cd "$(dirname "$0")"
exec .venv/Scripts/python -u scripts/test_btc_5m_session_exit_sl.py --profile "$PROFILE" --execute
