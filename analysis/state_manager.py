#!/usr/bin/env python3
"""
State manager for incremental backtest/optimization.
Tracks which 5-min slots have been processed, persists results.
"""

import json
import os
from pathlib import Path

STATE_FILE = Path(__file__).parent / ".backtest_state.json"


def load_state():
    """Load persisted state. Returns dict with defaults if no state exists."""
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, KeyError):
            pass
    return {
        "last_bucket": 0,
        "total_slots_processed": 0,
        "slot_results": {}  # {bucket_ts: {pnl_pct, won, direction, ...}}
    }


def save_state(state):
    """Persist state to disk."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def get_new_buckets(candles, last_bucket):
    """Filter candles to only include 5-min slots newer than last_bucket."""
    from datetime import datetime, timezone

    def bucket_5m(ts):
        return ts - (ts % 300)

    slots = {}
    for c in candles:
        b = bucket_5m(c['open_time'] // 1000)
        if b <= last_bucket:
            continue
        slots.setdefault(b, []).append(c)

    now_ts = int(datetime.now(timezone.utc).timestamp())
    completed = {}
    for b, sc in sorted(slots.items()):
        if len(sc) >= 3 and b + 300 < now_ts - 30:
            completed[b] = sc

    return completed


def update_state(state, new_results, latest_bucket):
    """Merge new backtest results into state."""
    state["last_bucket"] = max(state["last_bucket"], latest_bucket)
    for bucket_ts, result in new_results.items():
        if str(bucket_ts) not in state["slot_results"]:
            state["slot_results"][str(bucket_ts)] = result
            state["total_slots_processed"] += 1
    return state


def compute_cumulative_stats(state):
    """Compute aggregate stats from all stored slot results."""
    results = list(state["slot_results"].values())
    trades = [r for r in results if r.get("entered")]
    if not trades:
        return {"n_slots": len(results), "n_trades": 0}

    import statistics
    n = len(trades)
    n_wins = sum(1 for t in trades if t["won"])
    n_sl = sum(1 for t in trades if t.get("exit_reason") == "stop_loss")
    pnl_pcts = [t["pnl_pct"] for t in trades]
    avg_pnl = statistics.mean(pnl_pcts)
    median_pnl = statistics.median(pnl_pcts)
    cum_pnl = sum(pnl_pcts)
    std_pnl = statistics.stdev(pnl_pcts) if n >= 3 else 0
    sharpe = avg_pnl / std_pnl if std_pnl > 0 else 0

    cum = 0
    peak = 0
    max_dd = 0
    for p in pnl_pcts:
        cum += p
        peak = max(peak, cum)
        max_dd = min(max_dd, cum - peak)

    return {
        "n_slots": len(results),
        "n_trades": n,
        "n_wins": n_wins,
        "win_rate": round(n_wins / n * 100, 1) if n > 0 else 0,
        "n_stop_loss": n_sl,
        "avg_pnl": round(avg_pnl, 2),
        "median_pnl": round(median_pnl, 2),
        "cum_pnl": round(cum_pnl, 2),
        "std_pnl": round(std_pnl, 2),
        "sharpe": round(sharpe, 3),
        "max_dd": round(max_dd, 2),
    }


def reset_state():
    """Clear all stored state."""
    if STATE_FILE.exists():
        STATE_FILE.unlink()
