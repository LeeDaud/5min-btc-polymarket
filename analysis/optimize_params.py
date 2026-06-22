#!/usr/bin/env python3
"""
Parameter optimization for BTC 5-min Polymarket momentum strategy.
Grid search over threshold, max_entry_price, stop_loss, min_btc_move.
Evaluates cumulative return, win rate, Sharpe-like ratio, max consecutive loss.
"""

import requests
import json
import math
import statistics
from datetime import datetime, timezone, timedelta
from itertools import product

UTC = timezone.utc


# ============================================================
# Data layer (same as backtest_btc5m.py)
# ============================================================

def get_btc_klines(n_candles=500):
    r = requests.get('https://api.mexc.com/api/v3/klines', params={
        'symbol': 'BTCUSDT', 'interval': '1m', 'limit': min(n_candles, 500)
    }, timeout=15)
    r.raise_for_status()
    data = r.json()
    return [{
        'open_time': int(c[0]),
        'open': float(c[1]),
        'high': float(c[2]),
        'low': float(c[3]),
        'close': float(c[4]),
        'volume': float(c[5]),
    } for c in data]


def bucket_5m(ts):
    return ts - (ts % 300)


def compute_5min_slots(candles):
    slots = {}
    for c in candles:
        bucket = bucket_5m(c['open_time'] // 1000)
        slots.setdefault(bucket, []).append(c)
    return {b: sc for b, sc in sorted(slots.items()) if len(sc) >= 3}


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ============================================================
# Simulation
# ============================================================

def simulate_slot(slot_candles, threshold=0.70, max_entry_price=1.0,
                  stop_loss_pct=0.25, min_btc_move=40,
                  exit_before_sec=20, btc_5min_std=None, clob_spread=0.03):
    """
    Returns: (entered, result_dict) — entered=False if no trade.
    """
    slot_start = slot_candles[0]['open_time'] // 1000
    slot_end = slot_start + 300
    btc_start = slot_candles[0]['open']

    if btc_5min_std is None:
        btc_5min_std = 70.0

    for i, c in enumerate(slot_candles):
        ts = c['open_time'] // 1000
        seconds_left = slot_end - ts

        if seconds_left < 60 or seconds_left > 180:
            continue

        btc_now = c['close']
        btc_move = btc_now - btc_start

        if abs(btc_move) < min_btc_move:
            continue

        direction = 'UP' if btc_move > 0 else 'DOWN'

        remaining_sec = slot_end - ts
        remaining_vol = btc_5min_std * math.sqrt(remaining_sec / 300.0)
        abs_move = abs(btc_move)

        if remaining_vol > 0:
            fair_prob = norm_cdf(abs_move / remaining_vol)
        else:
            fair_prob = 0.5
        fair_prob = max(0.50, min(0.999, fair_prob))

        ask_price = min(0.999, fair_prob + clob_spread / 2.0)

        # Entry conditions
        if ask_price < threshold:
            continue
        if ask_price > max_entry_price:
            continue

        entry_price = ask_price
        sl_price = entry_price * (1.0 - stop_loss_pct)

        # Walk forward
        exit_price = None
        exit_reason = None

        for j in range(i + 1, len(slot_candles)):
            follow_c = slot_candles[j]
            follow_btc = follow_c['close']
            follow_remaining = slot_end - follow_c['open_time'] // 1000
            follow_remaining = max(10, follow_remaining)

            btc_move_from_start = follow_btc - btc_start
            if direction == 'UP':
                effective_move = btc_move_from_start
            else:
                effective_move = -btc_move_from_start

            follow_vol = btc_5min_std * math.sqrt(follow_remaining / 300.0)
            if follow_vol > 0:
                current_mid = norm_cdf(effective_move / follow_vol)
            else:
                current_mid = 0.5
            current_mid = max(0.001, min(0.999, current_mid))
            current_bid = max(0.001, current_mid - clob_spread / 2.0)

            if current_bid <= sl_price:
                exit_price = sl_price
                exit_reason = 'stop_loss'
                break

            follow_ts = follow_c['open_time'] // 1000
            if slot_end - follow_ts <= exit_before_sec:
                exit_price = current_bid
                exit_reason = 'time_exit'
                break

            if j == len(slot_candles) - 1:
                exit_price = current_bid
                exit_reason = 'end_of_data'

        if exit_price is None:
            final_btc = slot_candles[-1]['close']
            btc_final_move = final_btc - btc_start
            won = ((direction == 'UP' and btc_final_move > 0) or
                   (direction == 'DOWN' and btc_final_move < 0))
            exit_price = 1.0 if won else 0.0
            exit_reason = 'settlement'

        pnl_usd = exit_price - entry_price
        pnl_pct = (pnl_usd / entry_price) * 100

        return (True, {
            'direction': direction,
            'entry_price': round(entry_price, 4),
            'exit_price': round(exit_price, 4),
            'exit_reason': exit_reason,
            'pnl_pct': round(pnl_pct, 2),
            'won': exit_price > entry_price,
        })

    return (False, None)


# ============================================================
# Optimization
# ============================================================

def run_grid_search(slots, btc_5min_std, param_grid):
    """Run grid search and return sorted results."""
    results = []
    total_combos = 1
    for v in param_grid.values():
        total_combos *= len(v)
    print(f"  Testing {total_combos} parameter combinations...")

    count = 0
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]

    for combo in product(*values):
        params = dict(zip(keys, combo))
        count += 1
        if count % 500 == 0:
            print(f"    {count}/{total_combos}...")

        trades = []
        for bucket, sc in sorted(slots.items()):
            entered, result = simulate_slot(sc, btc_5min_std=btc_5min_std, **params)
            if entered:
                trades.append(result)

        if len(trades) < 5:
            continue

        n = len(trades)
        n_wins = sum(1 for t in trades if t['won'])
        n_stop_loss = sum(1 for t in trades if t['exit_reason'] == 'stop_loss')
        pnl_pcts = [t['pnl_pct'] for t in trades]
        avg_pnl = statistics.mean(pnl_pcts)
        median_pnl = statistics.median(pnl_pcts)
        cum_pnl = sum(pnl_pcts)
        std_pnl = statistics.stdev(pnl_pcts) if n >= 3 else 0
        sharpe = avg_pnl / std_pnl if std_pnl > 0 else 0

        # Max drawdown (cumulative)
        cum = 0
        peak = 0
        max_dd = 0
        for p in pnl_pcts:
            cum += p
            peak = max(peak, cum)
            max_dd = min(max_dd, cum - peak)

        # Consecutive losses
        max_consec_loss = 0
        cur_loss = 0
        for t in trades:
            if not t['won']:
                cur_loss += 1
                max_consec_loss = max(max_consec_loss, cur_loss)
            else:
                cur_loss = 0

        # Composite score: blend of cumulative return and stability
        # Penalize large drawdowns and stop losses
        score = (
            cum_pnl * 0.35 +
            sharpe * 100 * 0.25 +
            median_pnl * 0.20 +
            (n_wins / n * 100) * 0.10 +
            max_dd * 0.05 -
            n_stop_loss * 2 * 0.05
        )

        results.append({
            'params': params,
            'n_trades': n,
            'n_wins': n_wins,
            'win_rate': round(n_wins / n * 100, 1),
            'n_stop_loss': n_stop_loss,
            'avg_pnl': round(avg_pnl, 2),
            'median_pnl': round(median_pnl, 2),
            'cum_pnl': round(cum_pnl, 2),
            'std_pnl': round(std_pnl, 2),
            'sharpe': round(sharpe, 3),
            'max_dd': round(max_dd, 2),
            'max_consec_loss': max_consec_loss,
            'score': round(score, 2),
        })

    results.sort(key=lambda r: r['score'], reverse=True)
    return results


def main():
    print("=" * 90)
    print("BTC 5-MIN STRATEGY — PARAMETER OPTIMIZATION (GRID SEARCH)")
    print("=" * 90)

    # Fetch data
    print("\n[1] Fetching BTC data...")
    candles = get_btc_klines(500)
    print(f"    {len(candles)} candles")

    slots = compute_5min_slots(candles)
    now_ts = int(datetime.now(UTC).timestamp())
    completed_slots = {}
    for b, sc in slots.items():
        if b + 300 < now_ts - 30:
            completed_slots[b] = sc

    # Calibrate volatility
    five_min_returns = []
    for b, sc in sorted(completed_slots.items()):
        if len(sc) >= 3:
            five_min_returns.append(sc[-1]['close'] - sc[0]['open'])
    btc_5min_std = statistics.stdev(five_min_returns) if len(five_min_returns) >= 5 else 70.0

    print(f"    {len(completed_slots)} slots, BTC 5-min σ = ${btc_5min_std:.1f}\n")

    # Parameter grid
    param_grid = {
        'threshold':        [0.65, 0.68, 0.70, 0.72, 0.75],
        'max_entry_price':  [0.82, 0.85, 0.88, 0.90, 0.92],
        'stop_loss_pct':    [0.15, 0.20, 0.25, 0.30],
        'min_btc_move':     [30, 40, 50, 60],
    }

    print(f"[2] Running grid search...")
    all_results = run_grid_search(completed_slots, btc_5min_std, param_grid)

    # Top 20
    print(f"\n{'='*90}")
    print(f"TOP 20 PARAMETER COMBINATIONS (by composite score)")
    print(f"{'='*90}\n")

    header = (f"{'Rank':<5} {'Thresh':<8} {'MaxAsk':<8} {'SL%':<6} {'MinMove':<8} "
              f"{'Trades':<7} {'Win%':<7} {'Avg':<8} {'Med':<8} {'Cum%':<9} "
              f"{'Sharpe':<8} {'MaxDD':<8} {'StopN':<6} {'Score':<8}")
    print(header)
    print("-" * len(header))

    for i, r in enumerate(all_results[:20]):
        p = r['params']
        print(f"{i+1:<5} {p['threshold']:<8.2f} {p['max_entry_price']:<8.2f} "
              f"{p['stop_loss_pct']:<6.0%} {p['min_btc_move']:<8.0f} "
              f"{r['n_trades']:<7} {r['win_rate']:<7.1f}% "
              f"{r['avg_pnl']:<+8.2f} {r['median_pnl']:<+8.2f} {r['cum_pnl']:<+9.2f} "
              f"{r['sharpe']:<8.3f} {r['max_dd']:<+8.2f} {r['n_stop_loss']:<6} {r['score']:<8.2f}")

    # Compare to baseline (original strategy params)
    print(f"\n{'='*90}")
    print(f"BASELINE COMPARISON (original strategy params)")
    print(f"{'='*90}\n")

    baseline_params = {
        'threshold': 0.70,
        'max_entry_price': 1.0,  # no limit in original
        'stop_loss_pct': 0.25,
        'min_btc_move': 40,
    }
    trades_base = []
    for bucket, sc in sorted(completed_slots.items()):
        entered, result = simulate_slot(sc, btc_5min_std=btc_5min_std, **baseline_params)
        if entered:
            trades_base.append(result)

    if trades_base:
        pnl_base = [t['pnl_pct'] for t in trades_base]
        n_base = len(trades_base)
        w_base = sum(1 for t in trades_base if t['won'])
        sl_base = sum(1 for t in trades_base if t['exit_reason'] == 'stop_loss')
        avg_base = statistics.mean(pnl_base)
        cum_base = sum(pnl_base)
        std_base = statistics.stdev(pnl_base) if n_base >= 3 else 0
        sharpe_base = avg_base / std_base if std_base > 0 else 0

        print(f"  Baseline (thresh=0.70, max_ask=1.0, sl=25%, min_move=$40):")
        print(f"    Trades: {n_base} | Win: {w_base}/{n_base} ({w_base/n_base*100:.1f}%)")
        print(f"    Avg: {avg_base:+.2f}% | Cum: {cum_base:+.2f}% | Sharpe: {sharpe_base:.3f}")
        print(f"    Stop losses: {sl_base}")

        # Best optimized
        best = all_results[0]
        print(f"\n  Best optimized ({', '.join(f'{k}={v}' for k, v in best['params'].items())}):")
        print(f"    Trades: {best['n_trades']} | Win: {best['n_wins']}/{best['n_trades']} ({best['win_rate']}%)")
        print(f"    Avg: {best['avg_pnl']:+.2f}% | Cum: {best['cum_pnl']:+.2f}% | Sharpe: {best['sharpe']:.3f}")
        print(f"    Stop losses: {best['n_stop_loss']} | MaxDD: {best['max_dd']:+.2f}%")

        improvement = best['cum_pnl'] - cum_base
        print(f"\n  Improvement in cumulative return: {improvement:+.2f}%")

    # Sensitivity analysis: which params matter most?
    print(f"\n{'='*90}")
    print(f"PARAMETER SENSITIVITY (impact of each param on score, averaged)")
    print(f"{'='*90}\n")

    for param_name in param_grid.keys():
        param_values = param_grid[param_name]
        print(f"  {param_name}:")
        for val in param_values:
            matching = [r for r in all_results if r['params'][param_name] == val]
            if matching:
                avg_score = statistics.mean([r['score'] for r in matching])
                avg_cum = statistics.mean([r['cum_pnl'] for r in matching])
                avg_trades = statistics.mean([r['n_trades'] for r in matching])
                bar = '#' * max(1, int(avg_score / 2))
                print(f"    {str(val):>8s} → score={avg_score:7.2f}  cum={avg_cum:8.2f}%  "
                      f"trades={avg_trades:5.0f}  {bar}")

    print(f"\n{'='*90}")
    print("Done.")
    print(f"{'='*90}")


if __name__ == '__main__':
    main()
    input("\nPress Enter to exit...")
