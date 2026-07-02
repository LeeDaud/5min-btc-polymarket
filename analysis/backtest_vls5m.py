#!/usr/bin/env python3
"""
Dedicated VLS-5M backtest with detailed diagnostics.
Outputs win rate by squeeze color, sweep type, hour of day, and VWAP distance.
"""
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
from signal_engine import evaluate_vls_signal, VlsSignalResult
from indicators import compute_squeeze_momentum, detect_liquidity_sweep, compute_anchored_vwap

UTC = timezone.utc


def bucket_5m(ts):
    return ts - (ts % 300)


def get_btc_klines(n=500):
    r = requests.get('https://api.mexc.com/api/v3/klines', params={
        'symbol': 'BTCUSDT', 'interval': '1m', 'limit': min(n, 500)
    }, timeout=15)
    r.raise_for_status()
    return [{
        'open_time': int(c[0]), 'open': float(c[1]), 'high': float(c[2]),
        'low': float(c[3]), 'close': float(c[4]), 'volume': float(c[5]),
    } for c in r.json()]


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def simulate_vls_slot(all_candles, slot_candles, config, btc_5min_std=70.0, clob_spread=0.03):
    """Simulate one VLS-5M trade. Returns (entered, trade_dict, signal_dict)."""
    slot_start = slot_candles[0]['open_time'] // 1000
    slot_end = slot_start + 300
    btc_start = slot_candles[0]['open']

    history = [c for c in all_candles if c['open_time'] // 1000 < slot_start]
    if len(history) < 30:
        return (False, None, None)

    for i, c in enumerate(slot_candles):
        ts = c['open_time'] // 1000
        seconds_left = slot_end - ts
        if seconds_left < 60 or seconds_left > 180:
            continue

        btc_now = c['close']
        btc_move = btc_now - btc_start
        if abs(btc_move) < 40:
            continue

        current_history = history + slot_candles[:i + 1]

        # Model entry price first for correct stop/TP calculation
        remaining_sec = slot_end - ts
        remaining_vol = btc_5min_std * math.sqrt(remaining_sec / 300.0)
        abs_move = abs(btc_move)
        fair_prob = norm_cdf(abs_move / remaining_vol) if remaining_vol > 0 else 0.5
        fair_prob = max(0.50, min(0.999, fair_prob))
        entry_price = min(0.999, fair_prob + clob_spread / 2.0)

        vls = evaluate_vls_signal(current_history, btc_now, config, entry_token_price=entry_price)
        if not vls.passed:
            continue

        direction = vls.direction
        stop_token = vls.stop_loss_token_price or (entry_price * 0.85)
        tp1_token = vls.tp1_token_price or (entry_price * 1.08)
        tp2_token = vls.tp2_token_price or (entry_price * 1.15)

        exit_price = None
        exit_reason = None
        tp1_done = False
        shares_remaining = 1.0
        tp1_exit_price = None
        breakeven_active = False

        for j in range(i + 1, len(slot_candles)):
            fc = slot_candles[j]
            fb = fc['close']
            fr = max(10, slot_end - fc['open_time'] // 1000)
            eff = (fb - btc_start) if direction == 'LONG' else -(fb - btc_start)
            fv = btc_5min_std * math.sqrt(fr / 300.0)
            mid = norm_cdf(eff / fv) if fv > 0 else 0.5
            mid = max(0.001, min(0.999, mid))
            bid = max(0.001, mid - clob_spread / 2.0)

            effective_stop = entry_price if breakeven_active else stop_token
            if bid <= effective_stop:
                exit_price = effective_stop
                exit_reason = 'stop_loss'
                break

            if not tp1_done:
                tp1_hit = (direction == 'LONG' and bid >= tp1_token) or \
                          (direction == 'SHORT' and bid <= tp1_token)
                if tp1_hit:
                    tp1_exit_price = bid
                    tp1_done = True
                    breakeven_active = True
                    shares_remaining = 0.5

            tp2_hit = (direction == 'LONG' and bid >= tp2_token) or \
                      (direction == 'SHORT' and bid <= tp2_token)
            if tp2_hit:
                exit_price = bid
                exit_reason = 'tp2'
                break

            if slot_end - fc['open_time'] // 1000 <= 20:
                exit_price = bid
                exit_reason = 'time_exit'
                break

            if j == len(slot_candles) - 1:
                exit_price = bid
                exit_reason = 'end_of_data'

        if exit_price is None:
            fb = slot_candles[-1]['close']
            won = (direction == 'LONG' and fb > btc_start) or \
                  (direction == 'SHORT' and fb < btc_start)
            exit_price = 1.0 if won else 0.0
            exit_reason = 'settlement'

        total_pnl = 0.0
        if tp1_done and tp1_exit_price is not None:
            total_pnl += (tp1_exit_price - entry_price) * 0.5
        if shares_remaining > 0:
            total_pnl += (exit_price - entry_price) * shares_remaining
        pnl_pct = (total_pnl / entry_price) * 100.0

        trade = {
            'direction': direction,
            'entry_price': round(entry_price, 4),
            'exit_price': round(exit_price, 4),
            'exit_reason': exit_reason,
            'pnl_pct': round(pnl_pct, 2),
            'won': total_pnl > 0,
            'tp1_hit': tp1_done,
            'slot_ts': slot_start,
        }

        signal_diag = {
            'sweep_type': vls.sweep_type,
            'squeeze_color': vls.squeeze_color,
            'squeeze_momentum': round(vls.squeeze_momentum, 4),
            'vwap_distance_pct': round((btc_now - vls.vwap) / vls.vwap * 100, 3) if vls.vwap else None,
            'confidence': vls.confidence,
            'hour': datetime.fromtimestamp(slot_start, UTC).hour,
        }
        return (True, trade, signal_diag)

    return (False, None, None)


def main():
    print("=" * 80)
    print("VLS-5M Strategy — Dedicated Backtest")
    print("=" * 80)

    config = {
        'sweep_lookback_min': 15,
        'sweep_lookback_max': 30,
        'squeeze_period': 20,
        'stop_buffer_pct': 0.2,
        'tp1_rr_ratio': 1.5,
        'tp2_rr_ratio': 3.0,
        'btc_to_token_move_ratio': 0.0002,
    }
    print(f"\nConfig: sweep={config['sweep_lookback_min']}-{config['sweep_lookback_max']}min")
    print(f"        squeeze_period={config['squeeze_period']}")
    print(f"        stop_buffer={config['stop_buffer_pct']}%  TP1={config['tp1_rr_ratio']}R  TP2={config['tp2_rr_ratio']}R")
    print()

    print("[1] Fetching BTC 1-min klines...")
    try:
        candles = get_btc_klines(500)
    except Exception as e:
        print(f"  Error: {e}")
        return
    print(f"  {len(candles)} candles")

    # Group into 5-min slots
    slots = {}
    for c in candles:
        b = bucket_5m(c['open_time'] // 1000)
        slots.setdefault(b, []).append(c)
    completed = {b: sc for b, sc in sorted(slots.items()) if len(sc) >= 3}
    print(f"  {len(completed)} completed 5-min slots\n")

    # Calibrate volatility
    returns = []
    for _, sc in sorted(completed.items()):
        returns.append(sc[-1]['close'] - sc[0]['open'])
    btc_std = statistics.stdev(returns) if len(returns) >= 5 else 70.0
    print(f"  BTC 5-min std: ${btc_std:.1f}\n")

    print("[2] Running VLS-5M simulation...")
    trades = []
    signal_diags = []
    slot_list = sorted(completed.items())

    for bucket, sc in slot_list:
        entered, trade, diag = simulate_vls_slot(candles, sc, config, btc_std)
        if entered:
            trades.append(trade)
            signal_diags.append(diag)

    if not trades:
        print("  No trades generated. Check if market conditions match strategy requirements.")
        print("  (VLS-5M requires: VWAP alignment + Sweep + Squeeze fire + correct momentum color)")
        return

    # ============================================================
    # Overall stats
    # ============================================================
    pnl_pcts = [t['pnl_pct'] for t in trades]
    n = len(trades)
    n_wins = sum(1 for t in trades if t['won'])
    avg = statistics.mean(pnl_pcts)
    med = statistics.median(pnl_pcts)
    cum = sum(pnl_pcts)
    std_pnl = statistics.stdev(pnl_pcts) if n >= 3 else 0
    sharpe = avg / std_pnl if std_pnl > 0 else 0

    cum_track = 0; peak = 0; max_dd = 0
    for p in pnl_pcts:
        cum_track += p; peak = max(peak, cum_track)
        max_dd = min(max_dd, cum_track - peak)

    print(f"\n{'='*80}")
    print(f"OVERALL — {n} trades")
    print(f"{'='*80}")
    print(f"  Win rate:          {n_wins}/{n} ({n_wins/n*100:.1f}%)")
    print(f"  Avg return/trade:  {avg:+.2f}%")
    print(f"  Median return:     {med:+.2f}%")
    print(f"  Cumulative return: {cum:+.2f}%")
    print(f"  Sharpe-like:       {sharpe:.3f}")
    print(f"  Max drawdown:      {max_dd:+.2f}%")

    # ============================================================
    # Breakdown by exit reason
    # ============================================================
    reasons = {}
    for t in trades:
        r = t['exit_reason']
        reasons.setdefault(r, {'count': 0, 'wins': 0, 'pnl': []})
        reasons[r]['count'] += 1
        reasons[r]['wins'] += 1 if t['won'] else 0
        reasons[r]['pnl'].append(t['pnl_pct'])

    print(f"\n{'='*80}")
    print(f"BY EXIT REASON")
    print(f"{'='*80}")
    for r in sorted(reasons.keys()):
        d = reasons[r]
        wr = d['wins'] / d['count'] * 100 if d['count'] > 0 else 0
        avg_r = statistics.mean(d['pnl'])
        print(f"  {r:<18s} {d['count']:>4d} trades  win={wr:.0f}%  avg={avg_r:+.2f}%")

    # ============================================================
    # Breakdown by squeeze color
    # ============================================================
    by_color = {}
    for t, s in zip(trades, signal_diags):
        color = s['squeeze_color']
        by_color.setdefault(color, {'count': 0, 'wins': 0, 'pnl': []})
        by_color[color]['count'] += 1
        by_color[color]['wins'] += 1 if t['won'] else 0
        by_color[color]['pnl'].append(t['pnl_pct'])

    print(f"\n{'='*80}")
    print(f"BY SQUEEZE COLOR")
    print(f"{'='*80}")
    for color in sorted(by_color.keys()):
        d = by_color[color]
        wr = d['wins'] / d['count'] * 100 if d['count'] > 0 else 0
        avg_c = statistics.mean(d['pnl'])
        print(f"  {color:<18s} {d['count']:>4d} trades  win={wr:.0f}%  avg={avg_c:+.2f}%")

    # ============================================================
    # Breakdown by sweep type
    # ============================================================
    by_sweep = {}
    for t, s in zip(trades, signal_diags):
        st = s['sweep_type']
        by_sweep.setdefault(st, {'count': 0, 'wins': 0, 'pnl': []})
        by_sweep[st]['count'] += 1
        by_sweep[st]['wins'] += 1 if t['won'] else 0
        by_sweep[st]['pnl'].append(t['pnl_pct'])

    print(f"\n{'='*80}")
    print(f"BY SWEEP TYPE")
    print(f"{'='*80}")
    for st in sorted(by_sweep.keys()):
        d = by_sweep[st]
        wr = d['wins'] / d['count'] * 100 if d['count'] > 0 else 0
        avg_s = statistics.mean(d['pnl'])
        print(f"  {st:<18s} {d['count']:>4d} trades  win={wr:.0f}%  avg={avg_s:+.2f}%")

    # ============================================================
    # Breakdown by hour of day (UTC)
    # ============================================================
    by_hour = {}
    for t, s in zip(trades, signal_diags):
        h = s['hour']
        by_hour.setdefault(h, {'count': 0, 'wins': 0, 'pnl': []})
        by_hour[h]['count'] += 1
        by_hour[h]['wins'] += 1 if t['won'] else 0
        by_hour[h]['pnl'].append(t['pnl_pct'])

    print(f"\n{'='*80}")
    print(f"BY HOUR (UTC)")
    print(f"{'='*80}")
    for h in sorted(by_hour.keys()):
        d = by_hour[h]
        wr = d['wins'] / d['count'] * 100 if d['count'] > 0 else 0
        avg_h = statistics.mean(d['pnl'])
        bar = '#' * max(1, d['count'])
        print(f"  {h:02d}:00              {d['count']:>4d} trades  win={wr:.0f}%  avg={avg_h:+.2f}%  {bar}")

    # ============================================================
    # PnL distribution
    # ============================================================
    buckets = [(-100, -20), (-20, -5), (-5, 0), (0, 3), (3, 8), (8, 15), (15, 100)]
    print(f"\n{'='*80}")
    print(f"PnL DISTRIBUTION")
    print(f"{'='*80}")
    for lo, hi in buckets:
        count = sum(1 for p in pnl_pcts if lo <= p < hi)
        if count > 0:
            bar = '#' * count
            print(f"  [{lo:+4d}% ~ {hi:+4d}%): {count:3d}  {bar}")

    # ============================================================
    # Worst/best trades
    # ============================================================
    sorted_trades = sorted(zip(pnl_pcts, trades, signal_diags), key=lambda x: x[0])
    print(f"\n{'='*80}")
    print(f"WORST 5 TRADES")
    print(f"{'='*80}")
    for pnl, t, s in sorted_trades[:5]:
        t_str = datetime.fromtimestamp(t['slot_ts'], UTC).strftime('%m-%d %H:%M')
        print(f"  {t_str}  {t['direction']:>5s}  {pnl:+6.2f}%  "
              f"{t['exit_reason']:<12s}  sweep={s['sweep_type']:<8s}  color={s['squeeze_color']}")

    print(f"\n{'='*80}")
    print(f"BEST 5 TRADES")
    print(f"{'='*80}")
    for pnl, t, s in sorted_trades[-5:]:
        t_str = datetime.fromtimestamp(t['slot_ts'], UTC).strftime('%m-%d %H:%M')
        print(f"  {t_str}  {t['direction']:>5s}  {pnl:+6.2f}%  "
              f"{t['exit_reason']:<12s}  sweep={s['sweep_type']:<8s}  color={s['squeeze_color']}")

    print(f"\n{'='*80}")
    print("Tip: Run this backtest regularly to monitor VLS-5M performance.")
    print("Strategy doc recommends 100+ paper trades before live deployment.")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
