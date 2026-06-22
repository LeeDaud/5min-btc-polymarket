#!/usr/bin/env python3
"""
Backtest BTC 5-min Polymarket momentum strategy.
Uses MEXC 1-min BTC klines for price data.
Simulates entry/exit based on strategy rules.
"""

import requests
import json
import math
import statistics
from datetime import datetime, timezone, timedelta

UTC = timezone.utc

def get_btc_klines(n_candles=500):
    """Get BTCUSDT 1-minute klines from MEXC (max 500 returned)."""
    r = requests.get('https://api.mexc.com/api/v3/klines', params={
        'symbol': 'BTCUSDT',
        'interval': '1m',
        'limit': min(n_candles, 500)
    }, timeout=15)
    r.raise_for_status()
    data = r.json()
    candles = []
    for c in data:
        candles.append({
            'open_time': int(c[0]),
            'open': float(c[1]),
            'high': float(c[2]),
            'low': float(c[3]),
            'close': float(c[4]),
            'volume': float(c[5]),
        })
    return candles


def bucket_5m(ts):
    return ts - (ts % 300)


def compute_5min_slots(candles):
    """Group 1-min candles into 5-min slots."""
    slots = {}
    for c in candles:
        bucket = bucket_5m(c['open_time'] // 1000)
        if bucket not in slots:
            slots[bucket] = []
        slots[bucket].append(c)
    # Sort and only keep slots with at least 3 candles (mostly complete)
    result = {}
    for bucket in sorted(slots.keys()):
        candles_in_slot = slots[bucket]
        if len(candles_in_slot) >= 3:
            result[bucket] = candles_in_slot
    return result


def norm_cdf(x):
    """Standard normal CDF approximation."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def simulate_slot(slot_candles, threshold=0.70, stop_loss_pct=0.25,
                  entry_min_btc_move=70, exit_before_sec=20,
                  btc_5min_std=None, clob_spread=0.03):
    """
    Simulate the strategy on one 5-minute slot.

    Key insight: The Polymarket CLOB *ask* price is what triggers entry.
    The ask = fair_probability + spread/2 (approximately).
    So if fair probability >= threshold - spread/2, the ask triggers.

    Returns: (signal, result_dict) or (None, None) if no trade
    """
    slot_start = slot_candles[0]['open_time'] // 1000
    slot_end = slot_start + 300  # 5 minutes later

    # Use calibrated 5-min std if available, else default
    if btc_5min_std is None:
        btc_5min_std = 70.0

    # BTC price at slot start (use open of first candle)
    btc_start = slot_candles[0]['open']

    # Walk through the slot minute by minute, simulating monitoring
    for i, c in enumerate(slot_candles):
        ts = c['open_time'] // 1000
        seconds_elapsed = ts - slot_start
        seconds_left = slot_end - ts

        # Entry window: prefer ~120 seconds left (roughly T-2 min)
        if seconds_left < 60 or seconds_left > 180:
            continue

        btc_now = c['close']
        btc_move = btc_now - btc_start

        # Check momentum condition: minimal BTC move
        if abs(btc_move) < 40:
            continue

        # Determine direction
        direction = 'UP' if btc_move > 0 else 'DOWN'

        # Model the FAIR probability for the winning side
        # BTC move is the signal; remaining uncertainty is scaled volatility
        remaining_sec = slot_end - ts
        remaining_vol = btc_5min_std * math.sqrt(remaining_sec / 300.0)

        # P(BTC stays in same direction at close) =
        #   P(final_move > 0 | current_move)
        # This is a biased random walk: expected final = current + drift
        # Under zero drift: P(win) = norm.cdf(current_move / remaining_vol)
        abs_move = abs(btc_move)
        if remaining_vol > 0:
            z = abs_move / remaining_vol
            fair_prob = norm_cdf(z)
        else:
            fair_prob = 0.5

        # Fair prob naturally in [0.5, 1.0] since we follow momentum direction
        # Clamp to reasonable range
        fair_prob = max(0.50, min(0.999, fair_prob))

        # CLOB ask = fair_prob + half-spread (market makers post above fair)
        # Spread is typically 0.01-0.05 for liquid Polymarket markets
        ask_price = min(0.999, fair_prob + clob_spread / 2.0)

        # Strategy only enters if ask price >= threshold
        if ask_price < threshold:
            continue

        entry_price = ask_price

        # === ENTRY ===
        # Now trace through remaining candles to see what happens

        # Entry at current candle's close price
        entry_btc = btc_now
        entry_ts = ts
        entry_price_final = entry_price  # Polymarket token price at entry

        # Stop loss in Polymarket price terms
        sl_price = entry_price_final * (1.0 - stop_loss_pct)

        # Walk through remaining candles (high/low resolution for stop check)
        result = {
            'slot_start': slot_start,
            'direction': direction,
            'btc_start': btc_start,
            'btc_entry': entry_btc,
            'btc_move_at_entry': btc_move,
            'entry_price_model': round(entry_price_final, 4),
            'entry_ts': entry_ts,
            'seconds_left_at_entry': seconds_left,
        }

        # Check minute by minute from entry to close
        exit_price = None
        exit_reason = None

        for j in range(i + 1, len(slot_candles)):
            follow_c = slot_candles[j]
            follow_btc = follow_c['close']
            follow_remaining = slot_end - follow_c['open_time'] // 1000
            follow_remaining = max(10, follow_remaining)

            # Current fair probability based on BTC move from start
            btc_move_from_start = follow_btc - btc_start
            if direction == 'UP':
                effective_move = btc_move_from_start
            else:
                effective_move = -btc_move_from_start

            follow_vol = btc_5min_std * math.sqrt(follow_remaining / 300.0)
            if follow_vol > 0:
                z = effective_move / follow_vol
                current_mid = norm_cdf(z)
            else:
                current_mid = 0.5
            current_mid = max(0.001, min(0.999, current_mid))

            # The price we can sell at (bid) = mid - spread/2
            current_bid = max(0.001, current_mid - clob_spread / 2.0)

            # Check stop loss
            if current_bid <= sl_price:
                exit_price = sl_price
                exit_reason = f'stop_loss'
                break

            # Check time exit (20s before close)
            follow_ts = follow_c['open_time'] // 1000
            if slot_end - follow_ts <= exit_before_sec:
                exit_price = current_bid
                exit_reason = f'time_exit'
                break

            # Last candle: force exit at fair value
            if j == len(slot_candles) - 1:
                exit_price = current_bid
                exit_reason = 'end_of_data'

        # If still no exit, use final outcome based on actual BTC close
        if exit_price is None:
            final_btc = slot_candles[-1]['close']
            btc_final_move = final_btc - btc_start
            won = (direction == 'UP' and btc_final_move > 0) or \
                  (direction == 'DOWN' and btc_final_move < 0)
            exit_price = 1.0 if won else 0.0
            exit_reason = 'settlement'

        # Calculate PnL
        if exit_price is not None:
            pnl_usd = (exit_price - entry_price_final)  # per $1 of position
            pnl_pct = pnl_usd / entry_price_final

            result.update({
                'exit_price': round(exit_price, 4),
                'exit_reason': exit_reason,
                'won': exit_price > entry_price_final,
                'pnl_per_dollar': round(pnl_usd, 4),
                'pnl_pct': round(pnl_pct * 100, 2),
            })
        else:
            result.update({
                'exit_price': None,
                'exit_reason': 'no_exit',
                'won': False,
                'pnl_per_dollar': 0,
                'pnl_pct': 0,
            })

        return ('ENTRY', result)

    # No entry signal found
    return ('NO_ENTRY', None)


def main():
    print("=" * 70)
    print("BTC 5-Minute Polymarket Momentum Strategy - Backtest")
    print("=" * 70)

    # Fetch BTC data
    print("\n[1] Fetching BTC 1-min klines from MEXC...")
    candles = get_btc_klines(500)
    print(f"    Got {len(candles)} candles")
    print(f"    Range: {datetime.fromtimestamp(candles[0]['open_time']//1000, UTC)} → "
          f"{datetime.fromtimestamp(candles[-1]['open_time']//1000, UTC)}")

    # Compute 5-min slots
    slots = compute_5min_slots(candles)
    print(f"    Grouped into {len(slots)} five-minute slots\n")

    # Filter to only completed slots (closed = slot end is in the past)
    now_ts = int(datetime.now(UTC).timestamp())
    completed_slots = {}
    for bucket, sc in slots.items():
        if bucket + 300 < now_ts - 30:  # at least 30s past close
            completed_slots[bucket] = sc

    print(f"    {len(completed_slots)} completed slots available for backtest\n")

    # Calibrate: compute actual BTC 5-minute volatility from data
    five_min_returns = []
    for bucket, sc in sorted(completed_slots.items()):
        if len(sc) >= 3:
            ret = sc[-1]['close'] - sc[0]['open']
            five_min_returns.append(ret)
    btc_5min_std = statistics.stdev(five_min_returns) if len(five_min_returns) >= 5 else 70.0
    print(f"    Calibrated BTC 5-min std dev: ${btc_5min_std:.1f}\n")

    # Run simulation on all completed slots
    test_slots = dict(sorted(completed_slots.items()))
    total_slots = len(test_slots)

    print(f"[2] Running backtest on {total_slots} completed slots...\n")

    results = []
    skipped = 0
    for bucket in sorted(test_slots.keys()):
        sc = test_slots[bucket]
        slot_time = datetime.fromtimestamp(bucket, UTC)
        btc_open = sc[0]['open']
        btc_close = sc[-1]['close']
        btc_move_total = btc_close - btc_open

        signal, result = simulate_slot(sc, btc_5min_std=btc_5min_std)

        if result:
            results.append(result)
            direction_icon = "UP" if result['direction'] == 'UP' else "DN"
            win_mark = "W" if result['won'] else "L"
            pnl_str = f"{result['pnl_pct']:+6.1f}%"
            print(f"  {slot_time.strftime('%H:%M')} {direction_icon} {win_mark} "
                  f"entry@{result['entry_price_model']:.3f} "
                  f"exit@{result['exit_price']:.3f} "
                  f"({result['exit_reason']:12s}) | {pnl_str}")
        else:
            skipped += 1

    # Summary statistics
    print(f"\n{'='*70}")
    print(f"RESULTS SUMMARY — {total_slots} slots, {skipped} skipped, {len(results)} trades")
    print(f"{'='*70}")
    if results:
        n_trades = len(results)
        n_wins = sum(1 for r in results if r['won'])
        n_losses = n_trades - n_wins
        win_rate = n_wins / n_trades * 100 if n_trades > 0 else 0

        pnl_pcts = [r['pnl_pct'] for r in results]
        pnl_per_dollar = [r['pnl_per_dollar'] for r in results]
        avg_pnl = statistics.mean(pnl_pcts)
        total_pnl = sum(pnl_pcts)
        median_pnl = statistics.median(pnl_pcts)

        n_stop_loss = sum(1 for r in results if 'stop_loss' in str(r.get('exit_reason', '')))

        # Avg by exit reason
        exit_stats = {}
        for r in results:
            reason = r.get('exit_reason', 'unknown')
            exit_stats.setdefault(reason, []).append(r['pnl_pct'])
        exit_summary = {k: (len(v), statistics.mean(v)) for k, v in exit_stats.items()}

        print(f"")
        print(f"  Win rate:          {n_wins}/{n_trades} ({win_rate:.1f}%)")
        print(f"  Stop-loss hits:    {n_stop_loss}")
        print(f"  Avg return/trade:  {avg_pnl:+.2f}%")
        print(f"  Median return:     {median_pnl:+.2f}%")
        print(f"  Cumulative return: {total_pnl:+.2f}%")
        print(f"  Avg $ PnL per $1:  ${statistics.mean(pnl_per_dollar):+.4f}")

        # Best/worst
        best = max(results, key=lambda r: r['pnl_pct'])
        worst = min(results, key=lambda r: r['pnl_pct'])
        print(f"  Best trade:        {best['pnl_pct']:+.1f}% (entry {best['entry_price_model']:.3f})")
        print(f"  Worst trade:       {worst['pnl_pct']:+.1f}% (entry {worst['entry_price_model']:.3f})")

        # PnL distribution
        buckets = [(-100, -20), (-20, -5), (-5, 0), (0, 3), (3, 8), (8, 15), (15, 100)]
        print(f"\n  PnL distribution:")
        for lo, hi in buckets:
            count = sum(1 for p in pnl_pcts if lo <= p < hi)
            if count > 0:
                bar = '#' * max(1, count)
                print(f"    [{lo:+4d}% ~ {hi:+4d}%): {count:3d} trades  {bar}")

        print(f"\n  By exit reason:")
        for reason, (count, avg) in sorted(exit_summary.items()):
            print(f"    {reason:20s}: {count:2d} trades, avg {avg:+.2f}%")

        # Sharpe-like ratio (return / std, annualized not applicable here)
        if len(pnl_pcts) >= 3:
            pnl_std = statistics.stdev(pnl_pcts)
            if pnl_std > 0:
                sharpe_like = avg_pnl / pnl_std
                print(f"\n  Return/Std ratio:  {sharpe_like:+.2f} (higher = better risk-adjusted)")
    else:
        print("  No trades generated")

    print(f"\n{'='*70}")
    print("Model assumptions:")
    print("  - Entry price: norm.cdf(BTC_move / remaining_vol) + spread/2")
    print("  - Exit price:  norm.cdf(BTC_move / remaining_vol) - spread/2")
    print("  - Stop loss: -25% from entry price")
    print("  - Data: MEXC 1-min BTC klines, no CLOB order book history")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
    input("\nPress Enter to exit...")
