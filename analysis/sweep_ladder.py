#!/usr/bin/env python3
"""Grid-search optimal TP ladder configuration (thresholds + ratios)."""
import sys, statistics, itertools, random, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from backtest_btc5m import get_btc_klines, bucket_5m, compute_5min_slots, norm_cdf
from state_manager import load_state


def simulate_slot_ladder(slot_candles, threshold=0.70, hard_stop_pct=0.07,
                         trail_activation_pct=0.05, trail_callback_pct=0.05,
                         tp_ladder=None, exit_before_sec=20,
                         btc_5min_std=None, clob_spread=0.03):
    """Simulate with multi-step TP ladder."""
    if tp_ladder is None:
        tp_ladder = [[0.10, 0.20], [0.20, 0.30], [0.30, 0.50]]

    slot_start = slot_candles[0]['open_time'] // 1000
    slot_end = slot_start + 300
    if btc_5min_std is None:
        btc_5min_std = 70.0
    btc_start = slot_candles[0]['open']
    noise_std = clob_spread * 0.8
    gap_prob = 0.15
    gap_mag = clob_spread * 2.5

    for i, c in enumerate(slot_candles):
        ts = c['open_time'] // 1000
        seconds_left = slot_end - ts
        if seconds_left < 60 or seconds_left > 180:
            continue

        btc_now = c['close']
        btc_move = btc_now - btc_start
        if abs(btc_move) < 40:
            continue

        direction = 'UP' if btc_move > 0 else 'DOWN'
        remaining_sec = slot_end - ts
        remaining_vol = btc_5min_std * (remaining_sec / 300.0) ** 0.5
        abs_move = abs(btc_move)
        fair_prob = norm_cdf(abs_move / remaining_vol) if remaining_vol > 0 else 0.5
        fair_prob = max(0.50, min(0.999, fair_prob))
        ask_price = min(0.999, fair_prob + clob_spread / 2.0)
        if ask_price < threshold:
            continue

        entry_price = ask_price
        entry_btc = btc_now
        hard_stop = entry_price * (1.0 - hard_stop_pct)
        highest_price = entry_price
        trailing_active = False
        trail_stop = hard_stop
        original_shares = 1.0  # normalized
        remaining_shares = 1.0
        ladder_executed = [False] * len(tp_ladder)
        ladder_records = []
        exit_price = None
        exit_reason = None

        for j in range(i + 1, len(slot_candles)):
            follow_c = slot_candles[j]
            follow_btc = follow_c['close']
            follow_remaining = max(10, slot_end - follow_c['open_time'] // 1000)

            btc_move_from_start = follow_btc - btc_start
            eff_move = btc_move_from_start if direction == 'UP' else -btc_move_from_start
            follow_vol = btc_5min_std * (follow_remaining / 300.0) ** 0.5
            base_mid = norm_cdf(eff_move / follow_vol) if follow_vol > 0 else 0.5
            base_mid = max(0.001, min(0.999, base_mid))
            noise = random.gauss(0, noise_std)
            gap = random.gauss(0, gap_mag) if random.random() < gap_prob else 0
            current_mid = max(0.001, min(0.999, base_mid + noise + gap))
            current_bid = max(0.001, current_mid - clob_spread / 2.0)

            pnl_pct = (current_bid - entry_price) / entry_price

            # Activate trailing
            if not trailing_active and pnl_pct >= trail_activation_pct:
                trailing_active = True
                highest_price = current_bid
                trail_stop = highest_price * (1.0 - trail_callback_pct)

            if trailing_active and current_bid > highest_price:
                highest_price = current_bid
                trail_stop = highest_price * (1.0 - trail_callback_pct)

            # Ladder TP
            for li, (lvl_pct, lvl_ratio) in enumerate(tp_ladder):
                if ladder_executed[li]:
                    continue
                if pnl_pct >= lvl_pct:
                    sell_shares = original_shares * lvl_ratio
                    sell_shares = min(sell_shares, remaining_shares)
                    if sell_shares > 0:
                        slip = 1.0 - random.uniform(0.01, 0.06)
                        sell_price = current_bid * slip
                        ladder_records.append({
                            'level': lvl_pct, 'ratio': lvl_ratio,
                            'shares': sell_shares, 'price': sell_price,
                            'pnl': (sell_price - entry_price) / entry_price,
                        })
                        remaining_shares -= sell_shares
                    ladder_executed[li] = True

            if remaining_shares <= 0:
                exit_reason = 'ladder_complete'
                exit_price = ladder_records[-1]['price'] if ladder_records else current_bid
                break

            # Hard stop
            if current_bid <= hard_stop:
                if random.random() < 0.30:
                    gap_slip = 1.0 - random.uniform(0.02, 0.15)
                else:
                    gap_slip = 0.98
                exit_price = hard_stop * gap_slip
                exit_reason = f'hard_stop'
                break

            # Trailing stop
            if trailing_active and current_bid <= trail_stop:
                if random.random() < 0.30:
                    gap_slip = 1.0 - random.uniform(0.02, 0.15)
                else:
                    gap_slip = 0.98
                exit_price = trail_stop * gap_slip
                exit_reason = f'trail_stop'
                break

            if slot_end - follow_c['open_time'] // 1000 <= exit_before_sec:
                exit_price = current_bid
                exit_reason = 'time_exit'
                break

            if j == len(slot_candles) - 1:
                exit_price = current_bid
                exit_reason = 'end_of_data'

        if exit_price is None:
            final_btc = slot_candles[-1]['close']
            btc_final_move = final_btc - btc_start
            won = (direction == 'UP' and btc_final_move > 0) or (direction == 'DOWN' and btc_final_move < 0)
            exit_price = 1.0 if won else 0.0
            exit_reason = 'settlement'

        # Composite PnL: ladder sells + final exit on remaining
        total_pnl = 0.0
        for lr in ladder_records:
            total_pnl += lr['pnl'] * lr['shares']
        if remaining_shares > 0:
            total_pnl += (exit_price - entry_price) / entry_price * remaining_shares
        total_pnl_pct = total_pnl * 100

        return {
            'direction': direction,
            'entry_price': round(entry_price, 4),
            'exit_price': round(exit_price, 4),
            'exit_reason': exit_reason,
            'won': total_pnl_pct > 0,
            'pnl_pct': round(total_pnl_pct, 2),
            'ladder_sells': len(ladder_records),
            'remaining_shares': round(remaining_shares, 4),
        }

    return None


def run_test(params, candles):
    slots = compute_5min_slots(candles)
    returns = []
    for _, sc in sorted(slots.items()):
        if len(sc) >= 3:
            returns.append(sc[-1]['close'] - sc[0]['open'])
    btc_5min_std = statistics.stdev(returns) if len(returns) >= 5 else 70.0

    slot_list = sorted(slots.items())
    pnl_list = []
    wins = 0
    trades = 0
    reasons = {}
    ladder_sells_total = 0
    for bucket, sc in slot_list:
        r = simulate_slot_ladder(sc, btc_5min_std=btc_5min_std, hard_stop_pct=params.get('hard_stop_pct', 0.07),
                                 trail_activation_pct=params.get('trail_activation_pct', 0.05),
                                 trail_callback_pct=params.get('trail_callback_pct', 0.05),
                                 tp_ladder=params.get('tp_ladder'))
        if r:
            trades += 1
            pnl_list.append(r['pnl_pct'])
            if r['won']:
                wins += 1
            reason = r['exit_reason'].split('_')[0] if r['exit_reason'] else '?'
            reasons[reason] = reasons.get(reason, 0) + 1
            ladder_sells_total += r['ladder_sells']

    if trades < 3:
        return None
    return {
        'params': params,
        'n_trades': trades,
        'win_rate': wins / trades * 100,
        'avg_pnl': statistics.mean(pnl_list),
        'med_pnl': statistics.median(pnl_list),
        'cum_pnl': sum(pnl_list),
        'reasons': reasons,
        'avg_ladder_sells': ladder_sells_total / trades,
    }


def ladder_label(lad):
    return "|".join(f"+{l[0]*100:.0f}%:{l[1]*100:.0f}%" for l in lad)


def main():
    print("Fetching BTC klines...")
    candles = get_btc_klines(500)
    print(f"Got {len(candles)} candles\n")

    # Ladder configs to test
    ladder_configs = [
        # Single TP (baseline)
        [[0.20, 0.50]],
        [[0.25, 0.50]],
        [[0.30, 0.50]],
        # 2-step ladders
        [[0.10, 0.25], [0.20, 0.75]],
        [[0.10, 0.33], [0.25, 0.67]],
        [[0.15, 0.30], [0.25, 0.70]],
        [[0.15, 0.40], [0.30, 0.60]],
        [[0.10, 0.50], [0.25, 0.50]],
        # 3-step ladders
        [[0.10, 0.20], [0.20, 0.30], [0.30, 0.50]],  # current
        [[0.10, 0.25], [0.20, 0.25], [0.30, 0.50]],
        [[0.10, 0.20], [0.15, 0.30], [0.25, 0.50]],
        [[0.10, 0.33], [0.20, 0.33], [0.30, 0.34]],
        [[0.15, 0.20], [0.25, 0.30], [0.35, 0.50]],
        [[0.10, 0.20], [0.20, 0.40], [0.30, 0.40]],
        [[0.05, 0.15], [0.15, 0.35], [0.30, 0.50]],
        [[0.05, 0.20], [0.15, 0.30], [0.25, 0.50]],
        # 4-step ladders
        [[0.05, 0.15], [0.10, 0.25], [0.20, 0.30], [0.30, 0.30]],
        [[0.10, 0.15], [0.15, 0.25], [0.20, 0.30], [0.30, 0.30]],
        [[0.10, 0.20], [0.15, 0.25], [0.25, 0.25], [0.35, 0.30]],
    ]

    # Test across different hard_stop values
    hard_stops = [0.07, 0.10]

    print("=" * 85)
    print("SWEEP: TP Ladder Configurations (hard_stop=7% & 10%, trail=5%/5%)")
    print("=" * 85)

    all_results = []
    for hs in hard_stops:
        for lad in ladder_configs:
            params = {
                'hard_stop_pct': hs,
                'trail_activation_pct': 0.05,
                'trail_callback_pct': 0.05,
                'tp_ladder': lad,
            }
            r = run_test(params, candles)
            if r:
                all_results.append(r)

    all_results.sort(key=lambda x: x['cum_pnl'], reverse=True)

    # Sort by cum_pnl
    print(f"\n{'='*85}")
    print("TOP 15 BY CUMULATIVE PnL")
    print(f"{'='*85}")
    print(f"{'#':<3} {'hard':<6} {'ladder':<45} {'cum':>8} {'avg':>7} {'win':>6} {'n':>4} {'sells':>6} {'exits'}")
    print("-" * 85)
    for i, r in enumerate(all_results[:15]):
        p = r['params']
        lad_str = ladder_label(p['tp_ladder'])
        print(f"{i+1:<3} {p['hard_stop_pct']:.0%}     {lad_str:<45} {r['cum_pnl']:>+7.1f}% {r['avg_pnl']:>+6.2f}% {r['win_rate']:>5.1f}% {r['n_trades']:>4} {r['avg_ladder_sells']:>5.1f}  {r['reasons']}")

    print(f"\n{'='*85}")
    print("TOP 15 BY WIN RATE")
    print(f"{'='*85}")
    by_win = sorted(all_results, key=lambda x: x['win_rate'], reverse=True)
    for i, r in enumerate(by_win[:15]):
        p = r['params']
        lad_str = ladder_label(p['tp_ladder'])
        print(f"{i+1:<3} {p['hard_stop_pct']:.0%}     {lad_str:<45} {r['win_rate']:>5.1f}% {r['cum_pnl']:>+7.1f}% {r['avg_pnl']:>+6.2f}% {r['n_trades']:>4} {r['avg_ladder_sells']:>5.1f}")

    print(f"\n{'='*85}")
    print("TOP 15 BY AVG LADDER SELLS (most profit-locking)")
    print(f"{'='*85}")
    by_sells = sorted(all_results, key=lambda x: x['avg_ladder_sells'], reverse=True)
    for i, r in enumerate(by_sells[:15]):
        p = r['params']
        lad_str = ladder_label(p['tp_ladder'])
        print(f"{i+1:<3} {p['hard_stop_pct']:.0%}     {lad_str:<45} sells={r['avg_ladder_sells']:.1f}  cum={r['cum_pnl']:>+7.1f}%  win={r['win_rate']:>5.1f}%")


if __name__ == '__main__':
    main()
