#!/usr/bin/env python3
"""Grid-search optimal stop-loss / trailing / partial-TP parameters using backtest engine."""
import sys, statistics, itertools
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from backtest_btc5m import get_btc_klines, bucket_5m, compute_5min_slots, norm_cdf, simulate_slot
from state_manager import load_state


def simulate_slot_advanced(slot_candles, threshold=0.70, hard_stop_pct=0.07,
                           trail_activation_pct=0.05, trail_callback_pct=0.05,
                           tp_partial_pct=0.20, tp_partial_ratio=0.50,
                           entry_min_btc_move=40, exit_before_sec=20,
                           btc_5min_std=None, clob_spread=0.03):
    """
    Extended simulation with hard stop + trailing stop + partial TP.
    Same BTC model as original, but with our layered exit logic.
    """
    import random
    slot_start = slot_candles[0]['open_time'] // 1000
    slot_end = slot_start + 300
    if btc_5min_std is None:
        btc_5min_std = 70.0
    btc_start = slot_candles[0]['open']

    # CLOB microstructure: price noise + gap-through risk
    # Thin books cause ±1-3% random jitter per tick, and occasional gap jumps
    noise_std = clob_spread * 0.8  # per-tick noise
    gap_prob = 0.15  # 15% chance per tick of a sudden jump
    gap_mag = clob_spread * 2.5  # gap magnitude

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
        entry_ts = ts

        # Our exit system
        hard_stop = entry_price * (1.0 - hard_stop_pct)
        highest_price = entry_price
        trailing_active = False
        trail_stop = hard_stop
        partial_tp_done = False
        exit_price = None
        exit_reason = None
        partial_tp_record = None

        for j in range(i + 1, len(slot_candles)):
            follow_c = slot_candles[j]
            follow_btc = follow_c['close']
            follow_remaining = max(10, slot_end - follow_c['open_time'] // 1000)

            btc_move_from_start = follow_btc - btc_start
            eff_move = btc_move_from_start if direction == 'UP' else -btc_move_from_start
            follow_vol = btc_5min_std * (follow_remaining / 300.0) ** 0.5
            base_mid = norm_cdf(eff_move / follow_vol) if follow_vol > 0 else 0.5
            base_mid = max(0.001, min(0.999, base_mid))
            # Add CLOB noise + gap risk
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

            # Update trailing
            if trailing_active and current_bid > highest_price:
                highest_price = current_bid
                trail_stop = highest_price * (1.0 - trail_callback_pct)

            # Partial TP
            if tp_partial_pct > 0 and not partial_tp_done and pnl_pct >= tp_partial_pct:
                # Close ratio of position at current bid with variable slip
                tp_slip = 1.0 - random.uniform(0.01, 0.06)  # 1-6% slip
                partial_exit_price = current_bid * tp_slip
                partial_tp_record = {
                    'shares_closed_pct': tp_partial_ratio,
                    'exit_price': partial_exit_price,
                    'pnl_on_closed': (partial_exit_price - entry_price) / entry_price,
                }
                partial_tp_done = True
                # Update entry_price to reflect remaining position (cost basis doesn't change for PnL calc)
                # Actually for PnL we just track both parts separately

            # Check exits with gap-through simulation
            if current_bid <= hard_stop:
                # 30% chance of gap-through: fill well below stop level
                if random.random() < 0.30:
                    gap_slip = 1.0 - random.uniform(0.02, 0.15)
                else:
                    gap_slip = 0.98  # normal 2% slip
                exit_price = hard_stop * gap_slip
                exit_reason = f'hard_stop_{int(hard_stop_pct*100)}pct'
                break

            if trailing_active and current_bid <= trail_stop:
                if random.random() < 0.30:
                    gap_slip = 1.0 - random.uniform(0.02, 0.15)
                else:
                    gap_slip = 0.98
                exit_price = trail_stop * gap_slip
                exit_reason = f'trail_stop_{int(trail_callback_pct*100)}pct'
                break

            # Time exit
            if slot_end - follow_c['open_time'] // 1000 <= exit_before_sec:
                exit_price = current_bid
                exit_reason = 'time_exit'
                break

            # Last candle
            if j == len(slot_candles) - 1:
                exit_price = current_bid
                exit_reason = 'end_of_data'

        if exit_price is None:
            final_btc = slot_candles[-1]['close']
            btc_final_move = final_btc - btc_start
            won = (direction == 'UP' and btc_final_move > 0) or (direction == 'DOWN' and btc_final_move < 0)
            exit_price = 1.0 if won else 0.0
            exit_reason = 'settlement'

        # PnL with partial TP
        if partial_tp_record:
            partial_pnl = partial_tp_record['pnl_on_closed'] * tp_partial_ratio
            main_pnl = (exit_price - entry_price) / entry_price * (1.0 - tp_partial_ratio)
            total_pnl_pct = (partial_pnl + main_pnl) * 100
        else:
            total_pnl_pct = (exit_price - entry_price) / entry_price * 100

        return {
            'direction': direction,
            'entry_price': round(entry_price, 4),
            'exit_price': round(exit_price, 4),
            'exit_reason': exit_reason,
            'won': exit_price > entry_price,
            'pnl_pct': round(total_pnl_pct, 2),
            'partial_tp': partial_tp_record,
            'trailing_active': trailing_active,
        }

    return None


def run_sweep(param_grid, candles, title=""):
    slots = compute_5min_slots(candles)
    five_min_returns = []
    for bucket, sc in sorted(slots.items()):
        if len(sc) >= 3:
            five_min_returns.append(sc[-1]['close'] - sc[0]['open'])
    btc_5min_std = statistics.stdev(five_min_returns) if len(five_min_returns) >= 5 else 70.0

    slot_list = sorted(slots.items())
    results = []
    for params in param_grid:
        p_label = ", ".join(f"{k}={v}" for k, v in params.items())
        pnl_list = []
        wins = 0
        trades = 0
        reasons = {}
        for bucket, sc in slot_list:
            r = simulate_slot_advanced(sc, btc_5min_std=btc_5min_std, **params)
            if r:
                trades += 1
                pnl_list.append(r['pnl_pct'])
                if r['won']:
                    wins += 1
                reason = r['exit_reason'].split('_')[0] if r['exit_reason'] else '?'
                reasons[reason] = reasons.get(reason, 0) + 1
        if trades >= 3:
            cum = sum(pnl_list)
            avg = statistics.mean(pnl_list)
            med = statistics.median(pnl_list)
            win_rate = wins / trades * 100
            results.append({
                'params': params,
                'n_trades': trades,
                'win_rate': win_rate,
                'avg_pnl': avg,
                'med_pnl': med,
                'cum_pnl': cum,
                'reasons': reasons,
            })
    results.sort(key=lambda x: x['cum_pnl'], reverse=True)
    return results


def main():
    print("Fetching BTC klines...")
    candles = get_btc_klines(500)
    print(f"Got {len(candles)} candles\n")

    # sweep hard stop (fixed) + trailing (activation=callback) combos
    hard_stops = [0.05, 0.07, 0.10, 0.12, 0.15]
    trail_params = [(0.03, 0.03), (0.05, 0.05), (0.07, 0.07), (0.10, 0.10)]
    tp_partials = [0.15, 0.20, 0.25, 0.30]

    print("=" * 80)
    print("SWEEP: hard_stop × trail(act/cb) × partial_tp")
    print("=" * 80)

    best = []
    for hs, (ta, tc), tpp in itertools.product(hard_stops, trail_params, tp_partials):
        params = {
            'hard_stop_pct': hs,
            'trail_activation_pct': ta,
            'trail_callback_pct': tc,
            'tp_partial_pct': tpp,
            'tp_partial_ratio': 0.50,
        }
        results = run_sweep([params], candles)
        if results:
            r = results[0]
            best.append(r)
            print(f"  hard={hs:.0%} trail_act={ta:.0%} trail_cb={tc:.0%} tp={tpp:.0%}  "
                  f"trades={r['n_trades']:3d}  win={r['win_rate']:4.1f}%  "
                  f"avg={r['avg_pnl']:+6.2f}%  cum={r['cum_pnl']:+7.2f}%  "
                  f"exits={r['reasons']}")

    best.sort(key=lambda x: x['cum_pnl'], reverse=True)

    print(f"\n{'='*80}")
    print("TOP 10 BY CUMULATIVE PnL")
    print(f"{'='*80}")
    for i, r in enumerate(best[:10]):
        p = r['params']
        print(f"  #{i+1} hard={p['hard_stop_pct']:.0%} trail_act={p['trail_activation_pct']:.0%} "
              f"trail_cb={p['trail_callback_pct']:.0%} tp={p['tp_partial_pct']:.0%}  "
              f"cum={r['cum_pnl']:+.2f}%  avg={r['avg_pnl']:+.2f}%  "
              f"win={r['win_rate']:.1f}%  n={r['n_trades']}")

    print(f"\n{'='*80}")
    print("TOP 10 BY WIN RATE")
    print(f"{'='*80}")
    by_win = sorted(best, key=lambda x: x['win_rate'], reverse=True)
    for i, r in enumerate(by_win[:10]):
        p = r['params']
        print(f"  #{i+1} hard={p['hard_stop_pct']:.0%} trail_act={p['trail_activation_pct']:.0%} "
              f"trail_cb={p['trail_callback_pct']:.0%} tp={p['tp_partial_pct']:.0%}  "
              f"win={r['win_rate']:.1f}%  cum={r['cum_pnl']:+.2f}%  avg={r['avg_pnl']:+.2f}%")


if __name__ == '__main__':
    main()
