#!/usr/bin/env python3
"""Sweep ratios for 3-step ladder: +15% / +20% / +25%"""
import sys, statistics, random, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from backtest_btc5m import get_btc_klines, compute_5min_slots, norm_cdf


def sim(slot_candles, hard_stop_pct, tp_ratios, btc_5min_std=None, clob_spread=0.03):
    slot_start = slot_candles[0]['open_time'] // 1000
    slot_end = slot_start + 300
    if btc_5min_std is None: btc_5min_std = 70.0
    btc_start = slot_candles[0]['open']
    ns, gp, gm = clob_spread * 0.8, 0.15, clob_spread * 2.5

    for i, c in enumerate(slot_candles):
        ts = c['open_time'] // 1000
        seconds_left = slot_end - ts
        if seconds_left < 60 or seconds_left > 180: continue
        btc_now, btc_move = c['close'], c['close'] - btc_start
        if abs(btc_move) < 40: continue

        direction = 'UP' if btc_move > 0 else 'DOWN'
        rv = btc_5min_std * ((slot_end - ts) / 300.0) ** 0.5
        fp = norm_cdf(abs(btc_move) / rv) if rv > 0 else 0.5
        ask = min(0.999, max(0.50, fp) + clob_spread / 2.0)
        if ask < 0.70: continue

        entry = ask
        hard_stop = entry * (1.0 - hard_stop_pct)
        highest, trail_active, trail_stop = entry, False, hard_stop
        rem = 1.0
        levels = [(0.15, tp_ratios[0]), (0.20, tp_ratios[1]), (0.25, tp_ratios[2])]
        done = [False, False, False]
        recs, exit_price, exit_reason = [], None, None

        for j in range(i + 1, len(slot_candles)):
            fc = slot_candles[j]
            fr = max(10, slot_end - fc['open_time'] // 1000)
            em = fc['close'] - btc_start
            em = em if direction == 'UP' else -em
            fv = btc_5min_std * (fr / 300.0) ** 0.5
            mid = norm_cdf(em / fv) if fv > 0 else 0.5
            mid = max(0.001, min(0.999, mid + random.gauss(0, ns) + (random.gauss(0, gm) if random.random() < gp else 0)))
            bid = max(0.001, mid - clob_spread / 2.0)
            pnl = (bid - entry) / entry

            if not trail_active and pnl >= 0.05: trail_active = True; highest = bid; trail_stop = highest * 0.95
            if trail_active and bid > highest: highest = bid; trail_stop = highest * 0.95

            for li, (lv, lr) in enumerate(levels):
                if done[li]: continue
                if pnl >= lv:
                    sell = min(1.0 * lr, rem)
                    if sell > 0:
                        recs.append({'pnl': (bid * (1.0 - random.uniform(0.01, 0.06)) - entry) / entry * sell})
                        rem -= sell
                    done[li] = True

            if rem <= 0: exit_reason = 'ladder'; break

            if bid <= hard_stop:
                exit_price = hard_stop * (1.0 - random.uniform(0.02, 0.15) if random.random() < 0.30 else 0.02)
                exit_reason = 'hard'; break

            if trail_active and bid <= trail_stop:
                exit_price = trail_stop * (1.0 - random.uniform(0.02, 0.15) if random.random() < 0.30 else 0.02)
                exit_reason = 'trail'; break

            if slot_end - fc['open_time'] // 1000 <= 20: exit_price = bid; exit_reason = 'time'; break
            if j == len(slot_candles) - 1: exit_price = bid; exit_reason = 'end'

        if exit_price is None:
            won = (direction == 'UP' and slot_candles[-1]['close'] - btc_start > 0) or (direction == 'DOWN' and slot_candles[-1]['close'] - btc_start < 0)
            exit_price = 1.0 if won else 0.0; exit_reason = 'settle'

        total = sum(r['pnl'] for r in recs) + (exit_price - entry) / entry * rem
        return {'won': total > 0, 'pnl_pct': round(total * 100, 2), 'sells': len(recs), 'rem': round(rem, 4)}
    return None


def run(ladder_ratios, hs, candles):
    slots = compute_5min_slots(candles)
    rets = [sc[-1]['close'] - sc[0]['open'] for _, sc in sorted(slots.items()) if len(sc) >= 3]
    std = statistics.stdev(rets) if len(rets) >= 5 else 70.0
    pnls, wins, trades, sells = [], 0, 0, 0
    for _, sc in sorted(slots.items()):
        if len(sc) < 3: continue
        r = sim(sc, hs, ladder_ratios, std)
        if r: trades += 1; pnls.append(r['pnl_pct']); wins += r['won']; sells += r['sells']
    if trades < 3: return None
    return {'avg': statistics.mean(pnls), 'cum': sum(pnls), 'win': wins/trades*100, 'n': trades, 'sells': sells/trades}


def main():
    candles = get_btc_klines(500)
    print(f"Got {len(candles)} candles\n")

    ratios = []
    for a in [15, 20, 25, 30, 35, 40]:
        for b in [15, 20, 25, 30, 35, 40]:
            c = 100 - a - b
            if c >= 15 and c <= 50:
                ratios.append((a/100, b/100, c/100))

    print(f"Testing {len(ratios)} ratio combos for +15%/+20%/+25% ladder\n")
    print(f"{'#':<4} {'ratios':<20} {'hs=7% cum':>10} {'hs=7% win':>9} {'hs=10% cum':>10} {'hs=10% win':>9} {'sells':>7}")
    print("-" * 72)

    results = []
    for r1, r2, r3 in ratios:
        res7 = run((r1, r2, r3), 0.07, candles)
        res10 = run((r1, r2, r3), 0.10, candles)
        if res7 and res10:
            results.append({'r': (r1, r2, r3), 'r7': res7, 'r10': res10})

    results.sort(key=lambda x: x['r7']['cum'], reverse=True)

    for i, x in enumerate(results[:20]):
        r1, r2, r3 = x['r']
        r7, r10 = x['r7'], x['r10']
        s7 = f"cum={r7['cum']:+.1f}% win={r7['win']:.0f}%"
        s10 = f"cum={r10['cum']:+.1f}% win={r10['win']:.0f}%"
        print(f"{i+1:<4} +15%:{r1*100:.0f}% +20%:{r2*100:.0f}% +25%:{r3*100:.0f}%  {s7:>22}  {s10:>22}  s7={r7['sells']:.1f} s10={r10['sells']:.1f}")

    print(f"\n--- TOP BY hs=7% avg PnL ---")
    results.sort(key=lambda x: x['r7']['avg'], reverse=True)
    for i, x in enumerate(results[:10]):
        r1, r2, r3 = x['r']
        print(f"{i+1}. +15%:{r1*100:.0f}% +20%:{r2*100:.0f}% +25%:{r3*100:.0f}%  avg={x['r7']['avg']:+.2f}%  win={x['r7']['win']:.0f}%  cum={x['r7']['cum']:+.1f}%")

    print(f"\n--- TOP BY hs=7% win rate ---")
    results.sort(key=lambda x: x['r7']['win'], reverse=True)
    for i, x in enumerate(results[:10]):
        r1, r2, r3 = x['r']
        print(f"{i+1}. +15%:{r1*100:.0f}% +20%:{r2*100:.0f}% +25%:{r3*100:.0f}%  win={x['r7']['win']:.0f}%  cum={x['r7']['cum']:+.1f}%  avg={x['r7']['avg']:+.2f}%")


if __name__ == '__main__':
    main()
