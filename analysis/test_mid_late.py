import sys, statistics, math
sys.path.insert(0, 'analysis')
from backtest_btc5m import get_btc_klines, compute_5min_slots, norm_cdf

candles = get_btc_klines(500)
slots = compute_5min_slots(candles)

results = []
for bucket, sc in sorted(slots.items()):
    if len(sc) < 4: continue
    btc_start = sc[0]['open']
    btc_end = sc[-1]['close']
    for c in sc:
        ts = c['open_time'] // 1000
        sec_left = bucket + 300 - ts
        if sec_left < 55 or sec_left > 95: continue
        btc_now = c['close']
        btc_move = btc_now - btc_start
        remaining_vol = 70.0 * (sec_left / 300.0) ** 0.5
        z = abs(btc_move) / remaining_vol if remaining_vol > 0 else 0
        fair_prob = norm_cdf(z)
        # 50/50: market is flat
        if 0.48 <= fair_prob <= 0.55 and abs(btc_move) < 60:
            rest = btc_end - btc_now
            results.append({'rest': rest, 'sec_left': sec_left})
        break

if not results:
    print('No 50/50 signals')
else:
    up = sum(1 for r in results if r['rest'] > 20)
    dn = sum(1 for r in results if r['rest'] < -20)
    flat = sum(1 for r in results if abs(r['rest']) < 30)
    big = sum(1 for r in results if abs(r['rest']) > 80)
    print(f'50/50 at T-60~90s: {len(results)} signals')
    print(f'  UP >$20: {up} ({up/len(results)*100:.0f}%)')
    print(f'  DOWN <-$20: {dn} ({dn/len(results)*100:.0f}%)')
    print(f'  Flat: {flat} ({flat/len(results)*100:.0f}%)')
    print(f'  Big >$80: {big} ({big/len(results)*100:.0f}%)')
    print(f'  Avg |rest|: ${statistics.mean([abs(r["rest"]) for r in results]):.0f}')

# Directional vs flat comparison
print()
for label, min_move in [('Directional >$40', 40), ('Flat <$30', 999)]:
    r2 = []
    for bucket, sc in sorted(slots.items()):
        if len(sc) < 4: continue
        btc_start = sc[0]['open']
        btc_end = sc[-1]['close']
        for c in sc:
            ts = c['open_time'] // 1000
            sec_left = bucket + 300 - ts
            if sec_left < 55 or sec_left > 95: continue
            btc_now = c['close']
            btc_move = btc_now - btc_start
            if label.startswith('Directional') and abs(btc_move) < min_move: continue
            if label.startswith('Flat') and abs(btc_move) >= 30: continue
            d = 'UP' if btc_move > 0 else 'DOWN'
            rest = btc_end - btc_now
            won = (d == 'UP' and rest > 0) or (d == 'DOWN' and rest < 0)
            r2.append({'won': won})
            break
    if r2:
        w = sum(1 for r in r2 if r['won'])
        print(f'{label}: {len(r2)} signals, win={w/len(r2)*100:.0f}%')
