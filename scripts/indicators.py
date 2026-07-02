#!/usr/bin/env python3
"""
Pure technical indicators for BTC 5-min Polymarket strategies.
Zero I/O, zero state. Works with candle dicts (open/high/low/close/volume/open_time).
"""
import math
import statistics
from dataclasses import dataclass
from typing import Optional


# ============================================================
# Data models
# ============================================================

@dataclass
class SqueezeState:
    is_squeezing: bool       # BB inside KC
    just_fired: bool         # was squeezing on prev bar, not now
    momentum_val: float      # linear regression oscillator value
    prev_momentum_val: float
    color: str               # bright_green / dark_green / bright_red / dark_red / black
    bb_upper: float
    bb_lower: float
    kc_upper: float
    kc_lower: float


@dataclass
class SweepResult:
    bullish_sweep: bool      # fake breakdown — price broke below low, recovered
    bearish_sweep: bool      # fake breakout — price broke above high, reversed
    sweep_level: float       # the low (bullish) or high (bearish) that was swept
    sweep_candle_idx: int    # which candle (index from end, negative) triggered the sweep


# ============================================================
# ATR
# ============================================================

def compute_atr(candles: list, period: int = 5) -> Optional[float]:
    """Average True Range over `period` bars. Candles can be dicts or objects."""
    if len(candles) < period + 1:
        return None
    tr_values = []
    for i in range(1, len(candles)):
        c, cp = candles[i], candles[i - 1]
        h = _v(c, 'high'); l = _v(c, 'low'); pc = _v(cp, 'close')
        tr = max(h - l, abs(h - pc), abs(l - pc))
        tr_values.append(tr)
    tr_values = tr_values[-period:]
    return statistics.mean(tr_values) if tr_values else None


# ============================================================
# Anchored VWAP
# ============================================================

def compute_anchored_vwap(candles: list, anchor_ts_ms: int = 0) -> Optional[float]:
    """
    Cumulative VWAP from `anchor_ts_ms` to the last candle.
    typical_price = (high + low + close) / 3.
    anchor_ts_ms = 0 means use all available candles.
    """
    if not candles:
        return None
    sum_pv = 0.0
    sum_vol = 0.0
    for c in candles:
        if anchor_ts_ms > 0 and c.get('open_time', 0) < anchor_ts_ms:
            continue
        h = _v(c, 'high'); l = _v(c, 'low'); cl = _v(c, 'close')
        vol = _v(c, 'volume')
        typical = (h + l + cl) / 3.0
        if vol > 0:
            sum_pv += typical * vol
            sum_vol += vol
    return sum_pv / sum_vol if sum_vol > 0 else None


def compute_daily_vwap(candles: list) -> Optional[float]:
    """VWAP anchored to UTC 00:00 of the first candle's day."""
    if not candles:
        return None
    first_ts = candles[0].get('open_time', 0)
    if first_ts <= 0:
        return compute_anchored_vwap(candles)
    # Floor to UTC midnight (open_time is in ms)
    midnight_ms = first_ts - (first_ts % (24 * 3600 * 1000))
    return compute_anchored_vwap(candles, midnight_ms)


# ============================================================
# Bollinger Bands
# ============================================================

def _sma(values: list, period: int) -> float:
    return statistics.mean(values[-period:]) if len(values) >= period else statistics.mean(values)


def _stdev(values: list, period: int) -> float:
    return statistics.stdev(values[-period:]) if len(values) >= period else 0.0


def compute_bollinger_bands(closes: list, period: int = 20, num_std: float = 2.0):
    """Returns (middle, upper, lower) for most recent bar."""
    if len(closes) < period:
        return None, None, None
    mid = _sma(closes, period)
    std = _stdev(closes, period)
    return mid, mid + num_std * std, mid - num_std * std


# ============================================================
# Keltner Channels
# ============================================================

def compute_keltner_channels(candles: list, period: int = 20, atr_mult: float = 1.5):
    """
    Keltner Channels: middle = SMA(close, period), bands = middle ± ATR(period) * atr_mult.
    Returns (middle, upper, lower) for most recent bar.
    """
    if len(candles) < period + 1:
        return None, None, None
    closes = [_v(c, 'close') for c in candles]
    mid = _sma(closes, period)
    atr = compute_atr(candles, period)
    if atr is None:
        return None, None, None
    return mid, mid + atr_mult * atr, mid - atr_mult * atr


# ============================================================
# Squeeze Momentum (LazyBear)
# ============================================================

def _linreg(values: list, period: int) -> Optional[float]:
    """Linear regression of `values[-period:]` — returns the endpoint value."""
    if len(values) < period:
        return None
    y = values[-period:]
    n_val = float(period)
    sum_x = 0.0
    sum_y = 0.0
    sum_xy = 0.0
    sum_x2 = 0.0
    for i, yi in enumerate(y):
        xi = float(i)
        sum_x += xi
        sum_y += yi
        sum_xy += xi * yi
        sum_x2 += xi * xi
    denom = n_val * sum_x2 - sum_x * sum_x
    if denom == 0:
        return y[-1]
    b = (n_val * sum_xy - sum_x * sum_y) / denom
    a = (sum_y - b * sum_x) / n_val
    return a + b * (period - 1)


def compute_squeeze_momentum(candles: list, period: int = 20,
                              bb_std: float = 2.0, kc_mult: float = 1.5) -> Optional[SqueezeState]:
    """
    LazyBear Squeeze Momentum Indicator.

    Squeeze condition: BB bands are inside KC bands (volatility contraction).
    Momentum: linear regression of (close - avg(highest_high, lowest_low, sma_close)).

    Returns SqueezeState for the most recent bar, or None if insufficient data.
    """
    if len(candles) < period + 1:
        return None

    closes = [_v(c, 'close') for c in candles]
    highs = [_v(c, 'high') for c in candles]
    lows = [_v(c, 'low') for c in candles]

    # Bollinger Bands (current and previous)
    bb_mid, bb_up, bb_lo = compute_bollinger_bands(closes, period, bb_std)
    if bb_mid is None:
        return None

    # For "previous" squeeze state: compute BB/KC on candles[:-1]
    prev_closes = closes[:-1]
    prev_candles = candles[:-1]
    _, prev_bb_up, prev_bb_lo = compute_bollinger_bands(prev_closes, period, bb_std)
    _, prev_kc_up, prev_kc_lo = compute_keltner_channels(prev_candles, period, kc_mult)

    # Keltner Channels (current)
    kc_mid, kc_up, kc_lo = compute_keltner_channels(candles, period, kc_mult)
    if kc_mid is None:
        return None

    # Squeeze detection: BB inside KC
    bb_width = bb_up - bb_lo
    kc_width = kc_up - kc_lo
    is_squeezing = bb_width < kc_width

    # Previous squeeze state
    prev_squeezing = True
    if prev_bb_up is not None and prev_bb_lo is not None and prev_kc_up is not None and prev_kc_lo is not None:
        prev_bb_w = prev_bb_up - prev_bb_lo
        prev_kc_w = prev_kc_up - prev_kc_lo
        prev_squeezing = prev_bb_w < prev_kc_w
    just_fired = prev_squeezing and not is_squeezing

    # Momentum oscillator (LazyBear method)
    # avgHL = avg(highest(high, period), lowest(low, period))
    # avgClose = sma(close, period)
    # avgBoth = avg(avgHL, avgClose)
    # diff = close - avgBoth
    # val = linreg(diff, period)
    highest_high = max(highs[-period:])
    lowest_low = min(lows[-period:])
    avg_hl = (highest_high + lowest_low) / 2.0
    avg_close = _sma(closes, period)
    avg_both = (avg_hl + avg_close) / 2.0

    diff_series = [closes[i] - avg_both for i in range(len(closes))]
    # Recompute avg_both for each bar for a proper diff series
    diff_series = []
    for i in range(len(closes)):
        if i >= period - 1:
            hh = max(highs[i - period + 1:i + 1])
            ll = min(lows[i - period + 1:i + 1])
            ahl = (hh + ll) / 2.0
            ac = _sma(closes[:i + 1], period)
            ab = (ahl + ac) / 2.0
            diff_series.append(closes[i] - ab)
        else:
            diff_series.append(0.0)

    mom_val = _linreg(diff_series, period)
    if mom_val is None:
        mom_val = 0.0

    # Previous momentum (for color direction)
    prev_diff = diff_series[:-1] if len(diff_series) > period else diff_series
    prev_mom = _linreg(prev_diff, period) if len(prev_diff) >= period else mom_val
    if prev_mom is None:
        prev_mom = mom_val

    # Color classification
    color = _squeeze_color(is_squeezing, mom_val, prev_mom)

    return SqueezeState(
        is_squeezing=is_squeezing,
        just_fired=just_fired,
        momentum_val=mom_val,
        prev_momentum_val=prev_mom,
        color=color,
        bb_upper=bb_up,
        bb_lower=bb_lo,
        kc_upper=kc_up,
        kc_lower=kc_lo,
    )


def _squeeze_color(squeezing: bool, mom: float, prev_mom: float) -> str:
    """LazyBear color logic for squeeze momentum histogram."""
    rising = mom > prev_mom
    positive = mom > 0

    if squeezing:
        # During squeeze: muted colors but technically the histogram still has direction
        if positive and rising:
            return "aqua"
        elif positive and not rising:
            return "blue"
        elif not positive and not rising:
            return "red"
        else:
            return "yellow"
    else:
        # Squeeze released: full colors
        if positive and rising:
            return "bright_green"
        elif positive and not rising:
            return "dark_green"
        elif not positive and not rising:
            return "bright_red"
        else:
            return "dark_red"


# ============================================================
# Liquidity Sweep Detection
# ============================================================

def detect_liquidity_sweep(candles: list,
                            lookback_min: int = 15,
                            lookback_max: int = 30) -> SweepResult:
    """
    Detect fake breakdown (bullish sweep) or fake breakout (bearish sweep).

    Bullish sweep: price briefly broke below the trailing N-minute low,
                   then the most recent candle closed above that low (recovery).
    Bearish sweep: price broke above the trailing N-minute high,
                   then closed below that high (reversal).

    Checks the last 3 candles for sweep events against the lookback range low/high.
    Returns SweepResult (all False if no sweep detected).
    """
    n = len(candles)
    if n < lookback_min:
        return SweepResult(False, False, 0.0, 0)

    # Check last 3 candles for sweep events
    for check_offset in range(1, min(3, n - lookback_min) + 1):
        sweep_candles = candles[:n - check_offset + 1] if check_offset > 0 else candles
        recent = candles[-check_offset] if check_offset > 0 else candles[-1]

        lo_start = max(0, len(sweep_candles) - lookback_max - 1)
        lo_end = len(sweep_candles) - check_offset - 1
        if lo_end <= lo_start:
            lo_end = lo_start + 1

        prior = sweep_candles[lo_start:lo_end]
        if not prior:
            continue

        prior_low = min(_v(c, 'low') for c in prior)
        prior_high = max(_v(c, 'high') for c in prior)

        cur_low = _v(recent, 'low')
        cur_high = _v(recent, 'high')
        cur_close = _v(recent, 'close')

        # Bullish sweep: broke below prior low, closed back above
        if cur_low < prior_low and cur_close > prior_low:
            return SweepResult(
                bullish_sweep=True, bearish_sweep=False,
                sweep_level=prior_low,
                sweep_candle_idx=-check_offset,
            )

        # Bearish sweep: broke above prior high, closed back below
        if cur_high > prior_high and cur_close < prior_high:
            return SweepResult(
                bullish_sweep=False, bearish_sweep=True,
                sweep_level=prior_high,
                sweep_candle_idx=-check_offset,
            )

    return SweepResult(False, False, 0.0, 0)


# ============================================================
# Utility
# ============================================================

def _v(obj, key: str) -> float:
    """Get value from dict or object attribute."""
    if isinstance(obj, dict):
        return float(obj.get(key, 0) or 0)
    return float(getattr(obj, key, 0) or 0)
