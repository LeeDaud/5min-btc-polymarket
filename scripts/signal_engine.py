#!/usr/bin/env python3
"""
BTC signal quality engine for 5-min Polymarket momentum strategy.
Active gates: window delta tier + pulse detection (delta-pulse mode).
VLS-5M mode: VWAP + Squeeze Momentum + Liquidity Sweep.
"""
import time
import statistics
from dataclasses import dataclass
from typing import Optional

import requests

from indicators import (
    compute_anchored_vwap, compute_daily_vwap,
    compute_bollinger_bands, compute_keltner_channels,
    compute_squeeze_momentum, detect_liquidity_sweep,
    compute_atr as _indicator_atr, SqueezeState, SweepResult,
)


# ============================================================
# Data models
# ============================================================

@dataclass
class Candle1m:
    open_time: int      # unix ms
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


@dataclass
class DeltaTier:
    label: str
    score: int          # 0 / 3 / 5 / 7
    min_pct: float      # abs(delta%) threshold


DELTA_TIERS = [
    DeltaTier("SKIP",   0, 0.0),
    DeltaTier("WEAK",   3, 0.05),
    DeltaTier("STRONG", 5, 0.10),
    DeltaTier("MEGA",   7, 1.0),
]


@dataclass
class SignalResult:
    passed: bool
    window_delta_pct: float
    delta_tier: DeltaTier
    direction_aligned: bool
    micro_momentum_pass: bool
    atr_pass: bool
    pulse_pass: bool
    pulse_ratio: float
    confidence: float       # 0-100
    reason: str             # "" if passed, else failure reason


@dataclass
class VlsSignalResult:
    passed: bool
    direction: str              # "LONG" / "SHORT" / "NONE"
    vwap: Optional[float]
    price_above_vwap: bool
    vwap_aligned: bool          # price>VWAP for LONG, price<VWAP for SHORT
    sweep_detected: bool
    sweep_type: str             # "bullish" / "bearish" / ""
    sweep_level: Optional[float]
    squeeze_firing: bool        # squeeze just released (was squeezing, now not)
    squeeze_color: str          # bright_green / bright_red / dark_green / dark_red / etc
    squeeze_momentum: float
    entry_btc_price: Optional[float]
    stop_loss_btc: Optional[float]
    stop_loss_token_price: Optional[float]
    tp1_token_price: Optional[float]     # 1.5:1 R/R
    tp2_token_price: Optional[float]     # 3:1 R/R
    confidence: float           # 0-100
    reason: str                 # "" if passed, failure reason otherwise


# ============================================================
# BTC data feed (MEXC)
# ============================================================

class BtcDataFeed:
    TICKER_URL = "https://api.mexc.com/api/v3/ticker/price?symbol=BTCUSDT"
    KLINES_URL = "https://api.mexc.com/api/v3/klines?symbol=BTCUSDT&interval=1m&limit=12"

    def __init__(self):
        self._klines_cache: list[Candle1m] = []
        self._klines_ts: float = 0
        self._cache_ttl = 60.0
        self._consecutive_errors: int = 0
        self._max_consecutive_errors: int = 5
        self._last_error: Optional[str] = None

    def fetch_price(self) -> Optional[float]:
        try:
            r = requests.get(self.TICKER_URL, timeout=4)
            r.raise_for_status()
            self._consecutive_errors = 0
            self._last_error = None
            return float(r.json()["price"])
        except Exception as e:
            self._consecutive_errors += 1
            self._last_error = str(e)
            return None

    def fetch_klines(self, force: bool = False) -> list[Candle1m]:
        now = time.time()
        if not force and self._klines_cache and (now - self._klines_ts) < self._cache_ttl:
            return self._klines_cache
        try:
            r = requests.get(self.KLINES_URL, timeout=6)
            r.raise_for_status()
            rows = r.json()
            candles = []
            for row in rows:
                candles.append(Candle1m(
                    open_time=int(row[0]),
                    open=float(row[1]),
                    high=float(row[2]),
                    low=float(row[3]),
                    close=float(row[4]),
                    volume=float(row[5]),
                ))
            self._klines_cache = candles
            self._klines_ts = now
            self._consecutive_errors = 0
            self._last_error = None
            return candles
        except Exception as e:
            self._consecutive_errors += 1
            self._last_error = str(e)
            return self._klines_cache

    def fetch_klines_daily(self, force: bool = False) -> list[Candle1m]:
        """Fetch up to 500 1-min klines for VWAP and squeeze calculation (needs ~8h data)."""
        now = time.time()
        if not force and self._klines_cache and (now - self._klines_ts) < self._cache_ttl:
            return self._klines_cache
        try:
            r = requests.get(
                "https://api.mexc.com/api/v3/klines?symbol=BTCUSDT&interval=1m&limit=500",
                timeout=10,
            )
            r.raise_for_status()
            rows = r.json()
            candles = []
            for row in rows:
                candles.append(Candle1m(
                    open_time=int(row[0]),
                    open=float(row[1]),
                    high=float(row[2]),
                    low=float(row[3]),
                    close=float(row[4]),
                    volume=float(row[5]),
                ))
            self._klines_cache = candles
            self._klines_ts = now
            self._consecutive_errors = 0
            self._last_error = None
            return candles
        except Exception as e:
            self._consecutive_errors += 1
            self._last_error = str(e)
            return self._klines_cache

    def get_window_open(self, bucket_ts: int) -> Optional[float]:
        candles = self.fetch_klines()
        bucket_start_ms = bucket_ts * 1000
        for c in candles:
            if c.open_time >= bucket_start_ms:
                return c.open
            if c.open_time + 60000 >= bucket_start_ms:
                return c.close
        return candles[0].open if candles else None

    def is_healthy(self) -> bool:
        return self._consecutive_errors < self._max_consecutive_errors


# ============================================================
# Signal calculations (pure functions, independently testable)
# ============================================================

def compute_window_delta(current_price: float, window_open: float) -> float:
    if window_open <= 0:
        return 0.0
    return (current_price - window_open) / window_open * 100.0


def classify_delta(delta_pct: float) -> DeltaTier:
    abs_d = abs(delta_pct)
    result = DELTA_TIERS[0]
    for tier in DELTA_TIERS:
        if abs_d >= tier.min_pct:
            result = tier
    return result


def check_direction_alignment(delta_pct: float, side: str) -> bool:
    if side == "UP":
        return delta_pct > 0
    elif side == "DOWN":
        return delta_pct < 0
    return False


def check_micro_momentum(candles: list[Candle1m], side: str, n_candles: int = 2) -> bool:
    if len(candles) < n_candles:
        return False
    recent = candles[-n_candles:]
    for c in recent:
        if side == "UP" and c.close <= c.open:
            return False
        if side == "DOWN" and c.close >= c.open:
            return False
    return True


def check_pulse(candles: list[Candle1m], n_lookback: int = 3, threshold: float = 0.5) -> tuple[bool, float]:
    """
    Detect impulse (pulse) moves where the last candle dominates.
    Returns (is_pulse, concentration_ratio).
    is_pulse=True means the move is too concentrated — likely to reverse.
    """
    if len(candles) < n_lookback:
        return (False, 0.0)
    window = candles[-n_lookback:]
    total_move = sum(abs(c.open - c.close) for c in window)
    if total_move <= 0:
        return (False, 0.0)
    last_move = abs(window[-1].open - window[-1].close)
    ratio = last_move / total_move
    return (ratio > threshold, ratio)


def compute_atr(candles: list[Candle1m], period: int = 5) -> Optional[float]:
    if len(candles) < period + 1:
        return None
    tr_values = []
    for i in range(1, len(candles)):
        high = candles[i].high
        low = candles[i].low
        prev_close = candles[i - 1].close
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        tr_values.append(tr)
    tr_values = tr_values[-period:]
    return statistics.mean(tr_values) if tr_values else None


def check_atr_filter(candles: list[Candle1m], multiplier: float = 1.5,
                     period: int = 5) -> tuple[bool, Optional[float], Optional[float]]:
    atr = compute_atr(candles, period)
    if atr is None:
        return (False, None, None)
    latest = candles[-1]
    current_range = latest.high - latest.low
    return (current_range > atr * multiplier, current_range, atr)


def compute_confidence(delta_tier: DeltaTier, momentum_pass: bool,
                       late_window_boost: bool = False) -> float:
    raw = delta_tier.score
    if momentum_pass:
        raw += 2
    confidence = (raw / 9.0) * 100.0
    if late_window_boost:
        confidence *= 1.25
    return min(100.0, max(0.0, confidence))


# ============================================================
# Composite signal evaluation
# ============================================================

def evaluate_signal(
    current_price: float,
    window_open: float,
    candles: list[Candle1m],
    side: str,
    config: dict,
    seconds_left: float = 120,
) -> SignalResult:
    enable_delta = config.get("enable_window_delta", True)
    require_momentum = config.get("require_micro_momentum", False)
    atr_mult = float(config.get("atr_filter_multiplier", 0) or 0)
    min_confidence = float(config.get("min_confidence", 0) or 0)
    min_tier_label = str(config.get("window_delta_min_tier", "WEAK"))

    delta_pct = compute_window_delta(current_price, window_open)
    delta_tier = classify_delta(delta_pct)
    dir_aligned = check_direction_alignment(delta_pct, side)

    # Gate 1: window delta
    if enable_delta:
        min_tier = next((t for t in DELTA_TIERS if t.label == min_tier_label), DELTA_TIERS[1])
        if delta_tier.score < min_tier.score:
            return SignalResult(False, delta_pct, delta_tier, dir_aligned, False, True, True, 0.0,
                                0.0, f"delta_tier_{delta_tier.label}_below_{min_tier_label}")
        if not dir_aligned:
            return SignalResult(False, delta_pct, delta_tier, dir_aligned, False, True, True, 0.0,
                                0.0, "direction_misaligned")

    # Gate 2: micro momentum
    mom_pass = True
    if require_momentum:
        mom_pass = check_micro_momentum(candles, side)

    # Gate 3: ATR volatility
    atr_pass = True
    if atr_mult > 0:
        skip, _, _ = check_atr_filter(candles, atr_mult)
        if skip:
            atr_pass = False

    # Gate 4: pulse detection (last candle dominates = likely reversal)
    pulse_pass = True
    pulse_ratio = 0.0
    if config.get("enable_pulse_filter", True):
        is_pulse, pulse_ratio = check_pulse(candles)
        if is_pulse:
            pulse_pass = False

    # Confidence score
    late_boost = (seconds_left < 60)
    confidence = compute_confidence(delta_tier, mom_pass, late_boost)

    if confidence < min_confidence:
        return SignalResult(False, delta_pct, delta_tier, dir_aligned, mom_pass, atr_pass, pulse_pass, pulse_ratio,
                            confidence, f"confidence_{confidence:.0f}_below_{min_confidence}")
    if not mom_pass:
        return SignalResult(False, delta_pct, delta_tier, dir_aligned, mom_pass, atr_pass, pulse_pass, pulse_ratio,
                            confidence, "micro_momentum_fail")
    if not atr_pass:
        return SignalResult(False, delta_pct, delta_tier, dir_aligned, mom_pass, atr_pass, pulse_pass, pulse_ratio,
                            confidence, "atr_high_volatility")
    if not pulse_pass:
        return SignalResult(False, delta_pct, delta_tier, dir_aligned, mom_pass, atr_pass, pulse_pass, pulse_ratio,
                            confidence, f"pulse_detected_{pulse_ratio:.0%}")

    return SignalResult(True, delta_pct, delta_tier, dir_aligned, mom_pass, atr_pass, pulse_pass, pulse_ratio,
                        confidence, "")


# ============================================================
# VLS-5M signal evaluation (VWAP + Liquidity Sweep + Squeeze)
# ============================================================

def evaluate_vls_signal(
    candles: list,
    current_price: float,
    config: dict,
    side_preference: str = "",
    entry_token_price: Optional[float] = None,
) -> VlsSignalResult:
    """
    VLS-5M three-gate signal: VWAP trend → Liquidity Sweep → Squeeze Momentum.

    Uses candle dicts (open/high/low/close/volume/open_time) from MEXC klines.
    Returns VlsSignalResult with direction, stop/tp levels, and confidence.
    """
    if len(candles) < 30:
        return VlsSignalResult(False, "NONE", None, False, False, False, "", None,
                               False, "", 0.0, None, None, None, None, None, 0.0,
                               "insufficient_data")

    n = len(candles)
    recent = candles[-1]
    prev = candles[-2] if n >= 2 else recent

    # ---- VWAP (daily anchor) ----
    vwap = compute_daily_vwap(candles)
    if vwap is None:
        return VlsSignalResult(False, "NONE", None, False, False, False, "", None,
                               False, "", 0.0, None, None, None, None, None, 0.0,
                               "vwap_unavailable")
    price_above_vwap = current_price > vwap

    # ---- Liquidity Sweep ----
    sweep_lookback_min = int(config.get("sweep_lookback_min", 15))
    sweep_lookback_max = int(config.get("sweep_lookback_max", 30))
    sweep = detect_liquidity_sweep(candles, sweep_lookback_min, sweep_lookback_max)

    # ---- Squeeze Momentum ----
    squeeze_period = int(config.get("squeeze_period", 20))
    squeeze = compute_squeeze_momentum(candles, squeeze_period)
    if squeeze is None:
        return VlsSignalResult(False, "NONE", vwap, price_above_vwap, False, False, "", None,
                               False, "", 0.0, None, None, None, None, None, 0.0,
                               "squeeze_unavailable")

    # ---- Gate 1: VWAP alignment ----
    # LONG requires price > VWAP, SHORT requires price < VWAP
    long_vwap_ok = price_above_vwap
    short_vwap_ok = not price_above_vwap

    # ---- Gate 2: Squeeze must be firing (released from squeeze) ----
    squeeze_firing = squeeze.just_fired
    if not squeeze_firing:
        return VlsSignalResult(False, "NONE", vwap, price_above_vwap, False,
                               sweep.bullish_sweep or sweep.bearish_sweep,
                               "bullish" if sweep.bullish_sweep else ("bearish" if sweep.bearish_sweep else ""),
                               sweep.sweep_level if sweep.sweep_level else None,
                               False, squeeze.color, squeeze.momentum_val,
                               current_price, None, None, None, None,
                               0.0, "squeeze_not_fired")

    # ---- Gate 3: Sweep must match direction ----
    # Bullish sweep (fake breakdown) → LONG, Bearish sweep (fake breakout) → SHORT
    long_sweep_ok = sweep.bullish_sweep
    short_sweep_ok = sweep.bearish_sweep

    # ---- Gate 4: Momentum color (after squeeze release) ----
    # bright_green = positive + rising (LONG), bright_red = negative + falling (SHORT)
    long_color_ok = squeeze.color == "bright_green"
    short_color_ok = squeeze.color == "bright_red"

    # ---- Determine direction ----
    # If side_preference is given, only check that direction
    if side_preference == "UP":
        short_vwap_ok = False; short_sweep_ok = False; short_color_ok = False
    elif side_preference == "DOWN":
        long_vwap_ok = False; long_sweep_ok = False; long_color_ok = False

    long_all = long_vwap_ok and long_sweep_ok and long_color_ok
    short_all = short_vwap_ok and short_sweep_ok and short_color_ok

    if not long_all and not short_all:
        # Build specific rejection reason
        reasons = []
        direction = "NONE"
        # Determine intended direction based on sweep
        if sweep.bullish_sweep:
            direction = "LONG"
            if not long_vwap_ok:
                reasons.append("vwap_not_above")
            if not long_color_ok:
                reasons.append(f"wrong_color_{squeeze.color}")
        elif sweep.bearish_sweep:
            direction = "SHORT"
            if not short_vwap_ok:
                reasons.append("vwap_not_below")
            if not short_color_ok:
                reasons.append(f"wrong_color_{squeeze.color}")
        elif price_above_vwap:
            direction = "LONG"
            reasons.append("no_sweep")
        else:
            direction = "SHORT"
            reasons.append("no_sweep")

        return VlsSignalResult(False, direction, vwap, price_above_vwap, False,
                               sweep.bullish_sweep or sweep.bearish_sweep,
                               "bullish" if sweep.bullish_sweep else ("bearish" if sweep.bearish_sweep else ""),
                               sweep.sweep_level if sweep.sweep_level else None,
                               squeeze_firing, squeeze.color, squeeze.momentum_val,
                               current_price, None, None, None, None,
                               0.0, "|".join(reasons) if reasons else "signal_not_met")

    # ---- Determine entry direction ----
    if long_all:
        direction = "LONG"
        signal_side = "UP"
        sweep_level = sweep.sweep_level
    else:
        direction = "SHORT"
        signal_side = "DOWN"
        sweep_level = sweep.sweep_level

    # ---- Calculate stop loss and take profit ----
    stop_buffer_pct = float(config.get("stop_buffer_pct", 0.2)) / 100.0
    tp1_rr = float(config.get("tp1_rr_ratio", 1.5))
    tp2_rr = float(config.get("tp2_rr_ratio", 3.0))

    if direction == "LONG":
        # Stop loss at sweep low minus buffer%
        stop_btc = sweep_level * (1.0 - stop_buffer_pct)
        # Distance from entry to stop (in BTC $)
        stop_distance = current_price - stop_btc
    else:
        stop_btc = sweep_level * (1.0 + stop_buffer_pct)
        stop_distance = stop_btc - current_price

    if stop_distance <= 0:
        stop_distance = 50.0  # fallback: $50 BTC move

    # Convert BTC distance to token price distance
    btc_to_token_ratio = float(config.get("btc_to_token_move_ratio", 0.0002))
    token_stop_distance = stop_distance * btc_to_token_ratio

    if entry_token_price is None or entry_token_price <= 0:
        entry_token_price = 0.55  # reasonable default

    if direction == "LONG":
        stop_token = entry_token_price - token_stop_distance
        tp1_token = entry_token_price + token_stop_distance * tp1_rr
        tp2_token = entry_token_price + token_stop_distance * tp2_rr
    else:
        stop_token = entry_token_price + token_stop_distance
        tp1_token = entry_token_price - token_stop_distance * tp1_rr
        tp2_token = entry_token_price - token_stop_distance * tp2_rr

    stop_token = max(0.02, min(0.98, stop_token))
    tp1_token = max(0.02, min(0.98, tp1_token))
    tp2_token = max(0.02, min(0.98, tp2_token))

    # ---- Confidence score ----
    # VWAP distance (how far price is from VWAP, normalized)
    vwap_distance_pct = abs(current_price - vwap) / vwap * 100.0
    vwap_score = min(3.0, vwap_distance_pct / 0.1)  # 0.1% away = 1 point, cap at 3

    # Sweep quality (larger wick relative to body = clearer sweep)
    sweep_score = 3.0 if (sweep.bullish_sweep or sweep.bearish_sweep) else 0.0

    # Squeeze momentum strength
    mom_abs = abs(squeeze.momentum_val)
    mom_score = min(3.0, mom_abs / 1.0)  # normalize, cap at 3

    confidence = (vwap_score + sweep_score + mom_score) / 9.0 * 100.0
    confidence = min(100.0, max(0.0, confidence))

    # Late window boost
    late_boost = config.get("late_window_boost", False)
    if late_boost:
        confidence *= 1.25
        confidence = min(100.0, confidence)

    return VlsSignalResult(
        passed=True,
        direction=direction,
        vwap=vwap,
        price_above_vwap=price_above_vwap,
        vwap_aligned=(direction == "LONG" and price_above_vwap) or (direction == "SHORT" and not price_above_vwap),
        sweep_detected=True,
        sweep_type="bullish" if sweep.bullish_sweep else "bearish",
        sweep_level=sweep_level,
        squeeze_firing=True,
        squeeze_color=squeeze.color,
        squeeze_momentum=squeeze.momentum_val,
        entry_btc_price=current_price,
        stop_loss_btc=stop_btc,
        stop_loss_token_price=round(stop_token, 4),
        tp1_token_price=round(tp1_token, 4),
        tp2_token_price=round(tp2_token, 4),
        confidence=round(confidence, 1),
        reason="",
    )


# ============================================================
# Dynamic position sizing
# ============================================================

def compute_position_size(confidence: float, max_position: float,
                          base_frac: float = 0.05, conf_mult: float = 0.10) -> float:
    capped_conf = min(confidence / 100.0, 0.45)
    fraction = base_frac + capped_conf * conf_mult
    return min(max_position, max(1.0, max_position * fraction))
