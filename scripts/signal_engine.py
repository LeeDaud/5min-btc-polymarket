#!/usr/bin/env python3
"""
BTC signal quality engine for 5-min Polymarket momentum strategy.
Provides: Binance data feed, window delta, micro-momentum, ATR filter,
composite confidence scoring, and dynamic position sizing.

All signal gates default to disabled for backward compatibility.
"""

import time
import statistics
from dataclasses import dataclass, field
from typing import Optional

import requests


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
    confidence: float       # 0-100
    reason: str             # "" if passed, else failure reason


# ============================================================
# Multi-source BTC data feed (Binance + MEXC fallback)
# ============================================================

class BtcDataFeed:
    # Primary: Binance (may be geo-blocked in some regions)
    BINANCE_TICKER = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
    BINANCE_KLINES = "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1m&limit=12"
    # Fallback: MEXC (accessible from mainland China)
    MEXC_TICKER = "https://api.mexc.com/api/v3/ticker/price?symbol=BTCUSDT"
    MEXC_KLINES = "https://api.mexc.com/api/v3/klines?symbol=BTCUSDT&interval=1m&limit=12"

    def __init__(self):
        self._klines_cache: list[Candle1m] = []
        self._klines_ts: float = 0
        self._cache_ttl = 60.0
        self._consecutive_errors: int = 0
        self._max_consecutive_errors: int = 5
        self._last_error: Optional[str] = None
        self._using_fallback: bool = False

    def _raw_ticker(self, url: str, timeout: float) -> Optional[float]:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return float(r.json()["price"])

    def _raw_klines(self, url: str, timeout: float) -> list[Candle1m]:
        r = requests.get(url, timeout=timeout)
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
        return candles

    def fetch_price(self) -> Optional[float]:
        for label, url in [("binance", self.BINANCE_TICKER), ("mexc", self.MEXC_TICKER)]:
            try:
                price = self._raw_ticker(url, timeout=4)
                self._consecutive_errors = 0
                self._last_error = None
                if label == "mexc" and not self._using_fallback:
                    self._using_fallback = True
                return price
            except Exception as e:
                if label == "binance":
                    continue  # try fallback immediately
                self._consecutive_errors += 1
                self._last_error = str(e)
                return None

    def fetch_klines(self, force: bool = False) -> list[Candle1m]:
        now = time.time()
        if not force and self._klines_cache and (now - self._klines_ts) < self._cache_ttl:
            return self._klines_cache
        for label, url in [("binance", self.BINANCE_KLINES), ("mexc", self.MEXC_KLINES)]:
            try:
                candles = self._raw_klines(url, timeout=6)
                self._klines_cache = candles
                self._klines_ts = now
                self._consecutive_errors = 0
                self._last_error = None
                if label == "mexc" and not self._using_fallback:
                    self._using_fallback = True
                return candles
            except Exception as e:
                if label == "binance":
                    continue
                self._consecutive_errors += 1
                self._last_error = str(e)
        return self._klines_cache  # return stale cache as last resort

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

def bucket_5m_eng(ts: int) -> int:
    return ts - (ts % 300)


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
            return SignalResult(False, delta_pct, delta_tier, dir_aligned, False, True,
                                0.0, f"delta_tier_{delta_tier.label}_below_{min_tier_label}")
        if not dir_aligned:
            return SignalResult(False, delta_pct, delta_tier, dir_aligned, False, True,
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

    # Confidence score
    late_boost = (seconds_left < 60)
    confidence = compute_confidence(delta_tier, mom_pass, late_boost)

    if confidence < min_confidence:
        return SignalResult(False, delta_pct, delta_tier, dir_aligned, mom_pass, atr_pass,
                            confidence, f"confidence_{confidence:.0f}_below_{min_confidence}")
    if not mom_pass:
        return SignalResult(False, delta_pct, delta_tier, dir_aligned, mom_pass, atr_pass,
                            confidence, "micro_momentum_fail")
    if not atr_pass:
        return SignalResult(False, delta_pct, delta_tier, dir_aligned, mom_pass, atr_pass,
                            confidence, "atr_high_volatility")

    return SignalResult(True, delta_pct, delta_tier, dir_aligned, mom_pass, atr_pass,
                        confidence, "")


# ============================================================
# Dynamic position sizing
# ============================================================

def compute_position_size(confidence: float, max_position: float,
                          base_frac: float = 0.05, conf_mult: float = 0.10) -> float:
    capped_conf = min(confidence / 100.0, 0.45)
    fraction = base_frac + capped_conf * conf_mult
    return min(max_position, max(1.0, max_position * fraction))
