#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from typing import Any, Optional
from pathlib import Path

import requests

import logging
logging.getLogger('py_clob_client_v2').setLevel(logging.CRITICAL)

from py_clob_client_v2.client import ClobClient
from py_clob_client_v2.constants import POLYGON
from py_clob_client_v2.clob_types import ApiCreds

sys.path.insert(0, str(Path(__file__).resolve().parent))
from signal_engine import BtcDataFeed, evaluate_signal, compute_position_size, SignalResult

UTC = dt.timezone.utc
CST = dt.timezone(dt.timedelta(hours=8))  # Shanghai


def now_utc() -> dt.datetime:
    return dt.datetime.now(UTC)


def ts_utc() -> str:
    return now_utc().isoformat().replace('+00:00', 'Z')


def ts_local() -> str:
    return dt.datetime.now(CST).strftime('%H:%M:%S')


def parse_json_objects(text: str) -> list[dict[str, Any]]:
    out = []
    cur = []
    depth = 0
    for ch in text:
        if ch == '{':
            depth += 1
        if depth > 0:
            cur.append(ch)
        if ch == '}' and depth > 0:
            depth -= 1
            if depth == 0:
                s = ''.join(cur)
                cur = []
                try:
                    out.append(json.loads(s))
                except Exception:
                    pass
    return out


def bucket_5m(ts: int) -> int:
    return ts - (ts % 300)


def fetch_event(slug: str) -> Optional[dict[str, Any]]:
    for attempt in range(3):
        try:
            r = requests.get('https://gamma-api.polymarket.com/events', params={'slug': slug}, timeout=8)
            r.raise_for_status()
            arr = r.json()
            return arr[0] if arr else None
        except Exception as e:
            if attempt < 2:
                time.sleep(1)
            else:
                raise


def resolve_active_current_5m_market() -> Optional[dict[str, Any]]:
    """Return active BTC 5m market for the current slot only."""
    now = int(time.time())
    cur = bucket_5m(now)
    slug = f'btc-updown-5m-{cur}'

    try:
        ev = fetch_event(slug)
    except Exception:
        return None
    if not ev:
        return None

    mkts = ev.get('markets') or []
    if not mkts:
        return None

    m = mkts[0]
    if m.get('closed') is True:
        return None
    if m.get('active') is False:
        return None

    end_iso = str(m.get('endDate') or m.get('endDateIso') or '')
    try:
        end_ts = dt.datetime.fromisoformat(end_iso.replace('Z', '+00:00')).timestamp()
    except Exception:
        return None

    sec_left = end_ts - time.time()
    if sec_left <= 5:
        return None

    mm = dict(m)
    mm['_event_slug'] = slug
    mm['_seconds_left'] = sec_left
    return mm


def parse_json_field(v):
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            return v
    return v


def market_side_prices(market: dict[str, Any]) -> tuple[float, float, str, str, str, str]:
    outcomes = parse_json_field(market.get('outcomes')) or []
    prices = parse_json_field(market.get('outcomePrices')) or []
    token_ids = parse_json_field(market.get('clobTokenIds')) or []
    if len(prices) < 2 or len(token_ids) < 2:
        raise RuntimeError('missing outcomePrices/clobTokenIds')

    up_i, down_i = 0, 1
    labs = [str(x).lower() for x in outcomes[:2]] if isinstance(outcomes, list) else []
    if len(labs) >= 2 and ('up' in labs[1] or 'yes' in labs[1]):
        up_i, down_i = 1, 0

    up_p = float(prices[up_i])
    dn_p = float(prices[down_i])
    up_t = str(token_ids[up_i])
    dn_t = str(token_ids[down_i])
    return up_p, dn_p, up_t, dn_t, str(market.get('slug') or market.get('_event_slug') or ''), str(market.get('endDate') or market.get('endDateIso') or '')


def _best_bid_ask(book) -> tuple[Optional[float], Optional[float]]:
    # V2 returns dict; V1 returns object. Handle both.
    if isinstance(book, dict):
        bids = book.get('bids') or []
        asks = book.get('asks') or []
    else:
        bids = getattr(book, 'bids', []) or []
        asks = getattr(book, 'asks', []) or []
    best_bid = None
    best_ask = None
    for b in bids:
        p = float(b.get('price', 0) if isinstance(b, dict) else (getattr(b, 'price', 0) or 0))
        if best_bid is None or p > best_bid:
            best_bid = p
    for a in asks:
        p = float(a.get('price', 0) if isinstance(a, dict) else (getattr(a, 'price', 0) or 0))
        if best_ask is None or p < best_ask:
            best_ask = p
    return best_bid, best_ask


def clob_side_prices(up_token: str, down_token: str, clob_base: str = 'https://clob.polymarket.com') -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Return trigger prices from CLOB orderbooks: UP ask, DOWN ask, spread of picked side when available."""
    pub = ClobClient(host=clob_base, chain_id=POLYGON)
    up_book = pub.get_order_book(str(up_token))
    dn_book = pub.get_order_book(str(down_token))
    up_bid, up_ask = _best_bid_ask(up_book)
    dn_bid, dn_ask = _best_bid_ask(dn_book)

    picked_spread = None
    # Side picked later by max ask; keep a generic sanity spread estimate
    if up_ask is not None and up_bid is not None:
        picked_spread = max(0.0, up_ask - up_bid)
    if dn_ask is not None and dn_bid is not None:
        s = max(0.0, dn_ask - dn_bid)
        picked_spread = s if picked_spread is None else min(picked_spread, s)

    return up_ask, dn_ask, picked_spread


def clob_best_bid(token_id: str, clob_base: str = 'https://clob.polymarket.com') -> Optional[float]:
    pub = ClobClient(host=clob_base, chain_id=POLYGON)
    book = pub.get_order_book(str(token_id))
    best_bid, _ = _best_bid_ask(book)
    return best_bid


CREDS_CACHE = Path(__file__).resolve().parents[1] / '.api_creds.json'

def _load_cached_creds():
    try:
        if CREDS_CACHE.exists():
            with open(CREDS_CACHE) as f:
                return json.load(f)
    except Exception:
        pass
    return None

def _save_cached_creds(creds):
    try:
        data = {'api_key': creds.api_key, 'api_secret': creds.api_secret, 'api_passphrase': creds.api_passphrase}
        with open(CREDS_CACHE, 'w') as f:
            json.dump(data, f)
    except Exception:
        pass

_auth_client_cache: Optional[ClobClient] = None

def auth_clob_client(clob_base: str = 'https://clob.polymarket.com') -> Optional[ClobClient]:
    global _auth_client_cache
    try:
        if _auth_client_cache is not None:
            try:
                _auth_client_cache.get_server_time()
                return _auth_client_cache
            except Exception:
                _auth_client_cache = None

        key = os.getenv('PM_PRIVATE_KEY') or ''
        funder = os.getenv('PM_DEPOSIT_WALLET') or os.getenv('PM_FUNDER') or None
        sig = int(os.getenv('PM_SIGNATURE_TYPE', '3'))
        if not key:
            return None

        cached = _load_cached_creds()
        if cached:
            try:
                creds = ApiCreds(api_key=cached['api_key'], api_secret=cached['api_secret'], api_passphrase=cached['api_passphrase'])
                c = ClobClient(host=clob_base, chain_id=POLYGON, key=key, signature_type=sig, funder=funder, creds=creds)
                _ = c.get_server_time()
                _auth_client_cache = c
                return c
            except Exception:
                pass

        c = ClobClient(host=clob_base, chain_id=POLYGON, key=key, signature_type=sig, funder=funder)
        creds = c.create_or_derive_api_key()
        if creds:
            _save_cached_creds(creds)
            c = ClobClient(host=clob_base, chain_id=POLYGON, key=key, signature_type=sig, funder=funder, creds=creds)
            _auth_client_cache = c
            return c
        return None
    except Exception:
        return None


def poll_order_status(client: Optional[ClobClient], order_id: str, wait_sec: float = 6.0, step_sec: float = 1.0) -> tuple[str, Optional[dict[str, Any]]]:
    if client is None or not order_id:
        return '', None
    deadline = time.time() + max(0.0, float(wait_sec))
    last = None
    while time.time() <= deadline:
        try:
            last = client.get_order(order_id)
            st = str((last or {}).get('status') or '').upper()
            if st and st not in ('LIVE', 'OPEN'):
                return st, last
        except Exception:
            pass
        time.sleep(max(0.2, float(step_sec)))
    try:
        last = client.get_order(order_id)
    except Exception:
        pass
    st = str((last or {}).get('status') or '').upper()
    return st, last


def cancel_token_orders(client: Optional[ClobClient], token_id: str) -> Optional[dict[str, Any]]:
    if client is None:
        return None
    try:
        return client.cancel_market_orders(asset_id=str(token_id))
    except Exception as e:
        return {'error': str(e)}


def run_open(repo: str, slug: str, side: str, stake: float, execute: bool, trigger_price: float = 0.5) -> tuple[str, list[dict[str, Any]]]:
    cmd = [
        _venv_python(repo),
        'src/live/pm_live_trade_runner.py',
        '--market-slug', slug,
        '--force-side', side,
        '--start-equity', '100',
        '--risk-frac', str(stake / 100.0),
        '--max-notional-usd', str(stake),
        '--trigger-price', str(trigger_price),
    ]
    if execute:
        cmd.append('--execute')
    env = os.environ.copy()
    env.setdefault('PM_MAX_SPREAD', '1')
    env.setdefault('PM_MIN_TOP_ASK_NOTIONAL_USD', '0')
    env.setdefault('PM_ORDER_TYPE', 'FAK')
    p = subprocess.run(cmd, cwd=repo, capture_output=True, text=True, env=env)
    out = (p.stdout or '') + '\n' + (p.stderr or '')
    return out, parse_json_objects(out)


def run_close(
    repo: str,
    slug: str,
    token_id: str,
    shares: float,
    execute: bool,
    close_order_type: str = 'FAK',
    close_limit_price: float | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    cmd = [
        _venv_python(repo),
        'src/live/pm_live_trade_runner.py',
        '--market-slug', slug,
        '--close-token-id', token_id,
        '--close-shares', f'{shares:.8f}',
    ]
    if close_limit_price is not None and close_limit_price > 0:
        cmd += ['--close-limit-price', f'{close_limit_price:.6f}']
    if execute:
        cmd.append('--execute')
    env = os.environ.copy()
    env['PM_CLOSE_ORDER_TYPE'] = str(close_order_type or 'FAK').upper()
    p = subprocess.run(cmd, cwd=repo, capture_output=True, text=True, env=env)
    out = (p.stdout or '') + '\n' + (p.stderr or '')
    return out, parse_json_objects(out)


def get_side_price_from_slug(slug: str, side: str) -> Optional[float]:
    try:
        ev = fetch_event(slug)
        if not ev:
            return None
        mkts = ev.get('markets') or []
        if not mkts:
            return None
        up, dn, *_ = market_side_prices(mkts[0])
        return up if side == 'UP' else dn
    except Exception:
        return None


PROFILES: dict[str, dict[str, Any]] = {
    'conservative': {
        'threshold': 0.72,
        'max_entry_price': 0.88,
        'stake_usd': 1.0,
        'trail_stop_pct': 0.07,
        'take_profit_pct': 0.0,
        'tp_partial_pct': 0.20,
        'tp_partial_ratio': 0.50,
        'hedge_ratio': 0.0,
        'exit_before_sec': 40,
        'emergency_exit_sec': 15,
        'min_entry_seconds_left': 60,
        'entry_timeout_min': 60,
        'poll_sec': 1.0,
        'net_fail_limit': 2,
        # BTC signal quality
        'enable_btc_signal': True,
        'btc_delta_min_tier': 'WEAK',
        'btc_require_momentum': True,
        'btc_atr_multiplier': 1.5,
        'btc_min_confidence': 30.0,
        # Dynamic sizing
        'enable_dynamic_sizing': False,
        'max_position_usd': 5.0,
        'sizing_base_frac': 0.05,
        'sizing_conf_mult': 0.10,
    },
    'aggressive': {
        'threshold': 0.68,
        'max_entry_price': 0.90,
        'stake_usd': 1.0,
        'trail_stop_pct': 0.10,
        'take_profit_pct': 0.0,
        'tp_partial_pct': 0.25,
        'tp_partial_ratio': 0.50,
        'hedge_ratio': 0.0,
        'exit_before_sec': 40,
        'emergency_exit_sec': 15,
        'min_entry_seconds_left': 60,
        'entry_timeout_min': 60,
        'poll_sec': 1.0,
        'net_fail_limit': 2,
        # BTC signal quality (looser)
        'enable_btc_signal': True,
        'btc_delta_min_tier': 'WEAK',
        'btc_require_momentum': False,
        'btc_atr_multiplier': 2.0,
        'btc_min_confidence': 25.0,
        # Dynamic sizing
        'enable_dynamic_sizing': False,
        'max_position_usd': 5.0,
        'sizing_base_frac': 0.05,
        'sizing_conf_mult': 0.10,
    },
}


def load_profile_from_yaml(profile_name: str) -> Optional[dict[str, Any]]:
    yaml_path = Path(__file__).resolve().parents[1] / 'config' / 'btc_5m_profiles.yaml'
    if not yaml_path.exists():
        return None
    try:
        import yaml
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        raw = data.get('profiles', {}).get(profile_name)
        if not raw:
            return None
        flat: dict[str, Any] = {}
        sig = raw.get('signal', {})
        flat['threshold'] = float(sig.get('threshold_price', 0.70))
        flat['max_entry_price'] = float(sig.get('max_entry_price', 1.0))
        siz = raw.get('sizing', {})
        flat['stake_usd'] = float(siz.get('stake_usd', 1.0))
        ts = raw.get('trail_stop', {})
        flat['trail_stop_pct'] = float(ts.get('trail_stop_pct', 0.15))
        tp = raw.get('take_profit', {})
        flat['take_profit_pct'] = float(tp.get('take_profit_pct', 0.0))
        flat['tp_partial_pct'] = float(tp.get('tp_partial_pct', 0.20))
        flat['tp_partial_ratio'] = float(tp.get('tp_partial_ratio', 0.50))
        hg = raw.get('hedge', {})
        flat['hedge_ratio'] = float(hg.get('hedge_ratio', 0.0))
        ee = raw.get('emergency_exit', {})
        flat['emergency_exit_sec'] = int(ee.get('exit_sec', 10))
        flat['exit_before_sec'] = int(data.get('shared_rules', {}).get('session_timing', {}).get('exit_before_sec', 40))
        flat['min_entry_seconds_left'] = int(data.get('shared_rules', {}).get('session_timing', {}).get('min_entry_seconds_left', 60))
        flat['entry_timeout_min'] = 60
        flat['poll_sec'] = 1.0
        flat['net_fail_limit'] = 2
        # BTC signal quality
        sq = raw.get('btc_signal_quality', {})
        if sq:
            flat['enable_btc_signal'] = bool(sq.get('enabled', False))
            flat['btc_delta_min_tier'] = str(sq.get('window_delta_min_tier', 'WEAK'))
            flat['btc_require_momentum'] = bool(sq.get('require_micro_momentum', False))
            flat['btc_atr_multiplier'] = float(sq.get('atr_filter_multiplier', 0))
            flat['btc_min_confidence'] = float(sq.get('min_confidence', 0))
        ds = raw.get('dynamic_sizing', {})
        if ds:
            flat['enable_dynamic_sizing'] = bool(ds.get('enabled', False))
            flat['max_position_usd'] = float(ds.get('max_position_usd', 5.0))
            flat['sizing_base_frac'] = float(ds.get('base_fraction', 0.05))
            flat['sizing_conf_mult'] = float(ds.get('confidence_multiplier', 0.10))
        return flat
    except Exception:
        return None


def apply_profile(args: argparse.Namespace) -> argparse.Namespace:
    # Try YAML first, fall back to hardcoded PROFILES
    prof = load_profile_from_yaml(args.profile or 'conservative')
    if prof is None:
        prof = PROFILES.get(args.profile or 'conservative', PROFILES['conservative'])
    if args.threshold is None:
        args.threshold = float(prof['threshold'])
    if args.max_entry_price is None:
        args.max_entry_price = float(prof.get('max_entry_price', 1.0))
    if args.stake_usd is None:
        args.stake_usd = float(prof['stake_usd'])
    if args.trail_stop_pct is None:
        args.trail_stop_pct = float(prof.get('trail_stop_pct', 0.15))
    if args.tp_partial_pct is None:
        args.tp_partial_pct = float(prof.get('tp_partial_pct', 0))
    if args.tp_partial_ratio is None:
        args.tp_partial_ratio = float(prof.get('tp_partial_ratio', 0.5))
    if args.take_profit_pct is None:
        args.take_profit_pct = float(prof.get('take_profit_pct', 0))
    if args.hedge_ratio is None:
        args.hedge_ratio = float(prof.get('hedge_ratio', 0))
    if args.exit_before_sec is None:
        args.exit_before_sec = int(prof['exit_before_sec'])
    if args.emergency_exit_sec is None:
        args.emergency_exit_sec = int(prof.get('emergency_exit_sec', 10))
    if args.min_entry_seconds_left is None:
        args.min_entry_seconds_left = int(prof['min_entry_seconds_left'])
    if args.entry_timeout_min is None:
        args.entry_timeout_min = int(prof['entry_timeout_min'])
    if args.poll_sec is None:
        args.poll_sec = float(prof['poll_sec'])
    # BTC signal quality (CLI overrides when provided)
    if getattr(args, 'enable_btc_signal', None) is not None:
        args.enable_btc_signal = str(args.enable_btc_signal).lower() in ('true', '1', 'yes')
    else:
        args.enable_btc_signal = bool(prof.get('enable_btc_signal', False))
    if getattr(args, 'btc_delta_min_tier', None) is None:
        args.btc_delta_min_tier = str(prof.get('btc_delta_min_tier', 'WEAK'))
    args.btc_require_momentum = bool(prof.get('btc_require_momentum', False))
    if getattr(args, 'btc_atr_multiplier', None) is None:
        args.btc_atr_multiplier = float(prof.get('btc_atr_multiplier', 0) or 0)
    if getattr(args, 'btc_min_confidence', None) is None:
        args.btc_min_confidence = float(prof.get('btc_min_confidence', 0) or 0)
    # Dynamic sizing (CLI overrides when provided)
    if getattr(args, 'enable_dynamic_sizing', None) is not None:
        args.enable_dynamic_sizing = str(args.enable_dynamic_sizing).lower() in ('true', '1', 'yes')
    else:
        args.enable_dynamic_sizing = bool(prof.get('enable_dynamic_sizing', False))
    if getattr(args, 'max_position_usd', None) is None:
        args.max_position_usd = float(prof.get('max_position_usd', 5.0))
    args.sizing_base_frac = float(prof.get('sizing_base_frac', 0.05))
    args.sizing_conf_mult = float(prof.get('sizing_conf_mult', 0.10))
    return args


def default_repo_path() -> str:
    env_repo = os.environ.get('BTC5M_REPO')
    if env_repo:
        return env_repo
    return str(Path(__file__).resolve().parents[1])  # repo root (self-contained)


def _venv_python(repo: str) -> str:
    """Cross-platform path to venv Python."""
    if sys.platform == 'win32':
        return str(Path(repo) / '.venv' / 'Scripts' / 'python.exe')
    return str(Path(repo) / '.venv' / 'bin' / 'python')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default=default_repo_path())
    ap.add_argument('--profile', choices=['conservative', 'aggressive'], default='conservative')
    ap.add_argument('--threshold', type=float, default=None)
    ap.add_argument('--max-entry-price', type=float, default=None, help='Skip entry if CLOB ask > this price (no upside)')
    ap.add_argument('--stake-usd', type=float, default=None)
    ap.add_argument('--trail-stop-pct', type=float, default=None, help='0.15 = trailing stop 15%% below peak')
    ap.add_argument('--tp-partial-pct', type=float, default=None, help='0.20 = partial TP at +20%% PnL')
    ap.add_argument('--tp-partial-ratio', type=float, default=None, help='0.50 = sell 50%% at partial TP')
    ap.add_argument('--take-profit-pct', type=float, default=None, help='0.20 = +20%% take-profit')
    ap.add_argument('--hedge-ratio', type=float, default=None, help='0.10 = 10%% hedge on opposite side')
    ap.add_argument('--exit-before-sec', type=int, default=None)
    ap.add_argument('--emergency-exit-sec', type=int, default=None, help='T-N seconds last-ditch FAK sell')
    ap.add_argument('--min-entry-seconds-left', type=int, default=None, help='Do not open if less seconds remain in current 5m slot')
    ap.add_argument('--entry-timeout-min', type=int, default=None)
    ap.add_argument('--poll-sec', type=float, default=None)
    ap.add_argument('--close-retry-max', type=int, default=18, help='Max close retries when position is not yet visible / not immediately closable')
    ap.add_argument('--close-retry-delay-sec', type=float, default=2.0, help='Delay between close retries')
    ap.add_argument('--enable-btc-signal', type=str, default=None, help='Enable BTC signal filters: true/false (default from profile)')
    ap.add_argument('--btc-delta-min-tier', type=str, default=None, choices=['SKIP','WEAK','STRONG','MEGA'], help='Minimum window delta tier')
    ap.add_argument('--btc-min-confidence', type=float, default=None, help='Minimum composite confidence 0-100')
    ap.add_argument('--btc-atr-multiplier', type=float, default=None, help='ATR multiplier for volatility filter (0=disabled)')
    ap.add_argument('--enable-dynamic-sizing', type=str, default=None, help='Enable dynamic position sizing: true/false')
    ap.add_argument('--max-position-usd', type=float, default=None, help='Max position size for dynamic sizing (USD)')
    ap.add_argument('--execute', action='store_true')
    args = apply_profile(ap.parse_args())

    report: dict[str, Any] = {
        'started_at': ts_utc(),
        'params': {
            'profile': args.profile,
            'threshold': args.threshold,
            'max_entry_price': args.max_entry_price,
            'stake_usd': args.stake_usd,
            'trail_stop_pct': args.trail_stop_pct,
            'tp_fixed': args.take_profit_pct,
            'exit_before_sec': args.exit_before_sec,
            'min_entry_seconds_left': args.min_entry_seconds_left,
            'entry_timeout_min': args.entry_timeout_min,
            'poll_sec': args.poll_sec,
            'close_retry_max': args.close_retry_max,
            'close_retry_delay_sec': args.close_retry_delay_sec,
            'execute': args.execute,
            'enable_btc_signal': getattr(args, 'enable_btc_signal', False),
            'btc_delta_min_tier': getattr(args, 'btc_delta_min_tier', 'WEAK'),
            'btc_min_confidence': getattr(args, 'btc_min_confidence', 0),
            'btc_atr_multiplier': getattr(args, 'btc_atr_multiplier', 0),
            'enable_dynamic_sizing': getattr(args, 'enable_dynamic_sizing', False),
            'max_position_usd': getattr(args, 'max_position_usd', 5.0),
        },
        'attempts': [],
    }

    deadline = time.time() + args.entry_timeout_min * 60
    opened = None

    # Prevent re-entering the same 5-min slot
    traded_slugs = set()
    last_trade_file = Path(__file__).resolve().parents[1] / '.last_trade.json'
    try:
        if last_trade_file.exists():
            with open(last_trade_file) as f:
                data = json.load(f)
                if isinstance(data, list):
                    for t in data:
                        traded_slugs.add(t.get('slug', ''))
    except Exception:
        pass

    # Initialize BTC data feed for signal quality checks
    btc_feed = None
    btc_signal_config = {
        'enable_window_delta': True,
        'window_delta_min_tier': str(getattr(args, 'btc_delta_min_tier', 'WEAK')),
        'require_micro_momentum': bool(getattr(args, 'btc_require_momentum', False)),
        'atr_filter_multiplier': float(getattr(args, 'btc_atr_multiplier', 0) or 0),
        'min_confidence': float(getattr(args, 'btc_min_confidence', 0) or 0),
    }
    if getattr(args, 'enable_btc_signal', False):
        try:
            btc_feed = BtcDataFeed()
            _ = btc_feed.fetch_klines()  # pre-warm cache
            print(f"  [BTC] Signal filters enabled: delta>={btc_signal_config['window_delta_min_tier']} "
                  f"momentum={btc_signal_config['require_micro_momentum']} "
                  f"atr_mult={btc_signal_config['atr_filter_multiplier']} "
                  f"min_conf={btc_signal_config['min_confidence']:.0f}%")
        except Exception as e:
            print(f"  [WARN] BTC data feed init failed: {e}. Continuing without BTC filters.")
            btc_feed = None

    while time.time() < deadline:
        try:
            m = resolve_active_current_5m_market()
            if not m:
                report['attempts'].append({'ts': ts_utc(), 'status': 'heartbeat_no_current_market'})
                time.sleep(args.poll_sec)
                continue

            g_up, g_dn, up_t, dn_t, slug, end_iso = market_side_prices(m)

            end_ts = None
            sec_left = None
            try:
                end_ts = dt.datetime.fromisoformat(end_iso.replace('Z', '+00:00')).timestamp()
                sec_left = max(0.0, end_ts - time.time())
            except Exception:
                pass

            if sec_left is None:
                report['attempts'].append({'ts': ts_utc(), 'slug': slug, 'status': 'heartbeat_bad_market_end'})
                time.sleep(args.poll_sec)
                continue

            # Do not open if less than N seconds remain in current slot.
            if sec_left < args.min_entry_seconds_left:
                report['attempts'].append({
                    'ts': ts_utc(),
                    'slug': slug,
                    'status': 'skip_too_late_to_enter',
                    'seconds_left': sec_left,
                    'min_entry_seconds_left': args.min_entry_seconds_left,
                })
                time.sleep(args.poll_sec)
                continue

            # CLOB-based trigger price (best ask of selected side), not Gamma outcomePrices.
            try:
                up_ask, dn_ask, min_spread = clob_side_prices(up_t, dn_t)
            except Exception as e:
                report['attempts'].append({'ts': ts_utc(), 'slug': slug, 'status': 'skip_clob_unavailable', 'error': str(e)})
                time.sleep(args.poll_sec)
                continue

            report['attempts'].append({
                'ts': ts_utc(),
                'slug': slug,
                'status': 'heartbeat',
                'gamma_up': g_up,
                'gamma_down': g_dn,
                'clob_up_ask': up_ask,
                'clob_down_ask': dn_ask,
                'seconds_left': sec_left,
                'min_spread': min_spread,
            })

            if slug in traded_slugs:
                continue

            print(f"[{ts_local()}] {slug} UP_ask={up_ask} DOWN_ask={dn_ask} sec_left={sec_left:.0f}", flush=True)

            candidates: list[tuple[str, float]] = []
            max_entry = float(args.max_entry_price or 1.0)
            if up_ask is not None and float(up_ask) >= args.threshold and float(up_ask) <= max_entry:
                candidates.append(('UP', float(up_ask)))
            if dn_ask is not None and float(dn_ask) >= args.threshold and float(dn_ask) <= max_entry:
                candidates.append(('DOWN', float(dn_ask)))

            if not candidates:
                report['attempts'].append({
                    'ts': ts_utc(),
                    'slug': slug,
                    'status': 'skip_price_below_threshold',
                    'threshold': args.threshold,
                    'clob_up_ask': up_ask,
                    'clob_down_ask': dn_ask,
                    'seconds_left': sec_left,
                })
                print(f"  -> SKIP (need ask in [{args.threshold:.2f}, {max_entry:.2f}])", flush=True)
                time.sleep(args.poll_sec)
                continue

            side, trigger_price = sorted(candidates, key=lambda x: x[1], reverse=True)[0]
            print(f"  -> CLOB OK {side} ask={trigger_price:.4f}", flush=True)

            # ===== BTC signal quality check =====
            signal_result = None
            if btc_feed is not None:
                if btc_feed.is_healthy():
                    btc_price = btc_feed.fetch_price()
                    candles = btc_feed.fetch_klines()
                    window_open = btc_feed.get_window_open(int(time.time()) - (int(time.time()) % 300))
                    if btc_price is not None and window_open is not None and candles:
                        signal_result = evaluate_signal(
                            current_price=btc_price,
                            window_open=window_open,
                            candles=candles,
                            side=side,
                            config=btc_signal_config,
                            seconds_left=sec_left,
                        )
                        if not signal_result.passed:
                            print(f"  -> BTC REJECT ({signal_result.reason}) "
                                  f"delta={signal_result.window_delta_pct:+.3f}% "
                                  f"tier={signal_result.delta_tier.label} "
                                  f"conf={signal_result.confidence:.0f}", flush=True)
                            report['attempts'].append({
                                'ts': ts_utc(), 'slug': slug, 'side': side,
                                'status': 'skip_btc_signal',
                                'reason': signal_result.reason,
                                'window_delta_pct': round(signal_result.window_delta_pct, 4),
                                'delta_tier': signal_result.delta_tier.label,
                                'confidence': round(signal_result.confidence, 1),
                            })
                            time.sleep(args.poll_sec)
                            continue
                        print(f"  -> BTC OK delta={signal_result.window_delta_pct:+.3f}% "
                              f"tier={signal_result.delta_tier.label} "
                              f"mom={signal_result.micro_momentum_pass} "
                              f"atr={signal_result.atr_pass} "
                              f"conf={signal_result.confidence:.0f}%", flush=True)
                    else:
                        print(f"  -> BTC REJECT: data unavailable, skip entry", flush=True)
                        report['attempts'].append({
                            'ts': ts_utc(), 'slug': slug, 'side': side,
                            'status': 'skip_btc_signal',
                            'reason': 'btc_data_unavailable',
                        })
                        time.sleep(args.poll_sec)
                        continue
                else:
                    err = btc_feed._last_error or 'unknown'
                    print(f"  -> BTC REJECT: feed unhealthy ({err}), skip entry", flush=True)
                    report['attempts'].append({
                        'ts': ts_utc(), 'slug': slug, 'side': side,
                        'status': 'skip_btc_signal',
                        'reason': f'btc_feed_unhealthy',
                    })
                    time.sleep(args.poll_sec)
                    continue

            # ===== Dynamic position sizing =====
            entry_stake = args.stake_usd
            if getattr(args, 'enable_dynamic_sizing', False) and signal_result is not None:
                entry_stake = compute_position_size(
                    confidence=signal_result.confidence,
                    max_position=float(getattr(args, 'max_position_usd', 5.0)),
                    base_frac=float(getattr(args, 'sizing_base_frac', 0.05)),
                    conf_mult=float(getattr(args, 'sizing_conf_mult', 0.10)),
                )
                print(f"  -> Dynamic sizing: ${entry_stake:.2f} (conf={signal_result.confidence:.0f}%)")

            opened = None
            for attempt in range(2):
                if attempt > 0:
                    print(f"  -> RETRY open...", flush=True)
                    time.sleep(1.5)
                out, objs = run_open(args.repo, slug, side, entry_stake, args.execute, trigger_price)
                post = None
                runner = None
                for o in objs:
                    if isinstance(o, dict) and 'order_post_result' in o:
                        runner = o
                        post = o.get('order_post_result') or {}
                if post and post.get('success') is True and str(post.get('status', '')).lower() == 'matched':
                    token_id = str(runner.get('token_id') or (up_t if side == 'UP' else dn_t))
                    shares = float(post.get('takingAmount') or 0)
                    cost = float(post.get('makingAmount') or 0)
                    entry_price = float(runner.get('entry_price') or trigger_price)
                    opened = {
                        'opened_at': ts_utc(),
                        'market_slug': slug,
                        'market_end_iso': end_iso,
                        'side': side,
                        'token_id': token_id,
                        'entry_price': entry_price,
                        'shares': shares,
                        'cost_usdc': cost,
                        'open_order_id': post.get('orderID'),
                        'open_tx': (post.get('transactionsHashes') or [None])[0],
                    }
                    report['open_raw'] = out[-4000:]
                    print(f"  -> ORDER SUCCESS: {side} @ {entry_price:.4f}  tx={opened['open_tx']}", flush=True)
                    break
                else:
                    if attempt == 0:
                        # First failure — will retry
                        err = str(post.get('error', ''))[:80] if post else 'no response'
                        print(f"  -> ORDER FAILED (attempt 1/2): {err}", flush=True)
                        continue
                    # Second failure — give up
                    report['last_open_try'] = out[-2000:]
                    if post and isinstance(post, dict):
                        status_str = str(post.get('status', 'unknown'))
                        error_str = str(post.get('error', ''))
                        print(f"  -> ORDER FAILED (attempt 2/2): status={status_str} error={error_str[:120]}", flush=True)
                    else:
                        print(f"  -> ORDER FAILED: no valid response from runner", flush=True)
                    print(f"  -> Raw: {out[-500:]}", flush=True)
                if opened:
                    break
        except Exception as e:
            report['attempts'].append({'ts': ts_utc(), 'status': 'error', 'error': str(e)})
            print(f"  -> EXCEPTION: {e}", flush=True)
        if opened:
            # Entered in this slot — prevent re-entry and proceed to monitoring
            traded_slugs.add(opened['market_slug'])
            break
        time.sleep(args.poll_sec)

    if not opened:
        report['finished_at'] = ts_utc()
        report['result'] = 'no_entry_timeout'
        print(f'[{ts_local()}] TIMEOUT - no valid entry signal found', flush=True)
        return

    report['opened'] = opened

    print(f"\n{'='*60}")
    print(f"POSITION OPENED: {opened['side']} @ {opened['entry_price']:.4f}")
    print(f"  Shares: {opened['shares']:.4f}  Cost: ${opened['cost_usdc']:.2f}")
    print(f"  Market ends: {opened['market_end_iso']}")
    print(f"{'='*60}\n")

    # Micro hedge: buy opposite side for insurance
    hedge_opened = None
    hedge_ratio = getattr(args, 'hedge_ratio', 0) or 0
    if hedge_ratio > 0 and args.execute:
        hedge_side = 'UP' if opened['side'] == 'DOWN' else 'DOWN'
        hedge_notional = opened['cost_usdc'] * hedge_ratio
        if hedge_notional >= 1.0:
            try:
                h_slug = opened['market_slug']
                h_tid, h_price, h_info = resolve_token_id(h_slug, hedge_side)
                if h_tid:
                    from py_clob_client_v2.clob_types import MarketOrderArgs as MA, OrderType as OT2
                    from py_clob_client_v2 import Side as S2
                    h_signed = client.create_market_order(
                        MA(token_id=h_tid, amount=round(hedge_notional, 2), side=S2.BUY, order_type=OT2.FAK),
                        options=PartialCreateOrderOptions(tick_size='0.01'),
                    ) if client else None
                    if h_signed and client:
                        h_result = client.post_order(h_signed)
                        if isinstance(h_result, dict) and h_result.get('success'):
                            hedge_opened = {
                                'side': hedge_side,
                                'token_id': h_tid,
                                'notional': hedge_notional,
                                'tx': (h_result.get('transactionsHashes') or [None])[0],
                            }
                            print(f"  [HEDGE] {hedge_side} ${hedge_notional:.2f} tx={str(hedge_opened['tx'])[:20]}...")
            except Exception as e:
                print(f"  [HEDGE] failed: {e}")
    report['hedge'] = hedge_opened

    # monitor after open: trailing stop + on-chain GTC safety net
    end_ts = None
    try:
        end_ts = dt.datetime.fromisoformat(opened['market_end_iso'].replace('Z', '+00:00')).timestamp()
    except Exception:
        end_ts = time.time() + 300

    trail_pct = args.trail_stop_pct
    highest_price = opened['entry_price']
    trail_stop = highest_price * (1.0 - trail_pct)
    tp_price = opened['entry_price'] * (1.0 + args.take_profit_pct) if args.take_profit_pct > 0 else None
    cooldown_until = time.time() + 5

    # Place on-chain GTC safety sell at initial trail stop price
    client = auth_clob_client()  # init before GTC safety + close logic
    gtc_safety_id: Optional[str] = None
    gtc_safety_price = 0.0
    if client and opened['shares'] > 0 and args.execute:
        try:
            from py_clob_client_v2.clob_types import OrderArgs, OrderType as OT2
            from py_clob_client_v2 import Side as S2
            sx = client.create_order(OrderArgs(
                token_id=opened['token_id'], price=round(trail_stop - 0.02, 2),
                size=opened['shares'], side=S2.SELL,
            ))
            gtc_result = client.post_order(sx)
            if isinstance(gtc_result, dict) and gtc_result.get('orderID'):
                gtc_safety_id = str(gtc_result['orderID'])
                gtc_safety_price = trail_stop
                print(f"  [SAFETY] GTC sell placed @ {trail_stop:.4f}  order={gtc_safety_id[:20]}...")
        except Exception as e:
            print(f"  [SAFETY] GTC placement failed: {e}")

    report['trail_stop_pct'] = trail_pct
    report['initial_trail_stop'] = trail_stop
    print(f"  Trail Stop: {trail_pct*100:.0f}% below peak  Initial: {trail_stop:.4f}  Cooldown: 5s  FailLimit: {getattr(args, 'net_fail_limit', 2)}  ExitBefore: {args.exit_before_sec}s")

    close_reason = None
    partial_tp_done = False
    net_fails = 0
    emergency_sec = getattr(args, 'emergency_exit_sec', 10)
    try:
     while True:
        now = time.time()
        sec_left = end_ts - now
        if now >= (end_ts - emergency_sec):
            print(f'  [EMERGENCY] T-{emergency_sec}s last-ditch exit')
            close_reason = f'emergency_exit_{emergency_sec}s'
            break
        if now >= (end_ts - args.exit_before_sec):
            close_reason = f'time_exit_{args.exit_before_sec}s_before_end'
            break

        try:
            side_px = clob_best_bid(opened['token_id'])
            net_fails = 0
        except Exception:
            side_px = None
            net_fails += 1
            fail_limit = getattr(args, 'net_fail_limit', 2)
            if net_fails >= fail_limit:
                print(f'  [ALERT] 3 consecutive network failures, force-closing...')
                close_reason = 'force_close_network_loss'
                break
        report['last_side_price'] = side_px
        report['last_check_at'] = ts_utc()

        if side_px is not None:
            if side_px > highest_price:
                highest_price = side_px
                new_trail = highest_price * (1.0 - trail_pct)
                # Update on-chain GTC when trail stop moves up
                if new_trail > gtc_safety_price + 0.01 and client and args.execute:
                    try:
                        if gtc_safety_id:
                            client.cancel(gtc_safety_id)
                        new_price = round(new_trail - 0.02, 2)
                        from py_clob_client_v2.clob_types import OrderArgs as OA
                        from py_clob_client_v2 import Side as S2
                        sx = client.create_order(OA(
                            token_id=opened['token_id'], price=new_price,
                            size=opened['shares'], side=S2.SELL,
                        ))
                        gtc_result = client.post_order(sx)
                        if isinstance(gtc_result, dict) and gtc_result.get('orderID'):
                            gtc_safety_id = str(gtc_result['orderID'])
                            gtc_safety_price = new_price
                            print(f"  [SAFETY] GTC updated to {new_price:.4f}  order={gtc_safety_id[:20]}...")
                    except Exception:
                        pass
                trail_stop = new_trail
                print(f"  [TRAIL] lock at {trail_stop:.4f} (+{(trail_stop - opened['entry_price'])/opened['entry_price']*100:.1f}%)", flush=True)

            pnl_pct = (side_px - opened['entry_price']) / opened['entry_price'] * 100
            pnl_usd = opened['cost_usdc'] * pnl_pct / 100
            print(f"  {opened['side']} | in={opened['entry_price']:.4f} now={side_px:.4f} | PnL={pnl_usd:+.2f}$ ({pnl_pct:+.1f}%) | T-{sec_left:.0f}s", flush=True)

        if now >= cooldown_until:
            if side_px is not None and side_px <= trail_stop:
                close_reason = f"trail_stop_{int(trail_pct * 100)}pct"
                break

            if tp_price and side_px is not None and side_px >= tp_price:
                close_reason = f"take_profit_{int(args.take_profit_pct * 100)}pct"
                break

            # Partial take-profit: lock profit on part of position
            tp_partial_pct = getattr(args, 'tp_partial_pct', 0) or 0
            tp_partial_ratio = getattr(args, 'tp_partial_ratio', 0.5) or 0.5
            if tp_partial_pct > 0 and not partial_tp_done and side_px is not None:
                if pnl_pct >= tp_partial_pct * 100:
                    close_shares = opened['shares'] * tp_partial_ratio
                    print(f"  [PARTIAL TP] +{pnl_pct:.1f}% reached, closing {tp_partial_ratio*100:.0f}% ({close_shares:.4f} shares)", flush=True)
                    try:
                        out_p, objs_p = run_close(args.repo, opened['market_slug'], opened['token_id'], close_shares, args.execute, close_order_type='FAK')
                        last_p = objs_p[-1] if objs_p else {}
                        post_p = last_p.get('order_post_result') or {}
                        if post_p.get('success') and str(post_p.get('status','')).lower() == 'matched':
                            close_rev = float(post_p.get('takingAmount') or 0)
                            opened['shares'] -= close_shares
                            opened['cost_usdc'] -= close_rev * (opened['cost_usdc'] / (opened['shares'] + close_shares))
                            if opened['shares'] < 0:
                                opened['shares'] = 0
                            partial_tp_done = True
                            report['partial_tp'] = {'shares_closed': close_shares, 'revenue': close_rev}
                            print(f"  [PARTIAL TP] done, remaining {opened['shares']:.4f} shares", flush=True)
                            # Also cancel/update GTC safety for remaining shares
                            if gtc_safety_id and client:
                                try: client.cancel(gtc_safety_id)
                                except: pass
                                gtc_safety_id = None
                        else:
                            print(f"  [PARTIAL TP] order failed, status={post_p.get('status','?')}")
                    except Exception as e:
                        print(f"  [PARTIAL TP] failed: {e}")
        time.sleep(args.poll_sec)
    except Exception as e:
        print(f'  [CRASH] Recovery: {e}', flush=True)
        close_reason = f'crash_recovery_{str(e)[:30]}'
    close_debug: list[dict[str, Any]] = []
    close_obj: dict[str, Any] = {}
    out = ''
    fallback_used = None
    force_close_used = None

    for i in range(max(1, int(args.close_retry_max))):
        bb = None
        try:
            bb = clob_best_bid(opened['token_id'])
        except Exception:
            pass
        print(f"  [CLOSE #{i+1}] shares={opened['shares']:.4f} best_bid={bb}", flush=True)
        out, objs = run_close(
            args.repo,
            opened['market_slug'],
            opened['token_id'],
            opened['shares'],
            args.execute,
            close_order_type='FAK',
        )
        close_obj = objs[-1] if objs else {}
        post = close_obj.get('order_post_result') or {}
        status = str(post.get('status') or '').lower()
        skipped = str(close_obj.get('close_skipped') or '')
        close_debug.append({
            'ts': ts_utc(),
            'attempt': i + 1,
            'order_type': 'FAK',
            'status': status,
            'close_skipped': skipped,
        })
        if post.get('success') is True and status == 'matched':
            break

        # common transient path right after open: token balance not yet visible
        if skipped == 'zero_effective_shares':
            time.sleep(float(args.close_retry_delay_sec))
            continue

        # fallback: if FAK has no instant match, try a GTC limit close near current side price
        txt = ((out or '') + '\n' + json.dumps(close_obj, ensure_ascii=False)).lower()
        if 'no orders found to match with fak order' in txt:
            px = get_side_price_from_slug(opened['market_slug'], opened['side'])
            if px is None:
                px = report.get('last_side_price')
            if px is None:
                px = opened['entry_price']
            bb = None
            try:
                bb = clob_best_bid(opened['token_id'])
            except Exception:
                bb = None
            limit_px = max(0.01, min(0.99, float((bb - 0.01) if bb is not None else px)))
            fallback_used = {'type': 'GTC_LIMIT', 'price': limit_px}
            out2, objs2 = run_close(
                args.repo,
                opened['market_slug'],
                opened['token_id'],
                opened['shares'],
                args.execute,
                close_order_type='GTC',
                close_limit_price=limit_px,
            )
            close_obj2 = objs2[-1] if objs2 else {}
            post2 = close_obj2.get('order_post_result') or {}
            status2 = str(post2.get('status') or '').lower()
            close_debug.append({
                'ts': ts_utc(),
                'attempt': i + 1,
                'order_type': 'GTC',
                'status': status2,
                'close_skipped': str(close_obj2.get('close_skipped') or ''),
                'limit_price': limit_px,
            })
            close_obj = close_obj2
            out = out2
            if post2.get('success') is True and status2 == 'matched':
                break

            # If GTC is accepted but still live, force-close flow: poll status, cancel, repost aggressive.
            if post2.get('success') is True and status2 == 'live':
                oid2 = str(post2.get('orderID') or '')
                st_upd, ord_upd = poll_order_status(client, oid2, wait_sec=min(8.0, max(2.0, float(args.close_retry_delay_sec) * 2)), step_sec=1.0)
                close_debug.append({
                    'ts': ts_utc(),
                    'attempt': i + 1,
                    'order_type': 'GTC_POLL',
                    'status': st_upd.lower() if st_upd else '',
                    'order_id': oid2,
                })
                if st_upd == 'MATCHED':
                    post2['status'] = 'matched'
                    close_obj['order_post_result'] = post2
                    break

                cancel_info = cancel_token_orders(client, opened['token_id'])
                bb2 = None
                try:
                    bb2 = clob_best_bid(opened['token_id'])
                except Exception:
                    bb2 = None
                force_px = max(0.01, min(0.99, float((bb2 - 0.02) if bb2 is not None else 0.01)))
                force_close_used = {
                    'type': 'FORCE_GTC_LIMIT',
                    'price': force_px,
                    'cancel_info': cancel_info,
                }
                out3, objs3 = run_close(
                    args.repo,
                    opened['market_slug'],
                    opened['token_id'],
                    opened['shares'],
                    args.execute,
                    close_order_type='GTC',
                    close_limit_price=force_px,
                )
                close_obj3 = objs3[-1] if objs3 else {}
                post3 = close_obj3.get('order_post_result') or {}
                status3 = str(post3.get('status') or '').lower()
                close_debug.append({
                    'ts': ts_utc(),
                    'attempt': i + 1,
                    'order_type': 'FORCE_GTC',
                    'status': status3,
                    'close_skipped': str(close_obj3.get('close_skipped') or ''),
                    'limit_price': force_px,
                })
                close_obj = close_obj3
                out = out3
                if post3.get('success') is True and status3 == 'matched':
                    break

                # Last resort: dump sell at any price
                dump_px = 0.05
                dump_close_used = {'type': 'DUMP_GTC', 'price': dump_px}
                out4, objs4 = run_close(
                    args.repo, opened['market_slug'], opened['token_id'],
                    opened['shares'], args.execute,
                    close_order_type='GTC', close_limit_price=dump_px,
                )
                close_obj4 = objs4[-1] if objs4 else {}
                post4 = close_obj4.get('order_post_result') or {}
                status4 = str(post4.get('status') or '').lower()
                close_debug.append({
                    'ts': ts_utc(), 'attempt': i + 1,
                    'order_type': 'DUMP_GTC',
                    'status': status4,
                    'close_skipped': str(close_obj4.get('close_skipped') or ''),
                    'limit_price': dump_px,
                })
                close_obj = close_obj4
                out = out4
                force_close_used = dump_close_used
                if post4.get('success') is True and status4 == 'matched':
                    break

        time.sleep(float(args.close_retry_delay_sec))

    post = close_obj.get('order_post_result') or {}
    post_status = str(post.get('status') or '').lower()
    close_usdc = float(post.get('takingAmount') or 0)
    closed = {
        'close_reason': close_reason,
        'closed_at': ts_utc(),
        'close_success': bool(post.get('success') is True and (post_status == 'matched' or close_usdc > 0)),
        'close_status': post.get('status'),
        'close_order_id': post.get('orderID'),
        'close_tx': (post.get('transactionsHashes') or [None])[0],
        'close_shares': float(post.get('makingAmount') or 0),
        'close_usdc': close_usdc,
        'close_skipped': close_obj.get('close_skipped'),
    }
    # Cancel safety GTC only after close succeeds
    if gtc_safety_id and client and closed['close_success']:
        try:
            client.cancel(gtc_safety_id)
        except Exception:
            pass

    report['close_debug'] = close_debug
    if fallback_used:
        report['close_fallback'] = fallback_used
    if force_close_used:
        report['close_force'] = force_close_used
    report['close_raw'] = out[-4000:]
    report['closed'] = closed

    # Close hedge position if open
    hedge_pnl = 0.0
    if hedge_opened and client:
        try:
            from py_clob_client_v2.clob_types import MarketOrderArgs as MA, OrderType as OT2
            from py_clob_client_v2 import Side as S2
            h_signed = client.create_market_order(
                MA(token_id=hedge_opened['token_id'], amount=100, side=S2.SELL, order_type=OT2.FAK),
                options=PartialCreateOrderOptions(tick_size='0.01'),
            )
            h_result = client.post_order(h_signed)
            if isinstance(h_result, dict) and h_result.get('success'):
                h_revenue = float(h_result.get('makingAmount') or 0)
                hedge_pnl = round(h_revenue - hedge_opened['notional'], 6)
                print(f'  [HEDGE] closed, PnL=${hedge_pnl:+.2f}')
        except Exception as e:
            print(f'  [HEDGE] close failed: {e}')

    pnl = None
    if closed['close_usdc']:
        pnl = round(closed['close_usdc'] - opened['cost_usdc'] + hedge_pnl, 6)
    report['realized_cashflow_pnl_usdc'] = pnl
    report['finished_at'] = ts_utc()
    report['result'] = 'done'

    # Compact summary
    pnl_str = f'${pnl:+.2f}' if pnl else 'N/A'
    side = opened.get('side', '?')
    entry = opened.get('entry_price', 0)
    reason = closed.get('close_reason', '?')
    close_price = closed['close_usdc'] / closed['close_shares'] if closed.get('close_usdc') and closed.get('close_shares') else 0
    pnl_pct = (pnl / opened['cost_usdc'] * 100) if pnl and opened.get('cost_usdc') else 0
    tx_short = str(opened.get('open_tx', '?'))[:20]
    print(f'\nTRADE DONE | {side} entry=@{entry:.3f} close=@{close_price:.3f} PnL={pnl_str}({pnl_pct:+.1f}%) exit={reason}', flush=True)

    # Save trade to prevent re-entry in same slot
    try:
        sorted_slugs = list(traded_slugs)
        sorted_slugs.append(opened.get('market_slug', ''))
        sorted_slugs = sorted_slugs[-5:]
        with open(last_trade_file, 'w') as f:
            json.dump([{'slug': s} for s in sorted_slugs], f)
    except Exception:
        pass

    # Auto-save trade log
    try:
        _save_trade_log(report)
    except Exception:
        pass


def _clear_screen():
    os.system('cls' if sys.platform == 'win32' else 'clear')

def _save_trade_log(report: dict):
    log_dir = Path(__file__).resolve().parents[1] / 'logs'
    log_dir.mkdir(exist_ok=True)
    ts = dt.datetime.now(UTC).strftime('%Y%m%d_%H%M%S')
    fname = f"trade_{report.get('opened',{}).get('side','NONE')}_{ts}.json"
    with open(log_dir / fname, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f'[LOG] saved to logs/{fname}', flush=True)

def _next_slot_sec() -> float:
    """Seconds until the start of the next 5-min boundary + 10s buffer."""
    now = int(time.time())
    bucket = now - (now % 300)
    next_start = bucket + 300 + 10  # T+10s into next window
    return max(1, next_start - now)

if __name__ == '__main__':
    while True:
        _clear_screen()
        print(f'{"="*50}')
        print(f'BTC 5m Live — {dt.datetime.now(CST).strftime("%Y-%m-%d %H:%M:%S")} CST')
        print(f'{"="*50}\n')
        try:
            main()
        except KeyboardInterrupt:
            print('\n[STOP] Keyboard interrupt, exiting.')
            break
        except Exception as e:
            print(f'\n[FATAL] {e}', flush=True)
            time.sleep(5)
        wait = _next_slot_sec()
        print(f'\n[RESTART] Next cycle at {ts_local()} (in {wait:.0f}s)...\n', flush=True)
        time.sleep(wait)
