#!/usr/bin/env python3
"""
Polymarket CLOB trade execution runner (V2 API).
Stateless: receives order parameters via CLI, executes ONE operation, outputs JSON to stdout.
"""

import argparse
import json
import os
import sys
from typing import Any, Optional

import requests
from dotenv import load_dotenv

load_dotenv()


# ============================================================
# Authentication
# ============================================================

def get_clob_client() -> Optional[Any]:
    """Create an authenticated ClobClient using V2 API."""
    try:
        from py_clob_client_v2.client import ClobClient
        from py_clob_client_v2.constants import POLYGON
        from py_clob_client_v2.clob_types import ApiCreds
    except ImportError:
        print(json.dumps({"error": "py_clob_client_v2 not installed. pip install py-clob-client-v2"}))
        return None

    key = os.getenv("PM_PRIVATE_KEY") or ""
    # V2 deposit wallet flow: use signature_type=3 (POLY_1271)
    sig = int(os.getenv("PM_SIGNATURE_TYPE", "3"))
    # funder = Polymarket deposit wallet (NOT the EOA)
    funder = os.getenv("PM_DEPOSIT_WALLET") or os.getenv("PM_FUNDER") or None

    if not key:
        print(json.dumps({"error": "missing credentials", "detail": "Set PM_PRIVATE_KEY in .env"}))
        return None

    try:
        c = ClobClient(host="https://clob.polymarket.com", chain_id=POLYGON, key=key, signature_type=sig, funder=funder)
        creds = c.create_or_derive_api_key()
        if creds:
            c = ClobClient(host="https://clob.polymarket.com", chain_id=POLYGON, key=key, signature_type=sig, funder=funder, creds=creds)
            return c

        print(json.dumps({"error": "auth failed — could not derive API key"}))
        return None
    except Exception as e:
        print(json.dumps({"error": f"auth failed: {e}"}))
        return None


# ============================================================
# Market resolution
# ============================================================

def fetch_event(slug: str) -> Optional[dict]:
    r = requests.get("https://gamma-api.polymarket.com/events", params={"slug": slug}, timeout=12)
    r.raise_for_status()
    arr = r.json()
    return arr[0] if arr else None


def resolve_token_id(slug: str, side: str) -> tuple[Optional[str], Optional[float], dict]:
    ev = fetch_event(slug)
    if not ev:
        return None, None, {"error": "event not found", "slug": slug}

    mkts = ev.get("markets") or []
    if not mkts:
        return None, None, {"error": "no markets in event", "slug": slug}

    m = mkts[0]
    outcomes_raw = m.get("outcomes") or "[]"
    prices_raw = m.get("outcomePrices") or "[]"
    token_ids_raw = m.get("clobTokenIds") or "[]"

    outcomes = json.loads(outcomes_raw) if isinstance(outcomes_raw, str) else outcomes_raw
    prices = json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw
    token_ids = json.loads(token_ids_raw) if isinstance(token_ids_raw, str) else token_ids_raw

    if len(prices) < 2 or len(token_ids) < 2:
        return None, None, {"error": "missing outcomePrices/clobTokenIds", "slug": slug}

    up_i, down_i = 0, 1
    if len(outcomes) >= 2:
        labs = [str(x).lower() for x in outcomes[:2]]
        if "up" in labs[1] or "yes" in labs[1]:
            up_i, down_i = 1, 0

    idx = up_i if side.upper() == "UP" else down_i
    token_id = str(token_ids[idx])
    price = float(prices[idx])

    return token_id, price, {
        "slug": slug, "side": side, "token_id": token_id,
        "outcome_price": price,
        "market_end": m.get("endDate") or m.get("endDateIso"),
        "market_closed": m.get("closed"),
    }


# ============================================================
# Position open
# ============================================================

def open_position(args) -> dict:
    side = (args.force_side or "").upper()
    if side not in ("UP", "DOWN"):
        return {"error": f"invalid --force-side: {args.force_side}"}

    token_id, ref_price, info = resolve_token_id(args.market_slug, side)
    if token_id is None:
        return info

    max_notional = float(args.max_notional_usd or 5.0)
    if args.risk_frac and args.start_equity:
        risk_based = float(args.start_equity) * float(args.risk_frac)
        max_notional = min(max_notional, risk_based)

    order_type = os.getenv("PM_ORDER_TYPE", "FAK").upper()

    if not args.execute:
        return {
            "dry_run": True,
            "order_post_result": {
                "success": True, "status": "dry_run",
                "detail": f"Would {order_type} BUY {side} token={token_id} notional={max_notional:.2f}",
            },
            "token_id": token_id, "entry_price": ref_price,
            "dry_run_info": info,
        }

    client = get_clob_client()
    if client is None:
        return {"error": "authentication failed"}

    try:
        from py_clob_client_v2.clob_types import MarketOrderArgs, OrderType, PartialCreateOrderOptions
        from py_clob_client_v2 import Side

        ot = OrderType.FAK if order_type == "FAK" else OrderType.GTC

        signed = client.create_market_order(
            MarketOrderArgs(token_id=token_id, amount=max_notional, side=Side.BUY, price=0.5, order_type=ot),
            options=PartialCreateOrderOptions(tick_size="0.01"),
        )
        result = client.post_order(signed)

        entry_price = ref_price
        if isinstance(result, dict):
            making = float(result.get("makingAmount") or 0)
            taking = float(result.get("takingAmount") or 0)
            if taking > 0:
                entry_price = making / taking / 1e6

        return {
            "order_post_result": result if isinstance(result, dict) else {"raw": str(result)},
            "token_id": token_id, "entry_price": round(entry_price, 6),
        }
    except Exception as e:
        return {"error": f"open order failed: {e}", "token_id": token_id}


# ============================================================
# Position close
# ============================================================

def close_position(args) -> dict:
    token_id = args.close_token_id
    shares = float(args.close_shares or 0)
    order_type = os.getenv("PM_CLOSE_ORDER_TYPE", "FAK").upper()
    limit_price = float(args.close_limit_price) if args.close_limit_price else None

    if not token_id or shares <= 0:
        return {
            "order_post_result": {"success": False, "status": "skipped"},
            "close_skipped": "zero_effective_shares",
        }

    if not args.execute:
        limit_str = f"limit={limit_price:.6f}" if limit_price else "market"
        return {
            "dry_run": True,
            "order_post_result": {
                "success": True, "status": "dry_run",
                "detail": f"Would {order_type} SELL token={token_id} shares={shares:.8f} {limit_str}",
            },
        }

    client = get_clob_client()
    if client is None:
        return {"error": "authentication failed"}

    try:
        from py_clob_client_v2.clob_types import MarketOrderArgs, OrderArgs, OrderType, PartialCreateOrderOptions
        from py_clob_client_v2 import Side

        if order_type == "FAK":
            signed = client.create_market_order(
                MarketOrderArgs(token_id=token_id, amount=shares, side=Side.SELL, price=0.5, order_type=OrderType.FAK),
                options=PartialCreateOrderOptions(tick_size="0.01"),
            )
            result = client.post_order(signed)
        else:
            price = limit_price or 0.01
            signed = client.create_order(OrderArgs(
                token_id=token_id, price=price,
                size=shares, side=Side.SELL,
            ))
            result = client.post_order(signed)

        close_skipped = None
        if isinstance(result, dict):
            status = str(result.get("status") or "").lower()
            if status == "skipped" or not result.get("success"):
                close_skipped = "zero_effective_shares"

        out = {"order_post_result": result if isinstance(result, dict) else {"raw": str(result)}}
        if close_skipped:
            out["close_skipped"] = close_skipped
        return out
    except Exception as e:
        return {
            "order_post_result": {"success": False, "status": "error", "error": str(e)},
            "close_skipped": str(e),
        }


# ============================================================
# CLI
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="Polymarket CLOB trade execution runner (V2)")
    ap.add_argument("--market-slug", type=str, default=None, help="Gamma event slug")
    ap.add_argument("--force-side", type=str, default=None, choices=["UP", "DOWN"])
    ap.add_argument("--start-equity", type=float, default=None)
    ap.add_argument("--risk-frac", type=float, default=None)
    ap.add_argument("--max-notional-usd", type=float, default=None)
    ap.add_argument("--close-token-id", type=str, default=None)
    ap.add_argument("--close-shares", type=float, default=None)
    ap.add_argument("--close-limit-price", type=float, default=None)
    ap.add_argument("--execute", action="store_true", default=False)
    args = ap.parse_args()

    if not args.market_slug:
        print(json.dumps({"error": "--market-slug is required"}))
        sys.exit(1)

    if args.close_token_id:
        result = close_position(args)
    elif args.force_side:
        result = open_position(args)
    else:
        print(json.dumps({"error": "specify --force-side to open or --close-token-id to close"}))
        sys.exit(1)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
