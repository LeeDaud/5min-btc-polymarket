#!/usr/bin/env python3
"""
Polymarket CLOB trade execution runner.
Stateless: receives order parameters via CLI, executes ONE operation, outputs JSON to stdout.

Open position:
  pm_live_trade_runner.py --market-slug <s> --force-side UP|DOWN
      --start-equity 100 --risk-frac 0.05 --max-notional-usd 5 [--execute]

Close position:
  pm_live_trade_runner.py --market-slug <s> --close-token-id <id>
      --close-shares <n> [--close-limit-price <p>] [--execute]
"""

import argparse
import json
import os
import sys
from typing import Any, Optional

import requests
from dotenv import load_dotenv

# Load .env from repo root or current dir
load_dotenv()


# ============================================================
# Authentication
# ============================================================

def get_clob_client() -> Optional[Any]:
    """Create an authenticated ClobClient, deriving L2 creds from wallet if needed."""
    try:
        from py_clob_client.client import ClobClient
        from py_clob_client.constants import POLYGON
        from py_clob_client.clob_types import ApiCreds
    except ImportError:
        print(json.dumps({"error": "py_clob_client not installed"}))
        return None

    key = os.getenv("PM_PRIVATE_KEY") or ""
    funder = os.getenv("PM_FUNDER") or os.getenv("PM_ADDRESS") or None
    sig = int(os.getenv("PM_SIGNATURE_TYPE", "2"))

    if not key:
        print(json.dumps({"error": "missing credentials", "detail": "Set PM_PRIVATE_KEY in .env"}))
        return None

    try:
        c = ClobClient(host="https://clob.polymarket.com", chain_id=POLYGON, key=key, signature_type=sig, funder=funder)

        # First try: use env vars if they look valid
        v2 = os.getenv("PM_API_SECRET") or ""
        if v2 and len(v2) >= 43 and not v2.startswith("0x"):
            v1 = os.getenv("PM_API_KEY") or ""
            v3 = os.getenv("PM_API_PASSPHRASE") or v2
            try:
                c.set_api_creds(ApiCreds(api_key=v1, api_secret=v2, api_passphrase=v3))
                c.get_server_time()
                return c
            except Exception:
                pass

        # Second try: derive L2 creds from wallet signature
        try:
            creds = c.create_or_derive_api_creds()
            if creds:
                c.set_api_creds(creds)
                _ = c.get_server_time()
                return c
        except Exception:
            pass

        print(json.dumps({"error": "auth failed — both env creds and wallet derivation failed"}))
        return None
    except Exception as e:
        print(json.dumps({"error": f"auth failed: {e}"}))
        return None


# ============================================================
# Market resolution
# ============================================================

def fetch_event(slug: str) -> Optional[dict]:
    r = requests.get(
        "https://gamma-api.polymarket.com/events",
        params={"slug": slug},
        timeout=12,
    )
    r.raise_for_status()
    arr = r.json()
    return arr[0] if arr else None


def resolve_token_id(slug: str, side: str) -> tuple[Optional[str], Optional[float], dict]:
    """Return (token_id, reference_price, market_dict) for the given side."""
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
        "slug": slug,
        "side": side,
        "token_id": token_id,
        "outcome_price": price,
        "market_end": m.get("endDate") or m.get("endDateIso"),
        "market_closed": m.get("closed"),
    }


# ============================================================
# Position open
# ============================================================

def open_position(args) -> dict:
    """Open a new position."""
    side = (args.force_side or "").upper()
    if side not in ("UP", "DOWN"):
        return {"error": f"invalid --force-side: {args.force_side}"}

    token_id, ref_price, info = resolve_token_id(args.market_slug, side)
    if token_id is None:
        return info

    # Calculate notional
    max_notional = float(args.max_notional_usd or 5.0)
    if args.risk_frac and args.start_equity:
        risk_based = float(args.start_equity) * float(args.risk_frac)
        max_notional = min(max_notional, risk_based)

    order_type = os.getenv("PM_ORDER_TYPE", "FAK").upper()

    if not args.execute:
        return {
            "dry_run": True,
            "order_post_result": {
                "success": True,
                "status": "dry_run",
                "detail": f"Would {order_type} BUY {side} token={token_id} notional={max_notional:.2f}",
            },
            "token_id": token_id,
            "entry_price": ref_price,
            "dry_run_info": info,
        }

    client = get_clob_client()
    if client is None:
        return {"error": "authentication failed"}

    try:
        from py_clob_client.clob_types import MarketOrderArgs, OrderType
        from py_clob_client.order_builder.constants import BUY

        ot = OrderType.FAK if order_type == "FAK" else OrderType.GTC

        signed = client.create_market_order(MarketOrderArgs(
            token_id=token_id,
            amount=max_notional,
            side=BUY,
            order_type=ot,
        ))
        result = client.post_order(signed)

        # Calculate entry price from fill
        entry_price = ref_price
        if isinstance(result, dict):
            making = float(result.get("makingAmount") or 0)
            taking = float(result.get("takingAmount") or 0)
            if taking > 0:
                entry_price = making / taking / 1e6  # making is in USDC wei
            elif making > 0 and making > 0:
                pass  # keep ref_price fallback

        return {
            "order_post_result": result if isinstance(result, dict) else {"raw": str(result)},
            "token_id": token_id,
            "entry_price": round(entry_price, 6),
        }
    except Exception as e:
        return {"error": f"open order failed: {e}", "token_id": token_id}


# ============================================================
# Position close
# ============================================================

def close_position(args) -> dict:
    """Close an existing position."""
    token_id = args.close_token_id
    shares = float(args.close_shares or 0)
    order_type = os.getenv("PM_CLOSE_ORDER_TYPE", "FAK").upper()
    limit_price = float(args.close_limit_price) if args.close_limit_price else None

    if not token_id or shares <= 0:
        # Check if shares are zero — this is the "zero_effective_shares" case
        return {
            "order_post_result": {"success": False, "status": "skipped"},
            "close_skipped": "zero_effective_shares",
        }

    if not args.execute:
        limit_str = f"limit={limit_price:.6f}" if limit_price else "market"
        return {
            "dry_run": True,
            "order_post_result": {
                "success": True,
                "status": "dry_run",
                "detail": f"Would {order_type} SELL token={token_id} shares={shares:.8f} {limit_str}",
            },
        }

    client = get_clob_client()
    if client is None:
        return {"error": "authentication failed"}

    try:
        from py_clob_client.clob_types import MarketOrderArgs, OrderArgs, OrderType
        from py_clob_client.order_builder.constants import SELL

        if order_type == "FAK":
            ot = OrderType.FAK
            signed = client.create_market_order(MarketOrderArgs(
                token_id=token_id,
                amount=shares,
                side=SELL,
                order_type=ot,
            ))
            result = client.post_order(signed)
        else:
            # GTC limit sell
            price = limit_price or 0.01
            signed = client.create_order(OrderArgs(
                token_id=token_id,
                price=price,
                size=shares,
                side=SELL,
            ))
            result = client.post_order(signed)

        close_skipped = None
        if isinstance(result, dict):
            status = str(result.get("status") or "").lower()
            if status == "skipped" or not result.get("success"):
                close_skipped = "zero_effective_shares"

        out = {
            "order_post_result": result if isinstance(result, dict) else {"raw": str(result)},
        }
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
    ap = argparse.ArgumentParser(description="Polymarket CLOB trade execution runner")
    # Open args
    ap.add_argument("--market-slug", type=str, default=None, help="Gamma event slug")
    ap.add_argument("--force-side", type=str, default=None, choices=["UP", "DOWN"])
    ap.add_argument("--start-equity", type=float, default=None)
    ap.add_argument("--risk-frac", type=float, default=None)
    ap.add_argument("--max-notional-usd", type=float, default=None)
    # Close args
    ap.add_argument("--close-token-id", type=str, default=None)
    ap.add_argument("--close-shares", type=float, default=None)
    ap.add_argument("--close-limit-price", type=float, default=None)
    # Global
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
        print(json.dumps({
            "error": "specify --force-side to open or --close-token-id to close"
        }))
        sys.exit(1)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
