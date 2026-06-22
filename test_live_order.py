import os, json, time, requests
from dotenv import load_dotenv
load_dotenv()

from py_clob_client.client import ClobClient
from py_clob_client.constants import POLYGON
from py_clob_client.clob_types import ApiCreds, MarketOrderArgs, OrderType
from py_clob_client.order_builder.constants import BUY

c = ClobClient(
    host="https://clob.polymarket.com", chain_id=POLYGON,
    key=os.getenv("PM_PRIVATE_KEY") or "",
    signature_type=2,
)
c.set_api_creds(ApiCreds(
    api_key=os.getenv("PM_API_KEY") or "",
    api_secret=os.getenv("PM_API_SECRET") or "",
    api_passphrase=os.getenv("PM_API_PASSPHRASE") or "",
))

now = int(time.time())
slug = f"btc-updown-5m-{now - now % 300}"
print(f"Market: {slug}")

r = requests.get("https://gamma-api.polymarket.com/events", params={"slug": slug}, timeout=15)
mkt = r.json()[0]["markets"][0]
tids = json.loads(mkt["clobTokenIds"])
tid = str(tids[0])
print(f"Token: {tid[:30]}...")
print(f"Amount: $1 (FAK BUY)")
print()

signed = c.create_market_order(MarketOrderArgs(
    token_id=tid, amount=1.0, side=BUY, order_type=OrderType.FAK,
))
result = c.post_order(signed, order_type=OrderType.FAK)
print(json.dumps(result, indent=2, default=str))
