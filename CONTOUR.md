# BTC 5m Skill Contour (single source of truth)

## Canonical execution path
- Strategy runner (canonical):
  - `scripts/test_btc_5m_session_exit_sl.py`
- Unified control entrypoint:
  - `scripts/btc5m_ctl.sh` (`start|status|stop|report|logs`)
- Compatibility wrapper (deprecated path, forwards to canonical):
  - `scripts/run_btc_5m_threshold_test.py`
- Chat/start helper:
  - `scripts/btc5m_hot.sh`
- Watch helper:
  - `scripts/watch_btc_5m_threshold_and_enter.sh`
- PnL/report utility:
  - `scripts/btc5m_report.py`
- Latest-run completion reporter:
  - `scripts/btc5m_latest_report.py`
- Optional docker control:
  - `scripts/btc5m_docker.sh`

## External dependency boundary
- Order placement/close engine (self-contained):
  - `src/live/pm_live_trade_runner.py`
- Auth source:
  - `.env` in repo root (or `BTC5M_ENV_FILE`)

## Runtime artifacts
- Primary runtime dir (skill-isolated):
  - `skills/btc-5m-live/runtime`
- BTC 5m run logs follow `btc5m_*` naming.

## Isolation guidance
- Keep BTC 5m cron/checkers scoped to this skill naming (`btc5m-*`).
- Avoid creating generic watchers in unrelated topics/chats.
- Keep all new BTC 5m automation pointing to canonical runner only.
- Active completion cron in this contour:
  - `btc5m-completion-autoreport-topic184` (`36d3b9e6-4638-4e93-80f6-abb268ebbe57`)
