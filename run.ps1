Param(
    [string]$Profile = "conservative",
    [float]$Stake = 1,
    [int]$TimeoutMin = 60
)
cd D:\AAA-Project\026-polymarket
$env:PYTHONUNBUFFERED = 1
while ($true) {
    try {
        & .venv\Scripts\python -u scripts\test_btc_5m_session_exit_sl.py --profile $Profile --stake-usd $Stake --entry-timeout-min $TimeoutMin --execute 2>&1
    } catch {
        Write-Host "ERROR: $_" -ForegroundColor Red
        Start-Sleep 5
    }
}
