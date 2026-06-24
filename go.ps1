# 快捷实盘:  .\go.ps1           (保守)
#            .\go.ps1 agg       (激进)
Param([string]$Profile = "conservative")
if ($Profile -eq "agg") { $Profile = "aggressive" }
$env:PYTHONUNBUFFERED = 1
Set-Location $PSScriptRoot
while ($true) {
    Write-Host "=== BTC 5m Live ($Profile) starting at $(Get-Date -Format 'yyyy-MM-ddTHH:mm:ssZ') ==="
    try {
        & .venv\Scripts\python -u scripts\test_btc_5m_session_exit_sl.py --profile $Profile --execute 2>&1
    } catch {
        Write-Host "ERROR: $_" -ForegroundColor Red
    }
    Write-Host "=== Exited at $(Get-Date -Format 'yyyy-MM-ddTHH:mm:ssZ'), restarting in 3s... ==="
    Start-Sleep 3
}
