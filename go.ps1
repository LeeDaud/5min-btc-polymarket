# 快捷实盘:  .\go.ps1           (保守)
#            .\go.ps1 agg       (激进)
Param([string]$Profile = "conservative")
if ($Profile -eq "agg") { $Profile = "aggressive" }
$env:PYTHONUNBUFFERED = 1
Set-Location $PSScriptRoot
& .venv\Scripts\python -u scripts\test_btc_5m_session_exit_sl.py --profile $Profile --execute 2>&1
