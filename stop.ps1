# Stop only this project's recorded background instance after all runs end.
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'scripts\windows\use-workspace.ps1')
$project = $PSScriptRoot
Set-Location -LiteralPath $project
$pidFile = Join-Path $project 'logs\server.pid'
if (!(Test-Path -LiteralPath $pidFile)) { Write-Host 'No recorded background service. Use Ctrl+C for a foreground service.'; exit 0 }
$launcherId = [int](Get-Content -LiteralPath $pidFile)
$launcher = Get-CimInstance Win32_Process -Filter "ProcessId = $launcherId"
if ($null -eq $launcher) { Write-Host 'Recorded launcher is no longer running.'; exit 0 }
function Test-ProjectService($item) {
    return $item.CommandLine -like '*wms serve*' -and $item.CommandLine.Contains($project)
}
if (!(Test-ProjectService $launcher)) { throw 'PID identity changed. Refusing to stop an unrelated process.' }
$check = @"
import httpx
r = httpx.get('http://127.0.0.1:8765/api/v1/runs', trust_env=False, timeout=5)
r.raise_for_status()
assert all(x['state'] in {'completed','failed','cancelled','interrupted'} for x in r.json()['items']), 'Runs are active; cancel or wait in Web before stopping'
"@
& (Join-Path $project '.venv\Scripts\python.exe') -c $check
if ($LASTEXITCODE -ne 0) { throw 'Service was not stopped; resolve the check above first.' }
$children = @(Get-CimInstance Win32_Process -Filter "ParentProcessId = $launcherId")
foreach ($child in $children) {
    if (!(Test-ProjectService $child)) { throw 'Unrecognized child process. Refusing to stop it.' }
}
foreach ($child in $children) { Stop-Process -Id $child.ProcessId -ErrorAction SilentlyContinue }
Stop-Process -Id $launcherId -ErrorAction SilentlyContinue
Write-Host 'WM-SafetyLab background service stopped. Saved runs and artifacts remain in data/.'
