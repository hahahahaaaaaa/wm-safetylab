param([int]$Port = 8765, [string]$StorageRoot = '', [string]$WorkerPython = '')
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'scripts\windows\use-workspace.ps1')
Set-Location -LiteralPath $PSScriptRoot
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (!$StorageRoot) { $StorageRoot = Join-Path $PSScriptRoot 'data' }
if (!$WorkerPython) { $WorkerPython = $python }
Write-Host "WM SafetyLab: http://127.0.0.1:$Port (Ctrl+C to stop)"
& $python -m wms serve --storage-root $StorageRoot --worker-python $WorkerPython --port $Port
exit $LASTEXITCODE
