$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'scripts\windows\use-workspace.ps1')
Set-Location -LiteralPath $PSScriptRoot
& "$PSScriptRoot\.venv\Scripts\python.exe" -m wms run native-modelscope --wait
exit $LASTEXITCODE
