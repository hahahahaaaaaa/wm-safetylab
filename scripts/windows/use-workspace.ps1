# Dot-source before installation or execution. All generated files stay in the workspace.
$WmsProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
if (Test-Path -LiteralPath 'K:\world-model\use-k-drive.ps1') {
    . 'K:\world-model\use-k-drive.ps1'
} else {
    $wmsWorkspace = if ($env:WMS_WORKSPACE_ROOT) { $env:WMS_WORKSPACE_ROOT } else { Join-Path $WmsProjectRoot '.workspace' }
    $wmsCaches = @{
        TEMP='tmp'; TMP='tmp'; PIP_CACHE_DIR='cache\pip'; npm_config_cache='cache\npm'
        UV_CACHE_DIR='cache\uv'; UV_PYTHON_INSTALL_DIR='runtime\python'; UV_PYTHON_BIN_DIR='runtime\bin'
        XDG_CACHE_HOME='cache'; HF_HOME='cache\huggingface'; HF_HUB_CACHE='cache\huggingface\hub'
        TORCH_HOME='cache\torch'; CUDA_CACHE_PATH='cache\cuda'; TORCHINDUCTOR_CACHE_DIR='cache\torchinductor'
        TRITON_CACHE_DIR='cache\triton'; MPLCONFIGDIR='cache\matplotlib'; WANDB_DIR='cache\wandb'
    }
    foreach ($entry in $wmsCaches.GetEnumerator()) {
        $cachePath = Join-Path $wmsWorkspace $entry.Value
        New-Item -ItemType Directory -Path $cachePath -Force | Out-Null
        [Environment]::SetEnvironmentVariable($entry.Key, $cachePath, 'Process')
    }
}
$env:PYTHONUTF8='1'
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONDONTWRITEBYTECODE='1'
