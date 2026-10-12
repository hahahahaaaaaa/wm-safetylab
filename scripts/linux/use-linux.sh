#!/usr/bin/env bash
# Source this file before installing or running project code on Linux.
WMS_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
WMS_ROOT="${WMS_ROOT:-$WMS_PROJECT_ROOT/.workspace}"
export WMS_PROJECT_ROOT WMS_ROOT
export TMPDIR="$WMS_ROOT/tmp" PIP_CACHE_DIR="$WMS_ROOT/cache/pip"
export UV_CACHE_DIR="$WMS_ROOT/cache/uv" UV_PYTHON_INSTALL_DIR="$WMS_ROOT/runtime/python"
export UV_PYTHON_BIN_DIR="$WMS_ROOT/runtime/bin" npm_config_cache="$WMS_ROOT/cache/npm"
export XDG_CACHE_HOME="$WMS_ROOT/cache" HF_HOME="$WMS_ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub" TORCH_HOME="$WMS_ROOT/cache/torch"
export CUDA_CACHE_PATH="$WMS_ROOT/cache/cuda" TORCHINDUCTOR_CACHE_DIR="$WMS_ROOT/cache/torchinductor"
export TRITON_CACHE_DIR="$WMS_ROOT/cache/triton" MPLCONFIGDIR="$WMS_ROOT/cache/matplotlib"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONUTF8=1
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$UV_PYTHON_BIN_DIR" "$npm_config_cache" "$HF_HUB_CACHE" "$TORCH_HOME" "$CUDA_CACHE_PATH" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$MPLCONFIGDIR"
