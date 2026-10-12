#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
ROOT="${WMS_ROOT:?Set WMS_ROOT to the absolute project storage root; source scripts/linux/use-linux.sh first.}"
SOURCE="$ROOT/repos/ManiSkill-tdmpc2-20261012"
ENVIRONMENT="$ROOT/envs/tdmpc2-maniskill-20261012"
BASE="${WMS_TDMPC2_BASE:-$ROOT/envs/maniskill-pushcube}"
FIX_CUDA_STUB="${WMS_FIX_CUDA_STUB:-false}"
COMMIT="62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3"
case "$ROOT" in
  /*) ;;
  *) echo 'WMS_ROOT must be an absolute path.' >&2; exit 1 ;;
esac
case "$FIX_CUDA_STUB" in
  true|false) ;;
  *) echo 'WMS_FIX_CUDA_STUB must be true or false (default: false).' >&2; exit 1 ;;
esac
if [ ! -x "$BASE/bin/python" ]; then
  echo "Missing ManiSkill base Python: $BASE/bin/python. Prepare the base environment first or set WMS_TDMPC2_BASE." >&2
  exit 1
fi
BASE_SITE="$("$BASE/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
if [ ! -d "$BASE_SITE" ]; then
  echo "Missing ManiSkill base site-packages: $BASE_SITE" >&2
  exit 1
fi
export TMPDIR="$ROOT/tmp" PIP_CACHE_DIR="$ROOT/cache/pip" XDG_CACHE_HOME="$ROOT/cache"
export TORCH_HOME="$ROOT/cache/torch" PYTHONDONTWRITEBYTECODE=1
mkdir -p "$PROJECT/results/tdmpc2-integration-20261012" "$TMPDIR"
if [ ! -d "$SOURCE/.git" ]; then
  git clone --depth 1 --filter=blob:none --sparse https://github.com/mani-skill/ManiSkill.git "$SOURCE"
  git -C "$SOURCE" sparse-checkout set examples/baselines/tdmpc2
  git -C "$SOURCE" fetch --depth 1 origin "$COMMIT"
  git -C "$SOURCE" checkout --detach "$COMMIT"
fi
test "$(git -C "$SOURCE" rev-parse HEAD)" = "$COMMIT" || { echo 'Existing checkout differs; preserve it and choose a separate source directory.'; exit 1; }
git -C "$SOURCE" rev-parse HEAD
git -C "$SOURCE" status --short
if [ ! -x "$ENVIRONMENT/bin/python" ]; then
  "$BASE/bin/python" -m venv "$ENVIRONMENT"
fi
# Reuse the already tested large CUDA/physics stack without changing its files.
# Additional dependencies are installed into this environment only. The complete
# base and overlay package versions are captured below; this overlay is not an
# independently portable environment and the final tutorial documents rebuilding.
SITE="$("$ENVIRONMENT/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
mkdir -p "$SITE"
printf '%s\n' "$BASE_SITE" > "$SITE/wms_maniskill_base.pth"
"$ENVIRONMENT/bin/python" -m pip install 'hydra-core==1.3.2' 'hydra-submitit-launcher==1.2.0' 'termcolor==2.4.0' 'tensordict==0.8.3' 'torchrl==0.8.1' 'moviepy==1.0.3' 'wandb==0.19.11'
"$ENVIRONMENT/bin/python" -m pip freeze > "$PROJECT/results/tdmpc2-integration-20261012/requirements-effective.lock.txt"
"$ENVIRONMENT/bin/python" -m pip check
"$ENVIRONMENT/bin/python" -c 'import torch, mani_skill, sapien, torchrl, tensordict; print("torch",torch.__version__,"cuda",torch.cuda.is_available()); print("torchrl",torchrl.__version__,"tensordict",tensordict.__version__); print("mani_skill",mani_skill.__file__)'

# SAPIEN's automatic download defaults outside the project. Prepare the exact
# release ourselves. A locally downloaded archive can be copied to ARCHIVE first.
PHYSX="$ROOT/cache/physx/105.1-physx-5.3.1.patch0"
ARCHIVE="$ROOT/downloads/physx-gpu-105.1-local.zip"
mkdir -p "$PHYSX" "$ROOT/downloads"
if [ ! -f "$PHYSX/libPhysXGpu_64.so" ]; then
  if [ ! -f "$ARCHIVE" ]; then
    curl -fL -C - --connect-timeout 20 --max-time 900 \
      https://github.com/sapien-sim/physx-precompiled/releases/download/105.1-physx-5.3.1.patch0/linux-so.zip -o "$ARCHIVE"
  fi
  printf '%s  %s\n' 167a01aad7381afef963b89169968c289e7b653880a7a823c116d87ee5c00fc6 "$ARCHIVE" | sha256sum -c -
  unzip -o "$ARCHIVE" -d "$PHYSX"
fi
# Opt in only after diagnosing libcuda.so resolving to a CUDA toolkit stub.
# This creates a project-local link; it never changes global driver configuration.
if [ "$FIX_CUDA_STUB" = true ]; then
  DRIVER_LIBRARY="${WMS_CUDA_DRIVER_LIBRARY:-/usr/lib/x86_64-linux-gnu/libcuda.so.1}"
  if [ ! -f "$DRIVER_LIBRARY" ]; then
    echo "Missing real CUDA driver library: $DRIVER_LIBRARY" >&2
    exit 1
  fi
  mkdir -p "$ROOT/cache/nvidia-driver"
  ln -sf "$DRIVER_LIBRARY" "$ROOT/cache/nvidia-driver/libcuda.so"
  echo "CUDA lookup link prepared. Set worker LD_LIBRARY_PATH to $ROOT/cache/nvidia-driver when needed."
fi
