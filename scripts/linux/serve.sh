#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/use-linux.sh"
cd "$WMS_PROJECT_ROOT"
exec "$WMS_PROJECT_ROOT/.venv/bin/python" -m wms serve \
  --host 127.0.0.1 --port "${WMS_PORT:-8765}" \
  --storage-root "${WMS_STORAGE_ROOT:-$WMS_PROJECT_ROOT/data-linux}" \
  --worker-python "${WMS_WORKER_PYTHON:-$WMS_PROJECT_ROOT/.venv/bin/python}" "$@"
