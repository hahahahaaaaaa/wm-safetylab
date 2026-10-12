"""Load existing native runtime JSON without package locks or GPU preflight."""
import json
from pathlib import Path


def load_runtime(value):
    if value is None:
        return None
    if isinstance(value, dict):
        raw, base = dict(value), Path.cwd()
    else:
        path = Path(value).resolve()
        raw, base = json.loads(path.read_text(encoding="utf-8-sig")), path.parent
    # Older deployment files keep useful metadata and input defaults in sidecars.
    # Read them as configuration, not as executable-version or integrity gates.
    for name in ("weights", "input"):
        key = name + "_lock"
        if name not in raw and raw.get(key):
            path = Path(raw[key])
            path = path if path.is_absolute() else base / path
            raw[name] = json.loads(path.read_text(encoding="utf-8-sig"))
    return raw
