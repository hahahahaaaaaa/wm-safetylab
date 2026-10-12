"""Plain JSON values and output-file references; no content-addressed storage."""
from pathlib import Path
import json


def json_value(value):
    """Convert paths and native arrays/scalars without importing ML libraries."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return json_value(value.tolist())
    return value


def output_file(path, kind="file", media_type=None):
    item = {"kind": kind, "path": str(path)}
    if media_type:
        item["media_type"] = media_type
    return item


def is_payload(value):
    return isinstance(value, dict) and value.get("kind") in ("inline", "file")


def resolve(value, base=None):
    """Read explicit JSON/text files; all other model values pass through."""
    if not is_payload(value):
        return value
    if value["kind"] == "inline":
        return value["value"]
    path = Path(value.get("path") or value["$file"])
    if not path.is_absolute() and base is not None:
        path = Path(base) / path
    encoding = value.get("encoding")
    if encoding == "json" or path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8-sig"))
    if encoding == "utf-8" or path.suffix == ".txt":
        return path.read_text(encoding="utf-8")
    return path
