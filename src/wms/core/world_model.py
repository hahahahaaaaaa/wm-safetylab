"""Small request helpers retained by the native prediction adapters.

Prediction arrays and images remain files. They are not translated into a common
latent state, hashed again, or treated as a safety verdict.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Protocol


class PredictionAdapter(Protocol):
    def describe(self) -> dict: ...
    def predict(self, request: dict, output_dir: Path) -> dict: ...


def content_hash(value):
    """Optional utility for research scripts; not a runtime requirement."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    """Optional utility for research scripts; not called by the run protocol."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifest(manifest):
    if not isinstance(manifest, dict) or not manifest.get("model_id"):
        raise ValueError("Prediction adapter needs a model_id")
    return manifest


def validate_request(request, manifest=None):
    if not isinstance(request.get("inputs"), dict):
        raise ValueError("Prediction inputs must be an object")
    horizon = request.get("horizon", {})
    if type(horizon.get("steps")) is not int or horizon["steps"] < 1:
        raise ValueError("Prediction horizon.steps must be a positive integer")
    if not horizon.get("unit"):
        raise ValueError("Prediction horizon needs its unit")
    return request


def validate_prediction_bundle(bundle, output_dir=None, **kwargs):
    if not isinstance(bundle, dict) or not isinstance(bundle.get("outputs"), list):
        raise ValueError("Prediction needs an outputs list")
    if output_dir is not None:
        for item in bundle["outputs"]:
            if not (Path(output_dir) / item["relative_path"]).is_file():
                raise FileNotFoundError(item["relative_path"])
    return bundle


def write_prediction_bundle(output_dir, request, manifest, artifacts, *, mode,
                            diagnostics=None, provenance=None, claims=None):
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    outputs = []
    for artifact in artifacts:
        item = deepcopy(artifact)
        path = Path(item.pop("path"))
        path = (path if path.is_absolute() else root / path).resolve()
        item["relative_path"] = path.relative_to(root).as_posix()
        outputs.append(item)
    bundle = {"schema_version": "wms.prediction/1.0", "status": "completed",
              **{key: deepcopy(request.get(key)) for key in ("request_id", "model_id", "seed", "horizon")},
              "families": manifest.get("families", []), "representations": manifest.get("representations", []),
              "mode": mode, "closed_loop": False, "safety_verdict": "not_evaluated",
              "outputs": outputs, "diagnostics": diagnostics or {},
              "provenance": provenance or manifest.get("provenance", {}), "claims": claims or []}
    (root / "prediction.json").write_text(json.dumps(bundle, ensure_ascii=False, indent=2,
                                                    allow_nan=False), encoding="utf-8")
    return bundle
