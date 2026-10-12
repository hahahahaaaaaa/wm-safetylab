"""Native training wrapper plus an actually runnable scalar least-squares example."""
from __future__ import annotations

from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
import platform
import random
import sys


def fit_teaching_gain(inputs, params, output_dir, context):
    """Fit gain from explicit transition rows, or clearly labelled synthetic rows.

    Real backend wrappers may use this same function signature. This small
    teaching backend does not import any training frameworks.
    """
    source = inputs.get("dataset_path")
    if source:
        path = Path(source).expanduser().resolve()
        rows = json.loads(path.read_text(encoding="utf-8"))
        dataset = {"path": str(path), "kind": "supplied_transition_rows"}
    else:
        rng = random.Random(int(params.get("seed", 0)))
        true_gain = float(params.get("synthetic_gain", 0.8))
        rows = []
        for _ in range(int(params.get("samples", 64))):
            state, action = rng.uniform(-1, 1), rng.uniform(-0.2, 0.2)
            rows.append({"observation": [state], "action": [action],
                         "next_observation": [state + true_gain * action]})
        dataset = {"kind": "synthetic_teaching_data", "synthetic_gain": true_gain}
    context.check_cancelled()
    denominator = sum(float(row["action"][0]) ** 2 for row in rows)
    if not rows or denominator <= 0:
        raise ValueError("Training needs non-empty transition rows with non-zero actions")
    gain = sum(float(row["action"][0]) *
               (float(row["next_observation"][0]) - float(row["observation"][0]))
               for row in rows) / denominator
    path = Path(output_dir) / "checkpoint.json"
    path.write_text(json.dumps({"format": "wms-teaching-gain/1", "gain": gain,
                               "learned_neural_world_model": False}, indent=2, allow_nan=False),
                    encoding="utf-8")
    # Save exactly what trained this checkpoint, not a summary of generated data.
    (Path(output_dir) / "training_data.json").write_text(
        json.dumps(rows, indent=2, allow_nan=False), encoding="utf-8")
    return {"checkpoint": str(path), "dataset": dataset,
            "training_rows": len(rows), "gain": gain,
            "algorithm": "scalar_ordinary_least_squares", "training_only": True}


def run(config, output_dir, context):
    """Platform task:native entrypoint; invoke one chosen backend in this worker."""
    params = {**config.get("params", {}), "seed": config.get("seed", 0)}
    entrypoint = params.pop("train_entrypoint", "wms_model_template.train:fit_teaching_gain")
    module_name, separator, function_name = entrypoint.partition(":")
    if not separator:
        raise ValueError("train_entrypoint must be module:function")
    backend = getattr(importlib.import_module(module_name), function_name)
    context.event("progress", {"phase": "train", "backend": entrypoint})
    trained = backend(config.get("inputs", {}), params, Path(output_dir), context)
    checkpoint = Path(trained["checkpoint"]).resolve()
    relative = checkpoint.relative_to(Path(output_dir).resolve())
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    provenance = {
        "backend": entrypoint, "seed": config.get("seed", 0),
        "python": sys.executable, "python_version": platform.python_version(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": config.get("inputs", {}), "params": params,
        "upstream_revision": params.get("upstream_revision", "not_applicable_teaching_example"),
        "parent_checkpoint": params.get("parent_checkpoint"),
        "backend_result": trained,
    }
    context.write_json("training.json", provenance)
    outputs = [{"kind": "checkpoint", "path": relative.as_posix()},
               {"kind": "provenance", "path": "training.json"}]
    if (Path(output_dir) / "training_data.json").is_file():
        outputs.append({"kind": "dataset", "path": "training_data.json"})
    return {"outputs": outputs, "data": {"training": trained}}
