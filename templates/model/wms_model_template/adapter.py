"""Runnable CPU teaching predictor, not a neural world-model baseline."""
from __future__ import annotations

import json
import math
from pathlib import Path


def _scalar_vector(value, name):
    if not isinstance(value, (list, tuple)) or len(value) != 1:
        raise ValueError(f"{name} must be a one-element vector")
    scalar = float(value[0])
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


class TeachingStateModel:
    """x[t+1] = x[t] + gain * action[t]; positions and increments in metres.

    The default gain is hand specified. A JSON checkpoint from train.py may
    contain a fitted gain. Neither option is a learned neural WM.
    """

    def __init__(self, config):
        self.config = dict(config)
        self.gain = float(config.get("gain", 1.0))
        self.checkpoint = config.get("checkpoint")
        if self.checkpoint:
            path = Path(self.checkpoint).expanduser().resolve()
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("format") != "wms-teaching-gain/1":
                raise ValueError("This adapter only loads the teaching gain JSON format")
            self.gain = float(value["gain"])
            self.checkpoint = str(path)
        if not math.isfinite(self.gain):
            raise ValueError("gain must be finite")

    def predict(self, request):
        position = _scalar_vector(request["observation"], "observation")
        predictions = []
        for index, action in enumerate(request["actions"]):
            position += self.gain * _scalar_vector(action, f"actions[{index}]")
            predictions.append([position])
        result = {
            "predictions": predictions,
            "reference": request.get("reference"),
            "metadata": {
                "model_kind": "teaching_scalar_transition",
                "learned_neural_world_model": False,
                "prediction_shape": [len(predictions), 1],
                "state_fields": ["position_m"],
                "action_fields": ["position_increment_m"],
                "alignment": "row t is state after action t, initial state excluded",
                "gain": self.gain,
                "checkpoint": self.checkpoint,
            },
        }
        # Keep this off for repeated planner calls; otherwise each candidate
        # would overwrite the same file. A normal predict recipe enables it.
        if request.get("save_predictions", False):
            path = Path(request["output_dir"]) / "prediction.json"
            path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
            result["outputs"] = [{"kind": "prediction", "path": path.name,
                                  "media_type": "application/json"}]
        return result

    def close(self):
        pass
