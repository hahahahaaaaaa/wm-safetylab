"""Small CPU tests for platform wiring; GPU behavior is tested on the server."""
import json
from pathlib import Path

from wms.engines.tdmpc2.native import _settings, _source


def test_run_references_resolve_checkpoint_and_evidence(tmp_path):
    root = tmp_path / "runs"
    train = root / "run_aa" / "output"
    evaluated = root / "run_bb" / "output"
    destination = root / "run_cc" / "output"
    train.mkdir(parents=True)
    evaluated.mkdir(parents=True)
    for filename in ("initial.pt", "model.pt"):
        (train / filename).write_bytes(b"fixture path only")
    (evaluated / "evaluation.json").write_text("{}")
    inputs = {"source_run": "run_aa", "evaluation_run": "run_bb"}
    assert _source(inputs, destination) == train / "model.pt"
    assert _source({**inputs, "initial_checkpoint": True}, destination) == train / "initial.pt"
    assert _source(inputs, destination, "evaluation_report") == evaluated / "evaluation.json"


def test_platform_device_override_does_not_rewrite_machine_file(tmp_path):
    runtime = tmp_path / "runtime.json"
    original = {"gpu_device": "0", "python": "/machine/python"}
    runtime.write_text(json.dumps(original))
    value, params, inputs = _settings({"runtime": {"config_file": str(runtime),
                                                   "env": {"CUDA_VISIBLE_DEVICES": "2"}},
                                      "params": {"episodes": 10}})
    assert value["gpu_device"] == "2"
    assert params == {"episodes": 10} and inputs == {}
    assert json.loads(runtime.read_text()) == original
