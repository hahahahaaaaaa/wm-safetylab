"""Exercise the copyable onboarding code against the current platform contracts."""
import json
from pathlib import Path
import sys
import time

from fastapi.testclient import TestClient
import pytest

from wms.app.api import create_app
from wms.core.registry import Registry


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "templates/model"
ENVIRONMENT = ROOT / "templates/environment"


def wait_completed(client, run_id, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/runs/{run_id}")
        assert response.status_code == 200, response.text
        run = response.json()
        if run["state"] not in {"queued", "running", "cancelling"}:
            assert run["state"] == "completed", run
            return run
        time.sleep(0.05)
    pytest.fail(f"Onboarding worker did not finish: {run_id}")


def submit(client, configuration):
    response = client.post("/api/v1/runs", json=configuration)
    assert response.status_code == 200, response.text
    return response.json()


def test_template_registries_coexist_and_resolve_paths_without_gym_import():
    registry = Registry(paths=[MODEL / "registry.json", ENVIRONMENT / "registry.json"])
    prediction = registry.resolve({"recipe": "onboarding-model-predict"})
    rollout = registry.resolve({"recipe": "template-pendulum-rollout"})
    assert str(MODEL.resolve()) in prediction["runtime"]["python_paths"]
    assert str(ENVIRONMENT.resolve()) in rollout["runtime"]["python_paths"]
    assert prediction["model"]["entrypoint"] == "wms_model_template.adapter:TeachingStateModel"
    assert rollout["environment"]["entrypoint"] == "lab_env.adapter:PendulumAdapter"
    assert prediction["inputs"]["observation"] == [0.0]
    assert rollout["model"]["config"]["torque_Nm"] == 0.5
    # An unimplemented real-backend scaffold must not pretend to be configured.
    if not (MODEL / "runtime.local.json").exists():
        assert registry.recipe("onboarding-model-upstream")["configured"] is False


def test_teaching_train_checkpoint_predict_and_independent_evaluation(tmp_path):
    storage = tmp_path / "store"
    with TestClient(create_app(storage_root=storage, worker_python=sys.executable,
                              registries=[MODEL / "registry.json"])) as client:
        training = submit(client, {"recipe": "onboarding-model-train", "seed": 9})
        trained = wait_completed(client, training["id"])
        checkpoint = storage / "runs" / trained["id"] / "output/checkpoint.json"
        values = json.loads(checkpoint.read_text(encoding="utf-8"))
        assert values["gain"] == pytest.approx(0.8)
        assert values["learned_neural_world_model"] is False
        prediction = submit(client, {
            "recipe": "onboarding-model-predict",
            "model": {"entrypoint": "wms_model_template.adapter:TeachingStateModel",
                      "config": {"checkpoint": str(checkpoint)}},
            "inputs": {"reference": [[0.08], [0.24], [0.16]]},
            "metrics": [],
        })
        predicted = wait_completed(client, prediction["id"])
        source = storage / "runs" / predicted["id"] / "output"
        before = {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
        result = predicted["result"]
        assert [row[0] for row in result["data"]["predictions"]] == pytest.approx([0.08, 0.24, 0.16])
        assert Path(result["data"]["metadata"]["checkpoint"]).resolve() == checkpoint.resolve()
        response = client.post(f"/api/v1/runs/{predicted['id']}/evaluate", json={"metrics": [{
            "entrypoint": "wms_model_template.metrics:prediction_metrics", "version": "1", "unit": "m"
        }]})
        assert response.status_code == 200, response.text
        evaluation = wait_completed(client, response.json()["id"])
        assert evaluation["source_run"] == predicted["id"]
        assert evaluation["result"]["metric_errors"] == {}
        assert evaluation["result"]["metrics"]["template_state_mae_m"] == pytest.approx(0.0, abs=1e-12)
        assert (storage / "runs" / evaluation["id"] / "output/prediction_errors.json").is_file()
        assert {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()} == before
        assert not (source / "prediction_errors.json").exists()


def test_pendulum_adapter_real_reset_step_and_truncation(monkeypatch):
    pytest.importorskip("gymnasium", reason="Gymnasium is an optional worker environment dependency")
    monkeypatch.syspath_prepend(str(ENVIRONMENT))
    from lab_env.adapter import PendulumAdapter
    from lab_env.policy import ConstantPolicy

    environment = PendulumAdapter({"max_episode_steps": 2})
    policy = ConstantPolicy({"torque_Nm": 0.5})
    try:
        initial = environment.reset(42, {})
        assert len(initial) == 3
        action = policy.act(initial)
        assert action == [0.5]
        first = environment.step(action)
        assert set(first) >= {"obs", "reward", "terminated", "truncated", "info"}
        assert first["info"]["safety_signal_status"] == "unavailable"
        assert "cost" not in first
        assert first["truncated"] is False
        second = environment.step(action)
        assert second["truncated"] is True
        assert environment.reset(42, {}) == initial
        assert environment.step(action) == first
        with pytest.raises(ValueError, match="within"):
            environment.step([2.1])
        with pytest.raises(ValueError, match="one finite"):
            environment.step([0.5, 0.5])
        assert environment.metadata()["dt_s"] == pytest.approx(0.05)
    finally:
        environment.close()
        environment.close()
    with pytest.raises(RuntimeError, match="closed"):
        environment.reset(42, {})
