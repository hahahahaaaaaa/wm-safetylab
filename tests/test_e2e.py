"""CPU HTTP-to-worker tests; no model downloads and no GPU required."""
import json
import os
from pathlib import Path
import sys
import time

from fastapi.testclient import TestClient
import pytest

from wms.app.api import create_app


TERMINAL = {"completed", "failed", "cancelled", "interrupted", "timed_out"}


def wait_run(client, run_id, *, states=TERMINAL, timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/runs/{run_id}")
        assert response.status_code == 200, response.text
        run = response.json()
        if run["state"] in states:
            return run
        time.sleep(0.05)
    pytest.fail(f"Run did not reach {states}: {run}")


def submit(client, **config):
    response = client.post("/api/v1/runs", json=config)
    assert response.status_code == 200, response.text
    return response.json()


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(storage_root=tmp_path / "store", worker_python=sys.executable)) as value:
        yield value


@pytest.mark.parametrize("task", ["generate", "predict", "rollout", "plan", "native"])
def test_five_tasks_execute_in_real_subprocess(client, task):
    submitted = submit(client, recipe=f"demo-{task}", metrics=[])
    finished = wait_run(client, submitted["id"])
    assert finished["state"] == "completed", finished
    assert finished["pid"] != os.getpid()
    assert finished["returncode"] == 0
    result = client.get(f"/api/v1/runs/{finished['id']}/result").json()
    assert result["status"] == "completed"
    outputs = client.get(f"/api/v1/runs/{finished['id']}/files").json()["items"]
    for output in outputs:
        file = client.get(f"/api/v1/runs/{finished['id']}/files/{output['path']}")
        assert file.status_code == 200, file.text
        assert file.content
    assert "\"phase\": \"start\"" in client.get(f"/api/v1/runs/{finished['id']}/logs").json()["text"]


def test_failed_task_does_not_block_queue_and_retry_is_new_run(client):
    broken = submit(client, recipe="demo-native", params={"fail": True})
    good = submit(client, recipe="demo-native", inputs={"message": "queue continues"})
    assert wait_run(client, broken["id"])["state"] == "failed"
    assert wait_run(client, good["id"])["state"] == "completed"
    response = client.post(f"/api/v1/runs/{good['id']}/retry")
    assert response.status_code == 200, response.text
    retried = response.json()
    assert retried["id"] != good["id"]
    assert retried["retry_of"] == good["id"]
    assert wait_run(client, retried["id"])["state"] == "completed"


def test_running_task_can_be_cancelled_then_queue_continues(client):
    slow = submit(client, recipe="demo-native", steps=100, params={"delay": 0.2})
    wait_run(client, slow["id"], states={"running"})
    # Wait for the actual worker, not just the queue's running state.
    deadline = time.monotonic() + 10
    while "\"phase\": \"start\"" not in client.get(f"/api/v1/runs/{slow['id']}/logs").json()["text"]:
        assert time.monotonic() < deadline
        time.sleep(0.05)
    assert client.post(f"/api/v1/runs/{slow['id']}/cancel").status_code == 200
    assert wait_run(client, slow["id"])["state"] == "cancelled"
    good = submit(client, recipe="demo-native")
    assert wait_run(client, good["id"])["state"] == "completed"


def test_source_outputs_survive_restart(tmp_path):
    root = tmp_path / "store"
    with TestClient(create_app(storage_root=root, worker_python=sys.executable)) as client:
        run = submit(client, recipe="demo-native", inputs={"message": "persistent"})
        assert wait_run(client, run["id"])["state"] == "completed"
        original = (root / "runs" / run["id"] / "output/result.json").read_bytes()
    with TestClient(create_app(storage_root=root, worker_python=sys.executable)) as client:
        assert client.get(f"/api/v1/runs/{run['id']}").json()["state"] == "completed"
        assert client.get(f"/api/v1/runs/{run['id']}/files/echo.json").json() == {"message": "persistent"}
        assert (root / "runs" / run["id"] / "output/result.json").read_bytes() == original


def test_external_model_recipe_runs_without_core_edit_or_service_import(tmp_path):
    module_dir = tmp_path / "extension"
    module_dir.mkdir()
    (module_dir / "independent_model.py").write_text('''import os, sys
def run(config, output_dir, context):
    context.write_json("observed.json", {"python": sys.executable,
        "config_file": config["runtime"]["config_file"], "gpu": os.environ["CUDA_VISIBLE_DEVICES"]})
    return {"outputs": [{"path": "observed.json", "kind": "json"}]}
''', encoding="utf-8")
    machine = module_dir / "machine.json"
    machine.write_text(json.dumps({"python": sys.executable, "cuda_visible_devices": "2"}), encoding="utf-8")
    registry = module_dir / "recipes.json"
    registry.write_text(json.dumps({"python_paths": ["."], "recipes": {"external-model": {
        "task": "native", "entrypoint": "independent_model:run",
        "runtime": {"config_file": "machine.json", "env": {"CUDA_VISIBLE_DEVICES": "7"}}
    }}}), encoding="utf-8")
    assert "independent_model" not in sys.modules
    with TestClient(create_app(storage_root=tmp_path / "store", registries=[registry])) as client:
        run = submit(client, recipe="external-model")
        finished = wait_run(client, run["id"])
        assert finished["state"] == "completed", finished
        observed = client.get(f"/api/v1/runs/{run['id']}/files/observed.json").json()
        assert Path(observed["python"]).resolve() == Path(sys.executable).resolve()
        assert observed["config_file"] == str(machine)
        assert observed["gpu"] == "7"
    assert "independent_model" not in sys.modules
