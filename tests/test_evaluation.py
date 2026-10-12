import json
import os
from pathlib import Path
import sys
import time

import pytest
from fastapi.testclient import TestClient

from wms.app.api import create_app
from test_e2e import wait_run, submit

HERE = Path(__file__).parent


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(storage_root=tmp_path / "store", worker_python=sys.executable)) as client:
        yield client


def source(client, metrics=None):
    run = submit(client, recipe="demo-native", runtime={"python_paths": [str(HERE)]}, metrics=metrics or [])
    return wait_run(client, run["id"])


def evaluate(client, run, **options):
    response = client.post(f"/api/v1/runs/{run['id']}/evaluate", json=options)
    assert response.status_code == 200, response.text
    return response.json()


def spec(name="metric", **params):
    return {"entrypoint": f"worker_fixtures:{name}", "params": params, "unit": "test", "version": "1"}


def test_auto_manual_scores_are_independent_and_source_unchanged(client):
    run = source(client, [spec()])
    raw = client.get(f"/api/v1/runs/{run['id']}/files/result.json").content
    assert run["result"]["metrics"] == {}
    auto = wait_run(client, run["automatic_evaluation_id"])
    assert auto["source_run"] == run["id"] and auto["task"] == "evaluate"
    assert auto["pid"] != run["pid"] and auto["pid"] != os.getpid()
    assert auto["result"]["metrics"] == {"score": 1}
    started = time.monotonic()
    manual = evaluate(client, run, metrics=[spec(delay=1.5, value=2)])
    assert time.monotonic() - started < 1
    assert manual["id"] != auto["id"]
    assert wait_run(client, manual["id"])["result"]["metrics"] == {"score": 2}
    assert client.get(f"/api/v1/runs/{run['id']}/files/result.json").content == raw
    assert len(client.get(f"/api/v1/runs/{run['id']}/evaluations").json()["items"]) == 2
    store = client.app.state.services["store"]
    assert not (store.output_dir(run["id"]) / "metric-worker.json").exists()
    assert (store.output_dir(manual["id"]) / "metric-worker.json").is_file()


def test_partial_error_is_latest_no_old_backfill_and_explicit_selection(client):
    run = source(client)
    first = wait_run(client, evaluate(client, run, metrics=[spec(key="old", value=9)])["id"])
    latest = wait_run(client, evaluate(client, run, metrics=[spec(key="new", value=2), spec("bad_metric")])["id"])
    assert latest["evaluation_status"] == "partial_failed"
    response = client.post("/api/v1/comparisons", json={"run_ids": [run["id"]]}).json()
    row = response["rows"][0]
    assert row["evaluation"]["id"] == latest["id"]
    assert row["metrics"] == {"new": 2} and row["metric_errors"]
    selected = client.post("/api/v1/comparisons", json={"run_ids": [run["id"]], "evaluations": {run["id"]: first["id"]}}).json()
    assert selected["rows"][0]["metrics"] == {"old": 9}
    assert selected["rows"][0]["latest_evaluation"]["id"] == latest["id"]


def test_evaluation_crash_timeout_cancel_do_not_change_model(client):
    run = source(client)
    raw = client.get(f"/api/v1/runs/{run['id']}/files/result.json").content
    crashed = wait_run(client, evaluate(client, run, metrics=[spec("crash_metric")])["id"])
    assert crashed["state"] == "failed"
    pending_comparison = client.post("/api/v1/comparisons", json={"run_ids": [run["id"]]}).json()
    assert pending_comparison["rows"][0]["evaluation"]["kind"] == "none"
    timed = wait_run(client, evaluate(client, run, metrics=[spec(delay=20)], timeout_s=0.6)["id"])
    assert timed["state"] == "failed" and timed["error_code"] == "TIMEOUT"
    cancelled = evaluate(client, run, metrics=[spec(delay=20)])
    wait_run(client, cancelled["id"], states={"running"})
    client.post(f"/api/v1/runs/{cancelled['id']}/cancel")
    assert wait_run(client, cancelled["id"])["state"] == "cancelled"
    assert client.get(f"/api/v1/runs/{run['id']}").json()["state"] == "completed"
    assert client.get(f"/api/v1/runs/{run['id']}/files/result.json").content == raw
    success = wait_run(client, evaluate(client, run, metrics=[spec()])["id"])
    assert success["state"] == "completed"
    later = wait_run(client, evaluate(client, run, metrics=[spec("crash_metric")])["id"])
    row = client.post("/api/v1/comparisons", json={"run_ids": [run["id"]]}).json()["rows"][0]
    assert row["evaluation"]["id"] == success["id"]
    assert row["latest_evaluation"]["id"] == later["id"] and row["warnings"]


def test_legacy_projection_and_protocol_mismatch(client):
    old, new = source(client), source(client)
    store = client.app.state.services["store"]
    original = (store.output_dir(old["id"]) / "result.json").read_bytes()
    store.update(old["id"], record_version=2, result={**old["result"], "metrics": {"legacy_score": 7}})
    wait_run(client, evaluate(client, new, metrics=[spec()])["id"])
    comparison = client.post("/api/v1/comparisons", json={"run_ids": [old["id"], new["id"]]}).json()
    assert comparison["rows"][0]["evaluation"]["kind"] == "legacy"
    assert comparison["rows"][0]["metrics"] == {"legacy_score": 7}
    assert comparison["warnings"]
    assert (store.output_dir(old["id"]) / "result.json").read_bytes() == original


def test_auto_evaluation_enqueues_at_tail_and_is_not_duplicated(tmp_path):
    root = tmp_path / "store"
    with TestClient(create_app(storage_root=root, worker_python=sys.executable)) as client:
        first = submit(client, recipe="demo-native", steps=5, params={"delay": 0.1},
                       runtime={"python_paths": [str(HERE)]}, metrics=[spec()])
        second = submit(client, recipe="demo-native", steps=5, params={"delay": 0.1})
        first = wait_run(client, first["id"])
        auto = wait_run(client, first["automatic_evaluation_id"])
        second = wait_run(client, second["id"])
        assert second["finished_at"] <= auto["started_at"]
    with TestClient(create_app(storage_root=root, worker_python=sys.executable)) as client:
        items = client.get(f"/api/v1/runs/{first['id']}/evaluations").json()["items"]
        assert len(items) == 1 and items[0]["id"] == auto["id"]


def test_nonzero_worker_exit_cannot_claim_completed(client):
    run = submit(client, task="native", entrypoint="worker_fixtures:false_completed",
                 runtime={"python_paths": [str(HERE)]}, metrics=[spec()])
    finished = wait_run(client, run["id"])
    assert finished["state"] == "failed"
    assert client.get(f"/api/v1/runs/{run['id']}/evaluations").json()["items"] == []


def test_evaluation_machine_file_overrides_source_environment(client, tmp_path):
    run = source(client)
    machine = tmp_path / "metric-runtime.json"
    machine.write_text(json.dumps({"python": sys.executable, "env": {"WMS_TEST_MACHINE": "evaluation"}}))
    score = evaluate(client, run, metrics=[spec()], runtime={"config_file": str(machine)})
    assert score["config"]["runtime"]["env"]["WMS_TEST_MACHINE"] == "evaluation"
    assert score["config"]["runtime"]["config_file"] == str(machine.resolve())
    assert wait_run(client, score["id"])["state"] == "completed"
