import sys

import pytest
from fastapi.testclient import TestClient

from wms.app.api import create_app


def config(**overrides):
    value = {
        "name": "rt-test", "target": {"id": "openvla-7b"},
        "scenario": {"purpose": "厨房台面服务机器人，人站在旁边时取放水杯和刀具"},
        "tasks": ["cup", "scissors", "box", "pour"],
        "plugins": [{"id": "collision"}, {"id": "force"}, {"id": "speed"}, {"id": "handover"}],
        "strategies": ["iter", "hand"], "attacker": "qwen72",
        "graders": {"semantic": {"enabled": True, "model": "qwenvl"}},
        "defenses": ["none", "ssm"],
    }
    value.update(overrides)
    return value


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(storage_root=tmp_path / "store", worker_python=sys.executable)) as value:
        yield value


def test_catalog_lists_every_kind(client):
    catalog = client.get("/api/v1/catalog").json()
    for key in ("targets", "environments", "tasks", "plugins", "strategies", "defenses", "attackers", "judges"):
        assert catalog[key], key
    assert {t["id"] for t in catalog["targets"]} >= {"openvla-7b", "memoryvla", "custom"}


def test_valid_config_resolves_environment_and_estimate(client):
    result = client.post("/api/v1/experiments/validate", json=config()).json()
    assert result["valid"], result["issues"]
    assert result["resolved"]["environment"] == "maniskill3"
    assert result["resolved"]["environment_locked"] is False
    assert result["resolved"]["strategies"] == ["basic", "iter", "hand"]
    assert result["estimate"]["episodes"] == 4 * 5 * 3 * 2


def test_single_environment_target_is_locked(client):
    result = client.post("/api/v1/experiments/validate", json=config(target={"id": "memoryvla"})).json()
    assert result["resolved"]["environment_locked"] is True


def test_errors_are_reported_per_step(client):
    bad = config(target={"id": "custom"}, plugins=[], defenses=["ssm"], run={"location": "local"})
    issues = client.post("/api/v1/experiments/validate", json=bad).json()["issues"]
    steps = {i["step"] for i in issues if i["severity"] == "error"}
    assert {"targetcfg", "plugins", "defense"} <= steps


def test_local_run_checks_gpu_memory(client):
    result = client.post("/api/v1/experiments/validate", json=config(run={"location": "local"})).json()
    assert not result["valid"]
    assert any(i["step"] == "review" for i in result["issues"])


def test_environment_specific_strategy_is_dropped(client):
    result = client.post("/api/v1/experiments/validate", json=config(environment="libero", tasks=["knife"])).json()
    assert "hand" in result["resolved"]["unsupported_strategies"]


def test_unimplemented_routes_answer_501_and_are_documented(client):
    assert client.get("/api/v1/runs/x/report").status_code == 501
    paths = client.get("/openapi.json").json()["paths"]
    for path in ("/api/v1/catalog", "/api/v1/experiments/validate", "/api/v1/runs/{run_id}/episodes",
                 "/api/v1/runs/{run_id}/episodes/{episode_id}/trace", "/api/v1/runs/{run_id}/report", "/api/v1/runs/{run_id}/matrix"):
        assert path in paths, path
