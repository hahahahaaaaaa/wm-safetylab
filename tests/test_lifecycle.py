"""Real service death/restart, owned-process cleanup and timeout integration."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import httpx
import psutil
import pytest
from fastapi.testclient import TestClient

from wms.app.api import create_app
from wms.core.processes import identity, stop_owned
from test_e2e import submit, wait_run

HERE = Path(__file__).parent


def alive(pid):
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def test_timeout_kills_owned_tree_and_queue_continues(tmp_path):
    root = tmp_path / "store"
    with TestClient(create_app(storage_root=root, worker_python=sys.executable)) as client:
        slow = submit(client, task="native", entrypoint="worker_fixtures:stubborn", timeout_s=1.5,
                      runtime={"python_paths": [str(HERE)]})
        good = submit(client, recipe="demo-native")
        done = wait_run(client, slow["id"])
        assert done["state"] == "failed" and done["error_code"] == "TIMEOUT"
        child = json.loads((root / "runs" / slow["id"] / "output/child.json").read_text())["pid"]
        assert not alive(done["pid"]) and not alive(child)
        assert wait_run(client, good["id"])["state"] == "completed"


@pytest.mark.parametrize("entry,state", [("fast_orphan", "completed"), ("crash_orphan", "failed")])
def test_fast_child_is_reaped_before_completion(tmp_path, entry, state):
    root = tmp_path / "store"
    with TestClient(create_app(storage_root=root, worker_python=sys.executable)) as client:
        run = submit(client, task="native", entrypoint=f"worker_fixtures:{entry}",
                     runtime={"python_paths": [str(HERE)]})
        done = wait_run(client, run["id"])
        assert done["state"] == state, done
        child = json.loads((root / "runs" / run["id"] / "output/child.json").read_text())["pid"]
        assert not alive(child)


def test_restart_cleanup_refuses_unrelated_or_reused_identity(tmp_path):
    options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], **options)
    try:
        record = identity(process.pid)
        assert not stop_owned(record, tmp_path)["matched_parent"]
        assert process.poll() is None
        stale = {**record, "created": record["created"] - 10}
        assert not stop_owned(stale, tmp_path, [stale])["target_pids"]
        assert process.poll() is None
    finally:
        process.kill()
        process.wait(timeout=5)


@pytest.mark.parametrize("evaluation", [False, True])
def test_abrupt_service_death_then_restart_cleans_owned_worker(tmp_path, evaluation):
    root = tmp_path / "store"
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join([str(HERE.parent / "src"), str(HERE)])
    command = [sys.executable, "-m", "wms", "serve", "--port", str(port), "--storage-root", str(root)]
    options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {"start_new_session": True}
    service = None
    worker = child = None
    log = (tmp_path / "service.log").open("wb")
    client = httpx.Client(base_url=f"http://127.0.0.1:{port}", trust_env=False, timeout=3)

    def start():
        process = subprocess.Popen(command, env=environment, stdout=log, stderr=log,
                                   cwd=HERE.parent, stdin=subprocess.DEVNULL, **options)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            assert process.poll() is None, (tmp_path / "service.log").read_text(errors="replace")
            try:
                if client.get("/api/v1/health").status_code == 200:
                    return process
            except httpx.TransportError:
                pass
            time.sleep(0.1)
        process.kill()
        pytest.fail("Service did not start")

    try:
        service = start()
        if evaluation:
            source = submit(client, recipe="demo-native")
            wait_run(client, source["id"])
            response = client.post(f"/api/v1/runs/{source['id']}/evaluate", json={
                "runtime": {"python_paths": [str(HERE)]},
                "metrics": [{"entrypoint": "worker_fixtures:metric", "params": {"delay": 120}}]})
            assert response.status_code == 200
            run = response.json()
            marker = root / "runs" / run["id"] / "output/metric-worker.json"
        else:
            run = submit(client, task="native", entrypoint="worker_fixtures:stubborn",
                         runtime={"python_paths": [str(HERE)]})
            marker = root / "runs" / run["id"] / "output/child.json"
        deadline = time.monotonic() + 10
        while not marker.is_file():
            assert time.monotonic() < deadline
            time.sleep(0.05)
        running = client.get(f"/api/v1/runs/{run['id']}").json()
        worker = running["pid"]
        child = json.loads(marker.read_text())["pid"]
        # venv launchers on Windows can introduce one intermediate process.
        service_children = psutil.Process(service.pid).children(recursive=True)
        owned = {worker, child}
        server_processes = [p for p in service_children if p.pid not in owned and "wms.runners.unified" not in " ".join(p.cmdline())]
        # Stop only the server process itself; leave the actual worker alive.
        for proc in server_processes:
            commandline = proc.cmdline()
            if "serve" in commandline and "wms" in commandline:
                proc.kill()
        service.kill()
        service.wait(timeout=5)
        assert alive(worker) or alive(child)
        service = start()
        resumed = client.get(f"/api/v1/runs/{run['id']}").json()
        assert resumed["state"] == "interrupted", resumed
        assert not alive(worker) and not alive(child)
        if evaluation:
            assert client.get(f"/api/v1/runs/{source['id']}").json()["state"] == "completed"
        good = submit(client, recipe="demo-native")
        assert wait_run(client, good["id"])["state"] == "completed"
    finally:
        client.close()
        if service and service.poll() is None:
            tree = psutil.Process(service.pid).children(recursive=True)
            for item in reversed(tree):
                try:
                    item.kill()
                except psutil.NoSuchProcess:
                    pass
            service.kill()
            service.wait(timeout=5)
        for pid in (worker, child):
            if pid and alive(pid):
                try:
                    psutil.Process(pid).kill()
                except psutil.NoSuchProcess:
                    pass
        log.close()
