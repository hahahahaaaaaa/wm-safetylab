"""A single-host FIFO queue; models and evaluations use the same worker path."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import traceback

from .evaluation import evaluation_config
from .processes import descendants, identity, stop_owned
from .storage import utcnow

TERMINAL = {"completed", "failed", "cancelled", "interrupted"}


class Runner:
    def __init__(self, store):
        self.store = store
        self._queue = queue.Queue()
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._active = None
        for run in store.list_runs():
            if run["state"] not in TERMINAL:
                cleanup = stop_owned(run.get("process"), store.run_dir(run["id"]), run.get("children", []))
                store.update(run["id"], state="interrupted", finished_at=utcnow(), cleanup=cleanup,
                             error="Service stopped before this run finished; retry explicitly")
        self._thread = threading.Thread(target=self._loop, name="wms-runner", daemon=True)
        self._thread.start()

    @staticmethod
    def _validate(config):
        timeout = config.get("timeout_s", config.get("runtime", {}).get("timeout_s"))
        if timeout is not None and (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
                                    or not math.isfinite(timeout) or timeout <= 0):
            raise ValueError("timeout_s must be a finite positive number")
        if not isinstance(config.get("metrics", []), list):
            raise ValueError("metrics must be a list")

    def submit(self, config):
        self._validate(config)
        with self._lock:
            if self._stop.is_set():
                raise RuntimeError("Runner is shutting down")
            run = self.store.create(config)
            self._queue.put(run["id"])
            return run

    def evaluate(self, source_run, metrics=None, runtime=None, timeout_s=None):
        config = evaluation_config(self.store, self.store.get(source_run), metrics, runtime, timeout_s)
        return self.submit(config)

    def cancel(self, run_id):
        with self._lock:
            run = self.store.get(run_id)
            if run["state"] in TERMINAL:
                return run
            if run["state"] == "queued":
                return self.store.update(run_id, state="cancelled", finished_at=utcnow())
            (self.store.output_dir(run_id) / "cancel.flag").touch()
            return self.store.update(run_id, state="cancelling")

    def retry(self, run_id):
        previous = self.store.get(run_id)
        if previous["state"] not in TERMINAL:
            raise ValueError("Only a finished run can be retried")
        run = self.submit(previous["config"])
        return self.store.update(run["id"], retry_of=run_id)

    def _loop(self):
        while not self._stop.is_set():
            try:
                run_id = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                with self._lock:
                    if self.store.get(run_id)["state"] != "queued":
                        continue
                    if self._stop.is_set():
                        self.store.update(run_id, state="cancelled", finished_at=utcnow())
                        continue
                    self.store.update(run_id, state="running")
                    self._active = run_id
                self._execute(run_id)
            except Exception as exc:
                with (self.store.run_dir(run_id) / "run.log").open("a", encoding="utf-8") as log:
                    traceback.print_exc(file=log)
                self.store.update(run_id, state="failed", finished_at=utcnow(),
                                  error=f"{type(exc).__name__}: {exc}")
            finally:
                with self._lock:
                    self._active = None
                self._queue.task_done()

    def _execute(self, run_id):
        run = self.store.get(run_id)
        config, runtime = run["config"], run["config"].get("runtime", {})
        directory, output = self.store.run_dir(run_id), self.store.output_dir(run_id)
        command = [runtime.get("python") or sys.executable, "-m", "wms.runners.unified",
                   "--config", str(directory / "config.json"), "--output-dir", str(output), "--managed"]
        environment = os.environ.copy()
        environment.update({str(k): str(v) for k, v in runtime.get("env", {}).items()})
        paths = [str(Path(__file__).resolve().parents[2]), *map(str, runtime.get("python_paths", []))]
        if environment.get("PYTHONPATH"):
            paths.append(environment["PYTHONPATH"])
        environment.update(PYTHONPATH=os.pathsep.join(paths), PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        environment.update(WMS_RUN_ID=run_id, WMS_RUN_DIR=str(directory))
        options = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {"start_new_session": True}
        process, record, children = None, None, []
        timeout = config.get("timeout_s", runtime.get("timeout_s"))
        timed_out = False
        with (directory / "run.log").open("ab", buffering=0) as log:
            log.write(("Command: " + subprocess.list2cmdline(command) + "\n").encode("utf-8"))
            try:
                process = subprocess.Popen(command, cwd=runtime.get("cwd") or directory,
                                           env=environment, stdout=log, stderr=subprocess.STDOUT,
                                           stdin=subprocess.DEVNULL, **options)
                started = time.monotonic()
                record = identity(process.pid)
                self.store.update(run_id, pid=process.pid, process=record, started_at=utcnow())
                (directory / "start.flag").touch()
                cancel_started, last_children = None, 0.0
                while process.poll() is None:
                    now = time.monotonic()
                    ready = (directory / "finish.ready").exists() and not (directory / "finish.flag").exists()
                    if ready or now - last_children > 0.5:
                        current_children = descendants(record)
                        known = {(item["pid"], item["created"]): item for item in children + current_children}
                        children = list(known.values())
                        self.store.update(run_id, children=children)
                        last_children = now
                    if ready:
                        # The worker remains alive until its final children
                        # are recorded, closing the fast-child sampling gap.
                        (directory / "finish.flag").touch()
                    if timeout is not None and now - started >= timeout:
                        timed_out = True
                        stop_owned(record, directory, children)
                        process.wait(timeout=5)
                        break
                    cancelled = self.store.get(run_id)["state"] == "cancelling" or self._stop.is_set()
                    if cancelled:
                        (output / "cancel.flag").touch()
                        if cancel_started is None:
                            cancel_started = now
                        elif now - cancel_started >= 2:
                            stop_owned(record, directory, children)
                            process.wait(timeout=5)
                            break
                    time.sleep(0.05)
                process.wait(timeout=5)
                # Reap any known subprocess left behind by a worker that exited.
                cleanup = stop_owned(record, directory, children)
                if cleanup["still_alive"]:
                    raise RuntimeError(f"Owned worker processes did not stop: {cleanup['still_alive']}")
                cancelled = self.store.get(run_id)["state"] == "cancelling" or self._stop.is_set()
                result_path = output / "result.json"
                try:
                    result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else None
                except (json.JSONDecodeError, UnicodeDecodeError):
                    if not (cancelled or timed_out):
                        raise
                    result = None
                if timed_out or cancelled:
                    self.store.update(run_id, state="failed" if timed_out else "cancelled", result=result,
                                      error=f"Execution exceeded timeout_s={timeout}" if timed_out else None,
                                      error_code="TIMEOUT" if timed_out else None, finished_at=utcnow(),
                                      returncode=process.returncode, cleanup=cleanup)
                    return
                if not isinstance(result, dict):
                    raise RuntimeError(f"Worker exited with code {process.returncode} without a valid result.json; see run.log")
                state = result.get("status")
                if state not in {"completed", "failed", "cancelled"}:
                    raise ValueError(f"Worker returned unsupported status: {state}")
                if process.returncode != 0 and state != "cancelled":
                    state = "failed"
                if state == "completed":
                    for item in result.get("outputs", []):
                        target = (output / item["path"]).resolve()
                        if not target.is_relative_to(output) or not target.is_file():
                            raise ValueError(f"Missing or invalid output: {item['path']}")
                error = result.get("error")
                if state == "failed" and not error:
                    error = f"Worker failed (exit code {process.returncode}); see run.log"
                with self._lock:
                    if self.store.get(run_id)["state"] == "cancelling" or self._stop.is_set():
                        state, error = "cancelled", None
                    fields = dict(state=state, result=result, error=error, returncode=process.returncode,
                                  finished_at=utcnow(), cleanup=cleanup,
                                  metric_errors=result.get("metric_errors", {}),
                                  evaluation_status=result.get("evaluation_status"))
                    auto_config = None
                    if state == "completed" and config["task"] != "evaluate" and config.get("metrics"):
                        auto_config = evaluation_config(self.store, {**run, **fields}, automatic=True)
                    _, evaluation = self.store.finish(run_id, fields, auto_config)
                    if evaluation:
                        self._queue.put(evaluation["id"])
            finally:
                if process is not None and process.poll() is None:
                    if record:
                        stop_owned(record, directory, children)
                    if process.poll() is None:
                        process.kill()
                    process.wait(timeout=5)

    def shutdown(self):
        with self._lock:
            self._stop.set()
            for run in self.store.list_runs():
                if run["state"] not in TERMINAL:
                    self.cancel(run["id"])
        self._thread.join(timeout=20)
        if self._thread.is_alive():
            raise RuntimeError("Runner has not stopped yet; keep its Store open")

