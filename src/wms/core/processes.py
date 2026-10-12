"""Identify owned workers before terminating them, including after a restart.

Only the service imports psutil; native model environments need no new package.
The worker waits for start.flag so it cannot create children before its identity
has been durably recorded by the service.
"""
from __future__ import annotations

import os
import signal
import time

import psutil


def identity(pid):
    process = psutil.Process(pid)
    return {"pid": pid, "created": process.create_time(), "command": process.cmdline()}


def matching(record):
    if not record:
        return None
    try:
        process = psutil.Process(record["pid"])
        if abs(process.create_time() - record["created"]) > 0.001:
            return None
        if process.status() == psutil.STATUS_ZOMBIE:
            return None
        return process
    except (psutil.NoSuchProcess, psutil.AccessDenied, KeyError):
        return None


def owned_worker(record, directory):
    process = matching(record)
    if process is None:
        return None
    try:
        command = process.cmdline()
        # Exact arguments, not substring matches to unrelated commands.
        if "wms.runners.unified" not in command or str(directory / "config.json") not in command:
            return None
        if str(directory / "output") not in command:
            return None
        return process
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None


def descendants(record):
    process = matching(record)
    if process is None:
        return []
    found = []
    try:
        for child in process.children(recursive=True):
            try:
                found.append(identity(child.pid))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except psutil.NoSuchProcess:
        pass
    return found


def marked_descendants(record, directory):
    """Find short-lived-parent orphans by their inherited run identity.

    The random run ID plus absolute directory is specific to this invocation.
    Workers must preserve these variables when creating native subprocesses.
    """
    if not record:
        return []
    found = []
    for process in psutil.process_iter(["pid", "create_time"]):
        try:
            created = process.info.get("create_time")
            if created is None or created < record["created"] - 0.001:
                continue
            env = process.environ()
            if env.get("WMS_RUN_ID") == directory.name and env.get("WMS_RUN_DIR") == str(directory):
                found.append(identity(process.pid))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return found


def stop_owned(record, directory, children=()):
    """Never use an unverified PID or a process-name based kill."""
    parent = owned_worker(record, directory)
    known = list(children) + marked_descendants(record, directory)
    if parent:
        known += descendants(record)
    targets = {item["pid"]: matching(item) for item in known}
    if parent:
        targets[parent.pid] = parent
    targets = [process for process in targets.values() if process is not None]
    group = None
    if parent and os.name != "nt":
        try:
            if os.getpgid(parent.pid) == parent.pid:
                group = parent.pid
                os.killpg(group, signal.SIGTERM)
        except ProcessLookupError:
            pass
    for process in reversed(targets):
        try:
            process.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(targets, timeout=2)
    for process in alive:
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass
    if group:
        # This group was established as ours immediately above; kill remaining
        # group members even if the parent exited while processing SIGTERM.
        try:
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            pass
    _, alive = psutil.wait_procs(alive, timeout=2)
    remaining = []
    for process in alive:
        try:
            if process.status() != psutil.STATUS_ZOMBIE:
                remaining.append(process.pid)
        except psutil.NoSuchProcess:
            pass
    return {"matched_parent": bool(parent), "target_pids": [p.pid for p in targets],
            "still_alive": remaining,
            "checked_at": time.time()}
