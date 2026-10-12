"""Small Python client for the WM SafetyLab service."""
__version__ = "0.3.0"


def run(config):
    from wms.app.api import request
    return request("POST", "/runs", config)


def get_run(run_id):
    from wms.app.api import request
    return request("GET", f"/runs/{run_id}")


def wait(run_id, timeout=3600):
    import time
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = get_run(run_id)
        if record["state"] not in {"queued", "running", "cancelling"}:
            return record
        time.sleep(0.25)
    raise TimeoutError(f"Run still active: {run_id}")


def cancel(run_id):
    from wms.app.api import request
    return request("POST", f"/runs/{run_id}/cancel")


def retry(run_id):
    from wms.app.api import request
    return request("POST", f"/runs/{run_id}/retry")


def evaluate(run_id, metrics=None, *, runtime=None, timeout_s=None):
    """Enqueue scoring; wait on the returned ID, not on the source model ID."""
    from wms.app.api import request
    options = {key: value for key, value in {"metrics": metrics, "runtime": runtime,
               "timeout_s": timeout_s}.items() if value is not None}
    return request("POST", f"/runs/{run_id}/evaluate", options)


def evaluations(run_id):
    from wms.app.api import request
    return request("GET", f"/runs/{run_id}/evaluations")["items"]


def compare(run_ids, *, evaluations=None):
    from wms.app.api import request
    return request("POST", "/comparisons", {"run_ids": list(run_ids), "evaluations": evaluations or {}})

