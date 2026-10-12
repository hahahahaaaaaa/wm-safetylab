"""Small subprocess behaviors for integration tests, never production recipes."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def metric(result, params):
    source = Path(result["source_output_dir"])
    assert source != Path(result["output_dir"])
    assert (source / "result.json").is_file()
    (Path(result["output_dir"]) / "metric-worker.json").write_text(json.dumps({"pid": os.getpid()}))
    time.sleep(params.get("delay", 0))
    return {params.get("key", "score"): params.get("value", 1)}


def bad_metric(result, params):
    raise ValueError("Intentional metric error")


def crash_metric(result, params):
    os._exit(73)


def stubborn(config, output, context):
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    context.write_json("child.json", {"pid": child.pid})
    time.sleep(120)
    return {"data": "should never finish"}


def false_completed(config, output, context):
    context.write_json("result.json", {"status": "completed", "metrics": {}, "outputs": []})
    os._exit(74)


def fast_orphan(config, output, context):
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    context.write_json("child.json", {"pid": child.pid})
    return {"outputs": [{"path": "child.json"}]}


def crash_orphan(config, output, context):
    fast_orphan(config, output, context)
    os._exit(75)
