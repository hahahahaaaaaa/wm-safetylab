"""Resolve and execute a local example, or submit its resolved config to an API.

Use from the project root. --runtime is optional: default is this interpreter.
API submission requires paths in the resolved config to exist on the API host.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from wms.core.registry import Registry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--runtime", help="Machine configuration JSON")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--api", help="Optional API root, e.g. http://127.0.0.1:19012")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    recipe = "template-pendulum-video" if args.video else "template-pendulum-rollout"
    request = {"recipe": recipe}
    if args.runtime:
        request["runtime"] = {"config_file": str(Path(args.runtime).resolve())}
    config = Registry(paths=[Path(__file__).with_name("registry.json")]).resolve(request)
    config_file = output / "config.json"
    if config_file.exists():
        raise FileExistsError(f"Use a new output directory to preserve previous evidence: {output}")
    config_file.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.api:
        import wms
        os.environ["WMS_BASE_URL"] = args.api.rstrip("/")
        run = wms.run(config)
        completed = wms.wait(run["id"], timeout=180)
        (output / "submitted-run.json").write_text(json.dumps(completed, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"id": run["id"], "state": completed["state"]}))
        if completed["state"] != "completed":
            raise RuntimeError(completed.get("error") or "Experiment failed")
        print("Automatic evaluation is a separate queued job; inspect wms.evaluations(run_id).")
        return
    runtime = config["runtime"]
    process_env = {**os.environ, **{k: str(v) for k, v in runtime.get("env", {}).items()}}
    subprocess.run([runtime["python"], "-m", "wms.runners.unified", "--config", str(config_file),
                    "--output-dir", str(output / "source")], check=True, env=process_env,
                   cwd=runtime.get("cwd") or Path(__file__).resolve().parents[2], timeout=150)
    evaluation = {"task": "evaluate", "source_run": "local-onboarding-example",
                  "source_output_dir": str(output / "source"), "metrics": config["metrics"], "runtime": runtime}
    eval_file = output / "evaluation-config.json"
    eval_file.write_text(json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8")
    subprocess.run([runtime["python"], "-m", "wms.runners.unified", "--config", str(eval_file),
                    "--output-dir", str(output / "evaluation")], check=True, env=process_env,
                   cwd=runtime.get("cwd") or Path(__file__).resolve().parents[2], timeout=150)
    result = json.loads((output / "evaluation" / "result.json").read_text(encoding="utf-8"))
    if result.get("metric_errors"):
        raise RuntimeError(result["metric_errors"])
    print(json.dumps({"output": str(output), "metrics": result["metrics"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
