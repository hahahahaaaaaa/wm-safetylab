"""Independent metric jobs and explicitly sourced comparisons."""
from __future__ import annotations

from copy import deepcopy
import importlib
import json
from pathlib import Path


def _load(entrypoint):
    module, separator, attribute = entrypoint.partition(":")
    if not separator:
        module, _, attribute = entrypoint.rpartition(".")
    if not module or not attribute:
        raise ValueError("Metric entrypoint must be module:function")
    target = importlib.import_module(module)
    for part in attribute.split("."):
        target = getattr(target, part)
    return target


def metric_specs(metrics):
    return [{**{"params": {}, "unit": None, "version": None},
             **({"entrypoint": item} if isinstance(item, str) else item)} for item in metrics]


def evaluation_config(store, source, metrics=None, runtime=None, timeout_s=None, automatic=False):
    if source["task"] == "evaluate":
        raise ValueError("Evaluate the source experiment, not another evaluation")
    if source["state"] != "completed" or source.get("result") is None:
        raise ValueError("Evaluation requires a completed source experiment")
    if not (store.output_dir(source["id"]) / "result.json").is_file():
        raise ValueError("Source result.json is missing")
    selected = source["config"].get("metrics", []) if metrics is None else metrics
    if not isinstance(selected, list):
        raise ValueError("metrics must be a list")
    inherited = deepcopy(source["config"].get("runtime", {}))
    if runtime is not None:
        if not isinstance(runtime, dict):
            raise ValueError("Evaluation runtime must be an object")
        env = {**inherited.get("env", {}), **runtime.get("env", {})}
        inherited.update(deepcopy(runtime))
        inherited["env"] = env
    config = {"name": f"Evaluate: {source['name']}", "task": "evaluate",
              "source_run": source["id"], "direction": source.get("direction", ""),
              "source_output_dir": str(store.output_dir(source["id"])),
              "metrics": selected, "runtime": inherited,
              "evaluation_origin": "automatic" if automatic else "manual"}
    timeout = timeout_s if timeout_s is not None else source["config"].get("timeout_s")
    if timeout is not None:
        config["timeout_s"] = timeout
    return config


def execute_evaluation(config, output_dir, context):
    """Worker-only metric imports. Source paths are read-only by convention."""
    source = Path(config["source_output_dir"])
    result = json.loads((source / "result.json").read_text(encoding="utf-8"))
    specs, values, errors = metric_specs(config.get("metrics", [])), {}, {}
    for index, spec in enumerate(specs):
        context.check_cancelled()
        name = spec.get("name") or spec.get("entrypoint", f"metric_{index}")
        try:
            metric_input = deepcopy(result)
            metric_input.update(source_output_dir=str(source), output_dir=str(output_dir))
            value = _load(spec["entrypoint"])(metric_input, spec.get("params", {}))
            json.dumps(value, allow_nan=False)
            updates = value if isinstance(value, dict) else {name: value}
            overlap = values.keys() & updates.keys()
            if overlap:
                raise ValueError(f"Duplicate metric output names: {sorted(overlap)}")
            values.update(updates)
        except Exception as exc:
            errors[name] = f"{type(exc).__name__}: {exc}"
        context.event("metric", {"name": name, "error": errors.get(name)})
    data = {"source_run": config["source_run"], "metrics": values,
            "metric_errors": errors, "metric_specs": specs,
            "evaluation_status": "partial_failed" if errors else "completed"}
    context.write_json("metrics.json", data)
    outputs = [{"kind": "metrics" if path.name == "metrics.json" else "evaluation_artifact",
                "path": path.relative_to(output_dir).as_posix()}
               for path in sorted(Path(output_dir).rglob("*"))
               if path.is_file() and path.name not in {"result.json", "cancel.flag"}]
    return {**data, "outputs": outputs, "data": {"source_run": config["source_run"]}}


def evaluations_for(store, source_run):
    store.get(source_run)
    return [run for run in store.list_runs() if run.get("source_run") == source_run]


def compare_runs(store, ids, evaluations=None):
    selections, rows, warnings, signatures = evaluations or {}, [], [], []
    for run_id in ids:
        run = store.get(run_id)
        if run["task"] == "evaluate":
            raise ValueError("Comparison run_ids must refer to source experiments")
        attempts = evaluations_for(store, run_id)
        latest = attempts[0] if attempts else None
        selected_id = selections.get(run_id)
        selected = next((item for item in attempts if item["state"] == "completed"), None)
        if selected_id and selected_id not in {"legacy", "native"}:
            selected = next((item for item in attempts if item["id"] == selected_id), None)
            if selected is None:
                raise ValueError(f"Evaluation {selected_id} does not belong to {run_id}")
            if selected["state"] != "completed":
                raise ValueError("Select a completed evaluation; unfinished attempts have no comparable scores")
        row_warnings = []
        legacy = run.get("record_version", 0) < 3
        fallback_kind = "legacy" if legacy else "native"
        if selected_id in {"legacy", "native"}:
            if selected_id != fallback_kind:
                raise ValueError(f"This source has {fallback_kind} metrics, not {selected_id}")
            selected = None
        fallback = selected_id in {"legacy", "native"} or not attempts
        if selected:
            result = selected.get("result") or {}
            specs = result.get("metric_specs", metric_specs(selected["config"].get("metrics", [])))
            evaluation = {"id": selected["id"], "kind": "independent", "state": selected["state"],
                          "created_at": selected["created_at"], "metric_specs": specs}
            errors = result.get("metric_errors", {})
            if errors:
                row_warnings.append("所选评价部分失败；缺失指标不从旧评价补值。")
        elif fallback:
            result = run.get("result") or {}
            errors = run.get("metric_errors", {})
            specs = []
            evaluation = {"id": fallback_kind, "kind": fallback_kind, "state": run["state"],
                          "created_at": run["created_at"], "metric_specs": []}
            row_warnings.append("旧版评价，指标版本未知。" if legacy else "模型原生指标；尚未选择独立评价。")
        else:
            result, errors, specs = {}, {}, []
            evaluation = {"id": None, "kind": "none", "state": None, "created_at": None, "metric_specs": []}
            row_warnings.append("已有评价尝试但尚无已完成评价；未回退到历史分数。")
        if latest and (not selected or latest["id"] != selected["id"]):
            row_warnings.append(f"最新评价尝试 {latest['id']} 状态为 {latest['state']}；所示分数来自明确选定的记录。")
        if specs and any(not item.get("version") for item in specs):
            row_warnings.append("部分指标未记录版本，不能保证实现相同。")
        signatures.append(json.dumps({"kind": evaluation["kind"], "specs": specs}, sort_keys=True))
        latest_info = ({key: latest.get(key) for key in
                        ("id", "state", "created_at", "error", "evaluation_status")} if latest else None)
        rows.append({"run_id": run_id, "name": run["name"], "task": run["task"],
                     "direction": run["direction"], "state": run["state"], "seed": run["config"].get("seed", 0),
                     "metrics": result.get("metrics", {}), "metric_errors": errors, "error": run.get("error"),
                     "evaluation": evaluation, "latest_evaluation": latest_info, "warnings": row_warnings})
    if len(set(signatures)) > 1:
        warnings.append("评价来源或指标入口、参数、单位、版本不同；请确认可比性，不自动排名。")
    if any(row["warnings"] for row in rows):
        warnings.append("部分运行存在评价来源或版本提示，请查看各行。")
    return {"rows": rows, "warnings": warnings}

