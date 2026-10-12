"""Measure saved TD-MPC2 evidence without importing models or simulators."""
import json
import math
from pathlib import Path
from statistics import fmean

VERSION = "tdmpc2/2"
_FILES = {"train": "training.json", "policy_evaluate": "evaluation.json",
          "predict": "prediction_summary.json"}


def _number(value, name, *, nonnegative=False):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _count(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _flag(value, name):
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")
    return int(value)


def _training(data):
    episodes = data["episodes"]
    values = {
        "environment_steps": _count(data["environment_steps"], "environment_steps"),
        "update_count": _count(data["update_count"], "update_count"),
        "training_episodes": len(episodes) if isinstance(episodes, list) else _count(episodes, "episodes"),
        "elapsed_seconds": _number(data["elapsed_seconds"], "elapsed_seconds", nonnegative=True),
        "parameter_delta_l2": _number(data["parameter_delta_l2"], "parameter_delta_l2", nonnegative=True),
        "finite_losses": _flag(data["finite_losses"], "finite_losses"),
    }
    losses = data["last_losses"]
    if not isinstance(losses, dict):
        raise ValueError("last_losses must be an object")
    for key in ("total_loss", "consistency_loss", "reward_loss", "value_loss"):
        values["last_" + key] = _number(losses[key], "last_losses." + key)
    return values, {"training_episode_count": values["training_episodes"]}


def _policy(data):
    episodes = data["episodes"]
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("Policy evaluation requires at least one episode")
    returns, steps, success_once, success_at_end, interventions, seeds = [], [], [], [], [], []
    for index, episode in enumerate(episodes):
        if not isinstance(episode, dict):
            raise ValueError(f"episodes[{index}] must be an object")
        label = f"episodes[{index}]."
        seeds.append(_count(episode["seed"], label + "seed"))
        returns.append(_number(episode["return"], label + "return"))
        length = _count(episode["steps"], label + "steps")
        if length == 0:
            raise ValueError(label + "steps must be positive")
        count = _count(episode["action_intervention_steps"], label + "action_intervention_steps")
        if count > length:
            raise ValueError(label + "action_intervention_steps cannot exceed steps")
        steps.append(length)
        interventions.append(count)
        success_once.append(_flag(episode["success_once"], label + "success_once"))
        success_at_end.append(_flag(episode["success_at_end"], label + "success_at_end"))
    values = {
        "episode_count": len(episodes),
        "return_mean": fmean(returns),
        "success_once_rate": fmean(success_once),
        "success_at_end_rate": fmean(success_at_end),
        "episode_steps_mean": fmean(steps),
        "action_intervention_steps": sum(interventions),
        "action_intervention_rate": sum(interventions) / sum(steps),
    }
    # Historical results may not include geometric diagnostics. Only compare
    # full episode cohorts; a missing measurement is never filled with zero.
    for field in ("final_goal_distance_m", "min_goal_distance_m"):
        if all(episode.get(field) is not None for episode in episodes):
            values[field + "_mean"] = fmean(
                _number(episode[field], f"episodes[{i}].{field}", nonnegative=True)
                for i, episode in enumerate(episodes))
    for field in ("region_success_at_end", "center_success_at_end", "center_stable_last_5"):
        if all(episode.get(field) is not None for episode in episodes):
            values[field + "_rate"] = fmean(
                _flag(episode[field], f"episodes[{i}].{field}")
                for i, episode in enumerate(episodes))
    return values, {"episode_seeds": seeds, "episodes": episodes,
                    "env_id": data.get("env_id"),
                    "success_protocol": data.get("protocol", {}).get("success_protocol"),
                    "reward_version": data.get("protocol", {}).get("reward_version"),
                    "aggregation": "Equal episode weights for return, success and length; intervention rate per actual step."}


def _prediction(data):
    metrics = data["metrics"]
    if not isinstance(metrics, dict) or not metrics:
        raise ValueError("Prediction metrics must be a nonempty object")
    values = {}
    for name, value in metrics.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Prediction metric names must be nonempty strings")
        values[name] = _number(value, "metrics." + name)
    return values, {"aggregation": "Values recorded by the fixed-action prediction task; no new model inference."}


def measure(result, params):
    """Read one source task's metadata and write only the evaluation directory."""
    source = Path(result["source_output_dir"]).resolve()
    output = Path(result["output_dir"]).resolve()
    if output == source or output.is_relative_to(source):
        raise ValueError("Evaluation output must be outside the source directory")
    task = result.get("data", {}).get("task") if isinstance(result.get("data"), dict) else None
    if task in _FILES:
        filename = _FILES[task]
    else:
        candidates = [name for name in _FILES.values() if (source / name).is_file()]
        if not candidates:
            raise FileNotFoundError("Missing TD-MPC2 source metadata: " + ", ".join(_FILES.values()))
        if len(candidates) != 1:
            raise ValueError("Ambiguous TD-MPC2 source metadata; source result.data.task must select one task")
        filename = candidates[0]
    data = json.loads((source / filename).read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict) or data.get("task") not in _FILES:
        raise ValueError("TD-MPC2 source metadata needs task: train, policy_evaluate or predict")
    task = data["task"]
    if _FILES[task] != filename:
        raise ValueError("TD-MPC2 metadata task does not match its filename")
    values, details = {"train": _training, "policy_evaluate": _policy, "predict": _prediction}[task](data)
    metrics = {"tdmpc2_" + key: _number(value, key) for key, value in values.items()}
    report = {"metric_version": VERSION, "source_output_dir": str(source),
              "source_file": filename, "task": task, "metrics": metrics,
              "details": details, "safety_status": "not_measured",
              "conclusion_limits": ["Task success does not measure safety.",
                                    "Latent prediction error is not a physical-position error.",
                                    "Training updates and parameter changes do not establish convergence."]}
    output.mkdir(parents=True, exist_ok=True)
    (output / "tdmpc2_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return metrics
