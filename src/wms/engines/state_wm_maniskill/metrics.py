"""Independent evaluation: read source data, write only this evaluation directory."""
from pathlib import Path
import json

VERSION = "state-wm-maniskill/1"


def measure(result, params):
    import numpy as np
    source = Path(result["source_output_dir"])
    output = Path(result["output_dir"])
    data = np.load(source / "predictions.npz")
    training = json.loads((source / "training.json").read_text(encoding="utf-8"))
    values, details = {}, {}
    groups = {"cube_mm": (slice(18, 21), 1000.0, "euclidean"),
              "tcp_mm": (slice(21, 24), 1000.0, "euclidean"),
              "arm_rad": (slice(0, 7), 1.0, "rmse")}
    horizon = training["horizon"]
    for mode in ("one_step", "rollout"):
        truth = data["truth"] if mode == "one_step" else data["rollout_truth"][:, -1]
        ids = data["episode_ids"] if mode == "one_step" else data["rollout_episode_ids"]
        predictions = {"persistence": (data["initial"] if mode == "one_step" else data["rollout_initial"])[None]}
        for variant in ("action_conditioned", "no_action"):
            predictions[variant] = data[f"{variant}_{mode}"] if mode == "one_step" else data[f"{variant}_{mode}"][:, :, -1]
        for variant, pred in predictions.items():
            if not np.isfinite(pred).all():
                raise ValueError(f"Non-finite {variant} {mode} prediction")
            for label, (section, scale, norm) in groups.items():
                delta = (pred[..., section] - truth[None, ..., section]) * scale
                errors = np.sqrt(np.square(delta).sum(-1) if norm == "euclidean" else np.square(delta).mean(-1))
                # Equal episode weight, then mean and spread over independently trained seeds.
                per_episode = np.stack([errors[:, ids == episode].mean(-1) for episode in np.unique(ids)], axis=1)
                per_seed = per_episode.mean(-1)
                key = f"{variant}.{mode}.{label}"
                values[key] = float(per_seed.mean())
                details[key] = {"mean": values[key], "training_seed_std": float(per_seed.std()),
                                "per_seed": per_seed.tolist(), "per_seed_episode": per_episode.tolist(),
                                "test_episode_ids": np.unique(ids).tolist(), "lower_is_better": True,
                                "aggregation": "mean over time within each episode; equal episode weight; mean over training seeds",
                                "horizon": 1 if mode == "one_step" else horizon,
                                "unit": "rad" if label.endswith("rad") else "mm"}
    report = {"metric_version": VERSION, "values": values, "details": details,
              "dataset_source": str(source), "training_seeds": training["seeds"],
              "conclusion_limits": ["Training seed spread is not a confidence interval.",
                                    "No safety claim follows from predictive accuracy alone.",
                                    "Fixed planner collected data; the learned model did not control the robot.",
                                    "No model update comparison, probability calibration or fixed-decision switch test in this first engineering experiment."]}
    (output / "state_wm_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return values
