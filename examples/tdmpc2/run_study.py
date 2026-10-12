"""Client-only example: train -> held-out control -> intervention -> WM audit.

Run against the deployed API (WMS_BASE_URL), not inside the CUDA environment.
Pass --training-run to reuse an already completed training run.
"""
import argparse
import json
from pathlib import Path
import wms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-run")
    parser.add_argument("--record", default="results/tdmpc2-study.json")
    args = parser.parse_args()
    records = {}

    def execute(name, config):
        run = wms.wait(wms.run(config)["id"], timeout=14400)
        records[name] = run
        target = Path(args.record)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
        if run["state"] != "completed":
            raise RuntimeError(f'{name}: {run["id"]}: {run.get("error")}')
        evaluation_id = run.get("automatic_evaluation_id")
        if evaluation_id:
            evaluation = wms.wait(evaluation_id)
            if evaluation["state"] != "completed" or evaluation.get("evaluation_status") == "partial_failed":
                raise RuntimeError(f'Metric evaluation failed: {evaluation_id}')
        print(name, run["id"], flush=True)
        return run

    training = wms.get_run(args.training_run) if args.training_run else execute(
        "training", {"recipe": "tdmpc2-pushcube-train", "seed": 0})
    if training["state"] != "completed":
        raise ValueError("--training-run must identify completed training")
    records["training"] = training
    for name, initial, noise in (("initial", True, 0.0), ("trained", False, 0.0), ("perturbed", False, 0.1)):
        execute(name, {"recipe": "tdmpc2-pushcube-eval", "name": "TD-MPC2 / " + name,
                       "inputs": {"source_run": training["id"], "initial_checkpoint": initial},
                       "params": {"episodes": 10, "seed_start": 1000, "video_episodes": 2,
                                  "action_noise_std": noise}})
    execute("prediction", {"recipe": "tdmpc2-pushcube-predict",
                           "inputs": {"source_run": training["id"], "evaluation_run": records["trained"]["id"]},
                           "params": {"horizons": [1, 5]}})
    print(json.dumps(wms.compare([records[name]["id"] for name in ("initial", "trained", "perturbed")]),
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
