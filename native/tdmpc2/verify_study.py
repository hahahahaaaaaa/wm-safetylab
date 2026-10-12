"""Audit completed real runs and export compact evidence; never alters runs."""
import argparse
import json
from pathlib import Path
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    for name in ("train", "initial", "trained", "perturbed", "predict", "generic-predict", "generic-rollout"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    import numpy as np
    import imageio.v2 as imageio

    def request(path, data=None):
        body = None if data is None else json.dumps(data).encode()
        req = urllib.request.Request("http://127.0.0.1:8765/api/v1" + path, data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=20) as response:
            return json.load(response)

    records, evaluations, videos = {}, {}, []
    for name in ("train", "initial", "trained", "perturbed", "predict", "generic_predict", "generic_rollout"):
        run = request("/runs/" + getattr(args, name))
        assert run["state"] == "completed", (name, run["state"], run.get("error"))
        records[name] = run
        (output / f"{name}-run.json").write_text(json.dumps(run, indent=2, ensure_ascii=False))
        (output / f"{name}.log").write_text(request(f'/runs/{run["id"]}/logs')["text"])
        if run.get("automatic_evaluation_id"):
            ev = request("/runs/" + run["automatic_evaluation_id"])
            assert ev["state"] == "completed" and ev.get("evaluation_status") == "completed", ev
            evaluations[name] = ev
            (output / f"{name}-metrics-run.json").write_text(json.dumps(ev, indent=2, ensure_ascii=False))

    def run_output(name):
        return root / "data-linux/runs" / records[name]["id"] / "output"

    training = records["train"]["result"]["data"]
    assert training["update_count"] > 0 and training["parameter_delta_l2"] > 0 and training["finite_losses"]
    assert training["observation_dim"] == 35 and training["action_dim"] == 8
    import torch
    checkpoint = torch.load(run_output("train") / "model.pt", map_location="cpu", weights_only=True)
    assert all(bool(torch.isfinite(value).all()) for value in checkpoint["model"].values())
    assert checkpoint["metadata"]["environment_steps"] == training["environment_steps"]
    cohorts = {}
    for name in ("initial", "trained", "perturbed"):
        report = records[name]["result"]["data"]
        assert [ep["seed"] for ep in report["episodes"]] == list(range(1000, 1010))
        assert report["controller"] == "pd_joint_delta_pos"
        for episode in report["episodes"]:
            trajectory = np.load(run_output(name) / episode["trajectory"])
            assert trajectory["observations"].shape == (51, 35)
            assert trajectory["actions"].shape == (50, 8)
            assert trajectory["rewards"].shape == (50,)
            assert np.max(np.abs(trajectory["actions"])) <= 1.00001
            assert all(np.isfinite(trajectory[key]).all() for key in trajectory.files)
            differences = np.max(np.abs(trajectory["actions"] - trajectory["proposed_actions"]), axis=-1)
            if name == "perturbed":
                assert (differences > 1e-7).sum() == 50
            else:
                assert np.max(differences) == 0
            if episode["video"]:
                video = run_output(name) / episode["video"]
                with imageio.get_reader(video) as reader:
                    frames = reader.count_frames()
                    first, last = reader.get_data(0), reader.get_data(frames - 1)
                    assert frames == 51 and first.std() > 5
                    if name == "trained" and episode["seed"] == 1000:
                        imageio.imwrite(output / "trained-first-frame.png", first)
                        imageio.imwrite(output / "trained-last-frame.png", last)
                videos.append({"run": records[name]["id"], "file": episode["video"],
                               "decoded_frames": frames, "shape": list(first.shape), "bytes": video.stat().st_size})
        cohorts[name] = evaluations[name]["result"]["metrics"]
    prediction = records["predict"]["result"]["data"]
    assert len(prediction["records"]) == 20
    assert all(np.isfinite(value) for value in prediction["metrics"].values())
    # Recompute the reported errors from saved arrays, independently of Torch.
    reconstructed = {}
    for item in prediction["records"]:
        arrays = np.load(run_output("predict") / item["file"])
        h = item["horizon"]
        for key, values in {
            f"latent_mse_h{h}": ((arrays["predicted"] - arrays["actual"]) ** 2).mean(-1),
            f"persistence_latent_mse_h{h}": ((arrays["initial"] - arrays["actual"]) ** 2).mean(-1),
            f"zero_action_latent_mse_h{h}": ((arrays["zero_action"] - arrays["actual"]) ** 2).mean(-1),
            f"reward_mae_h{h}": np.abs(arrays["predicted_reward"] - arrays["actual_reward"]),
        }.items():
            reconstructed.setdefault(key, []).extend(values.tolist())
    for key, values in reconstructed.items():
        assert np.isclose(np.mean(values), prediction["metrics"][key], rtol=1e-5, atol=1e-8), key
    generic = records["generic_predict"]["result"]["data"]
    assert generic["predicted"]["representation"] == "tdmpc2_latent"
    assert len(generic["predicted"]["predictions"]) == len(generic["real"]) == 3
    assert len(generic["predicted"]["predictions"][0]) == checkpoint["config"]["latent_dim"]
    assert len(generic["real"][0]) == 35
    assert records["generic_rollout"]["result"]["data"]["steps"] == 50
    comparison = request("/comparisons", {"run_ids": [records[k]["id"] for k in ("initial", "trained", "perturbed")]})
    (output / "platform-comparison.json").write_text(json.dumps(comparison, indent=2, ensure_ascii=False))
    summary = {"verified": True, "runs": {k: v["id"] for k, v in records.items()},
               "metric_runs": {k: v["id"] for k, v in evaluations.items()},
               "training": training, "evaluation": cohorts, "prediction": prediction["metrics"],
               "videos": videos, "generic_interfaces": {"predict": True, "rollout": True},
               "scope": "One training seed, 10 held-out seeds per condition; no calibrated safety claim."}
    (output / "acceptance.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
