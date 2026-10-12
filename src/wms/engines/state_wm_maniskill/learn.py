"""Train residual MLPs using train-only normalization and episode-held-out data."""
from pathlib import Path
import json


def train(output_dir, params, context):
    import numpy as np
    import torch
    from torch import nn

    output = Path(output_dir)
    torch.set_num_threads(int(params.get("cpu_threads", 4)))
    data = np.load(output / "dataset.npz")
    metadata = json.loads((output / "dataset.json").read_text(encoding="utf-8"))
    x, y, action = data["states"], data["next_states"], data["actions"]
    episode, step = data["episode_ids"], data["steps"]
    masks = {name: np.isin(episode, ids) for name, ids in metadata["split"].items()}
    if any(not mask.any() for mask in masks.values()):
        raise ValueError("An episode split has no transitions")
    seeds = [int(seed) for seed in params.get("training_seeds", [0, 1, 2])]
    epochs = int(params.get("epochs", 400))
    batch_size = int(params.get("batch_size", 256))
    horizon = int(params.get("horizon", 5))
    width = int(params.get("hidden_width", 128))
    target = y - x
    target_mean = target[masks["train"]].mean(0)
    target_scale = np.maximum(target[masks["train"]].std(0), 1e-4)
    test_indices = np.flatnonzero(masks["test"])
    # All rollouts start from a real test state and then recurse on model predictions.
    starts = [i for i in test_indices if i + horizon <= len(x) and
              episode[i + horizon - 1] == episode[i] and step[i + horizon - 1] == step[i] + horizon - 1]
    starts = np.asarray(starts, dtype=np.int64)
    if not len(starts):
        raise ValueError("No complete held-out rollout window")
    artifacts = {"truth": y[test_indices], "initial": x[test_indices],
                 "episode_ids": episode[test_indices], "steps": step[test_indices],
                 "rollout_truth": np.stack([y[starts + h] for h in range(horizon)], axis=1),
                 "rollout_initial": x[starts], "rollout_episode_ids": episode[starts],
                 "rollout_steps": step[starts], "test_indices": test_indices}
    train_rows, checkpoints = [], []
    for variant in ("action_conditioned", "no_action"):
        features = np.concatenate([x, action], axis=1) if variant == "action_conditioned" else x
        feature_mean = features[masks["train"]].mean(0)
        feature_scale = np.maximum(features[masks["train"]].std(0), 1e-4)
        inputs = torch.tensor((features - feature_mean) / feature_scale, dtype=torch.float32)
        targets = torch.tensor((target - target_mean) / target_scale, dtype=torch.float32)
        train_ix = torch.tensor(np.flatnonzero(masks["train"]), dtype=torch.long)
        val_ix = torch.tensor(np.flatnonzero(masks["validation"]), dtype=torch.long)
        one_step, rollouts = [], []
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = nn.Sequential(nn.Linear(features.shape[1], width), nn.SiLU(),
                                  nn.Linear(width, width), nn.SiLU(), nn.Linear(width, x.shape[1]))
            optimizer = torch.optim.Adam(model.parameters(), lr=float(params.get("lr", 0.001)))
            best_loss, best_state, best_epoch = float("inf"), None, 0
            for epoch in range(epochs):
                context.check_cancelled()
                model.train()
                order = train_ix[torch.randperm(len(train_ix))]
                for batch in order.split(batch_size):
                    optimizer.zero_grad(set_to_none=True)
                    loss = (model(inputs[batch]) - targets[batch]).square().mean()
                    loss.backward()
                    optimizer.step()
                model.eval()
                with torch.no_grad():
                    val_loss = float((model(inputs[val_ix]) - targets[val_ix]).square().mean())
                if val_loss < best_loss:
                    best_loss, best_epoch = val_loss, epoch + 1
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                if epoch % 100 == 0:
                    context.event("progress", {"phase": "train", "variant": variant,
                                               "seed": seed, "epoch": epoch + 1, "validation_loss": val_loss})
            model.load_state_dict(best_state)

            def predict(current, commands):
                f = np.concatenate([current, commands], axis=1) if variant == "action_conditioned" else current
                with torch.no_grad():
                    delta = model(torch.tensor((f - feature_mean) / feature_scale, dtype=torch.float32)).numpy()
                return current + delta * target_scale + target_mean

            one_step.append(predict(x[test_indices], action[test_indices]))
            current, path = x[starts].copy(), []
            for h in range(horizon):
                current = predict(current, action[starts + h])
                path.append(current.copy())
            rollouts.append(np.stack(path, axis=1))
            filename = f"{variant}_seed{seed}.pt"
            torch.save({"state_dict": best_state, "variant": variant, "seed": seed,
                        "hidden_width": width, "state_size": x.shape[1], "action_size": action.shape[1],
                        "feature_mean": torch.tensor(feature_mean), "feature_scale": torch.tensor(feature_scale),
                        "target_mean": torch.tensor(target_mean), "target_scale": torch.tensor(target_scale)}, output / filename)
            checkpoints.append(filename)
            train_rows.append({"variant": variant, "seed": seed, "epochs_budget": epochs,
                               "best_epoch": best_epoch, "validation_normalized_mse": best_loss,
                               "parameters": sum(p.numel() for p in model.parameters())})
        artifacts[f"{variant}_one_step"] = np.stack(one_step)
        artifacts[f"{variant}_rollout"] = np.stack(rollouts)
    np.savez_compressed(output / "predictions.npz", **artifacts)
    report = {"schema": "wms.state-wm-maniskill.training/1", "model": "residual MLP, SiLU, two hidden layers",
              "device": "cpu", "seeds": seeds, "horizon": horizon,
              "hyperparameters": {"epochs": epochs, "batch_size": batch_size, "hidden_width": width,
                  "optimizer": "Adam", "learning_rate": float(params.get("lr", 0.001)),
                  "cpu_threads": int(params.get("cpu_threads", 4)), "minimum_normalization_scale": 1e-4},
              "normalization": "feature and target delta statistics fitted to train episodes only",
              "checkpoint_selection": "minimum validation normalized delta MSE, never test score",
              "split_transition_counts": {name: int(mask.sum()) for name, mask in masks.items()},
              "heldout_rollout_windows": len(starts), "runs": train_rows, "checkpoints": checkpoints,
              "claim_limit": "Offline one-step/short-horizon predictive engineering validation; no online planning, calibrated distribution, update efficacy or safety guarantee."}
    (output / "training.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
