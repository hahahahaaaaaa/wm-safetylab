"""Real online training using upstream TD-MPC2 update(), Buffer and MPPI."""
from __future__ import annotations

import json
import math
from pathlib import Path
import time

from . import compat


def train(runtime, params, output, context, seed=0, checkpoint=None):
    import torch
    baseline = compat.prepare(runtime)
    from common.buffer import Buffer
    from tdmpc2 import TDMPC2
    compat.seed_all(seed)
    output = Path(output)
    cfg = compat.make_config(baseline, params, output, seed)
    env = compat.StateEnvironment(cfg, cfg.num_envs)
    agent = TDMPC2(cfg)
    if checkpoint:
        data = torch.load(checkpoint, map_location="cuda", weights_only=True)
        for key in ("control_mode", "obs_shape", "action_dim", "model_size"):
            if data["config"][key] != compat.plain_config(cfg)[key]:
                raise ValueError(f"Warm-start checkpoint {key} differs from training configuration")
        agent.load({"model": data["model"]})
    # Model-only warm start is deliberate: no claim of replay/optimizer/RNG resume.
    origin = compat.provenance(runtime)
    context.write_json("provenance.json", origin)
    context.write_json("config.resolved.json", compat.plain_config(cfg))
    initial = {key: value.detach().cpu().clone() for key, value in agent.model.state_dict().items()}
    compat.checkpoint_save(agent, cfg, output / "initial.pt", {"environment_steps": 0, "seed": seed})
    buffer = Buffer(cfg)
    steps, updates, episodes = 0, 0, 0
    latest, history = {}, []
    warmed = False
    started = time.monotonic()
    next_checkpoint = int(params.get("checkpoint_every", 25000))
    checkpoints = ["initial.pt"]
    if cfg.steps % (cfg.num_envs * env.horizon):
        raise ValueError("steps must be a multiple of num_envs * 50 so every collected transition enters replay")
    try:
        obs, _ = env.reset(seed)
        observations = [obs.detach().cpu().clone()]
        actions, rewards = [], []
        episode_return = torch.zeros(cfg.num_envs, device=obs.device)
        success_once = torch.zeros(cfg.num_envs, dtype=torch.bool, device=obs.device)
        while steps < cfg.steps:
            context.check_cancelled()
            if steps <= cfg.seed_steps:
                action = torch.rand((cfg.num_envs, cfg.action_dim), device=obs.device) * 2 - 1
            else:
                action = agent.act(obs, t0=not actions, eval_mode=False)
            new_obs, reward, terminated, truncated, info = env.step(action)
            observations.append(new_obs.detach().cpu().clone())
            actions.append(action.detach().cpu().clone())
            rewards.append(reward.detach().cpu().clone())
            episode_return += reward
            success_once |= info["success"].bool()
            steps += cfg.num_envs
            obs = new_obs
            # Insert complete episodes before sampling updates. No auto-reset means
            # observations[-1] is always the actual final simulator observation.
            if len(actions) == env.horizon:
                buffer.add(compat.episode_to_td(observations, actions, rewards))
                episodes += cfg.num_envs
                row = {"environment_steps": steps, "update_count": updates,
                       "episodes": episodes, "elapsed_seconds": time.monotonic() - started,
                       "collection_return": float(episode_return.mean()),
                       "collection_success_once": float(success_once.float().mean()),
                       **latest}
                history.append(row)
                with (output / "training_history.jsonl").open("a", encoding="utf-8") as file:
                    file.write(json.dumps(row, allow_nan=False) + "\n")
                context.event("progress", {"phase": "train", **row})
                obs, _ = env.reset()
                observations = [obs.detach().cpu().clone()]
                actions, rewards = [], []
                episode_return.zero_()
                success_once.zero_()
            if steps >= cfg.seed_steps and buffer.num_eps:
                count = int(cfg.seed_steps / cfg.steps_per_update) if not warmed else max(1, int(cfg.num_envs / cfg.steps_per_update))
                warmed = True
                for index in range(count):
                    if index % 16 == 0:
                        context.check_cancelled()
                    latest = agent.update(buffer)
                    updates += 1
                    if not all(math.isfinite(float(value)) for value in latest.values()):
                        raise RuntimeError(f"Non-finite training loss at update {updates}: {latest}")
            if next_checkpoint > 0 and steps >= next_checkpoint:
                name = f"checkpoint_{steps:07d}.pt"
                compat.checkpoint_save(agent, cfg, output / name,
                                       {"environment_steps": steps, "update_count": updates, "seed": seed})
                checkpoints.append(name)
                next_checkpoint += int(params.get("checkpoint_every", 25000))
        if not updates:
            raise RuntimeError("No gradient updates occurred; this is not a trained baseline")
        delta = sum(float((value.detach().cpu().double() - initial[key].double()).square().sum())
                    for key, value in agent.model.state_dict().items()) ** 0.5
        metadata = {"environment_steps": steps, "update_count": updates, "seed": seed,
                    "upstream_commit": origin["commit"]}
        compat.checkpoint_save(agent, cfg, output / "model.pt", metadata)
        checkpoints.append("model.pt")
        report = {"task": "train", **metadata, "episodes": episodes,
                  "elapsed_seconds": time.monotonic() - started, "parameter_delta_l2": delta,
                  "finite_losses": True, "last_losses": latest, "checkpoints": checkpoints,
                  "model_parameters": agent.model.total_params,
                  "warm_start_checkpoint": checkpoint, "optimizer_replay_resumed": False,
                  "observation_dim": env.obs_dim, "action_dim": env.action_dim,
                  "control_mode": cfg.control_mode, "sim_backend": "physx_cuda",
                  "training_budget_note": "Budgeted single-seed training; not a full paper reproduction",
                  "safety_status": "not_measured"}
        context.write_json("training.json", report)
        return report
    finally:
        env.close()


def act(agent, observation, first=True):
    """Evaluation calls preserve the official stochastic MPPI policy.

    Upstream checks _prev_mean even for evaluation; on a loaded agent only
    _prev_mean_eval exists. Set its presence for that guard, keeping actual
    train/eval means separate and all MPPI calculations unchanged.
    """
    if not first and hasattr(agent, "_prev_mean_eval") and not hasattr(agent, "_prev_mean"):
        agent._prev_mean = agent._prev_mean_eval
    return agent.act(observation, t0=first, eval_mode=True)


def intervene_action(action, *, noise_std=0.0, scale=1.0, generator=None):
    """S5 hook in normalized controller units; records clipping, never hides it."""
    import torch
    requested = action.detach().clone()
    noise = torch.randn(requested.shape, generator=generator, device=requested.device) * noise_std
    raw = requested * scale + noise
    applied = raw.clamp(-1, 1)
    return applied, {"changed": bool((applied - requested).abs().max() > 1e-7),
                     "clipped": bool((raw - applied).abs().max() > 0),
                     "max_delta": float((applied - requested).abs().max())}
