"""Real online training using upstream TD-MPC2 update(), Buffer and MPPI."""
from __future__ import annotations

import json
import math
from pathlib import Path
import time

from . import compat


def _check_warm_start_config(saved, current):
    """Old official checkpoints remain valid; cross-task transfer is explicit work."""
    for key in ("env_id", "control_mode", "obs_shape", "action_dim", "model_size"):
        if saved.get(key) != current.get(key):
            raise ValueError(f"Warm-start checkpoint {key} differs from training configuration")
    saved_reward = saved.get("reward_version")
    if saved_reward is None and saved.get("env_id") == "PushCube-v1":
        saved_reward = "maniskill-normalized-dense"
    current_reward = current.get("reward_version", "maniskill-normalized-dense")
    if saved_reward != current_reward:
        raise ValueError("Warm-start checkpoint reward_version differs from training configuration")


def _recording_starts(total_steps, batch_steps):
    """Choose existing episode batches only; recording never adds simulator resets."""
    if total_steps <= 0 or batch_steps <= 0 or total_steps % batch_steps:
        raise ValueError("Recording needs complete episode batches")
    return sorted({0, *((int(total_steps * fraction) // batch_steps) * batch_steps
                         for fraction in (.25, .5, .75)), total_steps - batch_steps})


def train(runtime, params, output, context, seed=0, checkpoint=None):
    import torch
    baseline = compat.prepare(runtime)
    from common.buffer import Buffer
    from tdmpc2 import TDMPC2
    compat.seed_all(seed)
    output = Path(output)
    cfg = compat.make_config(baseline, params, output, seed)
    record_training = bool(params.get("record_training", False))
    env = compat.StateEnvironment(cfg, cfg.num_envs, render=record_training)
    try:
        if cfg.steps % (cfg.num_envs * env.horizon):
            raise ValueError("steps must be a multiple of num_envs * episode horizon so every collected transition enters replay")
        agent = TDMPC2(cfg)
        if checkpoint:
            data = torch.load(checkpoint, map_location="cuda", weights_only=True)
            _check_warm_start_config(data["config"], compat.plain_config(cfg))
            if cfg.env_id == "WMS-PushCubeCenter-v1":
                if data.get("metadata", {}).get("task_sha256") != compat.task_definition(cfg.env_id)["task_sha256"]:
                    raise ValueError("Warm-start center task source differs; use a separately named transfer experiment")
            agent.load({"model": data["model"]})
        # Model-only warm start is deliberate: no claim of replay/optimizer/RNG resume.
        origin = compat.provenance(runtime, env_id=cfg.env_id)
        semantics = {"env_id": cfg.env_id,
                     "reward_version": cfg.get("reward_version", "maniskill-normalized-dense"),
                     "task_profile": cfg.get("task_profile"),
                     "task_sha256": origin["task_sha256"]}
        context.write_json("provenance.json", origin)
        context.write_json("config.resolved.json", compat.plain_config(cfg))
        initial = {key: value.detach().cpu().clone() for key, value in agent.model.state_dict().items()}
        compat.checkpoint_save(agent, cfg, output / "initial.pt",
                               {"environment_steps": 0, "seed": seed, **semantics})
        buffer = Buffer(cfg)
        steps, updates, episodes = 0, 0, 0
        latest, history = {}, []
        warmed = False
        started = time.monotonic()
        next_checkpoint = int(params.get("checkpoint_every", 25000))
        checkpoints = ["initial.pt"]
        selected_starts = set(_recording_starts(int(cfg.steps), int(cfg.num_envs) * env.horizon)) if record_training else set()
        recordings, frames = [], []
        recording_start, recording_updates, recording_random_steps = None, None, 0
        obs, _ = env.reset(seed)
        observations = [obs.detach().cpu().clone()]
        actions, rewards = [], []
        episode_return = torch.zeros(cfg.num_envs, device=obs.device)
        success_once = torch.zeros(cfg.num_envs, dtype=torch.bool, device=obs.device)
        while steps < cfg.steps:
            context.check_cancelled()
            if not actions and steps in selected_starts:
                recording_start, recording_updates = steps, updates
                recording_random_steps = 0
                frames = [env.frame().copy()]
            if steps <= cfg.seed_steps:
                action = torch.rand((cfg.num_envs, cfg.action_dim), device=obs.device) * 2 - 1
                if frames:
                    recording_random_steps += 1
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
            if frames:
                frames.append(env.frame().copy())
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
                if frames:
                    from .media import write_video
                    oracle = env.oracle()
                    cube = oracle["cube_xyz"][0].tolist()
                    goal = oracle["goal_xyz"][0].tolist()
                    recording = {"train_step_start": recording_start, "train_step_end": steps,
                                 "environment_index": 0, "episode_steps": len(actions),
                                 "update_count_start": recording_updates, "update_count_end": updates,
                                 "policy_mode": "exploratory_training_policy_env0",
                                 "random_action_steps": recording_random_steps,
                                 "return": float(episode_return[0].item()),
                                 "success_once": bool(success_once[0].item()),
                                 "success_at_end": bool(info["success"][0].item()),
                                 "final_cube_xyz": cube, "final_goal_xyz": goal,
                                 "final_goal_distance_m": math.hypot(cube[0] - goal[0], cube[1] - goal[1]),
                                 **semantics}
                    recording.update(write_video(output, f"training_{recording_start:07d}", frames,
                                                 int(env.native.control_freq), compatibility=True))
                    recordings.append(recording)
                    context.write_json("training_recordings.json", {
                        "task": "training_recordings", **semantics,
                        "selected_batch_start_steps": sorted(selected_starts),
                        "semantics": "Actual environment 0 during exploratory training, including random initialization actions; not an independent policy evaluation.",
                        "recordings": recordings})
                    context.event("progress", {"phase": "training_recording", **recording})
                    frames = []
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
                                       {"environment_steps": steps, "update_count": updates, "seed": seed, **semantics})
                checkpoints.append(name)
                next_checkpoint += int(params.get("checkpoint_every", 25000))
        if not updates:
            raise RuntimeError("No gradient updates occurred; this is not a trained baseline")
        delta = sum(float((value.detach().cpu().double() - initial[key].double()).square().sum())
                    for key, value in agent.model.state_dict().items()) ** 0.5
        metadata = {"environment_steps": steps, "update_count": updates, "seed": seed,
                    "upstream_commit": origin["commit"], **semantics}
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
        if record_training:
            report.update(training_recordings=recordings, training_recordings_file="training_recordings.json")
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
