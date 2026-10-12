"""Platform tasks: train, checkpoint evaluation, and fixed-action latent audit."""
from __future__ import annotations

import json
from pathlib import Path
import time

from . import compat


def _settings(config):
    from wms.engines.native_tasks import _runtime
    runtime = _runtime(config)
    if not runtime:
        raise ValueError("Select a TD-MPC2 runtime.config_file")
    return runtime, dict(config.get("params", {})), dict(config.get("inputs", {}))


def _source(inputs, output, field="checkpoint"):
    value = inputs.get(field)
    if value:
        path = Path(value).resolve()
    elif inputs.get("source_run" if field == "checkpoint" else "evaluation_run"):
        import re
        run_id = inputs["source_run" if field == "checkpoint" else "evaluation_run"]
        if not re.fullmatch(r"run_[0-9a-f]+", run_id):
            raise ValueError("Invalid source/evaluation run ID")
        filename = "model.pt" if field == "checkpoint" else "evaluation.json"
        if field == "checkpoint" and inputs.get("initial_checkpoint", False):
            filename = "initial.pt"
        path = Path(output).resolve().parents[1] / run_id / "output" / filename
    else:
        raise ValueError(f"inputs.{field} is required (or source_run for checkpoint)")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _result(output, data):
    import mimetypes
    files = []
    for path in sorted(Path(output).rglob("*")):
        if not path.is_file() or path.name in {"result.json", "cancel.flag"}:
            continue
        suffix = path.suffix
        kind = "checkpoint" if suffix == ".pt" else "video" if suffix == ".mp4" else "report" if suffix == ".html" else "evidence"
        files.append({"path": path.relative_to(output).as_posix(), "kind": kind,
                      "media_type": mimetypes.guess_type(path.name)[0] or "application/octet-stream"})
    return {"data": data, "outputs": files}


def train(config, output_dir, context):
    from .backend import train as train_backend
    runtime, params, inputs = _settings(config)
    report = train_backend(runtime, params, Path(output_dir), context,
                           seed=int(config.get("seed", 0)), checkpoint=inputs.get("checkpoint"))
    return _result(output_dir, report)


def policy_evaluate(config, output_dir, context):
    runtime, params, inputs = _settings(config)
    compat.prepare(runtime)
    import numpy as np
    import torch
    from .backend import act, intervene_action
    output = Path(output_dir)
    checkpoint = _source(inputs, output)
    agent, cfg, training_metadata = compat.checkpoint_load(checkpoint)
    seed_start = int(params.get("seed_start", 1000))
    seeds = [int(v) for v in inputs.get("seeds", range(seed_start, seed_start + int(params.get("episodes", 10))))]
    if not seeds:
        raise ValueError("At least one evaluation seed required")
    video_episodes = int(params.get("video_episodes", 2))
    env = compat.StateEnvironment(cfg, 1, render=video_episodes > 0)
    if env.obs_dim != agent.model._encoder["state"][0].in_features:
        env.close()
        raise ValueError("Checkpoint observation layout differs from actual environment")
    noise_std = float(params.get("action_noise_std", 0))
    obs_noise = float(params.get("observation_noise_std", 0))
    scale = float(params.get("action_scale", 1))
    if min(noise_std, obs_noise) < 0:
        raise ValueError("Noise standard deviations must be nonnegative")
    episodes, trajectories = [], []
    started = time.monotonic()
    origin = compat.provenance(runtime)
    context.write_json("provenance.json", origin)
    try:
        for episode_index, seed in enumerate(seeds):
            context.check_cancelled()
            compat.seed_all(seed)
            noise_generator = torch.Generator(device="cpu").manual_seed(seed + 1000000)
            obs_generator = torch.Generator(device="cpu").manual_seed(seed + 2000000)
            obs, _ = env.reset(seed)
            states = [obs[0].detach().cpu().numpy().copy()]
            oracle = env.oracle()
            cube, tcp = [oracle["cube_xyz"][0]], [oracle["tcp_xyz"][0]]
            actions, proposed, policy_obs, rewards, success, changes, clipped = [], [], [], [], [], [], []
            frames = [env.frame()] if episode_index < video_episodes else []
            for step in range(env.horizon):
                context.check_cancelled()
                seen = obs.detach().cpu().clone()
                if obs_noise:
                    seen += torch.randn(seen.shape, generator=obs_generator) * obs_noise
                requested = act(agent, seen, first=step == 0)
                applied, intervention = intervene_action(requested, noise_std=noise_std,
                                                         scale=scale, generator=noise_generator)
                next_obs, reward, terminated, truncated, info = env.step(applied)
                policy_obs.append(seen[0].numpy().copy())
                proposed.append(requested[0].numpy().copy())
                actions.append(applied[0].numpy().copy())
                rewards.append(float(reward.item()))
                success.append(bool(info["success"].item()))
                changes.append(intervention["changed"])
                clipped.append(intervention["clipped"])
                obs = next_obs
                states.append(obs[0].detach().cpu().numpy().copy())
                oracle = env.oracle()
                cube.append(oracle["cube_xyz"][0])
                tcp.append(oracle["tcp_xyz"][0])
                if frames:
                    frames.append(env.frame())
            trajectory = {"observations": np.asarray(states), "actions": np.asarray(actions),
                          "proposed_actions": np.asarray(proposed), "policy_observations": np.asarray(policy_obs),
                          "rewards": np.asarray(rewards), "success": np.asarray(success),
                          "cube_xyz": np.asarray(cube), "tcp_xyz": np.asarray(tcp), "seed": np.asarray(seed)}
            np.savez_compressed(output / f"trajectory_{episode_index:03d}.npz", **trajectory)
            trajectories.append(trajectory)
            row = {"seed": seed, "return": sum(rewards), "success_once": any(success),
                   "success_at_end": success[-1], "steps": len(actions),
                   "terminated_at_end": bool(terminated.item()), "truncated_at_end": bool(truncated.item()),
                   "action_intervention_steps": sum(changes), "action_clipped_steps": sum(clipped),
                   "trajectory": f"trajectory_{episode_index:03d}.npz",
                   "video": None, "video_frames": 0}
            if frames:
                import imageio.v2 as imageio
                name = f"episode_{seed}.mp4"
                fps = int(env.native.control_freq)
                imageio.mimsave(output / name, frames, fps=fps, macro_block_size=1,
                               codec="libx264", ffmpeg_params=["-pix_fmt", "yuv420p"])
                row.update(video=name, video_frames=len(frames), video_fps=fps)
            episodes.append(row)
            context.event("progress", {"phase": "policy_evaluate", "episode": episode_index + 1, **row})
        report = {"task": "policy_evaluate", "checkpoint": str(checkpoint),
                  "checkpoint_sha256": compat.file_sha256(checkpoint),
                  "task_sha256": origin["task_sha256"], "env_id": cfg.env_id,
                  "observation_dim": env.obs_dim, "action_dim": env.action_dim,
                  "training_metadata": training_metadata, "episodes": episodes,
                  "elapsed_seconds": time.monotonic() - started,
                  "controller": cfg.control_mode, "observation": "privileged simulator state",
                  "intervention": {"action_noise_std": noise_std, "action_scale": scale,
                                   "observation_noise_std": obs_noise},
                  "protocol": {"horizon": env.horizon, "sim_backend": "physx_cuda",
                               "reward_mode": "normalized_dense", "success_termination_ignored": True,
                               "planner": "official stochastic MPPI; separate episode seed"},
                  "safety_status": "not_measured"}
        context.write_json("evaluation.json", report)
        _html_evaluation(output, report)
        return _result(output, report)
    finally:
        env.close()


def predict(config, output_dir, context):
    """Use fixed actions from genuine held-out episodes; never step a simulator."""
    runtime, params, inputs = _settings(config)
    compat.prepare(runtime)
    import numpy as np
    import torch
    from common.math import two_hot_inv
    output = Path(output_dir)
    checkpoint = _source(inputs, output)
    episode_report = _source(inputs, output, "evaluation_report")
    agent, cfg, training_metadata = compat.checkpoint_load(checkpoint)
    source = json.loads(episode_report.read_text(encoding="utf-8"))
    horizons = [int(h) for h in params.get("horizons", [1, 5])]
    if not horizons or any(h <= 0 for h in horizons):
        raise ValueError("Prediction horizons must be positive")
    origin = compat.provenance(runtime)
    if (source.get("controller") != cfg.control_mode or source.get("env_id") != cfg.env_id
            or source.get("task_sha256") != origin["task_sha256"]
            or source.get("protocol", {}).get("reward_mode") != "normalized_dense"):
        raise ValueError("Trajectory controller/task/reward semantics differ from the selected model environment")
    if not source.get("episodes"):
        raise ValueError("No evaluation episodes provided")
    totals = {h: {"model": [], "hold": [], "zero_action": [], "reward": []} for h in horizons}
    records = []
    compat.seed_all(int(config.get("seed", 0)))
    for episode_index, episode in enumerate(source["episodes"]):
        context.check_cancelled()
        values = np.load(episode_report.parent / episode["trajectory"])
        obs = torch.as_tensor(values["observations"], device="cuda", dtype=torch.float32)
        actions = torch.as_tensor(values["actions"], device="cuda", dtype=torch.float32)
        rewards = torch.as_tensor(values["rewards"], device="cuda", dtype=torch.float32)
        if (obs.shape != (len(actions) + 1, cfg.obs_shape["state"][0])
                or actions.shape != (len(actions), cfg.action_dim) or rewards.shape != (len(actions),)):
            raise ValueError("Trajectory must contain T+1 observations, T actions and T rewards in the checkpoint layout")
        if not all(bool(torch.isfinite(v).all()) for v in (obs, actions, rewards)):
            raise ValueError("Non-finite trajectory evidence")
        with torch.no_grad():
            real_latents = agent.model.encode(obs, None)
            for horizon in horizons:
                count = len(actions) - horizon + 1
                if count <= 0:
                    raise ValueError("Prediction horizon exceeds episode length")
                initial = real_latents[:count]
                z, zero_z = initial.clone(), initial.clone()
                imagined_rewards = []
                for offset in range(horizon):
                    a = actions[offset:offset + count]
                    imagined_rewards.append(two_hot_inv(agent.model.reward(z, a, None), cfg).squeeze(-1))
                    z = agent.model.next(z, a, None)
                    zero_z = agent.model.next(zero_z, torch.zeros_like(a), None)
                target = real_latents[horizon:horizon + count]
                mse = (z - target).square().mean(-1)
                hold = (initial - target).square().mean(-1)
                zero = (zero_z - target).square().mean(-1)
                reward_error = (imagined_rewards[-1] - rewards[horizon - 1:horizon - 1 + count]).abs()
                for key, value in (("model", mse), ("hold", hold), ("zero_action", zero), ("reward", reward_error)):
                    totals[horizon][key].extend(value.cpu().tolist())
                name = f"latents_episode_{episode_index:03d}_h{horizon}.npz"
                np.savez_compressed(output / name, initial=initial.cpu().numpy(),
                                    predicted=z.cpu().numpy(), actual=target.cpu().numpy(),
                                    zero_action=zero_z.cpu().numpy(),
                                    predicted_reward=imagined_rewards[-1].cpu().numpy(),
                                    actual_reward=rewards[horizon - 1:horizon - 1 + count].cpu().numpy())
                records.append({"seed": episode["seed"], "horizon": horizon, "samples": count, "file": name})
    metrics = {}
    for h, values in totals.items():
        metrics.update({f"latent_mse_h{h}": float(np.mean(values["model"])),
                        f"persistence_latent_mse_h{h}": float(np.mean(values["hold"])),
                        f"zero_action_latent_mse_h{h}": float(np.mean(values["zero_action"])),
                        f"reward_mae_h{h}": float(np.mean(values["reward"]))})
    report = {"task": "predict", "checkpoint": str(checkpoint), "evaluation_report": str(episode_report),
              "checkpoint_sha256": compat.file_sha256(checkpoint), "task_sha256": origin["task_sha256"],
              "training_metadata": training_metadata, "metrics": metrics, "records": records,
              "semantics": "Within-checkpoint latent-space error; not Cartesian error, collision probability or cross-checkpoint ranking",
              "safety_status": "not_measured"}
    context.write_json("prediction_summary.json", report)
    return _result(output, report)


def _html_evaluation(output, report):
    import html
    rows = []
    for ep in report["episodes"]:
        video = f'<video controls preload="metadata" width="480" src="{ep["video"]}"></video>' if ep["video"] else "未录制"
        rows.append(f'<tr><td>{ep["seed"]}</td><td>{ep["return"]:.3f}</td><td>{ep["success_once"]}</td>'
                    f'<td>{ep["success_at_end"]}</td><td>{ep["action_intervention_steps"]}</td><td>{video}</td></tr>')
    text = '<!doctype html><meta charset="utf-8"><title>TD-MPC2 PushCube 评测</title>'
    text += '''<meta name="viewport" content="width=device-width,initial-scale=1"><style>
body{font:16px system-ui;margin:24px;background:#f6f8fc;color:#182538;overflow-wrap:anywhere}
table{border-collapse:collapse;background:white}td,th{padding:12px;border:1px solid #dce1e8}
video{width:480px;max-width:100%;display:block;border-radius:8px}
@media(max-width:700px){body{margin:16px}h1{font-size:26px}table,tbody,tr,td{display:block}
tr:first-child{display:none}tr{margin:16px 0;padding:12px;border:1px solid #dce1e8;border-radius:12px}
td{padding:5px 0;border:0;display:grid;grid-template-columns:110px 1fr;gap:8px}
td:before{color:#64748b}td:nth-child(1):before{content:'种子'}td:nth-child(2):before{content:'回报'}
td:nth-child(3):before{content:'曾成功'}td:nth-child(4):before{content:'最终成功'}
td:nth-child(5):before{content:'动作干预步数'}td:nth-child(6){display:block;margin-top:12px}}
</style>'''
    text += '<h1>TD-MPC2 × ManiSkill PushCube</h1><p>真实仿真 · 状态输入 · 潜空间规划 · 固定检查点评测</p>'
    text += '<p>任务成功不代表安全。当前未测量碰撞、接触力等安全信号；视频为真实仿真画面。</p>'
    text += '<p>检查点：' + html.escape(report["checkpoint"]) + '</p><p>干预：' + html.escape(json.dumps(report["intervention"])) + '</p>'
    text += '<table><tr><th>种子</th><th>回报</th><th>曾成功</th><th>最终成功</th><th>动作干预步数</th><th>录像</th></tr>' + ''.join(rows) + '</table>'
    (output / "visualization.html").write_text(text, encoding="utf-8")
