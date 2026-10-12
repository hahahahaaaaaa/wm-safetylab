"""Collect real PushCube transitions; simulator imports stay in this worker."""
from pathlib import Path
import importlib.metadata
import json
import os


def collect(output_dir, params, context, runtime):
    from wms.engines.maniskill.runtime import native_environment
    os.environ.update(native_environment(runtime))
    import numpy as np
    import torch
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    from mani_skill.examples.motionplanning.panda.solutions import push_cube

    output = Path(output_dir)
    count = int(params.get("episodes", 36))
    if count < 6:
        raise ValueError("At least six episodes are needed for separate train/validation/test sets")
    budget = int(params.get("max_episode_steps", 160))
    perturb_std = float(params.get("action_noise_std_rad", 0.003))
    if perturb_std < 0 or perturb_std > 0.05:
        raise ValueError("The engineering recipe supports joint-target noise in [0, 0.05] rad")
    start_seed = int(params.get("data_seed_start", 0))
    split_rng = np.random.default_rng(int(params.get("split_seed", 20261010)))
    perm = split_rng.permutation(count)
    n_test = max(1, count // 6)
    n_val = max(1, count // 6)
    split = {"train": sorted(perm[n_test + n_val:].tolist()),
             "validation": sorted(perm[n_test:n_test + n_val].tolist()),
             "test": sorted(perm[:n_test].tolist())}
    record_video = bool(params.get("record_video", False))
    recordings = []
    # Rendering is for evidence only; the controller and learned WM still use state.
    env = gym.make("PushCube-v1", robot_uids="panda", num_envs=1,
                   obs_mode="state", control_mode="pd_joint_pos",
                   sim_backend="physx_cpu", render_backend="gpu",
                   render_mode="rgb_array" if record_video else None,
                   max_episode_steps=budget)
    native = env.unwrapped
    episode_rows, states, next_states, planned, actions, episode_ids, times = [], [], [], [], [], [], []

    def array(value):
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        value = np.asarray(value)
        if not np.isfinite(value).all():
            raise ValueError("Non-finite simulator signal")
        return value

    def state():
        value = np.concatenate([array(native.agent.robot.get_qpos()).reshape(-1),
                               array(native.agent.robot.get_qvel()).reshape(-1),
                               array(native.obj.pose.p).reshape(-1),
                               array(native.agent.tcp.pose.p).reshape(-1)]).astype(np.float32)
        if value.shape != (24,):
            raise ValueError(f"Unexpected state dimension: {value.shape}")
        return value

    def capture():
        if not recording:
            return
        rgb = array(env.render())
        if rgb.ndim == 4 and rgb.shape[0] == 1:
            rgb = rgb[0]
        if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
            raise ValueError(f"Expected uint8 RGB render, got {rgb.shape} {rgb.dtype}")
        frames.append(rgb.copy())

    class EpisodeFinished(Exception):
        pass

    class Recorder(gym.Wrapper):
        def reset(self, **kwargs):
            context.check_cancelled()
            if row["steps"] or frames:
                raise RuntimeError("Do not silently reset or filter a recorded episode")
            observation, info = self.env.reset(**kwargs)
            row["goal_position"] = array(native.goal_region.pose.p).reshape(-1).tolist()
            capture()
            return observation, info

        def step(self, action):
            context.check_cancelled()
            if row["steps"] >= budget or row["terminated"] or row["truncated"]:
                raise EpisodeFinished()
            desired = array(action).reshape(-1).astype(np.float32)
            if desired.shape != (8,):
                raise ValueError(f"Unexpected action dimension: {desired.shape}")
            executed = desired.copy()
            executed[:7] += rng.normal(0, perturb_std, 7)
            bounds = self.env.unwrapped.single_action_space
            clipped = np.clip(executed, bounds.low, bounds.high).astype(np.float32)
            row["clipped_commands"] += int(not np.array_equal(clipped, executed))
            before = state()
            observation, reward, terminated, truncated, info = self.env.step(clipped)
            states.append(before)
            next_states.append(state())
            planned.append(desired)
            actions.append(clipped)
            episode_ids.append(episode)
            times.append(row["steps"])
            row["steps"] += 1
            row["terminated"] = bool(array(terminated).item())
            row["truncated"] = bool(array(truncated).item())
            row["success_once"] |= bool(array(info["success"]).item())
            row["final_success"] = bool(array(info["success"]).item())
            capture()
            return observation, reward, terminated, truncated, info

    wrapped = Recorder(env)
    try:
        for episode in range(count):
            recording = record_video and episode in split["test"]
            frames = []
            seed = start_seed + episode
            np.random.seed(seed)
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed + 1_000_000)
            row = {"episode": episode, "seed": seed, "perturb_seed": seed + 1_000_000,
                   "steps": 0, "terminated": False, "truncated": False,
                   "success_once": False, "final_success": False, "clipped_commands": 0,
                   "planner_status": "completed"}
            try:
                value = push_cube.solve(wrapped, seed=seed, debug=False, vis=False)
                if isinstance(value, (int, np.integer)) and value == -1:
                    row["planner_status"] = "no_plan"
            except EpisodeFinished:
                row["planner_status"] = "episode_ended"
            if row["steps"] == 0:
                raise RuntimeError(f"Episode {episode} produced no data; do not silently filter it")
            if recording:
                import imageio.v2 as imageio
                if len(frames) != row["steps"] + 1:
                    raise RuntimeError("Video must contain initial state plus every post-action state")
                video = output / "videos" / f"episode_{episode:03d}.mp4"
                video.parent.mkdir(parents=True, exist_ok=True)
                with imageio.get_writer(video, fps=float(native.control_freq), codec="libx264",
                                        quality=8, macro_block_size=1,
                                        ffmpeg_params=["-movflags", "+faststart"]) as writer:
                    for frame in frames:
                        context.check_cancelled()
                        writer.append_data(frame)
                recordings.append({"episode": episode, "seed": seed, "path": video.relative_to(output).as_posix(),
                    "frames": len(frames), "steps": row["steps"], "width": frames[0].shape[1],
                    "height": frames[0].shape[0], "goal_position": row["goal_position"],
                    "frame_alignment": "frame 0 initial state; frame k+1 next_states for action step k"})
            episode_rows.append(row)
            context.event("progress", {"phase": "collect", "episode": episode + 1,
                                        "episodes": count, "steps": row["steps"],
                                        "success": row["success_once"]})
        if record_video:
            recording_report = {"schema": "wms.state-wm-maniskill.recordings/1", "fps": float(native.control_freq),
                "selection": "all held-out test episodes; chosen by the original split before collection",
                "camera": "ManiSkill human render camera (evidence only; obs_mode remains state)",
                "episodes": recordings}
            (output / "recordings.json").write_text(json.dumps(recording_report, indent=2), encoding="utf-8")
        np.savez_compressed(output / "dataset.npz", states=np.stack(states),
                            next_states=np.stack(next_states), planned_actions=np.stack(planned),
                            actions=np.stack(actions), episode_ids=np.asarray(episode_ids),
                            steps=np.asarray(times))
        metadata = {"schema": "wms.state-wm-maniskill.dataset/1", "task": "PushCube-v1",
                    "robot": "panda", "observation_source": "privileged simulator state",
                    "state_features": [{"name": "qpos_arm", "slice": [0, 7], "unit": "rad"},
                        {"name": "qpos_fingers", "slice": [7, 9], "unit": "m"},
                        {"name": "qvel_arm", "slice": [9, 16], "unit": "rad/s"},
                        {"name": "qvel_fingers", "slice": [16, 18], "unit": "m/s"},
                        {"name": "cube_xyz", "slice": [18, 21], "unit": "m"},
                        {"name": "tcp_xyz", "slice": [21, 24], "unit": "m"}],
                    "action_semantics": "7 absolute arm joint targets in rad; 1 normalized gripper command",
                    "model_action_input": "actions = actually executed clipped target, not planned_actions",
                    "control_mode": "pd_joint_pos", "control_freq_hz": float(native.control_freq),
                    "sim_freq_hz": float(native.sim_freq), "sim_backend": str(native.backend.sim_backend),
                    "action_noise_std_rad": perturb_std, "max_episode_steps": budget,
                    "split_seed": int(params.get("split_seed", 20261010)), "split": split,
                    "episodes": episode_rows, "transition_count": len(states),
                    "versions": {name: importlib.metadata.version(name) for name in
                                 ("mani-skill", "sapien", "torch", "numpy", "gymnasium", "mplib")},
                    "claim_limit": "Partial state under a fixed controller; no visual perception, full Markov-state guarantee, safety guarantee or paper reproduction."}
        (output / "dataset.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return metadata
    finally:
        env.close()
