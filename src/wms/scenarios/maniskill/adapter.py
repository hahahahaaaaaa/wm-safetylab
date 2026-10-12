"""Panda PushCube adapter. Observations are simulator state, not image perception.

The control command is seven absolute arm joint targets plus one normalized
finger command. The fixed hold policy only exercises that interface.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

STATE_SIZE = 28
ACTION_SIZE = 8
ACTION_BOUNDS = [(-2.8973, 2.8973), (-1.7628, 1.7628), (-2.8973, 2.8973),
                 (-3.0718, -0.0698), (-2.8973, 2.8973), (-0.0175, 3.7525),
                 (-2.8973, 2.8973), (-1.0, 1.0)]


def finite_vector(value, size, label):
    if len(value) != size or any(not math.isfinite(float(x)) for x in value):
        raise ValueError(f"{label} requires {size} finite numbers")
    return [float(x) for x in value]


def validate_action(action, bounds=ACTION_BOUNDS):
    value = finite_vector(action, ACTION_SIZE, "Panda pd_joint_pos action")
    if any(x < low or x > high for x, (low, high) in zip(value, bounds)):
        raise ValueError("Panda action outside the controller's native bounds")
    return value


class HoldArmOpenGripperPolicy:
    """Fixed executor, not a learned policy or a world model."""
    def __init__(self, config):
        self.target = None

    def act(self, obs, instruction=None):
        if self.target is None:
            self.target = validate_action(finite_vector(obs, STATE_SIZE, "Panda state")[:7] + [1.0])
        return list(self.target)


class PandaPushCubeEnv:
    def __init__(self, config):
        from wms.engines.maniskill.runtime import load_config, native_environment
        runtime = config.get("runtime_config") or os.environ.get("WMS_MANISKILL_RUNTIME_CONFIG")
        self.runtime = load_config(runtime) if runtime else config
        os.environ.update(native_environment(self.runtime))
        os.environ.update({str(k): str(v) for k, v in config.get("env", {}).items()})
        self.env = None
        self.active = False
        self.max_steps = int(config.get("max_steps", 500))
        import numpy as np
        import torch
        import gymnasium as gym
        import mani_skill.envs  # registers PushCube-v1
        self.np, self.torch = np, torch
        try:
            self.env = gym.make("PushCube-v1", robot_uids="panda", num_envs=1,
                obs_mode="rgb", control_mode="pd_joint_pos", render_mode="rgb_array",
                max_episode_steps=self.max_steps,
                sim_backend=config.get("sim_backend", "physx_cpu"),
                render_backend=config.get("render_backend", "gpu"))
            self.native = self.env.unwrapped
            lows = self._array(self.env.action_space.low).reshape(-1)
            highs = self._array(self.env.action_space.high).reshape(-1)
            if len(lows) != ACTION_SIZE or len(highs) != ACTION_SIZE:
                raise ValueError("Panda pd_joint_pos needs eight action coordinates")
            self.bounds = list(zip(lows.tolist(), highs.tolist()))
        except BaseException:
            self.close()
            raise

    def _array(self, value):
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return self.np.asarray(value)

    def _scalar(self, value, cast):
        return cast(self._array(value).reshape(-1).item())

    def _state(self):
        n = self.native
        values = [n.agent.robot.get_qpos(), n.agent.robot.get_qvel(),
                  n.obj.pose.raw_pose, n.goal_region.pose.p]
        return finite_vector([x for value in values for x in self._array(value).reshape(-1).tolist()],
                             STATE_SIZE, "Panda qpos/qvel/cube_pose/goal state")

    def reset(self, seed, inputs):
        self.np.random.seed(seed)
        self.torch.manual_seed(seed)
        self.observation, self.reset_info = self.env.reset(seed=seed)
        self.steps, self.active = 0, True
        return self._state()

    def step(self, action):
        if not self.active:
            raise RuntimeError("Call reset before stepping a new episode")
        command = self.np.asarray(validate_action(action, self.bounds), dtype=self.np.float32)
        observation, reward, terminated, truncated, info = self.env.step(command.reshape(self.env.action_space.shape))
        self.observation = observation
        self.steps += 1
        terminated = self._scalar(terminated, bool)
        truncated = self._scalar(truncated, bool) or self.steps >= self.max_steps
        success = self._scalar(info["success"], bool)
        self.active = not (terminated or truncated)
        return {"obs": self._state(), "reward": self._scalar(reward, float),
                "terminated": terminated, "truncated": truncated,
                "events": ["success"] if success else [],
                "info": {"task_success": success, "observation_source": "privileged_simulator_state"}}

    def save_frame(self, path):
        import imageio.v2 as imageio
        rgb = self._array(self.observation["sensor_data"]["base_camera"]["rgb"])[0]
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(path, rgb)
        return str(path)

    def close(self):
        self.active = False
        if self.env is not None:
            self.env.close()
            self.env = None
