"""A real Gymnasium environment behind the platform's small Env interface."""
from __future__ import annotations


class PendulumAdapter:
    """One environment, one episode: obs=(cos(theta), sin(theta), rad/s).

    Actions are [torque_Nm], bounded to [-2, 2]. This example has no collision
    sensors or safety-cost definition. Missing safety measurements stay absent.
    """

    def __init__(self, config: dict):
        # These packages belong to runtime.python, never the Web process.
        import gymnasium as gym
        import numpy as np

        self.np = np
        self.config = dict(config)
        self.closed = False
        self.env = gym.make(
            "Pendulum-v1", g=float(config.get("gravity", 10.0)),
            render_mode="rgb_array" if config.get("record_video", False) else None,
            max_episode_steps=int(config.get("max_episode_steps", 200)),
        )
        self.reset_info = {}

    def reset(self, seed: int, inputs: dict):
        self._ensure_open()
        observation, info = self.env.reset(seed=int(seed), options=inputs.get("reset_options"))
        self.env.action_space.seed(int(seed))
        self.reset_info = dict(info)
        return observation.tolist()

    def step(self, action):
        self._ensure_open()
        value = self.np.asarray(action, dtype=self.np.float32)
        if value.shape != (1,) or not self.np.isfinite(value).all():
            raise ValueError("Pendulum action must be one finite torque: [torque_Nm]")
        if not self.env.action_space.contains(value):
            raise ValueError("Pendulum torque must be within [-2, 2] N m")
        observation, reward, terminated, truncated, info = self.env.step(value)
        return {
            "obs": observation.tolist(), "reward": float(reward),
            "terminated": bool(terminated), "truncated": bool(truncated),
            "info": {**info, "safety_signal_status": "unavailable"},
        }

    def render_frame(self):
        """Optional RGB capability; not part of the required Env interface."""
        self._ensure_open()
        if not self.config.get("record_video", False):
            raise ValueError("Create the adapter with record_video=true before rendering")
        return self.env.render()

    def metadata(self):
        import gymnasium
        native = self.env.unwrapped
        return {
            "environment": "Pendulum-v1", "gymnasium_version": gymnasium.__version__,
            "observation_fields": ["cos_theta", "sin_theta", "angular_velocity_rad_s"],
            "action_fields": ["torque_Nm"], "action_bounds": [[-2.0, 2.0]],
            "dt_s": float(native.dt), "gravity_m_s2": float(native.g),
            "safety_signal_status": "unavailable", "controller": "constant_torque_teaching_policy",
        }

    def _ensure_open(self):
        if self.closed:
            raise RuntimeError("Environment is closed")

    def close(self):
        if not self.closed:
            self.env.close()
            self.closed = True
