"""A sampling planner with an explicit (hand-written) world model, for wms.scenarios.toy.env:ToyPickEnv.

It exists so that S2-S4 plugins can be written and tested on a CPU before they meet
TD-MPC2: it has the same three probes (state, predict, plan) with the same meaning,
in a few lines of numpy. It is not learned and its results are not evidence about
any real world model.

Each step:
  1. state     latent z = [gripper_x, gripper_y, cube_x, cube_y, goal_x, goal_y, holding]: gripper
               from proprioception, cube and goal found by colour in the image, the last estimate
               kept when one is hidden (memory). Offered to on_state ("read") before planning.
  2. predict   the model: gripper += move * step_size (clipped to the unit square); a held cube
               moves with the gripper. Every rollout step of every candidate is offered to
               on_prediction (meta {"kind": "rollout", "h": h}, value (N, 7)); after the choice,
               the predicted next latent for the chosen move is offered as {"kind": "chosen_next"}
               (a report: monitors compare it with the next step's state).
  3. plan      N candidate move sequences of H steps (the straight line to the target, plus
               samples around the previous plan); score = -(final distance to target) -
               0.1 * mean distance. Offered to on_plan as {"candidates" (N, H, 2), "predicted"
               (N, H, 7), "scores" (N,), "target": "cube" | "goal"}; the highest score is executed.
  Gripper: close next to the cube, open on the goal (as CentroidPickPolicy).

config: samples 32, horizon 4, noise 0.5, step_size (from the environment), grasp_distance,
        place_distance, memory_max_age, min_pixels, colour_margin
"""
from __future__ import annotations

import numpy as np

from .policy import CentroidPickPolicy

LATENT = ["gripper_x", "gripper_y", "cube_x", "cube_y", "goal_x", "goal_y", "holding"]


class ToyPlannerPolicy:
    probes = ("state", "predict", "plan")

    def __init__(self, config):
        self.samples = int(config.get("samples", 32))
        self.horizon = int(config.get("horizon", 4))
        self.noise = float(config.get("noise", 0.5))
        self.eyes = CentroidPickPolicy({**config, "memory": True})
        self.step_size = self.eyes.step_size
        self.hooks = None
        self.reset(0)

    def bind_env(self, spec):
        self.eyes.bind_env(spec)
        self.step_size = self.eyes.step_size

    def set_hooks(self, hooks):
        self.hooks = hooks

    def reset(self, seed=None):
        self.eyes.reset(seed)
        self.rng = np.random.default_rng(0 if seed is None else int(seed))
        self.mean = np.zeros((self.horizon, 2))
        self.info = {}

    def export_state(self):
        return {"memory": self.eyes.export_state(), "plan_mean": self.mean.copy()}

    def import_state(self, state):
        if "memory" in state:
            self.eyes.import_state(state["memory"])
        if "plan_mean" in state:
            self.mean = np.asarray(state["plan_mean"], dtype=float).reshape(self.horizon, 2)

    # ------------------------------------------------------------ world model
    def predict(self, z, moves):
        """z (N, 7), moves (N, 2) -> next z (N, 7)."""
        z = np.array(z, dtype=float, copy=True)
        z[:, 0:2] = np.clip(z[:, 0:2] + np.clip(moves, -1, 1) * self.step_size, 0.0, 1.0)
        held = z[:, 6] > 0.5
        z[held, 2:4] = z[held, 0:2]
        return z

    def _hook(self, name, value, meta=None):
        if self.hooks is None or not self.hooks.wants("on_" + name):
            return value
        return getattr(self.hooks, name)(value, meta) if name != "state" else self.hooks.state(value, "read", meta)

    # ------------------------------------------------------------------ act
    def act(self, observation, instruction=None):
        image = np.asarray(observation["rgb"][self.eyes.camera])
        gripper = np.asarray(observation["proprio"][:2], dtype=float)
        holding = float(observation["proprio"][2] > 0.5)
        cube, cube_source, _ = self.eyes._estimate("cube", image, 0)
        goal, goal_source, _ = self.eyes._estimate("goal", image, 1)
        z = np.array([*gripper, *(cube if cube is not None else [np.nan] * 2),
                      *(goal if goal is not None else [np.nan] * 2), holding])
        state = self._hook("state", {"latent": z, "memory": self.eyes.export_state(), "plan_mean": self.mean.copy()})
        z = np.asarray(state["latent"], dtype=float)
        self.import_state({k: v for k, v in state.items() if k != "latent"})
        target_name = "goal" if z[6] > 0.5 else "cube"
        target = z[4:6] if target_name == "goal" else z[2:4]
        if not np.isfinite(target).all():
            self.info = {"phase": target_name + "_wait", "latent": z.tolist(), "plan": None}
            return [0.0, 0.0, 1.0 if z[6] > 0.5 else -1.0]

        n, h = self.samples, self.horizon
        straight = np.clip((target - z[0:2]) / self.step_size, -1, 1)
        candidates = np.clip(self.mean[None] + self.noise * self.rng.standard_normal((n, h, 2)), -1, 1)
        candidates[0] = straight
        rollout = np.repeat(z[None], n, axis=0)
        predicted = np.empty((n, h, len(LATENT)))
        for step in range(h):
            rollout = self._hook("prediction", self.predict(rollout, candidates[:, step]), {"kind": "rollout", "h": step})
            predicted[:, step] = rollout
        reach = np.linalg.norm(predicted[:, :, 0:2] - target, axis=2)
        scores = -reach[:, -1] - 0.1 * reach.mean(axis=1)
        plan = self._hook("plan", {"candidates": candidates, "predicted": predicted, "scores": scores,
                                   "target": target_name})
        scores = np.asarray(plan["scores"], dtype=float)
        chosen = int(np.argmax(scores))
        move = candidates[chosen, 0]
        self.mean = np.vstack([candidates[chosen, 1:], np.zeros((1, 2))])
        nxt = self.predict(z[None], move[None])[0]
        self._hook("prediction", nxt, {"kind": "chosen_next"})
        near = np.linalg.norm(nxt[0:2] - target)
        if z[6] > 0.5:
            grip = -1.0 if near <= self.eyes.place_distance else 1.0
        else:
            grip = 1.0 if near <= self.eyes.grasp_distance else -1.0
        self.info = {"phase": "carry" if z[6] > 0.5 else "reach", "latent": z.tolist(),
                     "estimate_cube": None if cube is None else cube.tolist(), "source_cube": cube_source,
                     "source_goal": goal_source, "chosen": chosen, "chosen_score": float(scores[chosen]),
                     "predicted_next": nxt.tolist()}
        return [float(move[0]), float(move[1]), grip]

    def close(self):
        pass
