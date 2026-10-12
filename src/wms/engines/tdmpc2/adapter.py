"""Optional small components for the platform's generic predict/rollout tasks.

The official MPPI policy stays inside act(). Generic platform CEM is not layered
over TD-MPC2 a second time. Predictions are explicitly latent representations.
"""
from . import compat


class ModelAdapter:
    def __init__(self, config):
        self.config = config
        self.agent = self.cfg = None
        self.first = True

    def _load(self, seed):
        if self.agent is None:
            from wms.engines.runtime import load_runtime
            runtime = load_runtime(self.config["runtime_config"])
            compat.prepare(runtime)
            compat.seed_all(seed)
            from .native import _source
            checkpoint = _source(self.config, self.config.get("output_dir", "."))
            self.config = {**self.config, "checkpoint": str(checkpoint)}
            self.agent, self.cfg, self.metadata = compat.checkpoint_load(checkpoint)
            # Checkpoint construction initializes then replaces weights. Use an
            # explicit policy RNG stream, independently of that initialization.
            compat.seed_all(seed)

    def predict(self, request):
        seed = int(request.get("seed", self.config.get("seed", 0)))
        self._load(seed)
        import torch
        from common.math import two_hot_inv
        obs = torch.as_tensor(request["observation"], device="cuda", dtype=torch.float32).reshape(1, -1)
        if obs.shape[1] != self.cfg.obs_shape["state"][0]:
            raise ValueError("Observation must match the checkpoint's full ManiSkill state layout")
        predictions, rewards = [], []
        with torch.no_grad():
            z = self.agent.model.encode(obs, None)
            for raw in request["actions"]:
                action = torch.as_tensor(raw, device="cuda", dtype=torch.float32).reshape(1, -1)
                if action.shape[1] != self.cfg.action_dim or not bool(torch.isfinite(action).all()) or bool((action.abs() > 1).any()):
                    raise ValueError("Actions must be normalized pd_joint_delta_pos with matching dimension")
                rewards.append(float(two_hot_inv(self.agent.model.reward(z, action, None), self.cfg).item()))
                z = self.agent.model.next(z, action, None)
                predictions.append(z[0].cpu().tolist())
        return {"predictions": predictions, "predicted_rewards": rewards,
                "representation": "tdmpc2_latent", "physical_state_decoded": False,
                "checkpoint": str(self.config["checkpoint"]), "seed": seed}

    def act(self, observation, instruction=None):
        self._load(int(self.config.get("seed", 0)))
        import torch
        from .backend import act
        obs = torch.as_tensor(observation, dtype=torch.float32).reshape(1, -1)
        action = act(self.agent, obs, first=self.first)
        self.first = False
        return action[0].tolist()

    def reset(self, seed=0):
        """For direct Python multi-episode use; generic rollout creates a fresh adapter.

        The generic runner does not call this hook; its recipe must explicitly
        set model.config.seed to the same value as the top-level episode seed.
        """
        self.config = {**self.config, "seed": int(seed)}
        self._load(seed)
        compat.seed_all(seed)
        self.first = True

    def close(self):
        self.agent = None


class EnvironmentAdapter:
    def __init__(self, config):
        from wms.engines.runtime import load_runtime
        compat.prepare(load_runtime(config["runtime_config"]))
        import torch
        from omegaconf import OmegaConf
        from .native import _source
        checkpoint = _source(config, config.get("output_dir", "."))
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        cfg = OmegaConf.create(saved["config"])
        cfg.num_envs = cfg.num_eval_envs = 1
        self.env = compat.StateEnvironment(cfg, 1)

    def reset(self, seed, inputs):
        obs, _ = self.env.reset(seed)
        return obs[0].detach().cpu().tolist()

    def step(self, action):
        import torch
        obs, reward, terminated, truncated, info = self.env.step(torch.as_tensor(action).reshape(1, -1))
        success = bool(info["success"].item())
        return {"obs": obs[0].detach().cpu().tolist(), "reward": float(reward.item()),
                "terminated": False, "truncated": self.env.steps >= self.env.horizon,
                "events": ["task_success"] if success else [],
                "info": {"task_success": success, "native_terminated": bool(terminated.item()),
                         "safety_signal_status": "not_measured", "controller": "pd_joint_delta_pos"}}

    def close(self):
        self.env.close()
