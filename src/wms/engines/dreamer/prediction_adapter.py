"""Read-only open-loop prediction using the pinned official DreamerV3 RSSM.

Run this adapter in the locked native Dreamer environment.  The platform core
only sees JSON and artifacts; importing this module does not import JAX.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
import time

from wms.engines.dreamer.runtime import load_config

MODEL_ID = "dreamerv3-cartpole-predict"
INPUT_FIELDS = {"context_observations", "context_actions", "actions"}


def validate_prediction_inputs(request: dict) -> tuple[list, list, list]:
    """Reject ambiguous action alignment, reset boundaries, and future targets."""
    from wms.core.world_model import validate_request

    validate_request(request)
    if request["model_id"] != MODEL_ID:
        raise ValueError("Request model_id does not identify this Dreamer adapter")
    if request["horizon"]["unit"] != "environment_step":
        raise ValueError("Dreamer horizon unit must be environment_step")
    steps = request["horizon"]["steps"]
    if not 1 <= steps <= 64:
        raise ValueError("Dreamer engineering profile supports 1..64 future steps")
    if request.get("parameters", {}) != {}:
        raise ValueError("This read-only Dreamer profile has no tunable parameters")
    inputs = request["inputs"]
    if set(inputs) != INPUT_FIELDS:
        raise ValueError("Dreamer inputs require only context_observations, context_actions, actions")
    observations = inputs["context_observations"]
    if not isinstance(observations, list) or not 1 <= len(observations) <= 128:
        raise ValueError("context_observations must contain 1..128 states")
    for state in observations:
        if not isinstance(state, list) or len(state) != 4:
            raise ValueError("Each CartPole state must contain four numeric coordinates")
        if any(type(x) not in (int, float) or not math.isfinite(x) for x in state):
            raise ValueError("CartPole state coordinates must be finite numbers")
    history, actions = inputs["context_actions"], inputs["actions"]
    for name, values, length in (
        ("context_actions", history, len(observations) - 1),
        ("actions", actions, steps),
    ):
        if not isinstance(values, list) or len(values) != length:
            raise ValueError(f"{name} must have length {length}")
        if any(type(value) is not int or value not in (0, 1) for value in values):
            raise ValueError(f"{name} must contain integer CartPole actions 0 or 1")
    return observations, history, actions


def _normalize_config(config: dict) -> dict:
    return load_config(config)


class DreamerPredictionAdapter:
    """Posterior context -> fixed actions -> stochastic prior predictions.

    This adapter deliberately does not call the learned actor, train the agent,
    or score physical safety.  The existing native infer/worker stays separate.
    """

    def __init__(self, config_dict: dict):
        self.config = _normalize_config(config_dict)

    def describe(self) -> dict:
        weights = self.config["weights"]
        return {
            "schema_version": "wms.model/1.0",
            "model_id": MODEL_ID,
            "name": "DreamerV3 CartPole engineering checkpoint: fixed-action prediction",
            "families": ["latent_state_space"],
            "representations": ["latent", "numeric_state"],
            "capabilities": ["predict", "trace"],
            "action_conditioning": "discrete_action",
            "limitations": [
                "Engineering checkpoint with 32 training updates; not converged or a DMC policy baseline.",
                "CartPole numeric-state observations only; one episode context, batch size one.",
                "Future states are decoded model predictions, not simulator ground truth.",
                "One stochastic latent trajectory per seed; no calibrated physical-state likelihood.",
                "Fixed external actions; no online planning, environment execution, or safety guarantee.",
                "Continuation head uses the checkpoint's discounted continuation training target.",
            ],
            "provenance": {
                "implementation": "official_danijar_dreamerv3",
                "source_commit": weights["source"]["commit"],
                "checkpoint_class": weights["checkpoint_class"],
                "converged": False,
                "training_updates": weights["training_updates"],
            },
        }

    def predict(self, request: dict, output_dir: Path) -> dict:
        from wms.core.world_model import validate_request, write_prediction_bundle

        observations, history, actions = validate_prediction_inputs(request)
        manifest = self.describe()
        validate_request(request, manifest)
        weights = self.config["weights"]
        environment = {"python": sys.version, "interpreter": sys.executable}
        output = Path(output_dir).resolve()
        checkpoint = Path(self.config["checkpoint_dir"]).resolve()
        source = Path(weights["source"]["source_dir"]).resolve()
        if output == checkpoint or output.is_relative_to(checkpoint) or output.is_relative_to(source):
            raise ValueError("Prediction output may not be inside checkpoint or upstream source")
        output.mkdir(parents=True, exist_ok=True)
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        os.environ.setdefault("OMP_NUM_THREADS", "4")
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
        import numpy as np
        import jax
        import jax.numpy as jnp
        import ninjax as nj
        import elements
        from dreamerv3.agent import Agent

        started = time.monotonic()
        agent_config = json.loads((checkpoint / "agent-config.json").read_text())
        agent_config.update(logdir=str(output), seed=request["seed"])
        obs_space = {
            "state": elements.Space(np.float32, (4,)),
            "reward": elements.Space(np.float32),
            "is_first": elements.Space(bool),
            "is_last": elements.Space(bool),
            "is_terminal": elements.Space(bool),
        }
        act_space = {"action": elements.Space(np.int32, (), 0, 2)}
        agent = Agent(obs_space, act_space, elements.Config(agent_config))
        with np.load(checkpoint / "checkpoint.npz", allow_pickle=False) as archive:
            parameters = {name: archive[name] for name in archive.files}
        expected = agent.save()["params"]
        if set(parameters) != set(expected):
            raise ValueError("Checkpoint parameter names do not match the pinned model")
        for name, value in parameters.items():
            if value.shape != expected[name].shape or value.dtype != expected[name].dtype:
                raise ValueError("Checkpoint tensor shape/dtype mismatch: " + name)
            if not np.isfinite(value).all():
                raise ValueError("Non-finite checkpoint tensor: " + name)
        counters = json.loads((checkpoint / "counters.json").read_text())
        agent.load({"params": parameters, "counters": counters})
        model = agent.model
        horizon = len(actions)
        context_length = len(observations)

        def forward():
            reset = jnp.asarray([[True] + [False] * (context_length - 1)])
            context = {"state": jnp.asarray([observations], dtype=jnp.float32)}
            previous = {"action": jnp.asarray([[0] + history], dtype=jnp.int32)}
            _, _, tokens = model.enc(model.enc.initial(1), context, reset, training=False)
            carry, _, posterior = model.dyn.observe(
                model.dyn.initial(1), tokens, previous, reset, training=False)
            context_prior_logits = model.dyn._prior(posterior["deter"])
            _, future, used_actions = model.dyn.imagine(
                carry, {"action": jnp.asarray([actions], dtype=jnp.int32)},
                length=horizon, training=False)
            _, _, decoded = model.dec(
                model.dec.initial(1), future,
                jnp.zeros((1, horizon), dtype=bool), training=False)
            tensor = model.feat2tensor(future)
            return {
                "context_deter": posterior["deter"],
                "context_stoch": posterior["stoch"],
                "context_posterior_logits": posterior["logit"],
                "context_prior_logits": context_prior_logits,
                "future_deter": future["deter"],
                "future_stoch": future["stoch"],
                "future_prior_logits": future["logit"],
                "future_decoded_state": decoded["state"].pred(),
                "future_reward": model.rew(tensor, 2).pred(),
                "future_continue": model.con(tensor, 2).prob(1),
                "future_actions": used_actions["action"],
            }

        next_params, values = jax.jit(nj.pure(forward))(agent.params, seed=request["seed"])
        values = {name: np.asarray(value) for name, value in jax.device_get(values).items()}
        if any(not np.isfinite(value).all() for value in values.values()):
            raise ValueError("Prediction contains non-finite outputs")
        if values["future_decoded_state"].shape != (1, horizon, 4):
            raise ValueError("Decoded future has an unexpected shape")
        if not np.array_equal(values["future_actions"], np.asarray([actions])):
            raise ValueError("RSSM did not use the requested fixed action sequence")
        continuation = values["future_continue"]
        if continuation.shape != (1, horizon) or np.any((continuation < 0) | (continuation > 1)):
            raise ValueError("Continuation output is not a probability array")

        artifacts = []
        for name, value in values.items():
            path = output / (name + ".npy")
            np.save(path, value, allow_pickle=False)
            is_latent = name.endswith(("deter", "stoch", "logits"))
            kind = "latent" if is_latent else "numeric_state" if "decoded_state" in name else "score" if name.endswith(("reward", "continue")) else "trace"
            axes = ["batch", "context_step" if name.startswith("context_") else "future_step"]
            axes += ["feature"] if value.ndim == 3 else ["categorical_variable", "category"] if value.ndim == 4 else []
            artifacts.append({
                "name": name, "path": path.name, "kind": kind,
                "media_type": "application/x-npy", "shape": list(value.shape),
                "axes": axes,
                "semantics": (
                    "Unnormalized categorical RSSM logits; distribution applies checkpoint unimix."
                    if name.endswith("logits") else
                    "Learned discounted continuation probability, not a safety probability."
                    if name == "future_continue" else
                    "Decoded predicted CartPole coordinates; not actual simulator state."
                    if name == "future_decoded_state" else name
                ),
            })
        trace = {
            "schema_version": "dreamer.prediction.trace/1.0",
            "context_length": context_length,
            "context_observations": observations,
            "context_actions": history,
            "action_alignment": "action[t] moves predicted state s[C-1+t] to s[C+t]",
            "future_observations_consumed": 0,
            "steps": [
                {"offset": index + 1, "action": actions[index],
                 "decoded_state": values["future_decoded_state"][0, index].tolist(),
                 "reward": float(values["future_reward"][0, index]),
                 "continuation": float(values["future_continue"][0, index])}
                for index in range(horizon)
            ],
        }
        (output / "trace.json").write_text(json.dumps(trace, indent=2, allow_nan=False), encoding="utf-8")
        artifacts.append({"name": "prediction_trace", "path": "trace.json", "kind": "trace", "media_type": "application/json"})
        provenance = {
            **manifest["provenance"],
            "checkpoint_sha256": weights["files"]["checkpoint.npz"],
            "checkpoint_files_sha256": weights["files"],
            "source_files_sha256": weights["source"]["files"],
            "dataset_revision": weights["dataset"]["revision"],
            "environment": environment,
            "closed_loop_evaluation": False,
        }
        return write_prediction_bundle(
            output, request, manifest, artifacts, mode="open_loop",
            diagnostics={
                "context_steps": context_length, "future_steps": horizon,
                "future_observations_consumed": 0,
                "fixed_actions_verified": True,
                "elapsed_seconds": time.monotonic() - started,
                "rssm_unimix": float(agent_config["dyn"]["rssm"]["unimix"]),
                "continuation_discount": 1 - 1 / agent_config["horizon"] if agent_config["contdisc"] else 1.0,
            },
            provenance=provenance,
            claims=[
                "Real posterior assimilation followed by official RSSM fixed-action imagination.",
                "Executable prediction path only; predictive quality is not validated.",
                "No calibrated safety probability, converged policy, closed-loop control, or DMC benchmark claim.",
            ],
        )
