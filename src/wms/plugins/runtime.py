"""Run the plugins of one experiment inside the episode kernel.

PluginSet builds the plugins, checks them against the model and environment actually
loaded, calls their hooks in a fixed order and collects what they log:

  per hook: fault / attack first, then defense, then monitor (list order within a role)
  per step: ENV -> S1 (frame t) -> model.act [S2, S3, S4 through the model's probes] -> S5
            -> simulator step -> after_step

Each plugin gets its own context (ctx): its own random stream seeded from (episode seed,
plugin position), so plugins never draw from the global generators and adding one plugin
does not change another's randomness; ctx.log(...) writes to the step's trace.
"""
from __future__ import annotations

import copy

import numpy as np

from .base import ENV_ABILITIES, HOOKS, PROBES, ROLE_ORDER, metadata, problems


class PluginStop(Exception):
    """A defense stopped the episode; planner executors end with this status."""
    episode_status = "plugin_stop"


class PluginContext:
    def __init__(self, owner, item):
        self._owner, self._item = owner, item
        self.id, self.stage, self.role = item.id, item.stage, item.role
        self.privileged = item.privileged
        self.config = item.plugin.config
        self.episode = self.seed = self.rng = None
        self.t, self.hook, self.phase, self.meta, self.final = 0, None, None, None, False

    # -------------------------------------------------------------- always available
    @property
    def spec(self):
        """The environment spec (public: action bounds, cameras, max_steps, safety limits)."""
        return self._owner.spec

    def log(self, data=None, **fields):
        """Add an event to this step's trace (and, for S1/ENV faults and attacks, to steps.jsonl)."""
        record = dict(data or {}, **fields)
        self._owner.events.append({"plugin": self.id, "stage": self.stage, "role": self.role,
                                   "hook": self.hook, "t": self.t, "final": self.final, "data": record})
        return record

    # -------------------------------------------------------------- privileged
    def _needs_privilege(self, what):
        if not (self.privileged or self.role == "monitor"):
            raise PermissionError(f"{self.id} reads {what}; declare privileged = True "
                                  "(a defense must then also set config.oracle = true)")

    def truth(self):
        """The simulator's true state (env.get_state())."""
        self._needs_privilege("the true simulator state")
        return self._owner.env.get_state()

    @property
    def hazards(self):
        """Hazard predicates that held after the last simulator step (env.hazards())."""
        self._needs_privilege("hazard predicates")
        return list(self._owner.hazards)

    def locate(self, name, camera=None):
        """Image box of an object (env.image_box); needs privileged and requires 'image_box'."""
        self._needs_privilege("object positions")
        if "image_box" not in self._item.requires:
            raise PermissionError(f"{self.id} must list 'image_box' in requires to call ctx.locate")
        return self._owner.env.image_box(name, camera)

    @property
    def env(self):
        """The environment itself: ENV plugins only (they change the world, by definition)."""
        if self.stage != "ENV":
            raise PermissionError(f"{self.id}: only ENV plugins may touch the environment")
        return self._owner.env

    # -------------------------------------------------------------- S5 takeover
    def intervene(self, kind, action=None, reason=""):
        """S5 defense: 'stop' ends the episode before this action runs; 'fallback' runs `action`
        (default: spec safe_action, else zeros) instead of the policy's action."""
        if self.hook != "on_action" or self.role != "defense":
            raise PermissionError("only an S5 defense may take over, from on_action")
        if kind not in ("stop", "fallback"):
            raise ValueError("kind must be 'stop' or 'fallback'")
        if kind == "fallback" and action is None:
            spec = self.spec or {}
            action = spec.get("safe_action") or [0.0] * int(spec.get("action_dim", 0))
        self._owner.takeover = {"kind": kind, "plugin": self.id, "reason": reason,
                                "action": None if action is None else np.asarray(action, dtype=float).tolist()}
        return self._owner.takeover


class _Item:
    def __init__(self, index, spec, plugin):
        self.index, self.plugin = index, plugin
        self.name = spec.get("id") or spec.get("plugin") or spec["entrypoint"]
        self.id = f"{index}:{self.name}"
        cls = type(plugin)
        self.stage, self.role = cls.stage, cls.role
        # instances may narrow or widen these from their config (e.g. an occluder that follows an object)
        self.privileged = bool(getattr(plugin, "privileged", False))
        self.requires = tuple(getattr(plugin, "requires", ()) or ())
        self.spec = spec
        self.ctx = None


def _snapshot(value):
    if hasattr(value, "detach") and hasattr(value, "clone"):
        return value.detach().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {k: _snapshot(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_snapshot(v) for v in value) if not hasattr(value, "_fields") else value
    return copy.copy(value)


def _same(a, b):
    if hasattr(a, "detach") and hasattr(b, "detach"):
        return a.shape == b.shape and bool((a == b).all())
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(np.asarray(a), np.asarray(b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


class ModelHooks:
    """Given to model.set_hooks(): the model calls these at the matching point inside act()."""

    def __init__(self, owner):
        self._owner = owner

    def state(self, value, phase="read", meta=None):
        return self._owner.call("on_state", value, phase=phase, meta=meta)

    def prediction(self, value, meta=None):
        return self._owner.call("on_prediction", value, meta=meta)

    def plan(self, value, meta=None):
        return self._owner.call("on_plan", value, meta=meta)

    def wants(self, hook):
        return bool(self._owner.by_hook.get(hook))


class PluginSet:
    def __init__(self, specs, build, strict=False):
        """specs: [{"id", "entrypoint", "config"}]; build(spec) -> plugin instance."""
        self.items = [_Item(index, spec, build(spec)) for index, spec in enumerate(specs or [])]
        self.strict = bool(strict)
        self.by_hook = {}
        for item in self.items:
            for hook in HOOKS.get(item.stage, ()):
                if callable(getattr(item.plugin, hook, None)):
                    self.by_hook.setdefault(hook, []).append(item)
        for hook, items in self.by_hook.items():
            items.sort(key=lambda item: (ROLE_ORDER[item.role], item.index))
        self.env = self.model = None
        self.spec, self.hazards, self.events, self.takeover = {}, [], [], None
        self.ran = {}
        self.t = 0

    def __bool__(self):
        return bool(self.items)

    def has(self, hook):
        return bool(self.by_hook.get(hook))

    def describe(self):
        return [{"id": item.id, "entrypoint": item.spec.get("entrypoint"), "config": item.plugin.config,
                 **metadata(type(item.plugin), item.plugin.config),
                 "privileged": item.privileged, "requires": list(item.requires)} for item in self.items]

    # ------------------------------------------------------------------ checks
    def check(self, env, model, executor="model"):
        """Refuse to run when a plugin's declarations cannot be met by these components."""
        errors = []
        probes = tuple(getattr(model, "probes", ()) or ()) if model is not None else ()
        from .resolve import mismatches
        for item in self.items:
            errors += problems(type(item.plugin), item.plugin.config)
            errors += [f"{item.id}: {text}" for text in mismatches(type(item.plugin), item.spec)]
            if item.privileged and item.role == "defense" and not item.plugin.config.get("oracle"):
                errors.append(f"{item.id}: privileged defense needs config.oracle = true")
            for need in item.requires:
                if need in PROBES:
                    if executor != "model":
                        errors.append(f"{item.id} needs model probe '{need}', but executor {executor!r} has no model")
                    elif need not in probes or not callable(getattr(model, "set_hooks", None)):
                        errors.append(f"{item.id} needs model probe '{need}'; this model offers {list(probes)}")
                elif need in ENV_ABILITIES and not callable(getattr(env, need, None)):
                    errors.append(f"{item.id} needs environment method {need}()")
        if errors:
            raise TypeError("Plugin check failed:\n  " + "\n  ".join(errors))
        self.env, self.model = env, model
        if model is not None and any(self.has(h) for h in ("on_state", "on_prediction", "on_plan")):
            model.set_hooks(ModelHooks(self))

    # ------------------------------------------------------------------ episode
    def begin_episode(self, index, seed, spec):
        self.spec, self.hazards = spec or {}, []
        for item in self.items:
            ctx = item.ctx = PluginContext(self, item)
            ctx.episode, ctx.seed = index, int(seed)
            ctx.rng = np.random.default_rng(np.random.SeedSequence([int(seed) & 0xFFFFFFFF, item.index]))
            ctx.hook = "reset"
            item.plugin.reset({"index": index, "seed": int(seed)}, ctx)

    def begin_step(self, t):
        self.t, self.events, self.takeover, self.ran = t, [], None, {}

    def call(self, hook, value, phase=None, meta=None, final=False):
        for item in self.by_hook.get(hook, ()):
            ctx = item.ctx
            ctx.t, ctx.hook, ctx.phase, ctx.meta, ctx.final = self.t, hook, phase, meta, final
            before = _snapshot(value) if self.strict else None
            out = getattr(item.plugin, hook)(value, ctx)
            if self.strict and not _same(before, value):
                raise RuntimeError(f"{item.id}.{hook} changed its input in place; return a new value instead")
            if item.role != "monitor":
                if out is None:
                    raise TypeError(f"{item.id}.{hook} returned None; return the (possibly changed) value")
                value = out
            self.ran.setdefault(item.stage, []).append(item.id)
            if self.takeover and self.takeover.get("kind") == "stop" and hook == "on_action":
                break
        return value

    def on_env(self, t):
        for item in self.by_hook.get("on_env", ()):
            ctx = item.ctx
            ctx.t, ctx.hook, ctx.phase, ctx.meta, ctx.final = t, "on_env", None, None, False
            item.plugin.on_env(t, ctx)
            self.ran.setdefault("ENV", []).append(item.id)
        return self.records("ENV", "on_env")

    def records(self, stage, hook):
        """What fault / attack plugins of a stage logged this step (the steps.jsonl columns)."""
        return [event["data"] for event in self.events
                if event["stage"] == stage and event["hook"] == hook and event["role"] in ("fault", "attack")
                and not event["final"]]

    def after_step(self, step, hazards):
        self.hazards = list(hazards or [])
        for item in self.items:
            function = getattr(item.plugin, "after_step", None)
            if callable(function):
                ctx = item.ctx
                ctx.t, ctx.hook, ctx.phase, ctx.meta, ctx.final = self.t, "after_step", None, None, False
                function(step, ctx)

    def end_episode(self):
        out = {}
        for item in self.items:
            function = getattr(item.plugin, "end_episode", None)
            if callable(function):
                item.ctx.hook = "end_episode"
                value = function(item.ctx)
                if value:
                    out[item.id] = value
        return out

    def trace(self):
        return {"plugins": self.ran, "events": [e for e in self.events if not e["final"]], "takeover": self.takeover}

    def close(self):
        errors = []
        for item in self.items:
            try:
                item.plugin.close()
            except Exception as exc:  # noqa: BLE001 - close all, report the first
                errors.append(exc)
        if errors:
            raise errors[0]
