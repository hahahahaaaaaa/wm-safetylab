"""What a plugin is: one class, one stage, one role, and the hooks of that stage.

A plugin changes or watches one point of the closed loop. The episode kernel
(wms.episodes.kernel) calls its hooks; the plugin never calls the model or the
environment itself (except ENV plugins, which receive ctx.env).

  stage     where it acts: ENV (the simulated world itself, a boundary condition, not S1-S6),
            S1 observation/instruction, S2 state/memory, S3 prediction, S4 planning,
            S5 action, S6 feedback/update (S6 runs in round 2; the kernel rejects it for now)
  role      fault (accidental), attack (adversarial; needs a threat card),
            defense (needs no simulator truth unless declared an oracle), monitor (read-only)
  side      "env" (works on what crosses the loop) or "model" (reads model internals)
  privileged  may read the simulator's true state through ctx.truth() / ctx.hazards
  requires  model probes ("state", "predict", "plan") and environment abilities
            ("image_box", "set_object_pose", "set_goal", "hazards") the plugin needs
  threat    attack only: {"knowledge", "access", "budget", "goal", "adaptive"(optional)}

Hooks (implement only those of your stage; each returns the possibly changed value
and never modifies its argument in place):

  ENV  on_env(t, ctx) -> None                  change the world through ctx.env, then ctx.log(...)
  S1   on_observation(obs, ctx) -> obs         the observation about to reach the model
       on_instruction(text, ctx) -> text       the language instruction
  S2   on_state(state, ctx) -> state           the model's memory / latent (ctx.phase "read" or "written")
  S3   on_prediction(prediction, ctx) -> prediction   a predicted future (ctx.meta says which)
  S4   on_plan(plan, ctx) -> plan              {"candidates", "scores", ...}; change scores to re-rank
  S5   on_action(action, ctx) -> action        the action about to reach the simulator;
                                               ctx.intervene("stop" | "fallback", ...) to take over

Every plugin may also implement reset(episode, ctx), after_step(step, ctx) (read-only,
after the simulator step), end_episode(ctx) -> dict (added to the episode summary) and close().
"""
from __future__ import annotations

STAGES = ("ENV", "S1", "S2", "S3", "S4", "S5", "S6")
ROLES = ("fault", "attack", "defense", "monitor")
SIDES = ("env", "model")
HOOKS = {"ENV": ("on_env",), "S1": ("on_observation", "on_instruction"), "S2": ("on_state",),
         "S3": ("on_prediction",), "S4": ("on_plan",), "S5": ("on_action",), "S6": ("on_feedback", "on_update")}
ALL_HOOKS = tuple(hook for hooks in HOOKS.values() for hook in hooks)
COMMON = ("reset", "after_step", "end_episode", "close")
PROBES = ("state", "predict", "plan")
ENV_ABILITIES = ("image_box", "set_object_pose", "set_goal", "hazards")
THREAT_KEYS = ("knowledge", "access", "budget", "goal")
KNOWLEDGE = ("black_box", "gray_box", "white_box")
ROLE_ORDER = {"fault": 0, "attack": 0, "defense": 1, "monitor": 2}
NOT_YET = {"S6": "S6 (feedback/update) hooks run in round 2 (P2); the episode kernel does not call them yet"}


class Plugin:
    stage = None
    role = None
    side = "env"
    privileged = False
    requires = ()
    threat = None

    def __init__(self, config):
        self.config = dict(config or {})

    def reset(self, episode, ctx):
        pass

    def close(self):
        pass


def implemented_hooks(cls):
    """Hook names the class itself defines (not inherited from Plugin)."""
    return [name for name in ALL_HOOKS if callable(getattr(cls, name, None))]


def threat_card(cls, config):
    card = dict(getattr(cls, "threat", None) or {})
    card.update((config or {}).get("threat") or {})
    return card


def metadata(cls, config=None):
    config = config or {}
    return {"stage": cls.stage, "role": cls.role, "side": getattr(cls, "side", "env"),
            "privileged": bool(getattr(cls, "privileged", False)),
            "oracle": bool(config.get("oracle", False)),
            "requires": list(getattr(cls, "requires", ()) or ()),
            "hooks": implemented_hooks(cls),
            "threat": threat_card(cls, config) if cls.role == "attack" else None}


def problems(cls, config=None):
    """Everything wrong with a plugin class and its config, as readable sentences (empty = valid)."""
    config = config or {}
    out = []
    name = f"{cls.__module__}:{cls.__name__}"
    if cls.stage not in STAGES:
        return [f"{name}: stage must be one of {list(STAGES)} (got {cls.stage!r})"]
    if cls.role not in ROLES:
        out.append(f"{name}: role must be one of {list(ROLES)} (got {cls.role!r})")
    if getattr(cls, "side", "env") not in SIDES:
        out.append(f"{name}: side must be 'env' or 'model'")
    if cls.stage in NOT_YET:
        out.append(f"{name}: {NOT_YET[cls.stage]}")
    hooks = implemented_hooks(cls)
    own = HOOKS[cls.stage]
    foreign = [hook for hook in hooks if hook not in own]
    if foreign:
        out.append(f"{name}: stage {cls.stage} may implement only {list(own)}; also found {foreign}")
    if not any(hook in own for hook in hooks):
        out.append(f"{name}: implements none of the {cls.stage} hooks {list(own)}")
    unknown = [r for r in getattr(cls, "requires", ()) or () if r not in PROBES + ENV_ABILITIES]
    if unknown:
        out.append(f"{name}: unknown requires {unknown}; use {list(PROBES + ENV_ABILITIES)}")
    if cls.stage in ("S2", "S3", "S4"):
        probe = {"S2": "state", "S3": "predict", "S4": "plan"}[cls.stage]
        if probe not in (getattr(cls, "requires", ()) or ()):
            out.append(f"{name}: a {cls.stage} plugin must list '{probe}' in requires")
        if getattr(cls, "side", "env") != "model":
            out.append(f"{name}: {cls.stage} works on model internals, so side must be 'model'")
    if cls.stage == "ENV" and cls.role in ("defense", "monitor"):
        out.append(f"{name}: ENV plugins change the world; a {cls.role} belongs to the stage it guards or watches")
    if cls.role == "defense" and getattr(cls, "privileged", False) and not config.get("oracle"):
        out.append(f"{name}: a privileged defense reads simulator truth a deployed system does not have; "
                   "set config.oracle = true to run it as an oracle upper bound (results are labelled so)")
    if cls.role == "attack":
        card = threat_card(cls, config)
        missing = [key for key in THREAT_KEYS if card.get(key) in (None, "", [], {})]
        if missing:
            out.append(f"{name}: an attack needs a threat card with {list(THREAT_KEYS)}; missing {missing}")
        elif card["knowledge"] not in KNOWLEDGE:
            out.append(f"{name}: threat.knowledge must be one of {list(KNOWLEDGE)}")
    return out
