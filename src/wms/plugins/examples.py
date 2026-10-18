"""Small teaching plugins: one per kind of hook, so a new plugin can start from the nearest one.

They are examples of the contract (wms.plugins.base), not experiments. They import numpy
only inside hooks, so the service can import them to check a submission.
"""
from __future__ import annotations

from .base import Plugin


class ActionScale(Plugin):
    """S5 fault: multiply the first `dims` action values by `scale` (e.g. a mis-calibrated controller).

    config: scale (default 0.5), dims (default: all), start / end steps (default: whole episode)
    """
    stage, role = "S5", "fault"
    neutral = {"scale": 1.0}

    def on_action(self, action, ctx):
        start, end = int(self.config.get("start", 0)), self.config.get("end")
        if ctx.t < start or (end is not None and ctx.t >= int(end)):
            return action
        values = [float(v) for v in action]
        dims = int(self.config.get("dims", len(values)))
        scale = float(self.config.get("scale", 0.5))
        changed = [v * scale if i < dims else v for i, v in enumerate(values)]
        ctx.log({"scale": scale, "dims": dims})
        return changed


class MemoryAgeMonitor(Plugin):
    """S2 monitor: how long the model has been running on remembered (not seen) object positions.

    Works with the toy policies' state probe ({"cube", "goal", "age"} or {"memory": {...}}).
    end_episode reports the longest age per object.
    """
    stage, role, side, requires = "S2", "monitor", "model", ("state",)

    def reset(self, episode, ctx):
        self.longest = {}

    def on_state(self, state, ctx):
        if ctx.phase != "written":
            return state
        memory = state.get("memory", state)
        for name, age in (memory.get("age") or {}).items():
            self.longest[name] = max(self.longest.get(name, 0), int(age))
            if age:
                ctx.log({"object": name, "age": int(age)})
        return state

    def end_episode(self, ctx):
        return {"longest_memory_age": self.longest}


def summarise(value):
    """Small JSON summary of a probe value: shapes and norms of arrays / tensors, lengths of lists."""
    if hasattr(value, "detach"):
        tensor = value.detach()
        return {"shape": list(tensor.shape), "norm": float(tensor.float().norm())}
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        import numpy as np
        array = np.asarray(value, dtype=float)
        finite = array[np.isfinite(array)]
        return {"shape": list(array.shape), "norm": float(np.linalg.norm(finite)) if finite.size else None}
    if isinstance(value, dict):
        return {str(k): summarise(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return {"len": len(value)}
    return value if isinstance(value, (int, float, str, bool)) or value is None else type(value).__name__


class StateSummary(Plugin):
    """S2 monitor: log a summary (shapes, norms, lengths) of the model state at each read / write."""
    stage, role, side, requires = "S2", "monitor", "model", ("state",)

    def on_state(self, state, ctx):
        ctx.log(phase=ctx.phase, state=summarise(state))
        return state


class PredictionSummary(Plugin):
    """S3 monitor: count rollout predictions per step; log the chosen next prediction's summary."""
    stage, role, side, requires = "S3", "monitor", "model", ("predict",)

    def reset(self, episode, ctx):
        self.count, self.step = 0, None

    def on_prediction(self, prediction, ctx):
        if ctx.meta.get("kind") == "chosen_next":
            ctx.log(kind="chosen_next", rollout_predictions=self.count, prediction=summarise(prediction))
            self.count = 0
        else:
            self.count += 1
        return prediction


class PlanSummary(Plugin):
    """S4 monitor: number of candidates and best / mean score of each planning call."""
    stage, role, side, requires = "S4", "monitor", "model", ("plan",)

    def on_plan(self, plan, ctx):
        import numpy as np
        scores = np.asarray(plan["scores"], dtype=float)
        finite = scores[np.isfinite(scores)]
        ctx.log(candidates=int(scores.size), best=float(finite.max()) if finite.size else None,
                mean=float(finite.mean()) if finite.size else None, meta=dict(ctx.meta or {}))
        return plan


class SurpriseSummary(Plugin):
    """S6 monitor: per step, how far the model's prediction for its chosen action was from what it
    then saw (|predicted_next - embed(next_obs)|, in the model's own space); end_episode the mean
    and max. A starting point for surprise detectors; thresholds must come from clean runs."""
    stage, role, side, requires = "S6", "monitor", "model", ("predict", "embed")

    def reset(self, episode, ctx):
        self.values = []

    def on_feedback(self, transition, ctx):
        import numpy as np
        predicted, actual = transition["predicted_next"], transition["next_state"]
        if predicted is None or actual is None:
            return transition
        gap = np.asarray(predicted, dtype=float).reshape(-1) - np.asarray(actual, dtype=float).reshape(-1)
        gap = gap[np.isfinite(gap)]
        value = float(np.linalg.norm(gap)) if gap.size else None
        if value is not None:
            self.values.append(value)
        ctx.log(surprise=value, taken_over=bool(transition["taken_over"]))
        return transition

    def end_episode(self, ctx):
        if not self.values:
            return {"steps": 0}
        return {"steps": len(self.values), "mean": sum(self.values) / len(self.values), "max": max(self.values)}
