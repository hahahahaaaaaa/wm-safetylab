"""Two optional trajectory-search algorithms with ordinary configuration dicts.

A planner predicts candidate action sequences, scores them with a cost function,
and returns the first action of the best sequence. Lower cost is better.
"""
from __future__ import annotations

import math


def _positive_int(config, name, default):
    value = config.get(name, default)
    number = int(value)
    if number < 1 or float(value) != number:
        raise ValueError(f"{name} must be a positive integer")
    return number


class PlannerContext:
    def __init__(self, *, predict, cost, action_range, rng, case, record):
        self.predict = predict     # (observation, action_sequence) -> predicted values
        self.cost = cost           # (observation, predicted, case) -> scalar cost
        self.action_range = [tuple(map(float, pair)) for pair in action_range]
        if not self.action_range or any(len(pair) != 2 or not all(math.isfinite(v) for v in pair)
                                        or pair[0] > pair[1] for pair in self.action_range):
            raise ValueError("action_range must contain finite [lower, upper] pairs")
        self.rng = rng
        self.case = case
        self.record = record

    def sample(self):
        return [self.rng.uniform(lo, hi) for lo, hi in self.action_range]

    def score(self, obs, actions):
        return float(self.cost(obs, self.predict(obs, actions), self.case))


class RandomShooting:
    """Uniformly sample action sequences and execute the best first action."""
    def __init__(self, config):
        self.horizon = _positive_int(config, "horizon", 5)
        self.candidates = _positive_int(config, "candidates", 24)

    def plan(self, obs, t, ctx):
        best, best_cost, valid = None, math.inf, 0
        for _ in range(self.candidates):
            actions = [ctx.sample() for _ in range(self.horizon)]
            cost = ctx.score(obs, actions)
            if math.isfinite(cost):
                valid += 1
                if best is None or cost < best_cost:
                    best, best_cost = actions, cost
        ctx.record.update(candidates=self.candidates, finite_candidates=valid)
        if best is None:
            raise ValueError("No candidate has a finite cost; cannot choose an action")
        ctx.record["best_cost"] = best_cost
        return best[0]


class CrossEntropy:
    """CEM: fit a Gaussian over action sequences to the best-scoring samples."""
    def __init__(self, config):
        self.horizon = _positive_int(config, "horizon", 5)
        self.population = _positive_int(config, "population", 24)
        self.elites = _positive_int(config, "elites", 6)
        self.iterations = _positive_int(config, "iterations", 3)
        self.init_std = float(config.get("init_std", 0.5))
        if self.elites > self.population:
            raise ValueError("elites cannot exceed population")
        if not math.isfinite(self.init_std) or self.init_std <= 0:
            raise ValueError("init_std must be finite and positive")

    def plan(self, obs, t, ctx):
        lows, highs = zip(*ctx.action_range)
        dim, horizon = len(lows), self.horizon
        mean = [[(lo + hi) / 2 for lo, hi in ctx.action_range] for _ in range(horizon)]
        std = [[(hi - lo) / 2 * self.init_std for lo, hi in ctx.action_range] for _ in range(horizon)]
        best, best_cost, evaluated, valid = None, math.inf, 0, 0
        for iteration in range(self.iterations):
            scored = []
            for _ in range(self.population):
                actions = [[min(highs[d], max(lows[d], ctx.rng.gauss(mean[h][d], std[h][d])))
                            for d in range(dim)] for h in range(horizon)]
                cost = ctx.score(obs, actions)
                evaluated += 1
                if math.isfinite(cost):
                    scored.append((cost, actions))
                    valid += 1
            ctx.record.update(candidates=evaluated, finite_candidates=valid, iterations=iteration + 1)
            if not scored:
                raise ValueError("No candidate has a finite cost; cannot choose an action")
            scored.sort(key=lambda item: item[0])
            elite = scored[:self.elites]
            if best is None or elite[0][0] < best_cost:
                best_cost, best = elite[0]
            for h in range(horizon):
                for d in range(dim):
                    values = [actions[h][d] for _, actions in elite]
                    center = sum(values) / len(values)
                    mean[h][d] = center
                    std[h][d] = max(1e-6, math.sqrt(sum((v - center) ** 2 for v in values) / len(values)))
        ctx.record["best_cost"] = best_cost
        return best[0]
