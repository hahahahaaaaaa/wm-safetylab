"""One observed outcome per group, deterministic L1 judge, explicit tie handling."""
import math


def _flatten(value):
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            result.extend(_flatten(item))
        return result
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("outcomes must contain finite numeric values")
    return [value]


def make_group(outcome, candidates, *, group_id="group", weight=1.0, candidate_ids=None, tie_policy="all_best"):
    """Construct one group under an explicit tie protocol.

    all_best compares all tied winners only with strictly worse candidates.
    first_best selects the first winner but still compares it only to strictly
    worse candidates. Equal-distance candidates never become fabricated losers.

    Each pair has conditional weight 1 / pair_count within this group. Groups
    with no strict pair contribute zero but remain in group-mean denominators.
    outcome is returned as diagnostic truth and must be excluded from the
    comparison learner's inputs.
    """
    y = _flatten(outcome)
    candidates = list(candidates)
    cs = [_flatten(c) for c in candidates]
    if len(cs) < 2 or not y or any(len(c) != len(y) for c in cs):
        raise ValueError("at least two candidates must match the outcome dimensions")
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0:
        raise ValueError("group weight must be positive and finite")
    ids = list(range(len(cs))) if candidate_ids is None else list(candidate_ids)
    if len(ids) != len(cs) or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in ids):
        raise ValueError("candidate_ids must contain one nonnegative integer per candidate")
    distances = [math.fsum(abs(a-b) for a,b in zip(y,c)) for c in cs]
    if tie_policy not in ("all_best", "first_best"):
        raise ValueError("tie_policy must be all_best or first_best")
    best = min(distances)
    best_mask = [d == best for d in distances]
    first_best = distances.index(best)
    mask = best_mask if tie_policy == "all_best" else [i == first_best for i in range(len(cs))]
    pairs = [{"winner": ids[i], "loser": ids[j], "winner_position": i, "loser_position": j}
             for i, winner in enumerate(mask) if winner
             for j, is_best in enumerate(best_mask) if not is_best]
    for pair in pairs:
        pair["weight"] = 1.0 / len(pairs)
    return {"group_id": str(group_id), "outcome": outcome, "candidates": candidates,
            "candidate_ids": ids, "distances": distances, "winner_mask": mask, "best_mask": best_mask,
            "strict_loser_mask": [not v for v in best_mask],
            "pair01_label": 1 if distances[0] < distances[1] else -1 if distances[0] > distances[1] else 0,
            "pairs": pairs, "weight": float(weight), "outcome_query_count": 1,
            "pair_count": len(pairs), "tie_policy": tie_policy,
            "judge": "L1", "all_tied": all(best_mask), "tie_broken": tie_policy == "first_best" and sum(best_mask) > 1}


def group_channel(outcomes, candidates, observed_winner_mask, *, tie_policy="all_best"):
    """H(observed winner mask | y,C) for a deterministic L1 judge.

    Only provide a complete enumerated model outcome space. Omitting an
    unspecified INVALID bucket does not constitute a complete feedback channel.
    """
    observed = list(observed_winner_mask)
    if len(observed) != len(candidates) or not any(observed) or any(type(v) is not bool for v in observed):
        raise ValueError("observed_winner_mask must be a nonempty boolean winner mask")
    return [float(make_group(y, candidates, tie_policy=tie_policy)["winner_mask"] == observed) for y in outcomes]


def feedback_observation(group):
    """Public joint U, excluding outcome truth and numeric judge distances."""
    return {"winner_mask": list(group["winner_mask"]),
            "strict_loser_mask": list(group["strict_loser_mask"]),
            "pair01_label": group["pair01_label"]}


def joint_group_channel(outcomes, candidates, observed_group, *, tie_policy="first_best"):
    """Known H for the entire public U on a complete enumerated outcome space.

    The selected first index is equivalent to the one-hot winner mask. Original
    candidates and the tie protocol are conditioned on; truth and distances are
    not silently added to the observation.
    """
    observed = feedback_observation(observed_group)
    return [float(feedback_observation(make_group(y, candidates, tie_policy=tie_policy)) == observed)
            for y in outcomes]
