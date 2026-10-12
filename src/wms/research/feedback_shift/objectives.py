"""JAX objectives on complete outcome/sequence log probabilities.

JAX is imported only inside calls. Inputs must be log probabilities from the
model, never a softmax over a sampled candidate pool. Objective weights and
channels are fixed protocol data, not differentiated parameters.
"""
import math


def _jax():
    import jax.numpy as jnp
    from jax.nn import softplus
    from jax.scipy.special import logsumexp
    return jnp, softplus, logsumexp


def _weights(weights, count, name):
    values = [1.0] * count if weights is None else list(weights)
    if count == 0 or len(values) != count or any(
            isinstance(v, bool) or not isinstance(v, (int, float))
            or not math.isfinite(v) or v < 0 for v in values) or math.fsum(values) <= 0:
        raise ValueError(f"{name} must have nonnegative finite weights with positive sum")
    return values


def _beta(beta):
    if isinstance(beta, bool) or not isinstance(beta, (int, float)) or not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be positive and finite")


def _indices(values, size):
    values = list(values)
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 or v >= size for v in values):
        raise ValueError("outcome indices must be in log_probs range")
    return values


def bt_loss(log_probs, winners, losers, *, pair_weights=None, beta=1.0):
    jnp, softplus, _ = _jax()
    _beta(beta)
    lp = jnp.asarray(log_probs)
    if lp.ndim != 1:
        raise ValueError("log_probs must be a vector of complete sequence log probabilities")
    winners = _indices(winners, lp.shape[0])
    losers = _indices(losers, lp.shape[0])
    if len(winners) != len(losers):
        raise ValueError("winner and loser lengths differ")
    weights = jnp.asarray(_weights(pair_weights, len(winners), "pair_weights"))
    losses = softplus(-beta * (lp[jnp.asarray(winners)] - lp[jnp.asarray(losers)]))
    return jnp.sum(weights * losses) / jnp.sum(weights)


def best_of_k_loss(log_probs, groups, *, beta=1.0, reduction="group_mean"):
    """Group mean keeps all-tied groups as zero-loss query groups.

    pair_mean instead weights each derived pair by its group weight. Actual
    within-group weights are honored in group_mean and explicit in every record.
    """
    jnp, softplus, _ = _jax()
    _beta(beta)
    lp = jnp.asarray(log_probs)
    if lp.ndim != 1:
        raise ValueError("log_probs must be a vector")
    groups = list(groups)
    _weights([g["weight"] for g in groups], len(groups), "group_weights")
    if reduction not in ("group_mean", "pair_mean"):
        raise ValueError("reduction must be group_mean or pair_mean")
    total = jnp.zeros((), dtype=lp.dtype)
    denominator = 0.0
    for group in groups:
        pairs = group["pairs"]
        gw = group["weight"]
        if reduction == "group_mean":
            denominator += gw
        if not pairs:
            continue
        winners = _indices([p["winner"] for p in pairs], lp.shape[0])
        losers = _indices([p["loser"] for p in pairs], lp.shape[0])
        pw = _weights([p["weight"] for p in pairs], len(pairs), "pair_weights")
        if not math.isclose(math.fsum(pw), 1.0, abs_tol=1e-9):
            raise ValueError("within-group pair weights must sum to 1")
        losses = softplus(-beta * (lp[jnp.asarray(winners)] - lp[jnp.asarray(losers)]))
        weights = jnp.asarray(pw if reduction == "group_mean" else [1.0]*len(pairs))
        total = total + gw * jnp.sum(weights * losses)
        if reduction == "pair_mean":
            denominator += gw * len(pairs)
    return total / denominator if denominator > 0 else total


def dpo_loss(log_probs, reference_log_probs, winners, losers, *, pair_weights=None, beta=1.0):
    jnp, _, _ = _jax()
    import jax
    lp, ref = jnp.asarray(log_probs), jnp.asarray(reference_log_probs)
    if lp.shape != ref.shape:
        raise ValueError("model and frozen reference support must match")
    return bt_loss(lp - jax.lax.stop_gradient(ref), winners, losers,
                   pair_weights=pair_weights, beta=beta)


def mle_loss(log_probs, outcome_indices, *, weights=None):
    jnp, _, _ = _jax()
    lp = jnp.asarray(log_probs)
    if lp.ndim != 1:
        raise ValueError("log_probs must be a vector")
    indices = _indices(outcome_indices, lp.shape[0])
    weights = jnp.asarray(_weights(weights, len(indices), "weights"))
    return -jnp.sum(weights * lp[jnp.asarray(indices)]) / jnp.sum(weights)


def _channels(values, support_size, name):
    rows = [list(row) for row in values]
    if not rows or any(len(row) != support_size for row in rows):
        raise ValueError(f"{name} must be group-by-complete-support")
    if any(isinstance(v, bool) or not isinstance(v, (int,float)) or not math.isfinite(v) or not 0 <= v <= 1
           for row in rows for v in row):
        raise ValueError(f"{name} must contain finite channel probabilities")
    if any(not any(v > 0 for v in row) for row in rows):
        raise ValueError(f"{name} contains impossible observed feedback")
    return rows


def group_channel_nll(log_probs, channel_probabilities, *, group_weights=None):
    """-log sum_y q(y) H(U|y,C); no multiplication of pair marginals."""
    jnp, _, logsumexp = _jax()
    lp = jnp.asarray(log_probs)
    if lp.ndim != 1:
        raise ValueError("log_probs must be a vector")
    rows = _channels(channel_probabilities, lp.shape[0], "channel_probabilities")
    weights = jnp.asarray(_weights(group_weights, len(rows), "group_weights"))
    channel = jnp.asarray(rows)
    log_channel = jnp.where(channel > 0, jnp.log(jnp.where(channel > 0, channel, 1.0)), -jnp.inf)
    losses = -logsumexp(lp[None, :] + log_channel, axis=1)
    return jnp.sum(weights * losses) / jnp.sum(weights)


def selection_aware_composite_nll(log_probs, joint_channels, selection_channels, *, pair_weights=None):
    """Conditional terms -log P_q(U,S|C) / P_q(S|C).

    Callers must derive both channels using the actual selection protocol and
    the same complete support. This helper does not infer selection from pairs.
    Correlated terms form a composite objective, not a joint likelihood.
    """
    jnp, _, logsumexp = _jax()
    lp = jnp.asarray(log_probs)
    if lp.ndim != 1:
        raise ValueError("log_probs must be a vector")
    joint = _channels(joint_channels, lp.shape[0], "joint_channels")
    selection = _channels(selection_channels, lp.shape[0], "selection_channels")
    if len(joint) != len(selection) or any(a > b for r,s in zip(joint,selection) for a,b in zip(r,s)):
        raise ValueError("joint events must be subsets of their selection events")
    weights = jnp.asarray(_weights(pair_weights, len(joint), "pair_weights"))
    def marginal(rows):
        c = jnp.asarray(rows)
        return logsumexp(lp[None,:] + jnp.where(c > 0, jnp.log(jnp.where(c > 0,c,1.0)), -jnp.inf), axis=1)
    return -jnp.sum(weights * (marginal(joint) - marginal(selection))) / jnp.sum(weights)
