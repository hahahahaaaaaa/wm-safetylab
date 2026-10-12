"""Finite outcome probability metrics; never normalize candidate scores."""
import math


def _probabilities(values, name):
    values = list(values)
    if not values:
        raise ValueError(f"{name} must be nonempty")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           or v < 0 or v > 1 for v in values):
        raise ValueError(f"{name} must contain finite probabilities in [0,1]")
    return values


def metrics(p, q, *, invalid_mass=None):
    """Measure a complete common support, optionally appending one INVALID bucket.

    p must sum to 1. If invalid_mass is supplied q is a subdistribution on the
    legal support and this explicit bucket closes it. No renormalization occurs.
    Brier distribution error is sum((q-p)**2), not the expected categorical
    Brier score. An aggregated INVALID category coarsens that metric.
    """
    p = _probabilities(p, "p")
    q = _probabilities(q, "q")
    if len(p) != len(q):
        raise ValueError("p and q must describe the same ordered support")
    if not math.isclose(math.fsum(p), 1.0, abs_tol=1e-6, rel_tol=0):
        raise ValueError("p must sum to 1")
    if invalid_mass is not None:
        invalid_mass = _probabilities([invalid_mass], "invalid_mass")[0]
        p = p + [0.0]
        q = q + [invalid_mass]
    if not math.isclose(math.fsum(q), 1.0, abs_tol=1e-6, rel_tol=0):
        raise ValueError("q including explicit INVALID must sum to 1")
    zero_on_support = any(pi > 0 and qi == 0 for pi, qi in zip(p, q))
    nll = None if zero_on_support else -math.fsum(
        pi * math.log(qi) for pi, qi in zip(p, q) if pi > 0)
    error = math.fsum((qi - pi)**2 for pi, qi in zip(p, q))
    return {
        "tv": 0.5 * math.fsum(abs(qi - pi) for pi, qi in zip(p, q)),
        "nll": nll,
        "nll_status": "infinite_zero_model_probability" if zero_on_support else "finite",
        "brier_distribution_error": error,
        "expected_categorical_brier": error + 1 - math.fsum(pi*pi for pi in p),
        "brier_scope": "supplied_support_with_aggregated_INVALID" if invalid_mass is not None
                       else "supplied_support",
        "invalid_mass": invalid_mass,
        "p_mass": math.fsum(p),
        "q_mass": math.fsum(q),
        "support_size": len(p),
    }
