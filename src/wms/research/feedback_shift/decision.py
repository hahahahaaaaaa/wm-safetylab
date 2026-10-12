"""A fixed event-risk rule; the threshold is supplied before evaluating methods."""
import math


def decide(event_probability, tau, *, invalid_mass=0.0, conservative_unknown=True):
    for name, value in (("event_probability", event_probability),
                        ("tau", tau), ("invalid_mass", invalid_mass)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be a finite probability")
    if event_probability + invalid_mass > 1 + 1e-6:
        raise ValueError("event and INVALID must be disjoint probability masses")
    if not isinstance(conservative_unknown, bool):
        raise ValueError("conservative_unknown must be boolean")
    risk = event_probability + (invalid_mass if conservative_unknown else 0.0)
    return {"decision": "execute" if risk <= tau else "defer", "accepted": risk <= tau,
            "event_probability": event_probability, "invalid_mass": invalid_mass,
            "decision_risk": risk, "tau": tau,
            "unknown_policy": "all_unsafe" if conservative_unknown else "exclude_explicitly"}
