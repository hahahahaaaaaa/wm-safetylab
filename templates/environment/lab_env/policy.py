"""A teaching policy, deliberately not a learned model or a world model."""
import math


class ConstantPolicy:
    def __init__(self, config: dict):
        self.torque = float(config.get("torque_Nm", 0.5))
        if not math.isfinite(self.torque) or not -2 <= self.torque <= 2:
            raise ValueError("torque_Nm must be finite and within [-2, 2]")

    def act(self, observation, instruction=None):
        return [self.torque]

    def close(self):
        pass
