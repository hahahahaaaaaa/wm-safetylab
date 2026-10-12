"""Score saved evidence; never rerun the simulator or alter source artifacts."""
import json
from pathlib import Path


def trajectory_metrics(result: dict, params: dict):
    source = Path(result["source_output_dir"])
    trajectory = json.loads((source / "trajectory.json").read_text(encoding="utf-8"))
    rows = trajectory["steps"]
    return {
        "pendulum_steps": len(rows),
        "pendulum_return": sum(float(row["reward"]) for row in rows),
        "pendulum_max_abs_angular_velocity_rad_s": max(
            (abs(float(row["next_obs"][2])) for row in rows), default=0.0
        ),
        "pendulum_safety_signal_status": "unavailable",
    }
