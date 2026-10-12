"""CPU interface examples. These are synthetic components, not research baselines."""
import json
import math
from pathlib import Path
import random
import time


class PointEnv:
    def __init__(self, config):
        self.config = config

    def reset(self, seed, inputs):
        self.state = [float(inputs.get("initial", random.Random(seed).uniform(-0.2, 0.2)))]
        self.goal = float(inputs.get("goal", 1.0))
        return list(self.state)

    def step(self, action):
        self.state[0] += float(action[0])
        distance = abs(self.goal - self.state[0])
        return {"obs": list(self.state), "reward": -distance, "cost": float(abs(self.state[0]) > 2),
                "terminated": distance < 0.01, "truncated": False, "info": {"distance": distance}}

    def close(self):
        pass


class LinearModel:
    def __init__(self, config):
        self.config = config

    def act(self, obs, instruction=None):
        goal = float(self.config.get("goal", 1))
        return [max(-0.2, min(0.2, goal - obs[0]))]

    def predict(self, request):
        state = float(request.get("observation", [0])[0])
        states = []
        for action in request.get("actions", [[0.1], [0.1]]):
            state += float(action[0]) + float(self.config.get("bias", 0))
            states.append([state])
        return {"predictions": states, "source": "synthetic_linear_dynamics"}

    def generate(self, request):
        output = Path(request["output_dir"])
        (output / "sample.txt").write_text(str(request.get("prompt", "CPU generation interface example")), encoding="utf-8")
        return {"outputs": [{"kind": "text", "path": "sample.txt"}],
                "data": {"source": "synthetic", "note": "This is not a video model."}}

    def close(self):
        pass


def distance_cost(observation, predicted, inputs):
    return abs(float(predicted[-1][0]) - float(inputs.get("goal", 1)))


def trajectory_metrics(result, params):
    path = Path(result.get("source_output_dir", result["output_dir"])) / "trajectory.json"
    trajectory = json.loads(path.read_text(encoding="utf-8"))
    return {"steps": len(trajectory["steps"]),
            "return": trajectory["summary"]["return"],
            "cost_sum": sum(row.get("cost", 0) for row in trajectory["steps"])}


def native_example(config, output_dir, context):
    params = config.get("params", {})
    for index in range(int(config.get("steps", 1))):
        context.check_cancelled()
        if params.get("fail"):
            raise RuntimeError("Requested example failure")
        time.sleep(float(params.get("delay", 0)))
        context.event("progress", {"step": index + 1})
    context.write_json("echo.json", config.get("inputs", {}))
    return {"outputs": [{"kind": "json", "path": "echo.json"}], "data": {"source": "synthetic"}}

