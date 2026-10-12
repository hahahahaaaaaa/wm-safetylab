"""Real GPU smoke: original TD-MPC2 update and actual ManiSkill, no test doubles."""
import json
from pathlib import Path
import sys
from wms.runners.unified import RunContext
from wms.engines.tdmpc2.backend import train

runtime = json.loads(Path(sys.argv[1]).read_text())
output = Path(sys.argv[2])
context = RunContext(output)
result = train(runtime, {"num_envs": 8, "steps": 1600, "buffer_size": 5000,
                         "steps_per_update": 8, "batch_size": 64,
                         "num_samples": 64, "num_elites": 16, "num_pi_trajs": 8,
                         "iterations": 2, "checkpoint_every": 0}, output, context, seed=0)
print(json.dumps(result, indent=2))
