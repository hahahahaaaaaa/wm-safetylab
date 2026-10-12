"""Independent checkpoint reload and one real recorded simulation episode."""
import json
from pathlib import Path
import sys
from wms.runners.unified import RunContext
from wms.engines.tdmpc2.native import policy_evaluate

runtime, checkpoint, directory = sys.argv[1:]
config = {"runtime": {"config_file": runtime}, "seed": 0,
          "inputs": {"checkpoint": checkpoint},
          "params": {"episodes": 1, "seed_start": 1000, "video_episodes": 1}}
context = RunContext(directory)
result = policy_evaluate(config, Path(directory), context)
print(json.dumps(result, indent=2))
