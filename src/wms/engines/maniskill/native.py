"""Native task entrypoint; shared defaults live in native/maniskill/recipes.json."""
from copy import deepcopy

from wms.engines.native_tasks import _artifact, _runtime


def maniskill_rollout(config, output_dir, context):
    from wms.engines.maniskill.backend import rollout
    case = deepcopy(config.get("inputs", {}))
    if "steps" in config:
        case["max_steps"] = config["steps"]
    inputs, trajectory, report, video = rollout(_runtime(config), case, config.get("seed", 0), output_dir, context)
    return {"outputs": [_artifact(context.write_json("input.json", inputs), "input"),
                         _artifact(context.write_json("trajectory.json", trajectory), "trajectory"),
                         _artifact(context.write_json("runtime_report.json", report), "report"),
                         _artifact(video, "video", "video/mp4")]}
