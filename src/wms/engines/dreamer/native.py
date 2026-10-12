"""Native task entrypoint; shared defaults live in native/dreamer/recipes.json."""
from copy import deepcopy

from wms.engines.native_tasks import _artifact, _runtime


def dreamer_infer(config, output_dir, context):
    from wms.engines.dreamer.backend import infer
    case = deepcopy(config.get("inputs", {}))
    sequence, trajectory, report = infer(_runtime(config), case, config.get("seed", 0), output_dir, context)
    return {"outputs": [_artifact(context.write_json(name, value), kind) for name, value, kind in
        (("sequence.json", sequence, "input"), ("trajectory.json", trajectory, "trajectory"),
         ("runtime_report.json", report, "report"))]}

from wms.engines.dreamer.prediction_adapter import DreamerPredictionAdapter as PredictionAdapter

from wms.engines.native_tasks import world_model_predict as predict
