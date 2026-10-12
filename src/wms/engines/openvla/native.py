"""Native task entrypoint; shared defaults live in native/openvla/recipes.json."""

from wms.engines.native_tasks import _artifact, _runtime


def openvla_infer(config, output_dir, context):
    from wms.engines.openvla.backend import infer
    runtime = _runtime(config)
    case = {**config.get("params", {}).get("input_defaults", {}),
            **runtime.get("input", {}), **config.get("inputs", {})}
    prediction, report = infer(runtime, case, config.get("seed", 0), case["image_path"], context)
    return {"outputs": [_artifact(context.write_json("action.json", prediction), "action"),
                         _artifact(context.write_json("runtime_report.json", report), "report")],
            "data": {"action": prediction["action"]}}
