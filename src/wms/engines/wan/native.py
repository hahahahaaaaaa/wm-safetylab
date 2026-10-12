"""Native task entrypoint; shared defaults live in native/wan/recipes.json."""
from copy import deepcopy
from pathlib import Path

from wms.engines.native_tasks import _artifact, _runtime


def wan_generate(config, output_dir, context):
    from wms.engines.wan.backend import generate
    case = deepcopy(config.get("inputs", {}))
    video = Path(output_dir) / "video.mp4"
    report = generate(_runtime(config), case, config.get("seed", 0), video, context)
    report_path = context.write_json("runtime_report.json", report)
    return {"outputs": [_artifact(video, "video", "video/mp4"), _artifact(report_path, "report")]}
