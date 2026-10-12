"""Native task entrypoint; shared defaults live in native/renew/recipes.json."""
from copy import deepcopy
import os

from wms.engines.native_tasks import _runtime


def renew_pilot(config, output_dir, context):
    from wms.engines.renew.runtime import native_environment, limit_cpu_affinity
    from wms.engines.renew.backend import run_pilot
    runtime = _runtime(config)
    os.environ.update(native_environment(runtime))
    limit_cpu_affinity()
    case = deepcopy(config.get("inputs", {}))
    return {"outputs": run_pilot(runtime, case, config.get("seed", 0), output_dir, context)}
