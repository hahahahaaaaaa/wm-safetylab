"""Shared native prediction helpers; model task entries live beside each backend.

Historical native_tasks:<model> entry names remain lazy aliases for saved-run
retry. New recipes reference wms.engines.<model>.native directly.
"""
from copy import deepcopy
import importlib
import json
from pathlib import Path

from wms.engines.runtime import load_runtime


def _entrypoint(name):
    module, separator, attribute = name.partition(":")
    if not separator:
        raise ValueError("Adapter entrypoint must be 'module:class'")
    return getattr(importlib.import_module(module), attribute)


def _runtime(config):
    settings = config.get("runtime", {})
    runtime = load_runtime(settings.get("config_file") or config.get("params", {}).get("runtime_config", {}))
    # Backend helpers may set the device again. Pass the already-resolved choice
    # into their temporary config, without editing the machine's runtime file.
    device = settings.get("env", {}).get("CUDA_VISIBLE_DEVICES")
    if device is not None:
        runtime["cuda_visible_devices"] = str(device)
        if "gpu_device" in runtime:
            runtime["gpu_device"] = str(device)
    return runtime


def _artifact(path, kind, media="application/json"):
    return {"path": str(path), "kind": kind, "media_type": media}


def _profile(value):
    if isinstance(value, dict):
        profile, base = deepcopy(value), Path.cwd()
    else:
        path = Path(value).resolve()
        profile, base = json.loads(path.read_text(encoding="utf-8-sig")), path.parent
    # Preserve the adapter config filename so sidecar paths still resolve beside
    # that file. Requests are JSON values passed directly into the adapter.
    if isinstance(profile.get("config"), str):
        path = Path(profile["config"])
        profile["config"] = str(path if path.is_absolute() else (base / path).resolve())
    if isinstance(profile.get("request"), str):
        path = Path(profile["request"])
        profile["request"] = json.loads((path if path.is_absolute() else base / path).read_text(encoding="utf-8-sig"))
    return profile


def _prediction_result(bundle):
    outputs = [{**item, "path": item["relative_path"]} for item in bundle["outputs"]]
    for item in outputs:
        item.pop("relative_path", None)
    outputs.append(_artifact("prediction.json", "report"))
    return {"outputs": outputs, "data": {key: value for key, value in bundle.items() if key != "outputs"}}


def world_model_predict(config, output_dir, context):
    params = config.get("params", {})
    profile = _profile(config.get("runtime", {}).get("config_file") or params["profile"])
    entrypoint = params.get("adapter_entrypoint", profile.get("adapter_entrypoint"))
    if not entrypoint:
        # Compatibility with historical profiles, using a module convention
        # instead of a shared table that every new adapter must modify.
        entrypoint = f"wms.engines.{profile['adapter']}.native:PredictionAdapter"
    model = PredictionModel({"adapter_entrypoint": entrypoint, "runtime_config": profile["config"],
                             "request": profile.get("request", {})})
    request = {**config.get("inputs", {}), "seed": config.get("seed", 0),
               "output_dir": str(output_dir)}
    try:
        context.check_cancelled()
        return model.predict(request)
    finally:
        model.close()


class PredictionModel:
    """Adapt native predict(request, output_dir) to the single-request interface."""
    def __init__(self, config):
        self.config = config
        entrypoint = config.get("adapter_entrypoint")
        if not entrypoint and config.get("adapter"):
            entrypoint = f"wms.engines.{config['adapter']}.native:PredictionAdapter"
        self.adapter = _entrypoint(entrypoint)(load_runtime(config["runtime_config"]))

    def predict(self, request):
        fields = {"schema_version", "request_id", "model_id", "seed", "horizon", "parameters"}
        value = deepcopy(self.config.get("request", {}))
        value.update({key: request[key] for key in fields if key in request})
        value["inputs"] = {**value.get("inputs", {}), **request.get("inputs", {}),
                           **{key: item for key, item in request.items()
                              if key not in fields | {"inputs", "output_dir"}}}
        output = Path(request.get("output_dir", self.config.get("output_dir", ".")))
        manifest = self.adapter.describe()
        value.setdefault("request_id", "prediction")
        value.setdefault("model_id", manifest["model_id"])
        value.setdefault("schema_version", "wms.predict/1.0")
        value.setdefault("seed", 0)
        bundle = self.adapter.predict(value, output)
        return _prediction_result(bundle)

    def close(self):
        close = getattr(self.adapter, "close", None)
        if callable(close):
            close()


def __getattr__(name):
    """Resolve the six old function names only when retrying saved runs."""
    old_names = {"wan_generate", "modelscope_generate", "openvla_infer", "dreamer_infer", "maniskill_rollout", "renew_pilot"}
    if name not in old_names:
        raise AttributeError(name)
    engine = name.partition("_")[0]
    function = _entrypoint(f"wms.engines.{engine}.native:{name}")

    def historical(config, output_dir, context):
        from wms.core.registry import merge
        path = Path(__file__).resolve().parents[3] / "native" / engine / "recipes.json"
        defaults = json.loads(path.read_text(encoding="utf-8-sig"))["recipes"]["native-" + engine]
        config = deepcopy(config)
        config["inputs"] = merge(defaults.get("inputs", {}), config.get("inputs", {}))
        config["params"] = merge(defaults.get("params", {}), config.get("params", {}))
        return function(config, output_dir, context)

    return historical
