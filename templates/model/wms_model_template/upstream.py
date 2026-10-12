"""Real-model replacement scaffold. Intentionally cannot produce fake output."""
import json
from pathlib import Path


class UpstreamWorldModel:
    def __init__(self, config):
        self.config = dict(config)
        machine_file = config.get("runtime_config")
        settings = (json.loads(Path(machine_file).read_text(encoding="utf-8"))
                    if machine_file else {})
        # Registry resolves standard runtime keys only. Checkpoint and other
        # backend-specific settings are read here, inside the worker.
        self.checkpoint = config.get("checkpoint") or settings.get("checkpoint")
        if self.checkpoint:
            path = Path(self.checkpoint).expanduser()
            if not path.is_absolute():
                if not machine_file:
                    raise ValueError("Use an absolute checkpoint path, or resolve relative to a machine file")
                path = Path(machine_file).resolve().parent / path
            self.checkpoint = str(path.resolve())
        if not self.checkpoint or not Path(self.checkpoint).exists():
            raise FileNotFoundError("Set a real checkpoint in model.config or runtime.local.json")
        raise NotImplementedError(
            "Replace UpstreamWorldModel.__init__ with official backend imports and "
            "checkpoint loading; then implement predict using the model's true input/output semantics. "
            "See templates/model/README.zh-CN.md. This scaffold has not run a real model."
        )

    def predict(self, request):
        raise NotImplementedError("Implement observation/action conversion and real future prediction")

    def close(self):
        # Close only resources owned by this adapter. Do not kill shared jobs.
        pass
