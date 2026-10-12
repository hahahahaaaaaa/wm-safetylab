"""Platform entrypoint for real simulation data plus a small learned state WM."""
from pathlib import Path
import json
import shutil
import time


def run(config, output_dir, context):
    from wms.engines.native_tasks import _runtime
    from wms.engines.state_wm_maniskill.collect import collect
    from wms.engines.state_wm_maniskill.learn import train

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    params = dict(config.get("params", {}))
    runtime = _runtime(config)
    params.update(config.get("inputs", {}))
    started = time.monotonic()
    source = params.get("dataset_dir")
    if source and params.get("record_video"):
        raise ValueError("Video recording requires fresh simulation; cannot invent frames for a reused dataset")
    if source:
        source = Path(source).resolve()
        if source == output.resolve():
            raise ValueError("Dataset reuse source must be a previous, distinct run directory")
        for name in ("dataset.npz", "dataset.json"):
            shutil.copy2(source / name, output / name)
        dataset = json.loads((output / "dataset.json").read_text(encoding="utf-8"))
        context.event("progress", {"phase": "reuse_dataset", "source": str(source)})
    else:
        dataset = collect(output, params, context, runtime)
    training = train(output, params, context)
    visual_outputs = []
    if params.get("record_video"):
        from wms.engines.state_wm_maniskill.visualize import build_visualizations
        visual_outputs = build_visualizations(output, params, context)
    summary = {"task": "PushCube-v1", "model": "small action-conditioned state world model",
               "dataset_reused_from": str(source) if source else None,
               "episodes": len(dataset["episodes"]), "transitions": dataset["transition_count"],
               "training_seeds": training["seeds"], "elapsed_seconds": time.monotonic() - started,
               "video_recorded": bool(params.get("record_video", False)),
               "scope": "engineering validation, not a paper reproduction or a safety-effect claim"}
    context.write_json("experiment.json", summary)
    names = ["dataset.npz", "dataset.json", "predictions.npz", "training.json", "experiment.json", *training["checkpoints"]]
    outputs = [{"path": name, "kind": "checkpoint" if name.endswith(".pt") else "evidence",
                "media_type": "application/json" if name.endswith(".json") else "application/octet-stream"} for name in names]
    if params.get("record_video"):
        recordings = json.loads((output / "recordings.json").read_text(encoding="utf-8"))
        outputs.append({"path": "recordings.json", "kind": "evidence", "media_type": "application/json"})
        outputs.extend({"path": row["path"], "kind": "video", "media_type": "video/mp4"} for row in recordings["episodes"])
    return {"data": summary, "outputs": visual_outputs + outputs}
