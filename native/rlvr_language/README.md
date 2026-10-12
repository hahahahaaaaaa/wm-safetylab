# RLVR-World language entry point (platform 0.3.0)

Use the `predict-rlvr_language` recipe from the Experiments page, or follow the current Python, CLI and HTTP workflow in the [model and experiment manual](../../docs/MODELS_AND_EXPERIMENTS.zh-CN.md). The recipe calls `wms.engines.rlvr_language.native:predict` in a separately configured native environment.

This adapter performs one-transition Bytesized32 symbolic-state prediction with the official task-specific RLVR checkpoint. It preserves generated evidence and reconstructs the successor state using the upstream recovery function. It does not provide WebArena execution, robot control, training, closed-loop planning or a safety benchmark.

Configure the official source, model resources and local runtime before submitting a request. The machine-specific runtime, resource inventory and experiment evidence are local deployment files and are not published with this repository. The supported input example is `request.example.json`; inspect and prepare its input data according to the main manual.

The old `prepare_resources.py`, `inspect_case.py`, `run_smoke.py` and `prepare_server.sh` are historical local deployment scripts, excluded from publication. They are not the 0.3.0 entry points. Current shared settings are in `recipes.json`, and the adapter is in `../../src/wms/engines/rlvr_language/`.
