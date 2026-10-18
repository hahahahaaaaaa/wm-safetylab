"""wms plugin test <ID>: the checks every plugin must pass before review, on the CPU toy (seconds).

    python -m wms.plugins.conformance <registry ID | module:Class> [--config JSON] [--registry FILE ...]

Needs a Python with numpy (the toy runtime's). `wms plugin test` in the platform's own
environment forwards to it. Checks:

  declaration     stage, role, hooks, requires, privileged, threat card are consistent
                  (wms.plugins.base), and the registry entry agrees with the class
  runs            two toy episodes with the plugin finish, in strict mode: no hook may change
                  its input in place
  called          the plugin's hooks were called (trace.jsonl)
  reproducible    the same seeds give byte-identical steps.jsonl and trace.jsonl
  own randomness  the global random / numpy.random state is untouched (use ctx.rng)
  read-only       monitors only: every action equals the run without the plugin
  effect          others: on how many steps the executed action differs from the run without
                  the plugin (information, not pass/fail: a fault with no effect is legal)

Toy model: toy-planner (probes state, predict, plan) when the plugin requires a probe,
toy-centroid-memory otherwise. Passing says the plugin follows the contract; it says nothing
about the plugin's effect on a real model, which its task card's GPU run shows.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
import random
import re
import sys

ROOT = Path(__file__).resolve().parents[3]
TOY_ENV = {"entrypoint": "wms.scenarios.toy.env:ToyPickEnv",
           "capabilities": ["image_box", "set_object_pose", "set_goal", "hazards"],
           "config": {"image_size": 64, "max_steps": 60}}
TOY_MODELS = {
    "toy-planner": {"entrypoint": "wms.scenarios.toy.planner:ToyPlannerPolicy", "probes": ["state", "predict", "plan"],
                    "config": {"samples": 32, "horizon": 4}},
    "toy-centroid-memory": {"entrypoint": "wms.scenarios.toy.policy:CentroidPickPolicy", "probes": ["state"],
                            "config": {"memory": True}},
}


def _lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _run(spec, model, output, seed, episodes, with_plugin=True):
    from wms.runners.unified import UnifiedWorker
    params = {"episodes": episodes, "video": False, "save_raw": False, "plugin_strict": True,
              "plugins": [spec] if with_plugin else []}
    config = {"task": "native", "entrypoint": "wms.episodes.kernel:run", "seed": seed,
              "model": model, "environment": TOY_ENV, "params": params}
    with contextlib.redirect_stdout(io.StringIO()):          # progress events
        code = UnifiedWorker(config, output).run()
    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    return code, result


def _episodes(output, episodes):
    return [(output / "episodes" / f"ep_{i:03d}") for i in range(episodes)]


def check(spec, output, seed=0, episodes=2):
    """[(status, name, detail)] with status PASS / FAIL / INFO."""
    import numpy as np
    from wms.runners.unified import load_entrypoint
    from .base import PROBES, metadata, problems
    from .resolve import mismatches
    report = []
    try:
        cls = load_entrypoint(spec["entrypoint"])
    except Exception as exc:  # noqa: BLE001
        return [("FAIL", "declaration", f"cannot import {spec['entrypoint']}: {type(exc).__name__}: {exc}")]
    issues = problems(cls, spec.get("config")) + mismatches(cls, spec)
    report.append(("FAIL" if issues else "PASS", "declaration", "; ".join(issues) or json.dumps(
        {k: v for k, v in metadata(cls, spec.get("config")).items() if v not in (None, [], False)}, ensure_ascii=False)))
    if issues:
        return report
    model_name = "toy-planner" if any(r in PROBES for r in getattr(cls, "requires", ()) or ()) else "toy-centroid-memory"
    model = TOY_MODELS[model_name]
    report.append(("INFO", "toy model", model_name))

    state = (random.getstate(), np.random.get_state())
    code, result = _run(spec, model, output / "a", seed, episodes)
    touched = random.getstate() != state[0] or not all(
        np.array_equal(x, y) if isinstance(x, np.ndarray) else x == y for x, y in zip(np.random.get_state(), state[1]))
    if code != 0:
        report.append(("FAIL", "runs", str(result.get("error"))))
        return report
    report.append(("PASS", "runs", f"{episodes} episodes, strict mode"))
    report.append(("FAIL" if touched else "PASS", "own randomness",
                   "the global random / numpy.random state changed; draw from ctx.rng" if touched else "global state untouched"))

    plugin_id = f"0:{spec.get('id') or spec['entrypoint']}"
    calls = sum(plugin_id in ids for ep in _episodes(output / "a", episodes)
                for row in _lines(ep / "trace.jsonl") for ids in row.get("plugins", {}).values())
    report.append(("PASS" if calls else "FAIL", "called", f"{calls} hook calls" if calls else
                   "no hook of this plugin was called on the toy (does the toy offer what it waits for?)"))

    code_b, result_b = _run(spec, model, output / "b", seed, episodes)
    same = code_b == 0 and all((ep_a / name).read_bytes() == (ep_b / name).read_bytes()
                               for ep_a, ep_b in zip(_episodes(output / "a", episodes), _episodes(output / "b", episodes))
                               for name in ("steps.jsonl", "trace.jsonl"))
    report.append(("PASS" if same else "FAIL", "reproducible",
                   "identical steps.jsonl and trace.jsonl" if same else "a second run with the same seeds differs"))

    code_c, result_c = _run(spec, model, output / "baseline", seed, episodes, with_plugin=False)
    if code_c != 0:
        report.append(("FAIL", "baseline", str(result_c.get("error"))))
        return report
    differ = total = 0
    for ep_a, ep_c in zip(_episodes(output / "a", episodes), _episodes(output / "baseline", episodes)):
        a, c = _lines(ep_a / "steps.jsonl"), _lines(ep_c / "steps.jsonl")
        total += max(len(a), len(c))
        differ += sum(x["action"] != y["action"] for x, y in zip(a, c)) + abs(len(a) - len(c))
    if cls.role == "monitor":
        report.append(("PASS" if differ == 0 else "FAIL", "read-only",
                       "actions identical to the run without it" if differ == 0 else
                       f"a monitor changed the actions on {differ}/{total} steps"))
    else:
        report.append(("INFO", "effect", f"executed action differs on {differ}/{total} steps "
                       f"(success {result['data']['success_rate']:.2f} vs {result_c['data']['success_rate']:.2f} without)"))
    return report


def _spec(target, config, registries):
    from wms.core.registry import Registry
    from .resolve import resolve_specs
    registry = Registry(paths=registries)
    item = {"entrypoint": target} if ":" in target else {"plugin": target}
    return resolve_specs([{**item, "config": config}], registry.items["plugins"])[0]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Check one plugin against the plugin contract on the CPU toy.")
    parser.add_argument("plugin", help="registry ID or module:Class")
    parser.add_argument("--config", default="{}", help="JSON object (or a JSON file) merged over the registry config")
    parser.add_argument("--registry", action="append", default=[], help="extra recipes.json / registry.json")
    parser.add_argument("--output", default=None, help="where the toy runs go (default test-output/plugin-test/<ID>)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=2)
    args = parser.parse_args(argv)
    text = args.config.strip()
    if not text.startswith("{") and Path(text).is_file():          # a JSON file (easier to quote)
        text = Path(text).read_text(encoding="utf-8-sig")
    spec = _spec(args.plugin, json.loads(text), args.registry)
    for path in reversed(spec.get("python_paths", [])):
        sys.path.insert(0, path)
    output = Path(args.output or ROOT / "test-output" / "plugin-test" / re.sub(r"[^A-Za-z0-9_.-]+", "_", args.plugin))
    if output.exists() and any(output.iterdir()):
        import shutil
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    report = check(spec, output, args.seed, args.episodes)
    width = max(len(name) for _, name, _ in report)
    for status, name, detail in report:
        print(f"{status:4}  {name:<{width}}  {detail}")
    failed = any(status == "FAIL" for status, _, _ in report)
    (output / "report.json").write_text(json.dumps({"plugin": args.plugin, "spec": spec, "passed": not failed,
                                                    "checks": [dict(zip(("status", "check", "detail"), row)) for row in report]},
                                                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(("FAILED" if failed else "PASSED") + f"  (runs in {output})")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
