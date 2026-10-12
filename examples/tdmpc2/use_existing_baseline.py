"""Run the same trained TD-MPC2 through generic platform component interfaces."""
import argparse
import json
import wms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("training_run", help="Completed TD-MPC2 training run ID")
    parser.add_argument("--task", choices=("predict", "rollout"), default="predict")
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()
    config = {
        "name": "TD-MPC2 generic " + args.task, "task": args.task, "direction": "control",
        "runtime": {"config_file": "native/tdmpc2/runtime.json"}, "seed": args.seed, "steps": 50,
        "model": {"entrypoint": "wms.engines.tdmpc2.adapter:ModelAdapter",
                  "config": {"runtime_config": "$runtime.config_file", "source_run": args.training_run,
                             "seed": args.seed}},
        "environment": {"entrypoint": "wms.engines.tdmpc2.adapter:EnvironmentAdapter",
                        "config": {"runtime_config": "$runtime.config_file", "source_run": args.training_run}},
    }
    if args.task == "predict":
        config["inputs"] = {"actions": [[0.0] * 8 for _ in range(3)]}
    run = wms.wait(wms.run(config)["id"])
    print(json.dumps(run, ensure_ascii=False, indent=2))
    if run["state"] != "completed":
        raise RuntimeError(run.get("error"))


if __name__ == "__main__":
    main()
