"""Native process entry point; configuration is supplied by the trusted runtime."""
import argparse
import json
from pathlib import Path

from .adapter import RLVRVideoAdapter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
    request = json.loads(Path(args.request).read_text(encoding="utf-8-sig"))
    result = RLVRVideoAdapter(config).predict(request, Path(args.output))
    print(json.dumps({"status": result["status"], "model_id": result["model_id"],
                      "outputs": len(result["outputs"]), "diagnostics": result["diagnostics"]}))


if __name__ == "__main__":
    main()
