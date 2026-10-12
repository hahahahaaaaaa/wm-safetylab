"""Independent scoring: source directory is read-only by convention."""
import json
from pathlib import Path


def prediction_metrics(result, params):
    source = Path(result["source_output_dir"])
    destination = Path(result["output_dir"])
    # Larger real-model predictions should be read from source-relative files.
    # This example accepts its small inline values too.
    prediction_file = source / "prediction.json"
    data = (json.loads(prediction_file.read_text(encoding="utf-8"))
            if prediction_file.is_file() else result["data"])
    predicted, reference = data["predictions"], data.get("reference")
    if reference is None:
        raise ValueError("This metric requires an explicit reference; never treat missing truth as zero")
    if not predicted or len(predicted) != len(reference):
        raise ValueError("Predicted/reference time horizons must match and be non-empty")
    if any(len(a) != 1 or len(b) != 1 for a, b in zip(predicted, reference)):
        raise ValueError("This example metric is specifically for 1-D position in metres")
    errors = [abs(float(a[0]) - float(b[0])) for a, b in zip(predicted, reference)]
    values = {"template_state_mae_m": sum(errors) / len(errors)}
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "prediction_errors.json").write_text(
        json.dumps({"errors_m": errors, "metric": values}, indent=2, allow_nan=False),
        encoding="utf-8")
    return values
