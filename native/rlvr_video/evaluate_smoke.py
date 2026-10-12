"""CPU evaluator: only this separate process receives the future reference."""
import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image

from wms.core.world_model import file_hash, validate_prediction_bundle


def metrics(prediction, reference):
    difference = prediction.astype(np.float64) - reference.astype(np.float64)
    mse = float(np.mean(difference ** 2))
    return {"mae": float(np.mean(np.abs(difference))), "mse": mse,
            "psnr_db": float(-10 * np.log10(mse)) if mse > 0 else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    prediction_dir, data_root, out = map(Path, (args.prediction_dir, args.data_root, args.output_dir))
    out.mkdir(parents=True, exist_ok=True)
    bundle = json.loads((prediction_dir / "prediction.json").read_text())
    validate_prediction_bundle(bundle, prediction_dir)
    forecast = np.load(prediction_dir / "forecast.npy", allow_pickle=False)
    with np.load(data_root / "reference.npz", allow_pickle=False) as data:
        reference = data["image"].astype(np.float32) / 255
    with np.load(data_root / "context.npz", allow_pickle=False) as data:
        last_context = data["image"][-1:].astype(np.float32) / 255
    if forecast.shape != (1, 256, 320, 3) or forecast.shape != reference.shape:
        raise ValueError("Unexpected prediction or evaluation reference shape")
    if not np.isfinite(forecast).all() or forecast.min() < 0 or forecast.max() > 1:
        raise ValueError("Invalid predicted pixels")
    decoded = list(imageio.mimread(prediction_dir / "context_and_prediction.mp4"))
    if len(decoded) != 5:
        raise ValueError("Preview must contain exactly four context frames and one prediction")
    with np.load(prediction_dir / "token_trace.npz", allow_pickle=False) as data:
        if data["prompt"].shape != (1, 1333) or data["generated"].shape != (321,):
            raise ValueError("Unexpected native token sequence lengths")
    comparison = np.concatenate([last_context[0], forecast[0], reference[0]], axis=1)
    Image.fromarray((comparison * 255).astype(np.uint8)).save(out / "last_context_prediction_reference.png")
    report = {
        "evaluation_type": "single_real_sample_engineering_smoke",
        "sample_count": 1, "split_status": "split_unverified",
        "prediction_integrity_valid": True, "video_decoded_frames": len(decoded),
        "pixel_domain": "RGB float in [0,1] at 256x320",
        "model": metrics(forecast, reference), "copy_last_observation_baseline": metrics(last_context, reference),
        "reference_sha256": file_hash(data_root / "reference.npz"),
        "comparison_panel_order": ["last observed context", "RLVR model prediction", "real future reference"],
        "interpretation": "One sample checks real execution and data alignment. Errors are not safety scores; no generalization, RLVR improvement, or benchmark parity claim.",
    }
    (out / "evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
