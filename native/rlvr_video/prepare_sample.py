"""Extract one real RT-1 record; keep the evaluation target out of model inputs.

Input is a byte-range copy of the FIRST TFRecord in original shard 00000.
This is a reproducible smoke sample, not a verified held-out evaluation set.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import struct

import crc32c
import numpy as np
from PIL import Image
from tfrecord import example_pb2

ACTION_FIELDS = [("base_displacement_vector", 2), ("base_displacement_vertical_rotation", 1),
                 ("gripper_closedness_action", 1), ("rotation_delta", 3),
                 ("terminate_episode", 3), ("world_vector", 3)]


def masked_crc(data):
    crc = crc32c.crc32c(data)
    return (((crc >> 15) | (crc << 17)) + 0xa282ead8) & 0xffffffff


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    args = parser.parse_args()
    root = Path(args.data_root)
    raw = (root / "episode-00000.tfrecord").read_bytes()
    size = struct.unpack("<Q", raw[:8])[0]
    if len(raw) != size + 16:
        raise ValueError("Expected exactly one complete original TFRecord")
    if masked_crc(raw[:8]) != struct.unpack("<I", raw[8:12])[0] or masked_crc(raw[12:-4]) != struct.unpack("<I", raw[-4:])[0]:
        raise ValueError("TFRecord CRC mismatch")
    example = example_pb2.Example()
    example.ParseFromString(raw[12:-4])
    features = example.features.feature
    images = np.stack([np.asarray(Image.open(io.BytesIO(x)).convert("RGB"))
                       for x in features["steps/observation/image"].bytes_list.value])
    count = len(images)
    actions = []
    for field, width in ACTION_FIELDS:
        f = features["steps/action/" + field]
        values = f.float_list.value if f.WhichOneof("kind") == "float_list" else f.int64_list.value
        actions.append(np.array(values, dtype=np.float32).reshape(count, width))
    actions = np.concatenate(actions, axis=1)
    if count < 5 or actions.shape != (count, 13):
        raise ValueError("Incomplete RT-1 episode")
    # Mirror EvalDataset item 0's seeded start for a five-frame segment.
    start = int(np.random.RandomState(0).randint(count - 5 + 1))
    np.savez_compressed(root / "context.npz", image=images[start:start+4], action=actions[start:start+4])
    np.savez_compressed(root / "reference.npz", image=images[start+4:start+5])
    metadata = {
        "dataset": "fractal20220817_data", "version": "0.1.0", "original_split": "train",
        "source_url": "https://storage.googleapis.com/gresearch/robotics/fractal20220817_data/0.1.0/fractal20220817_data-train.tfrecord-00000-of-01024",
        "source_shard": 0, "record_in_shard": 0, "source_record_sha256": hashlib.sha256(raw).hexdigest(),
        "episode_frames": count, "original_image_shape": list(images.shape), "segment_start": start,
        "context_indices": list(range(start, start+4)), "reference_indices": [start+4],
        "action_fields_in_concatenation_order": ACTION_FIELDS,
        "action_order_basis": "Alphabetical TFDS nested dictionary iteration order; matches native 13-dimensional RT-1 range vector.",
        "split_status": "split_unverified",
        "split_explanation": "Released loader holds out sorted NPZ indices divisible by 100; shard0 record0 is consistent with converter episode0, but author checkpoint training episode membership is not released. No generalization claim.",
        "target_is_separate": True,
        "source_crc_verified": True,
    }
    key = "steps/observation/natural_language_instruction"
    if key in features:
        metadata["instruction"] = features[key].bytes_list.value[0].decode("utf-8", errors="replace")
    for filename in ("context.npz", "reference.npz", "dataset_info.json", "features.json"):
        metadata[filename + "_sha256"] = hashlib.sha256((root / filename).read_bytes()).hexdigest()
    (root / "sample_provenance.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
