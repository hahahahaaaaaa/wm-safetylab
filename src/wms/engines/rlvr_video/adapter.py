"""Typed evidence boundary around the released RLVR video checkpoint.

This adapter does not train, execute robot actions, or infer safety probabilities.
Only context frames and their actions enter the predictor; references stay outside.
"""
from __future__ import annotations

import io
from pathlib import Path
import sys
import time
from types import SimpleNamespace

from wms.core.world_model import validate_request, write_prediction_bundle

SOURCE_COMMIT = "e1b8b6f40ca0696919ce9b8dbe965c4153bcf5f1"
MODEL_ID = "rlvr-video-rt1-single-step-rlvr"
MODEL_REVISION = "ba6dae4049067c703f236f709720f6ff216c7e5f"
TOKENIZER_REVISION = "a41f1ad0fc30da2d24bfc346dee65d2f70703bfd"
MODEL_SHA256 = "185f21ed412e9a2ebc38402d8157f6abdb1a0f97b4bfcc622b584a9ab00b7eaa"
TOKENIZER_SHA256 = "ef5e927f2c45087066993d5a7cc42abf7ecba36ae044637df32c2204e2099c91"
ACTION_FIELDS = [
    ["base_displacement_vector", 2],
    ["base_displacement_vertical_rotation", 1],
    ["gripper_closedness_action", 1],
    ["rotation_delta", 3],
    ["terminate_episode", 3],
    ["world_vector", 3],
]


class RLVRVideoAdapter:
    def __init__(self, config_dict):
        self.config = dict(config_dict)

    def describe(self):
        return {
            "schema_version": "wms.model/1.0",
            "model_id": MODEL_ID,
            "name": "RLVR-World RT-1 single-step RLVR video model",
            "families": ["observation_generative_interactive"],
            "representations": ["video"],
            "capabilities": ["predict"],
            "action_conditioning": "rt1_13d_mixed_robot_action_discretized_256_bins",
            "limitations": [
                "Exactly four observed frames and four aligned RT-1 actions predict one future frame.",
                "Open-loop prediction only; no robot execution, safety verdict, or event probability.",
                "RT-1 action order and ranges are checkpoint-specific; other robots require an independently validated bridge.",
                "Transformers sampling backend differs from the authors' vLLM evaluation; this is not benchmark parity.",
                "Observed smoke data is split_unverified against the released weights; no generalization claim.",
                "No training or post-training capability is exposed by this inference adapter.",
            ],
            "provenance": {
                "source_url": "https://github.com/thuml/RLVR-World",
                "source_commit": SOURCE_COMMIT,
                "checkpoint": "thuml/rt1-world-model-single-step-rlvr",
                "checkpoint_revision": MODEL_REVISION,
                "checkpoint_sha256": MODEL_SHA256,
                "visual_tokenizer": "thuml/rt1-frame-tokenizer",
                "visual_tokenizer_revision": TOKENIZER_REVISION,
                "visual_tokenizer_sha256": TOKENIZER_SHA256,
                "license": "MIT",
                "native_processor": "vid_wm/ivideogpt/ivideogpt/processor.py:SimpleVideoProcessor",
                "action_fields": ACTION_FIELDS,
            },
        }

    def _validate(self, request, output_dir):
        validate_request(request, self.describe())
        if request["horizon"] != {"steps": 1, "unit": "video_frame"}:
            raise ValueError("This checkpoint adapter supports exactly one future video_frame")
        if "context_npz" not in request["inputs"]:
            raise ValueError("RLVR video requires context_npz")
        if request.get("parameters", {}):
            raise ValueError("Sampling is fixed to the documented native protocol; unsupported parameters")
        context = Path(request["inputs"]["context_npz"]).resolve()
        context_bytes = context.read_bytes()
        out = Path(output_dir).resolve()
        return context, out, context_bytes

    def predict(self, request, output_dir):
        context_path, out, context_bytes = self._validate(request, output_dir)
        # Import ML dependencies only in the isolated native worker.
        import numpy as np
        import torch
        import torchvision
        from PIL import Image
        import imageio.v2 as imageio
        import transformers
        from transformers import AutoConfig, AutoModelForCausalLM
        from safetensors.torch import load_file

        with np.load(io.BytesIO(context_bytes), allow_pickle=False) as data:
            if set(data.files) != {"image", "action"}:
                raise ValueError("context_npz must contain image and action only")
            frames, actions = data["image"], data["action"]
        if frames.ndim != 4 or frames.shape[0] != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
            raise ValueError("Context image must be uint8 [4, H, W, 3]")
        if min(frames.shape[1:3]) < 16 or max(frames.shape[1:3]) > 4096:
            raise ValueError("Unsupported context image dimensions")
        if actions.shape != (4, 13) or not np.issubdtype(actions.dtype, np.number) or not np.isfinite(actions).all():
            raise ValueError("Context action must be finite numeric [4,13] in declared RT-1 order")

        source = Path(self.config["source_root"]).resolve()
        model_path = Path(self.config["model_path"]).resolve()
        tokenizer_path = Path(self.config["tokenizer_path"]).resolve()
        native = source / "vid_wm" / "ivideogpt"
        sys.path.insert(0, str(native))
        from ivideogpt.tokenizer import CNNFSQModel256
        from ivideogpt.processor import SimpleVideoProcessor

        device = str(self.config.get("device", "cuda:0"))
        if not device.startswith("cuda") or not torch.cuda.is_available():
            raise ValueError("The validated native tokenizer path requires a CUDA runtime")
        torch.cuda.set_device(device)
        torch.manual_seed(request["seed"])
        torch.cuda.manual_seed_all(request["seed"])
        torch.set_num_threads(int(self.config.get("cpu_threads", 4)))
        torch.cuda.reset_peak_memory_stats(device)
        started = time.monotonic()
        tokenizer = CNNFSQModel256.from_pretrained(str(tokenizer_path), local_files_only=True).to(device).eval()
        # The official release has no safetensors metadata. Load the verified
        # tensor state directly rather than altering the upstream checkpoint.
        model_config = AutoConfig.from_pretrained(str(model_path), local_files_only=True)
        model = AutoModelForCausalLM.from_config(model_config, attn_implementation="sdpa")
        model.load_state_dict(load_file(str(model_path / "model.safetensors")), strict=True)
        model = model.to(device).eval()
        config = SimpleNamespace(action_ranges_path=str(native / "configs/vgpt/frac_action_ranges.pth"),
                                 action_bins=256, context_length=4, visual_token_num=4375,
                                 bos_token_id=4631, eos_token_id=4632)
        processor = SimpleVideoProcessor(config, tokenizer)
        processor.action_ranges = processor.action_ranges.to(device)
        pixels = torch.from_numpy(frames.copy()).permute(0, 3, 1, 2).float().div(255)
        # Same tensor resize as EvalDataset; preserve its explicit 256 x 320 shape.
        pixels = torchvision.transforms.functional.resize(pixels, [256, 320], antialias=True).unsqueeze(0).to(device)
        action_tensor = torch.from_numpy(actions.copy()).float().unsqueeze(0).to(device)
        with torch.inference_mode():
            # Build precisely the prefix used in eval_vgpt.py without encoding a future target.
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                visual_tokens = tokenizer.encode(pixels).reshape(1, 4, 320)
            action_tokens = processor._discretize_actions(action_tensor, 256).to(torch.long) + 4375
            prefix = torch.cat([visual_tokens, action_tokens], dim=-1).reshape(1, -1)
            prefix = torch.cat([prefix, torch.full((1, 1), 4631, device=device, dtype=torch.long)], dim=-1).long()
            generated = model.generate(
                input_ids=prefix, attention_mask=torch.ones_like(prefix), do_sample=True,
                temperature=1.0, top_k=100, top_p=1.0, min_new_tokens=321, max_new_tokens=321,
                eos_token_id=None, pad_token_id=0, use_cache=True,
            )[0, prefix.shape[1]:]
            visual_generated = generated[:-1]
            invalid = int(((visual_generated < 0) | (visual_generated >= 4375)).sum().item())
            # Native evaluation clamps non-visual ids before decoding; record them separately.
            decoded = tokenizer.decode(visual_generated.clamp(0, 4374).reshape(1, 1, 16, 20)).clamp(0, 1)
            forecast = decoded[0].permute(0, 2, 3, 1).float().cpu().numpy()
            context_resized = pixels[0].permute(0, 2, 3, 1).float().cpu().numpy()
        elapsed = time.monotonic() - started
        peak_memory = torch.cuda.max_memory_allocated(device)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "forecast.npy", forecast, allow_pickle=False)
        np.savez_compressed(out / "token_trace.npz", prompt=prefix.cpu().numpy(),
                            generated=generated.cpu().numpy(), action_tokens=action_tokens.cpu().numpy())
        Image.fromarray((forecast[0] * 255).astype(np.uint8)).save(out / "predicted_frame.png")
        video = (np.concatenate([context_resized, forecast]) * 255).astype(np.uint8)
        imageio.mimsave(out / "context_and_prediction.mp4", video, fps=4, macro_block_size=16)
        diagnostics = {
            "context_frames": 4, "predicted_frames": 1, "visual_tokens_per_frame": 320,
            "generated_token_count": int(generated.numel()), "invalid_visual_token_count": invalid,
            "native_invalid_token_handling": "clamp_to_visual_vocabulary_before_decode",
            "terminal_token": int(generated[-1]), "terminal_matches_eos": int(generated[-1]) == 4632,
            "generation_backend": "transformers", "torch_version": torch.__version__,
            "transformers_version": transformers.__version__, "elapsed_seconds": elapsed,
            "peak_allocated_gpu_bytes": peak_memory, "sampling": {"temperature": 1.0, "top_k": 100, "top_p": 1.0},
        }
        return write_prediction_bundle(out, request, self.describe(), [
            {"name": "forecast", "path": "forecast.npy", "kind": "video", "media_type": "application/x-npy",
             "shape": [1, 256, 320, 3], "axes": ["time", "height", "width", "channel"],
             "semantics": "predicted RGB float32 in [0,1]; exactly one future frame"},
            {"name": "predicted_frame", "path": "predicted_frame.png", "kind": "image", "media_type": "image/png"},
            {"name": "preview", "path": "context_and_prediction.mp4", "kind": "video", "media_type": "video/mp4",
             "semantics": "first four frames observed context; final frame predicted; display at 4 fps, not measured robot rate"},
            {"name": "token_trace", "path": "token_trace.npz", "kind": "trace", "media_type": "application/x-npz",
             "semantics": "native visual/action token ids, not environment-state probabilities"},
        ], mode="open_loop", diagnostics=diagnostics,
            provenance={**self.describe()["provenance"], "context_path": str(context_path),
                        "action_ranges_path": str(config.action_ranges_path),
                        "split_status": "split_unverified", "weights_loaded": "official_rlvr_not_base"},
            claims=["Official released RLVR weights produced a real action-conditioned next-frame prediction.",
                    "A smoke prediction is not evidence of safety, calibration, benchmark parity, or generalization."])
