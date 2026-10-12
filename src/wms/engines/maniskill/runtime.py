"""ManiSkill native process environment, without dependency/GPU preflight."""
import os
from pathlib import Path
from wms.engines.runtime import load_runtime as load_config

MANISKILL_VERSION = "3.0.1"


def native_environment(config):
    values = {"PYTHONDONTWRITEBYTECODE": "1", "MS_SKIP_ASSET_DOWNLOAD_PROMPT": "1"}
    if config.get("gpu_device") is not None:
        values["CUDA_VISIBLE_DEVICES"] = str(config["gpu_device"])
    if config.get("asset_dir"):
        values["MS_ASSET_DIR"] = config["asset_dir"]
    if os.environ.get("VK_ICD_FILENAMES"):
        values["VK_ICD_FILENAMES"] = os.environ["VK_ICD_FILENAMES"]
    else:
        for candidate in ("/etc/vulkan/icd.d/nvidia_icd.json", "/usr/share/vulkan/icd.d/nvidia_icd.json"):
            if Path(candidate).is_file():
                values["VK_ICD_FILENAMES"] = candidate
                break
    return values
