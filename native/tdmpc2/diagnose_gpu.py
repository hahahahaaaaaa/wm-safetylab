"""Read-only CUDA/SAPIEN deployment diagnosis; no simulator task is trained."""
import ctypes
import json
import sys
from pathlib import Path

runtime = json.loads(Path(sys.argv[1]).read_text())
driver = ctypes.CDLL("libcuda.so", ctypes.RTLD_GLOBAL)
print("driver", driver, "cuInit", driver.cuInit(0), flush=True)
count = ctypes.c_int()
print("cuDeviceGetCount", driver.cuDeviceGetCount(ctypes.byref(count)), count.value, flush=True)
import sapien
sapien.set_log_level("debug")
print("sapien", sapien.__version__, "physx", sapien.physx.version(), flush=True)
library = ctypes.CDLL(runtime["physx_gpu_library"], ctypes.RTLD_LOCAL)
sapien.physx._enable_gpu()
print("enabled", sapien.physx.is_gpu_enabled(), flush=True)
print("device", sapien.Device("cuda:0"), flush=True)
print("system", sapien.physx.PhysxGpuSystem(device=sapien.Device("cuda:0")), flush=True)
