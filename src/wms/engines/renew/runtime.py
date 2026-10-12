"""CPU settings for the existing RENEW mechanism experiment."""
import os
from wms.engines.runtime import load_runtime as load_config

SOURCE_COMMIT = "8f6e1ebf66a0200cc2015226cdd7d70900fc0cb7"
CPU_THREADS = 8


def native_environment(config=None):
    return {"JAX_PLATFORMS": "cpu", "CUDA_VISIBLE_DEVICES": "", "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": str(CPU_THREADS),
            "OPENBLAS_NUM_THREADS": str(CPU_THREADS), "MKL_NUM_THREADS": str(CPU_THREADS),
            "NUMEXPR_NUM_THREADS": str(CPU_THREADS), "TF_NUM_INTRAOP_THREADS": str(CPU_THREADS),
            "TF_NUM_INTEROP_THREADS": "1"}


def limit_cpu_affinity():
    if hasattr(os, "sched_getaffinity") and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:CPU_THREADS])
