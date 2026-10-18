"""Command-line entry points (`python -m asm_jepa.cli.<name>`).

BLAS/OpenMP pools are limited to one thread per process before numpy/torch load: pose preprocessing runs
in many DataLoader workers on a shared server, and per-sample arrays are far too small to profit from
multi-threaded BLAS. Export OMP_NUM_THREADS explicitly to override.
"""
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
