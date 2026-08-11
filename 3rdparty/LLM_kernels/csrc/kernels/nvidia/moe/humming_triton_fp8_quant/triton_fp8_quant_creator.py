#!/usr/bin/env python3
"""Batch-generate all humming FP8 quant Triton kernel variants.

Computes the unique N values needed across (model, tp, ep) configurations,
then iterates them in-process (single torch/triton import) and delegates each
N to both generators.

Usage:
    python triton_fp8_quant_creator.py <OUTPUT_DIR>
"""

import os
import sys

# Pin to the first visible GPU so compilation doesn't fan out across devices.
cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
if cuda_devices:
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices.split(",")[0]

# Allow import of the sibling generator modules.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from generate_triton_fp8_quant import generate_kernel_variant as generate_quant  # noqa: E402
from generate_triton_silu_mul_quant import generate_kernel_variant as generate_silu_mul_quant  # noqa: E402


# Model configs: name, H (hidden_size), I (inter_size).
MODELS = (
    ("Flash", 4096, 2048),
    ("Pro",   7168, 3072),
)

# TP / EP combinations to enumerate.
TPS = (1, 2, 4, 8)
EPS = (1, 2, 4, 8)


def collect_n_values():
    """Collect unique N values across all (model, tp, ep) combinations.

    For each combination we need two kernels:
      N1 = H                 (GEMM1: full hidden_size)
      N2 = ceil(I / moe_tp)  (GEMM2: inter_size sharded by moe_tp = max(tp/ep, 1))
    """
    n_set = set()
    for _name, h, i in MODELS:
        for tp in TPS:
            for ep in EPS:
                moe_tp = max(tp // ep, 1)
                n_set.add(h)
                n_set.add((i + moe_tp - 1) // moe_tp)  # ceil(I / moe_tp)
    return sorted(n_set)


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <OUTPUT_DIR>", file=sys.stderr)
        return 2

    output_dir = sys.argv[1]

    n_values = collect_n_values()
    total = len(n_values)
    print(f"Generating FP8 quant + SiluAndMul+FP8 quant kernels for {total} unique N values...")

    for idx, n in enumerate(n_values, start=1):
        print(f"[{idx}/{total}] N={n}: triton FP8 quant kernel")
        generate_quant(n=n, output_dir=output_dir)
        print(f"[{idx}/{total}] N={n}: fused SiluAndMul+FP8 quant kernel")
        generate_silu_mul_quant(n=n, output_dir=output_dir)

    print(f"Done. Generated kernels for {total} N values.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
