#!/usr/bin/env python3
"""Batch-generate all moe_fused_mul_sum Triton kernel variants.

Iterates the SM90 BF16 variant grid in-process (single torch/triton import)
and delegates each variant to generate_triton_moe_fused_mul_sum.generate_one.

Usage:
    python triton_moe_fused_mul_sum_creator.py <OUTPUT_DIR>
"""

import os
import sys

# Pin to the first visible GPU so compilation doesn't fan out across devices.
cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
if cuda_devices:
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices.split(",")[0]

# Allow import of the sibling generator module.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from generate_triton_moe_fused_mul_sum import compute_block_k, generate_kernel_variant  # noqa: E402


# Fixed top_k for KsanaLLM humming MoE.
TOP_K = 6

# Hidden sizes: Flash (4096) and Pro (7168).
SIZES = (4096, 7168)

# Expert-parallel on/off.
IS_EPS = (0, 1)

# BLOCK_M variants for SM90 (from _heuristic_config).
BLOCK_MS = (1, 2, 4)


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <OUTPUT_DIR>", file=sys.stderr)
        return 2

    output_dir = sys.argv[1]

    variants = [
        (size, is_ep, block_m)
        for size in SIZES
        for is_ep in IS_EPS
        for block_m in BLOCK_MS
    ]
    total = len(variants)
    print(f"Generating {total} moe_fused_mul_sum kernel variants...")

    for idx, (size, is_ep, block_m) in enumerate(variants, start=1):
        block_k = compute_block_k(size)
        print(
            f"[{idx}/{total}] size={size} is_ep={is_ep} "
            f"top_k={TOP_K} BLOCK_M={block_m} BLOCK_K={block_k}"
        )
        generate_kernel_variant(
            top_k=TOP_K,
            size=size,
            is_ep=is_ep,
            block_m=block_m,
            block_k=block_k,
            output_dir=output_dir,
        )

    print(f"Done. Generated {total} kernel variants.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
