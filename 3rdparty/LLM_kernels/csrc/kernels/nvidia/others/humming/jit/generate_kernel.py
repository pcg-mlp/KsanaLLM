#!/usr/bin/env python3
# Copyright 2024 Tencent Inc. All rights reserved.
#
# Generate a humming GEMM kernel .cu source file for KsanaLLM JIT compilation.
#
# Usage:
#   python3 generate_kernel.py \
#       --size_n 7168 --size_k 2048 --num_experts 256 \
#       --block_m 64 --gemm_type 1 --compute_dtype 1 \
#       --output /tmp/kernel.cu --meta_output /tmp/kernel_meta.json

import argparse
import json
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate humming JIT kernel source (.cu) for KsanaLLM"
    )
    parser.add_argument("--size_n", type=int, required=True, help="GEMM N dimension")
    parser.add_argument("--size_k", type=int, required=True, help="GEMM K dimension")
    parser.add_argument("--num_experts", type=int, required=True, help="Number of experts")
    parser.add_argument("--block_m", type=int, required=True, help="Block M for heuristics")
    parser.add_argument(
        "--gemm_type",
        type=int,
        required=True,
        choices=[1, 2],
        help="1=INDEXED, 2=GROUPED_CONTIGUOUS",
    )
    parser.add_argument(
        "--compute_dtype",
        type=int,
        required=True,
        choices=[0, 1],
        help="0=BF16, 1=FP8",
    )
    parser.add_argument("--output", type=str, required=True, help="Output .cu file path")
    parser.add_argument(
        "--meta_output", type=str, default=None, help="Output kernel metadata JSON path"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Import humming ──────────────────────────────────────────────────────
    try:
        from humming.kernel.humming import HummingKernel, KernelRuntime
        from humming.config.enum import GemmType, MmaType, WeightScaleType
        from humming import dtypes
        from humming.tune.sm90_h20 import Sm90H20Heuristics
        from humming.layer import HummingLayerMeta
    except ImportError as e:
        print(f"ERROR: Failed to import humming package: {e}", file=sys.stderr)
        print("Please ensure the humming package is installed.", file=sys.stderr)
        sys.exit(1)

    # ── Monkey-patch to avoid CUDA dependency during source generation ──────
    # We only need the rendered source code; compilation happens later in C++.
    KernelRuntime.prepare = lambda self: None
    KernelRuntime.init_sm_version = lambda self: (
        setattr(self, "sm_version", 90) or setattr(self, "sm_version_str", "90a")
    )

    # ── Map CLI args to humming enums ───────────────────────────────────────
    gemm_type_map = {
        1: GemmType.INDEXED,
        2: GemmType.GROUPED_CONTIGUOUS,
    }
    gemm_type = gemm_type_map[args.gemm_type]

    # Compute dtype: 0=BF16 activations, 1=FP8 activations
    if args.compute_dtype == 1:
        a_dtype = dtypes.float8e4m3
        # fused_e8m0_scale only supported for FP8 activations (the CUDA dequant kernel
        # requires Float8E4M3 or Int8 as target type, not BF16).
        use_fused_e8m0_scale = True
        # When use_fused_e8m0_scale=True, humming's HummingLayerMeta overrides
        # weight_scale_type to GROUP_TENSOR (per layer.py:151-153). LayerConfig (the
        # base of HummingKernel) does NOT apply this override, so we must pass it
        # explicitly here — otherwise is_tensor_weight_scale stays False and the
        # kernel skips global_scale application and the +6 exponent compensation.
        weight_scale_type = WeightScaleType.GROUP_TENSOR
    else:
        a_dtype = dtypes.bfloat16
        use_fused_e8m0_scale = False
        weight_scale_type = WeightScaleType.GROUP

    # ── Build HummingLayerMeta for heuristics ───────────────────────────────
    meta = HummingLayerMeta(
        shape_n=args.size_n,
        shape_k=args.size_k,
        num_experts=args.num_experts,
        b_dtype=dtypes.float4e2m1,
        a_dtype=a_dtype,
        c_dtype=dtypes.bfloat16,
        bs_dtype=dtypes.float8e8m0,
        weight_scale_group_size=32,
        weight_scale_type=weight_scale_type,
        has_zero_point=False,
        has_bias=False,
        mma_type=MmaType.WGMMA,
        use_fused_e8m0_scale=use_fused_e8m0_scale,
    )

    # ── Get tuning config from H20 heuristics ──────────────────────────────
    config = Sm90H20Heuristics.get_config(
        meta, shape_m=args.block_m, gemm_type=gemm_type
    )

    # ── Create HummingKernel ────────────────────────────────────────────────
    kernel = HummingKernel(
        shape_n=args.size_n,
        shape_k=args.size_k,
        num_experts=args.num_experts,
        b_dtype=dtypes.float4e2m1,
        a_dtype=a_dtype,
        c_dtype=dtypes.bfloat16,
        bs_dtype=dtypes.float8e8m0,
        weight_scale_group_size=32,
        weight_scale_type=weight_scale_type,
        has_zero_point=False,
        has_bias=False,
        mma_type=MmaType.WGMMA,
        use_fused_e8m0_scale=use_fused_e8m0_scale,
        gemm_type=gemm_type,
        block_shape=tuple(config["block_shape"]),
        warp_shape=tuple(config["warp_shape"]),
        use_stream_k=config["use_stream_k"],
        num_stages=config["num_stages"],
        num_ctas_per_sm=config.get("num_ctas_per_sm", 1),
        use_f16_accum=config.get("use_f16_accum", False),
    )

    # ── Get rendered source and kernel expression ───────────────────────────
    cuda_source = kernel.code
    kernel_expr = kernel.kernel_expr
    kernel_name = kernel.name  # "humming"

    # Append template instantiation to force the kernel symbol into the .cubin
    instantiation = (
        "\n// Force template instantiation for symbol extraction\n"
        f"auto __humming_kernel_ptr = reinterpret_cast<void*>(&{kernel_expr});\n"
    )
    cuda_source += instantiation

    # ── Write .cu output ────────────────────────────────────────────────────
    with open(args.output, "w") as f:
        f.write(cuda_source)
    print(f"Generated kernel source: {args.output}", file=sys.stderr)

    # ── Write metadata JSON ─────────────────────────────────────────────────
    if args.meta_output:
        metadata = {
            "kernel_name": kernel_name,
            "kernel_expr": kernel_expr,
            "size_n": args.size_n,
            "size_k": args.size_k,
            "num_experts": args.num_experts,
            "block_m": args.block_m,
            "gemm_type": args.gemm_type,
            "compute_dtype": args.compute_dtype,
            "block_shape": list(config["block_shape"]),
            "warp_shape": list(config["warp_shape"]),
            "num_stages": config["num_stages"],
            "num_ctas_per_sm": config.get("num_ctas_per_sm", 1),
            "use_stream_k": config["use_stream_k"],
            "use_f16_accum": config.get("use_f16_accum", False),
        }
        with open(args.meta_output, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Generated kernel metadata: {args.meta_output}", file=sys.stderr)


if __name__ == "__main__":
    main()
