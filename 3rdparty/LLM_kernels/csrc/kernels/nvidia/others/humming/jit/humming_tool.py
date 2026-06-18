#!/usr/bin/env python3
# Copyright 2026 Tencent Inc. All rights reserved.
#
# Unified humming CLI tool for KsanaLLM JIT compilation.
#
# Subcommands:
#   generate       Generate a humming GEMM kernel .cu source file
#   query-configs  Query humming heuristic for recommended kernel configurations
#
# Usage:
#   python3 humming_tool.py generate --size_n 7168 --size_k 2048 ...
#   python3 humming_tool.py query-configs --size_n 4096 --size_k 4096 ...

import argparse
import json
import os
import sys
import types
from typing import Any, Dict, List, Optional, Tuple

# ════════════════════════════════════════════════════════════════════════════════
# Pre-import: block humming.ops from loading.
#
# humming/__init__.py unconditionally does ``import humming.ops``, which:
#   1. Registers 9 torch custom ops (register_op x 9 ~ expensive)
#   2. Calls build_humming_launcher_in_bg() → spawns subprocess to JIT-compile
#      the torch C++ launcher extension (~6s even in background)
#
# We only need humming.tune / humming.layer / humming.kernel / humming.config.
# None of them require humming.ops.  Block it by pre-populating sys.modules
# with a stub so that ``import humming.ops`` inside humming/__init__.py becomes
# a no-op.
# ════════════════════════════════════════════════════════════════════════════════

os.environ.setdefault("HUMMING_DISABLE_PARALLEL_BUILD", "1")

_stub = types.ModuleType("humming.ops")
_stub.__path__ = []  # make it look like a package
_stub.__all__ = []
sys.modules["humming.ops"] = _stub
sys.modules["humming.ops.utils"] = types.ModuleType("humming.ops.utils")
sys.modules["humming.ops.input"] = types.ModuleType("humming.ops.input")
sys.modules["humming.ops.moe"] = types.ModuleType("humming.ops.moe")
sys.modules["humming.ops.weight"] = types.ModuleType("humming.ops.weight")
sys.modules["humming.ops.bench"] = types.ModuleType("humming.ops.bench")
sys.modules["humming.ops.hadamard"] = types.ModuleType("humming.ops.hadamard")

# ════════════════════════════════════════════════════════════════════════════════
# Shared humming imports and helpers
# ════════════════════════════════════════════════════════════════════════════════

# Lazy imports — only import when a subcommand actually needs them.
_humming_imports = None


def _import_humming():
    """Import humming package. Cached after first call."""
    global _humming_imports
    if _humming_imports is not None:
        return _humming_imports

    try:
        from humming.kernel.humming import HummingKernel, KernelRuntime
        from humming.config.enum import GemmType, MmaType, WeightScaleType
        from humming import dtypes
        from humming.tune import get_heuristics_class
        from humming.layer import HummingLayerMeta
    except ImportError as e:
        print(f"ERROR: Failed to import humming package: {e}", file=sys.stderr)
        print("Please ensure the humming package is installed.", file=sys.stderr)
        sys.exit(1)

    _humming_imports = {
        "HummingKernel": HummingKernel,
        "KernelRuntime": KernelRuntime,
        "GemmType": GemmType,
        "MmaType": MmaType,
        "WeightScaleType": WeightScaleType,
        "dtypes": dtypes,
        "get_heuristics_class": get_heuristics_class,
        "HummingLayerMeta": HummingLayerMeta,
    }
    return _humming_imports


def _get_dtype_config(compute_dtype: int) -> Tuple[Any, bool, Any]:
    """Map compute_dtype (0=BF16, 1=FP8) to (a_dtype, use_fused_e8m0_scale, weight_scale_type)."""
    m = _import_humming()
    if compute_dtype == 1:
        return (
            m["dtypes"].float8e4m3,
            True,
            m["WeightScaleType"].GROUP_TENSOR,
        )
    else:
        return (
            m["dtypes"].bfloat16,
            False,
            m["WeightScaleType"].GROUP,
        )


def _build_layer_meta(args, a_dtype, use_fused_e8m0_scale, weight_scale_type):
    """Build a HummingLayerMeta from CLI args."""
    m = _import_humming()
    return m["HummingLayerMeta"](
        shape_n=args.size_n,
        shape_k=args.size_k,
        num_experts=args.num_experts,
        b_dtype=m["dtypes"].float4e2m1,
        a_dtype=a_dtype,
        c_dtype=m["dtypes"].bfloat16,
        bs_dtype=m["dtypes"].float8e8m0,
        weight_scale_group_size=32,
        weight_scale_type=weight_scale_type,
        has_zero_point=False,
        has_bias=False,
        mma_type=m["MmaType"].WGMMA,
        use_fused_e8m0_scale=use_fused_e8m0_scale,
    )


def _monkey_patch_kernel_runtime():
    """Monkey-patch to avoid CUDA dependency during source generation.
    We only need the rendered source code; compilation happens later in C++.
    """
    m = _import_humming()
    m["KernelRuntime"].prepare = lambda self: None
    m["KernelRuntime"].init_sm_version = lambda self: (
        setattr(self, "sm_version", 90) or setattr(self, "sm_version_str", "90a")
    )


# ════════════════════════════════════════════════════════════════════════════════
# Subcommand: generate
# ════════════════════════════════════════════════════════════════════════════════

def _add_generate_args(subparsers):
    p = subparsers.add_parser("generate", help="Generate a humming GEMM kernel .cu source file")
    p.add_argument("--size_n", type=int, required=True, help="GEMM N dimension")
    p.add_argument("--size_k", type=int, required=True, help="GEMM K dimension")
    p.add_argument("--num_experts", type=int, required=True, help="Number of experts")
    p.add_argument("--block_m", type=int, required=True, help="Block M for heuristics")
    p.add_argument("--gemm_type", type=int, required=True, choices=[1, 2],
                   help="1=INDEXED, 2=GROUPED_CONTIGUOUS")
    p.add_argument("--compute_dtype", type=int, required=True, choices=[0, 1],
                   help="0=BF16, 1=FP8")
    p.add_argument("--output", type=str, required=True, help="Output .cu file path")
    p.add_argument("--meta_output", type=str, default=None, help="Output kernel metadata JSON path")
    return p


def _cmd_generate(args):
    m = _import_humming()
    _monkey_patch_kernel_runtime()

    a_dtype, use_fused_e8m0_scale, weight_scale_type = _get_dtype_config(args.compute_dtype)

    gemm_type_map = {1: m["GemmType"].INDEXED, 2: m["GemmType"].GROUPED_CONTIGUOUS}
    gemm_type = gemm_type_map[args.gemm_type]

    meta = _build_layer_meta(args, a_dtype, use_fused_e8m0_scale, weight_scale_type)

    config = m["get_heuristics_class"]().get_config(meta, shape_m=args.block_m, gemm_type=gemm_type)

    kernel = m["HummingKernel"](
        shape_n=args.size_n,
        shape_k=args.size_k,
        num_experts=args.num_experts,
        b_dtype=m["dtypes"].float4e2m1,
        a_dtype=a_dtype,
        c_dtype=m["dtypes"].bfloat16,
        bs_dtype=m["dtypes"].float8e8m0,
        weight_scale_group_size=32,
        weight_scale_type=weight_scale_type,
        has_zero_point=False,
        has_bias=False,
        mma_type=m["MmaType"].WGMMA,
        use_fused_e8m0_scale=use_fused_e8m0_scale,
        gemm_type=gemm_type,
        block_shape=tuple(config["block_shape"]),
        warp_shape=tuple(config["warp_shape"]),
        use_stream_k=config["use_stream_k"],
        num_stages=config["num_stages"],
        num_ctas_per_sm=config.get("num_ctas_per_sm", 1),
        use_f16_accum=config.get("use_f16_accum", False),
    )

    cuda_source = kernel.code
    kernel_expr = kernel.kernel_expr

    # Append template instantiation to force the kernel symbol into the .cubin
    cuda_source += (
        "\n// Force template instantiation for symbol extraction\n"
        f"auto __humming_kernel_ptr = reinterpret_cast<void*>(&{kernel_expr});\n"
    )

    with open(args.output, "w") as f:
        f.write(cuda_source)
    print(f"Generated kernel source: {args.output}", file=sys.stderr)

    if args.meta_output:
        metadata = {
            "kernel_name": kernel.name,
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


# ════════════════════════════════════════════════════════════════════════════════
# Subcommand: query-configs
# ════════════════════════════════════════════════════════════════════════════════

def _add_query_configs_args(subparsers):
    p = subparsers.add_parser("query-configs",
                              help="Query humming heuristic for recommended kernel configs")
    p.add_argument("--size_n", type=int, required=True, help="GEMM N dimension")
    p.add_argument("--size_k", type=int, required=True, help="GEMM K dimension")
    p.add_argument("--num_experts", type=int, required=True, help="Number of experts")
    p.add_argument("--compute_dtype", type=int, default=0, choices=[0, 1],
                   help="0=BF16, 1=FP8")
    return p


def _cmd_query_configs(args):
    try:
        m = _import_humming()
        a_dtype, use_fused_e8m0_scale, _weight_scale_type_unused = _get_dtype_config(args.compute_dtype)
        weight_scale_type = m["WeightScaleType"].GROUP  # Always GROUP for heuristic query

        meta = _build_layer_meta(args, a_dtype, use_fused_e8m0_scale, weight_scale_type)

        configs = m["get_heuristics_class"]().get_configs(meta, gemm_type=m["GemmType"].INDEXED)

        results = []
        for entry in configs:
            min_m, max_m, cfg = entry[0], entry[1], entry[2]
            block_m = cfg["block_shape"][0]
            if max_m > 1000000:
                rep_m = min_m + 1000
            else:
                rep_m = (min_m + max_m) // 2
            rep_m = max(rep_m, max(min_m, 1))

            results.append({
                "shape_m_representative": rep_m,
                "min_shape_m": min_m,
                "max_shape_m": max_m,
                "block_m": block_m,
                "use_stream_k": cfg["use_stream_k"],
                "num_stages": cfg["num_stages"],
                "num_ctas_per_sm": cfg.get("num_ctas_per_sm", 1),
            })

        # Deduplicate: merge consecutive entries with the same representative
        deduped = []
        seen_reps = {}
        for r in results:
            rep = r["shape_m_representative"]
            if rep in seen_reps:
                idx = seen_reps[rep]
                deduped[idx]["min_shape_m"] = min(deduped[idx]["min_shape_m"], r["min_shape_m"])
                deduped[idx]["max_shape_m"] = max(deduped[idx]["max_shape_m"], r["max_shape_m"])
            else:
                seen_reps[rep] = len(deduped)
                deduped.append(r)

        print(json.dumps(deduped))
    except Exception as e:
        # Fallback: return default configs (block_m=64,128)
        sys.stderr.write(f"Warning: humming heuristic query failed: {e}\n")
        sys.stderr.write("Falling back to default configs\n")
        results = [
            {"shape_m_representative": 64, "min_shape_m": 1, "max_shape_m": 64,
             "block_m": 64, "use_stream_k": True, "num_stages": 3, "num_ctas_per_sm": 1},
            {"shape_m_representative": 128, "min_shape_m": 65, "max_shape_m": 1073741824,
             "block_m": 128, "use_stream_k": True, "num_stages": 3, "num_ctas_per_sm": 1},
        ]
        print(json.dumps(results))


# ════════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Humming kernel tool for KsanaLLM JIT compilation"
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    _add_generate_args(subparsers)
    _add_query_configs_args(subparsers)

    args = parser.parse_args()

    if args.subcommand == "generate":
        _cmd_generate(args)
    elif args.subcommand == "query-configs":
        _cmd_query_configs(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
