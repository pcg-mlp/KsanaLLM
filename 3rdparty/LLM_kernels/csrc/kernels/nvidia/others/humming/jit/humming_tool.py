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
    # Problem shape
    p.add_argument("--size_n", type=int, required=True, help="GEMM N dimension")
    p.add_argument("--size_k", type=int, required=True, help="GEMM K dimension")
    p.add_argument("--num_experts", type=int, required=True, help="Number of experts")
    p.add_argument("--compute_dtype", type=int, required=True, choices=[0, 1],
                   help="0=BF16, 1=FP8")
    p.add_argument("--gemm_type", type=int, required=True, choices=[1, 2],
                   help="1=INDEXED, 2=GROUPED_CONTIGUOUS")
    # Block shape
    p.add_argument("--block_m", type=int, required=True, help="Block M")
    p.add_argument("--block_n", type=int, required=True, help="Block N")
    p.add_argument("--block_k", type=int, required=True, help="Block K")
    # Warp shape
    p.add_argument("--warp_shape_m", type=int, required=True, help="Warp shape M")
    p.add_argument("--warp_shape_n", type=int, required=True, help="Warp shape N")
    p.add_argument("--warp_shape_k", type=int, required=True, help="Warp shape K")
    # Scheduling
    p.add_argument("--use_stream_k", type=int, required=True, choices=[0, 1], help="Use stream-K")
    p.add_argument("--num_stages", type=int, required=True, help="Number of pipeline stages")
    p.add_argument("--num_ctas_per_sm", type=int, required=True, help="CTAs per SM")
    # TMA
    p.add_argument("--use_tma_a", type=int, required=True, choices=[0, 1])
    p.add_argument("--use_tma_b", type=int, required=True, choices=[0, 1])
    p.add_argument("--use_tma_c", type=int, required=True, choices=[0, 1])
    p.add_argument("--use_tma_bs", type=int, required=True, choices=[0, 1])
    # Other flags
    p.add_argument("--use_mbarrier", type=int, required=True, choices=[0, 1])
    p.add_argument("--use_warp_spec", type=int, required=True, choices=[0, 1])
    p.add_argument("--use_f16_accum", type=int, required=True, choices=[0, 1])
    # Output
    p.add_argument("--output", type=str, required=True, help="Output .cu file path")
    p.add_argument("--meta_output", type=str, default=None, help="Output kernel metadata JSON path")
    return p


def _cmd_generate(args):
    m = _import_humming()
    _monkey_patch_kernel_runtime()

    a_dtype, use_fused_e8m0_scale, weight_scale_type = _get_dtype_config(args.compute_dtype)

    gemm_type_map = {1: m["GemmType"].INDEXED, 2: m["GemmType"].GROUPED_CONTIGUOUS}
    gemm_type = gemm_type_map[args.gemm_type]

    block_shape = (args.block_m, args.block_n, args.block_k)
    warp_shape = (args.warp_shape_m, args.warp_shape_n, args.warp_shape_k)
    use_tma = bool(args.use_tma_a or args.use_tma_b or args.use_tma_c or args.use_tma_bs)

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
        block_shape=block_shape,
        warp_shape=warp_shape,
        use_stream_k=bool(args.use_stream_k),
        num_stages=args.num_stages,
        num_ctas_per_sm=args.num_ctas_per_sm,
        use_f16_accum=bool(args.use_f16_accum),
        use_tma=use_tma,
        use_tma_a=bool(args.use_tma_a),
        use_tma_b=bool(args.use_tma_b),
        use_tma_c=bool(args.use_tma_c),
        use_tma_bs=bool(args.use_tma_bs),
        use_mbarrier=bool(args.use_mbarrier),
        use_warp_spec=bool(args.use_warp_spec),
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
            "block_shape": list(block_shape),
            "warp_shape": list(warp_shape),
            "num_stages": args.num_stages,
            "num_ctas_per_sm": args.num_ctas_per_sm,
            "use_stream_k": bool(args.use_stream_k),
            "use_f16_accum": bool(args.use_f16_accum),
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


def _make_cache_key(cfg: Dict, size_n: int, size_k: int, num_experts: int,
                    compute_dtype: int, gemm_type: int) -> str:
    """Generate a unique cache key string for a kernel config."""
    parts = [
        f"n{size_n}", f"k{size_k}", f"e{num_experts}",
        f"dt{compute_dtype}", f"gt{gemm_type}",
        f"bm{cfg['block_m']}", f"bn{cfg['block_n']}", f"bk{cfg['block_k']}",
        f"wp{cfg['warp_shape'][0]}x{cfg['warp_shape'][1]}x{cfg['warp_shape'][2]}",
        f"sk{int(cfg['use_stream_k'])}",
        f"stages{cfg['num_stages']}",
        f"ctas{cfg['num_ctas_per_sm']}",
        f"tma{int(cfg['use_tma_a'])}{int(cfg['use_tma_b'])}{int(cfg['use_tma_c'])}{int(cfg['use_tma_bs'])}",
        f"mbar{int(cfg['use_mbarrier'])}",
        f"wspec{int(cfg['use_warp_spec'])}",
        f"f16acc{int(cfg['use_f16_accum'])}",
    ]
    return "_".join(parts)


def _config_identity(cfg: Dict) -> tuple:
    """Return a hashable identity tuple for dedup by final config values."""
    return (
        cfg["block_m"], cfg["block_n"], cfg["block_k"],
        tuple(cfg["warp_shape"]),
        cfg["use_stream_k"], cfg["num_stages"], cfg["num_ctas_per_sm"],
        cfg["use_tma_a"], cfg["use_tma_b"], cfg["use_tma_c"], cfg["use_tma_bs"],
        cfg["use_mbarrier"], cfg["use_warp_spec"], cfg["use_f16_accum"],
    )


def _normalize_config(cfg: Dict, min_m: int, max_m: int) -> Dict:
    """Extract a normalized config dict from a heuristic config entry."""
    return {
        "min_shape_m": min_m,
        "max_shape_m": max_m,
        "block_m": cfg["block_shape"][0],
        "block_n": cfg["block_shape"][1],
        "block_k": cfg["block_shape"][2],
        "warp_shape": list(cfg["warp_shape"]),
        "use_stream_k": cfg["use_stream_k"],
        "num_stages": cfg["num_stages"],
        "num_ctas_per_sm": cfg.get("num_ctas_per_sm", 1),
        "use_tma_a": cfg.get("use_tma_a", False),
        "use_tma_b": cfg.get("use_tma_b", False),
        "use_tma_c": cfg.get("use_tma_c", False),
        "use_tma_bs": cfg.get("use_tma_bs", False),
        "use_mbarrier": cfg.get("use_mbarrier", False),
        "use_warp_spec": cfg.get("use_warp_spec", False),
        "use_f16_accum": cfg.get("use_f16_accum", False),
    }


def _apply_override(base: Dict, stream_k_override=None, block_n_override=None,
                    stages_override=None, block_k_override=None,
                    num_ctas_per_sm_override=None) -> Dict:
    """Apply override fields to a base config, returning a new config dict."""
    result = dict(base)
    result["warp_shape"] = list(base["warp_shape"])  # defensive copy
    if stream_k_override is not None:
        result["use_stream_k"] = stream_k_override
    if block_n_override is not None:
        result["block_n"] = block_n_override
    if stages_override is not None:
        result["num_stages"] = stages_override
    if block_k_override is not None:
        result["block_k"] = block_k_override
        # warp_shape[2] must divide block_shape[2]; set it to block_k_override
        result["warp_shape"] = [result["warp_shape"][0], result["warp_shape"][1], block_k_override]
    if num_ctas_per_sm_override is not None:
        result["num_ctas_per_sm"] = num_ctas_per_sm_override
    return result


def _validate_config(cfg: Dict, size_n: int, size_k: int, num_experts: int,
                     compute_dtype: int, gemm_type: int) -> bool:
    """Validate a config by constructing HummingKernel (monkey-patched, no CUDA)."""
    m = _import_humming()
    try:
        a_dtype, use_fused_e8m0_scale, weight_scale_type = _get_dtype_config(compute_dtype)
        gemm_type_map = {1: m["GemmType"].INDEXED, 2: m["GemmType"].GROUPED_CONTIGUOUS}

        kernel = m["HummingKernel"](
            shape_n=size_n, shape_k=size_k, num_experts=num_experts,
            b_dtype=m["dtypes"].float4e2m1, a_dtype=a_dtype,
            c_dtype=m["dtypes"].bfloat16, bs_dtype=m["dtypes"].float8e8m0,
            weight_scale_group_size=32, weight_scale_type=weight_scale_type,
            has_zero_point=False, has_bias=False,
            mma_type=m["MmaType"].WGMMA,
            use_fused_e8m0_scale=use_fused_e8m0_scale,
            gemm_type=gemm_type_map[gemm_type],
            block_shape=(cfg["block_m"], cfg["block_n"], cfg["block_k"]),
            warp_shape=tuple(cfg["warp_shape"]),
            use_stream_k=cfg["use_stream_k"],
            num_stages=cfg["num_stages"],
            num_ctas_per_sm=cfg["num_ctas_per_sm"],
            use_f16_accum=cfg["use_f16_accum"],
            use_tma=cfg.get("use_tma"),
            use_tma_a=cfg["use_tma_a"],
            use_tma_b=cfg["use_tma_b"],
            use_tma_c=cfg["use_tma_c"],
            use_tma_bs=cfg["use_tma_bs"],
            use_mbarrier=cfg["use_mbarrier"],
            use_warp_spec=cfg["use_warp_spec"],
        )
        return True
    except Exception:
        return False


def _inject_gemm2_configs(heuristic_configs: list, size_n: int, size_k: int,
                          compute_dtype: int, num_experts: int) -> list:
    """Inject optimized configs for GEMM2 small-K large-N shapes.

    Migrated from C++ QueryKernelConfigs (humming_jit_compiler.cpp:850-873).
    The Python heuristic disables stream-K for K<=1024 and always uses block_n=128,
    both suboptimal for GEMM2's (small K, large N) shape at small-medium M.

    For each injected block_m, inherit the correct block_k, num_ctas_per_sm,
    warp_shape, and tma flags from the heuristic config with that block_m.
    Fall back to querying get_config for block_m values not in heuristic configs.
    """
    if not (size_k <= 512 and size_n >= 4096):
        return heuristic_configs

    # Remove heuristic configs for small-medium M (they use stream_k=false, block_n=128)
    filtered = [c for c in heuristic_configs if c["max_shape_m"] > 3072]

    # Build a map from block_m to a heuristic config with that block_m.
    # This lets us inherit the correct block_k, num_ctas_per_sm, warp_shape, and
    # tma flags for each block_m, rather than using a single base for all.
    bm_to_config = {}
    for c in heuristic_configs:
        bm = c["block_m"]
        if bm not in bm_to_config:
            bm_to_config[bm] = c

    # Set up for querying heuristic when a block_m is not in bm_to_config
    m = _import_humming()
    a_dtype, use_fused_e8m0_scale, weight_scale_type = _get_dtype_config(compute_dtype)

    class _Args:
        pass
    args = _Args()
    args.size_n = size_n
    args.size_k = size_k
    args.num_experts = num_experts
    args.compute_dtype = compute_dtype

    meta = _build_layer_meta(args, a_dtype, use_fused_e8m0_scale, weight_scale_type)
    gemm_type = m["GemmType"].INDEXED

    def _get_base_for_bm(bm):
        """Get a base config for the given block_m, querying heuristic if needed."""
        if bm in bm_to_config:
            return bm_to_config[bm]
        cfg = m["get_heuristics_class"]().get_config(meta, shape_m=bm, gemm_type=gemm_type)
        c = _normalize_config(cfg, 0, 0)
        # Heuristic may choose a different block_m for small shape_m; force it to bm
        c["block_m"] = bm
        c["warp_shape"][0] = bm
        return c

    injected = []
    # Small M (1-384): block_m=8 and 64
    for bm in [8, 64]:
        c = _apply_override(_get_base_for_bm(bm), stream_k_override=True,
                            block_n_override=256, stages_override=3)
        c["min_shape_m"] = 1
        c["max_shape_m"] = 384
        injected.append(c)
    # Medium M (385-1536): block_m=16, 32, and 64
    for bm in [16, 32, 64]:
        c = _apply_override(_get_base_for_bm(bm), stream_k_override=True,
                            block_n_override=256, stages_override=3)
        c["min_shape_m"] = 385
        c["max_shape_m"] = 1536
        injected.append(c)
    # Large-medium M (1537-3072): block_m=48 and 64
    for bm in [48, 64]:
        c = _apply_override(_get_base_for_bm(bm), stream_k_override=True,
                            block_n_override=256, stages_override=3)
        c["min_shape_m"] = 1537
        c["max_shape_m"] = 3072
        injected.append(c)

    return filtered + injected


def _generate_variants(heuristic_configs: list, size_n: int, size_k: int,
                       compute_dtype: int, hard_cap: int = 72) -> list:
    """Generate variant configs to expand the candidate space.

    Migrated from C++ QueryKernelConfigs (humming_jit_compiler.cpp:888-974).
    For each heuristic config, generate variants by overriding dimensions
    the heuristic may have chosen suboptimally for H20.
    """
    all_configs = list(heuristic_configs)
    seen_identities = {_config_identity(c) for c in all_configs}

    for base in heuristic_configs:
        variants = []
        # 1. stream_k variant (force stream_k ON; flip false→true only)
        if not base["use_stream_k"]:
            variants.append(_apply_override(base, stream_k_override=True))
        # 2. block_n variant (try 256 instead of heuristic 128)
        variants.append(_apply_override(base, block_n_override=256))
        # 3. stages variants (try 3, 4, 5 as alternatives)
        for stages in [3, 4, 5]:
            variants.append(_apply_override(base, stages_override=stages))
        # 4. block_k variant (try 64 for FP8 with K>64)
        if compute_dtype == 1 and size_k > 64:
            variants.append(_apply_override(base, block_k_override=64))
        # 5. num_ctas_per_sm variants (try 1, 2, 3, 4)
        for ctas in [1, 2, 3, 4]:
            variants.append(_apply_override(base, num_ctas_per_sm_override=ctas))
        # 6a. stages=4 + block_n=256
        variants.append(_apply_override(base, stages_override=4, block_n_override=256))
        # 6b. stages=5 + num_ctas_per_sm=2
        variants.append(_apply_override(base, stages_override=5, num_ctas_per_sm_override=2))

        # Variants must NOT inherit the base heuristic's M-range.
        # The heuristic range reflects the base config's optimality window, which
        # does not apply once dimensions (stream_k/block_n/stages/ctas) are overridden.
        # Variants get full range so ProfileSingleGemm evaluates them at every bucket
        # and profiling determines the actual optimal selection.
        for v in variants:
            v["min_shape_m"] = 1
            v["max_shape_m"] = 2**63 - 1  # INT64_MAX

        # Dedup variants against existing configs (by final config values)
        for v in variants:
            if len(all_configs) >= hard_cap:
                break
            identity = _config_identity(v)
            if identity not in seen_identities:
                seen_identities.add(identity)
                all_configs.append(v)
        if len(all_configs) >= hard_cap:
            break

    # Return only the variants (not the heuristic configs that are already in the input)
    return all_configs[len(heuristic_configs):]


def _cmd_query_configs(args):
    try:
        m = _import_humming()
        _monkey_patch_kernel_runtime()
        a_dtype, use_fused_e8m0_scale, _weight_scale_type_unused = _get_dtype_config(args.compute_dtype)
        weight_scale_type = m["WeightScaleType"].GROUP  # Always GROUP for heuristic query

        meta = _build_layer_meta(args, a_dtype, use_fused_e8m0_scale, weight_scale_type)
        configs = m["get_heuristics_class"]().get_configs(meta, gemm_type=m["GemmType"].INDEXED)

        # 1. Normalize heuristic configs
        heuristic_configs = []
        for entry in configs:
            min_m, max_m, cfg = entry[0], entry[1], entry[2]
            heuristic_configs.append(_normalize_config(cfg, min_m, max_m))

        # Deduplicate heuristic configs by final values
        seen = set()
        deduped_heuristic = []
        for c in heuristic_configs:
            identity = _config_identity(c)
            if identity not in seen:
                seen.add(identity)
                deduped_heuristic.append(c)
        heuristic_configs = deduped_heuristic

        # 2. GEMM2 injection (migrated from C++)
        heuristic_configs = _inject_gemm2_configs(
            heuristic_configs, args.size_n, args.size_k, args.compute_dtype, args.num_experts)

        # 3. Variant generation (migrated from C++)
        extend_configs = _generate_variants(
            heuristic_configs, args.size_n, args.size_k, args.compute_dtype)

        # 4. Validate extend configs (construct HummingKernel under monkey-patch)
        valid_extend = []
        for cfg in extend_configs:
            if _validate_config(cfg, args.size_n, args.size_k, args.num_experts,
                                args.compute_dtype, 1):  # gemm_type=1 (INDEXED)
                valid_extend.append(cfg)
        extend_configs = valid_extend

        # 5. Generate cache_key for each config
        for cfg in heuristic_configs:
            cfg["cache_key"] = _make_cache_key(
                cfg, args.size_n, args.size_k, args.num_experts, args.compute_dtype, 1)
            cfg["is_heuristic"] = True
        for cfg in extend_configs:
            cfg["cache_key"] = _make_cache_key(
                cfg, args.size_n, args.size_k, args.num_experts, args.compute_dtype, 1)
            cfg["is_heuristic"] = False

        print(json.dumps({"heuristic_configs": heuristic_configs,
                          "extend_configs": extend_configs}))
    except Exception as e:
        # Fallback: return default configs
        sys.stderr.write(f"Warning: humming heuristic query failed: {e}\n")
        sys.stderr.write("Falling back to default configs\n")
        fallback_cfgs = []
        for bm, min_m, max_m in [(64, 1, 64), (128, 65, 1073741824)]:
            cfg = {
                "cache_key": "",  # filled below
                "is_heuristic": True,
                "min_shape_m": min_m, "max_shape_m": max_m,
                "block_m": bm, "block_n": 128, "block_k": 64,
                "warp_shape": [16, 128, 64],
                "use_stream_k": True, "num_stages": 3, "num_ctas_per_sm": 1,
                "use_tma_a": True, "use_tma_b": True, "use_tma_c": True, "use_tma_bs": True,
                "use_mbarrier": True, "use_warp_spec": False, "use_f16_accum": False,
            }
            cfg["cache_key"] = _make_cache_key(
                cfg, args.size_n, args.size_k, args.num_experts, args.compute_dtype, 1)
            fallback_cfgs.append(cfg)
        print(json.dumps({"heuristic_configs": fallback_cfgs, "extend_configs": []}))


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
