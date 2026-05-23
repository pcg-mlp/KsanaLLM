#!/usr/bin/env python3
"""Query humming H20 heuristic for recommended kernel configurations.

Outputs JSON array of {shape_m_representative, min_shape_m, max_shape_m, block_m, use_stream_k}
for a given (size_n, size_k, num_experts) problem.

The `shape_m_representative` is the value to pass as `--block_m` to generate_kernel.py
(despite the name, generate_kernel.py uses it as shape_m for heuristic lookup).

Usage:
    python3 get_h20_configs.py --size_n 4096 --size_k 4096 --num_experts 256
"""
import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size_n", type=int, required=True)
    parser.add_argument("--size_k", type=int, required=True)
    parser.add_argument("--num_experts", type=int, required=True)
    parser.add_argument("--compute_dtype", type=int, default=0,
                        help="0=BF16, 1=FP8")
    args = parser.parse_args()

    try:
        from humming.tune.sm90_h20 import Sm90H20Heuristics
        from humming.layer import HummingLayerMeta
        from humming.config.enum import GemmType, WeightScaleType, MmaType
        from humming import dtypes

        h = Sm90H20Heuristics()

        if args.compute_dtype == 1:
            a_dtype = dtypes.float8e4m3
            use_fused_e8m0_scale = True
        else:
            a_dtype = dtypes.bfloat16
            use_fused_e8m0_scale = False

        meta = HummingLayerMeta(
            shape_n=args.size_n,
            shape_k=args.size_k,
            num_experts=args.num_experts,
            b_dtype=dtypes.float4e2m1,
            a_dtype=a_dtype,
            c_dtype=dtypes.bfloat16,
            bs_dtype=dtypes.float8e8m0,
            weight_scale_group_size=32,
            weight_scale_type=WeightScaleType.GROUP,
            has_zero_point=False,
            has_bias=False,
            mma_type=MmaType.WGMMA,
            use_fused_e8m0_scale=use_fused_e8m0_scale,
        )

        configs = h.get_configs(meta, gemm_type=GemmType.INDEXED)

        results = []
        for entry in configs:
            min_m, max_m, cfg = entry[0], entry[1], entry[2]
            block_m = cfg["block_shape"][0]
            # Pick a representative shape_m for kernel generation:
            # Use midpoint of range, clamped to [min_m, max_m]
            if max_m > 1000000:
                # For the unbounded last range, pick max_m of previous range + some offset
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

        # Deduplicate: if consecutive entries have the same representative,
        # merge their ranges (since they'd produce the same kernel)
        deduped = []
        seen_reps = {}
        for r in results:
            rep = r["shape_m_representative"]
            if rep in seen_reps:
                # Extend existing entry's range
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


if __name__ == "__main__":
    main()
