#!/usr/bin/env python3
# Copyright 2026 Tencent Inc.  All rights reserved.
"""Generate Triton cubin/json/ptx for attn_res kernel.

Usage:
    python generate_attn_res.py --h 7168 --max-blocks 8 --output_dir <dir>

Only (H, NB, HD, WB, ON, BL) are baked into the kernel file name — the C++
wrapper reconstructs the same name from those six keys. Everything else
that appears as `tl.constexpr` in the kernel signature is either derived
from (H, max_blocks) or fixed at generation time:

  * stride_prefix_m = stride_output_m = H
  * stride_delta_m  = H when HAS_DELTA else 0
  * stride_block_m  = max_blocks * H       (blocks bank has a fixed second dim)
  * stride_block_r  = H
  * hidden_size     = H
  * block_write_idx = NB when WB else -1
  * eps / output_norm_eps = FIXED_EPS (1e-6)
  * BL              = 1 or 4  (BL=1 -> num_warps=4, BL=4 -> num_warps=8)
"""

import argparse
import os
import sys

import torch
import triton

_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
sys.path.insert(0, _parent_dir)

from moe.common import dump_kernel, performance_kernel  # noqa: E402

from attn_res_kernel import _attn_res_kernel  # noqa: E402

FIXED_EPS = 1e-6
FIXED_NUM_STAGES = 2

# BL -> num_warps table; wrapper must derive num_warps from BL identically.
_BL_TO_NUM_WARPS = {1: 4, 4: 8}
_BL_VARIANTS = tuple(_BL_TO_NUM_WARPS.keys())


def _launch(
    h: int,
    max_blocks_bank: int,
    num_tokens: int,
    num_blocks: int,
    has_delta: bool,
    write_block: bool,
    apply_output_norm: bool,
    block_l: int,
):
    device = "cuda"
    dtype = torch.bfloat16

    prefix = torch.randn(num_tokens, h, device=device, dtype=dtype)
    delta = torch.randn_like(prefix) if has_delta else None
    blocks = torch.randn(num_tokens, max_blocks_bank, h, device=device, dtype=dtype)
    norm_weight = 1 + 0.1 * torch.randn(h, device=device, dtype=dtype)
    qk_weight = torch.randn(h, device=device, dtype=dtype) / (h ** 0.5)
    output_norm_weight = (
        1 + 0.1 * torch.randn(h, device=device, dtype=dtype)
        if apply_output_norm
        else None
    )

    output = prefix.new_empty(prefix.shape)
    block_write_idx = num_blocks if write_block else -1
    num_warps = _BL_TO_NUM_WARPS[block_l]

    kernel = _attn_res_kernel[(num_tokens,)](
        prefix,
        delta,
        blocks,
        norm_weight,
        qk_weight,
        output_norm_weight,
        output,
        h,                                 # stride_prefix_m
        h if has_delta else 0,             # stride_delta_m
        max_blocks_bank * h,               # stride_block_m
        h,                                 # stride_block_r
        h,                                 # stride_output_m
        num_blocks,
        h,                                 # hidden_size
        block_write_idx,
        FIXED_EPS,
        FIXED_EPS if apply_output_norm else 0.0,
        HAS_DELTA=has_delta,
        WRITE_BLOCK=write_block,
        APPLY_OUTPUT_NORM=apply_output_norm,
        BLOCK_L=block_l,
        BLOCK_D=triton.next_power_of_2(h),
        num_warps=num_warps,
        num_stages=FIXED_NUM_STAGES,
    )
    return output, kernel


def _kernel_name(
    h: int,
    num_blocks: int,
    has_delta: bool,
    write_block: bool,
    apply_output_norm: bool,
    block_l: int,
) -> str:
    return (
        f"_attn_res_kernel"
        f"_H_{h}"
        f"_NB_{num_blocks}"
        f"_HD_{int(has_delta)}"
        f"_WB_{int(write_block)}"
        f"_ON_{int(apply_output_norm)}"
        f"_BL_{block_l}"
    )


def _default_config(block_l: int) -> dict:
    return {
        "config": {
            "num_warps": _BL_TO_NUM_WARPS[block_l],
            "num_stages": FIXED_NUM_STAGES,
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Generate attn_res Triton kernel artifacts")
    parser.add_argument("--h", type=int, required=True, help="Hidden size (e.g. 7168)")
    parser.add_argument(
        "--max-blocks",
        type=int,
        required=True,
        help="Maximum bank rows — fixes the second dim of `blocks` at generation time. "
             "Also caps the num_blocks constexpr range.",
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=1024,
        help="Token count used only for JIT trigger; runtime grid is dynamic",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    combos = []
    for num_blocks in range(0, args.max_blocks + 1):
        for has_delta in (False, True):
            for apply_output_norm in (False, True):
                for write_block in (False, True):
                    if write_block and num_blocks >= args.max_blocks:
                        continue
                    for block_l in _BL_VARIANTS:
                        # BL=4 collapses to BL=1 when num_sources <= 1
                        # (num_blocks + 1 sources); skip the redundant cubin.
                        if block_l > 1 and num_blocks + 1 <= 1:
                            continue
                        combos.append(
                            (num_blocks, has_delta, write_block, apply_output_norm, block_l)
                        )

    for num_blocks, has_delta, write_block, apply_output_norm, block_l in combos:
        def fn(nb=num_blocks, hd=has_delta, wb=write_block, on=apply_output_norm, bl=block_l):
            out, k = _launch(
                args.h,
                args.max_blocks,
                args.num_tokens,
                nb,
                hd,
                wb,
                on,
                bl,
            )
            return (out, k)

        compiled, us, _ = performance_kernel(fn)
        name = _kernel_name(args.h, num_blocks, has_delta, write_block, apply_output_norm, block_l)
        dump_kernel(compiled, args.output_dir, name, _default_config(block_l))
        print(
            f"Generated {name}.{{cubin,json,ptx}} in {args.output_dir}"
            f" (kernel_time={us:.1f} us)"
        )


if __name__ == "__main__":
    main()
