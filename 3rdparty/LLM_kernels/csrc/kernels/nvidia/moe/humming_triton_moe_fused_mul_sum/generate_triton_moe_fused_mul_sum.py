#!/usr/bin/env python3
"""
Generate Triton cubin/json/ptx for sglang moe_fused_mul_sum kernel.

This script compiles the sglang-enhanced Triton moe_fused_mul_sum kernel
(supporting is_ep and routed_scaling_factor) with a specific set of
compile-time constants and dumps the resulting .cubin, .json, and .ptx artifacts.

Usage:
    python generate_triton_moe_fused_mul_sum.py \
        --top_k 6 --size 7168 --is_ep 0 \
        --block_m 2 --block_k 512 \
        --output_dir /path/to/triton_kernels

Follows the same pattern as generate_triton_fp8_quant.py: wrap the Triton kernel
call in a function, then use performance_kernel() to obtain the compiled kernel
handle, whose .asm/.metadata carry the artifacts for dump_kernel().

The kernel source is inlined from .analysis_repos/sglang because the full sglang
package has transitive dependencies (pybase64, etc.) not available at build time.
"""

import argparse
import os
import sys

import torch
import triton
import triton.language as tl

# Allow import from parent moe/ directory (for common.py).
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _parent_dir)

from common import dump_kernel, performance_kernel  # noqa: E402


# ---- Inlined sglang moe_fused_mul_sum_kernel ----
# Source: .analysis_repos/sglang/python/sglang/srt/layers/moe/fused_moe_triton/moe_fused_mul_sum.py
# Inlined because the sglang package __init__ has heavy transitive dependencies.
# Two additions over upstream humming:
#   - is_ep: tl.constexpr — when True, skips rows where topk_ids[n] < 0
#   - routed_scaling_factor: tl.constexpr — multiplies topk_weights in-kernel

@triton.jit
def moe_fused_mul_sum_kernel(
    inputs_ptr,
    topk_weights_ptr,
    outputs_ptr,
    top_ids_ptr,
    expert_map_ptr,
    num_tokens,
    stride_m,
    has_expert_map: tl.constexpr,
    is_ep: tl.constexpr,
    top_k: tl.constexpr,
    size: tl.constexpr,
    routed_scaling_factor: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_k = tl.program_id(0)
    pid_m = tl.program_id(1).to(tl.int64)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    m_mask = offs_m < num_tokens
    k_mask = offs_k < size
    mask = m_mask[:, None] & k_mask[None, :]

    a_base = inputs_ptr + (offs_m * stride_m)[:, None] + offs_k[None, :]
    b_base = topk_weights_ptr + offs_m * top_k

    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

    for n in tl.static_range(top_k):
        b_val = tl.load(b_base + n, mask=m_mask, other=0.0).to(tl.float32)
        if routed_scaling_factor != 1.0:
            b_val = b_val * routed_scaling_factor
        if has_expert_map:
            id_val = tl.load(top_ids_ptr + offs_m * top_k + n, mask=m_mask, other=0)
            expert_mask = tl.load(expert_map_ptr + id_val) >= 0
            a_vec = tl.load(
                a_base + n * size,
                mask=mask & expert_mask[:, None],
                other=0.0,
            ).to(tl.float32)
        elif is_ep:
            id_val = tl.load(top_ids_ptr + offs_m * top_k + n, mask=m_mask, other=0)
            expert_mask = id_val >= 0
            a_vec = tl.load(
                a_base + n * size,
                mask=mask & expert_mask[:, None],
                other=0.0,
            ).to(tl.float32)
        else:
            a_vec = tl.load(
                a_base + n * size,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
        acc += a_vec * b_val[:, None]

    out_ptrs = outputs_ptr + (offs_m * size)[:, None] + offs_k[None, :]
    tl.store(
        out_ptrs,
        acc.to(outputs_ptr.dtype.element_ty),
        mask=mask,
    )


def _heuristic_config_sm90(size: int, block_m: int, block_k: int):
    """Compute num_warps and num_stages for SM90 BF16.

    Simplified from sglang _heuristic_config, specific to SM90 (sm_major >= 9)
    with BF16 inputs (element_size <= 2).
    """
    total = block_m * block_k
    num_warps = min(max(4, total // 256), 8)
    num_stages = 4 if total <= 2048 else 2
    return num_warps, num_stages


def compute_block_k(size: int) -> int:
    """SM90 BF16 BLOCK_K rule: min(next_power_of_2(size), 512)."""
    return min(triton.next_power_of_2(size), 512)


def generate_moe_fused_mul_sum_kernel(
    inputs: torch.Tensor,
    topk_weights: torch.Tensor,
    outputs: torch.Tensor,
    topk_ids: torch.Tensor,
    num_tokens: int,
    top_k: int,
    size: int,
    block_m: int,
    block_k: int,
    is_ep: bool = False,
) -> tuple:
    """Wrapper that calls moe_fused_mul_sum_kernel directly.

    Returns (output, compiled_kernel_handle) for performance_kernel().
    """
    num_warps, num_stages = _heuristic_config_sm90(size, block_m, block_k)
    stride_m = top_k * size

    grid = (triton.cdiv(size, block_k), triton.cdiv(num_tokens, block_m))

    # KsanaLLM fixed values:
    #   has_expert_map = False  (no expert_map tensor)
    #   routed_scaling_factor = 1.0  (applied in TopK layer)
    kernel = moe_fused_mul_sum_kernel[grid](
        inputs,
        topk_weights,
        outputs,
        topk_ids,
        None,       # expert_map_ptr
        num_tokens,
        stride_m,
        False,      # has_expert_map
        is_ep,
        top_k,
        size,
        1.0,        # routed_scaling_factor
        block_m,
        block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return outputs, kernel


def generate_kernel_variant(
    top_k: int,
    size: int,
    is_ep: int,
    block_m: int,
    block_k: int,
    output_dir: str,
) -> None:
    """Compile one kernel variant and dump .cubin/.json/.ptx into output_dir.

    Shared by the CLI entry point (main) and the batch creator script so the
    Triton/torch runtime is imported only once per process.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Dummy input: num_tokens chosen per BLOCK_M so at least one M block runs.
    if block_m == 1:
        num_tokens = 2
    elif block_m == 2:
        num_tokens = 64
    else:
        num_tokens = 4096  # covers BLOCK_M=4 and larger

    dtype_input = torch.bfloat16
    dtype_weight = torch.float32
    dtype_ids = torch.int32

    inputs = torch.randn(num_tokens, top_k, size,
                         dtype=dtype_input, device="cuda")
    topk_weights = torch.randn(num_tokens, top_k,
                               dtype=dtype_weight, device="cuda")
    topk_ids = torch.randint(0, 64, (num_tokens, top_k),
                             dtype=dtype_ids, device="cuda")
    outputs = torch.empty(num_tokens, size,
                          dtype=dtype_input, device="cuda")

    is_ep_bool = bool(is_ep)

    def fn():
        out, k = generate_moe_fused_mul_sum_kernel(
            inputs, topk_weights, outputs, topk_ids,
            num_tokens, top_k, size,
            block_m, block_k, is_ep_bool,
        )
        return (out, k)

    # performance_kernel triggers JIT compile via graph capture.
    compiled_kernel, kernel_time_us, _ = performance_kernel(fn)
    # Call once more for output sanity check.
    out_tensor, _ = generate_moe_fused_mul_sum_kernel(
        inputs, topk_weights, outputs, topk_ids,
        num_tokens, top_k, size,
        block_m, block_k, is_ep_bool,
    )

    # Sanity checks.
    assert out_tensor.shape == (num_tokens, size), \
        f"Output shape {out_tensor.shape} != ({num_tokens}, {size})"
    assert out_tensor.dtype == dtype_input, \
        f"Output dtype {out_tensor.dtype} != {dtype_input}"

    kernel_name = (
        f"moe_fused_mul_sum_kernel"
        f"_is_ep_{is_ep}"
        f"_top_k_{top_k}"
        f"_size_{size}"
        f"_BLOCK_M_{block_m}"
        f"_BLOCK_K_{block_k}"
    )

    num_warps, num_stages = _heuristic_config_sm90(size, block_m, block_k)

    default_config = {
        "config": {
            "num_warps": num_warps,
            "num_stages": num_stages,
        }
    }

    dump_kernel(compiled_kernel, output_dir, kernel_name, default_config)
    print(
        f"Generated {kernel_name}.{{cubin,json,ptx}} in {output_dir} "
        f"(kernel_time={kernel_time_us:.1f} us, "
        f"num_warps={num_warps}, num_stages={num_stages})"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate sglang moe_fused_mul_sum Triton kernel artifacts")
    parser.add_argument("--top_k", type=int, required=True,
                        help="Number of experts per token (e.g. 6)")
    parser.add_argument("--size", type=int, required=True,
                        help="Hidden size (e.g. 4096, 7168)")
    parser.add_argument("--is_ep", type=int, required=True, choices=[0, 1],
                        help="Expert-parallel mode (0=off, 1=on)")
    parser.add_argument("--block_m", type=int, required=True,
                        help="BLOCK_M tile size (1, 2, or 4)")
    parser.add_argument("--block_k", type=int, required=True,
                        help="BLOCK_K tile size (typically 512)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for .cubin, .json, .ptx files")
    args = parser.parse_args()

    generate_kernel_variant(
        top_k=args.top_k,
        size=args.size,
        is_ep=args.is_ep,
        block_m=args.block_m,
        block_k=args.block_k,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
