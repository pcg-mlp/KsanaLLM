#!/usr/bin/env python3
"""
Generate Triton cubin/json/ptx for fused SiluAndMul + FP8 quantization kernel.

This kernel fuses two operations into one:
  1. SiluAndMul: y = silu(gate) * up  (input: [M, inter*2] BF16, gate+up concatenated)
  2. FP8 Quant:  y_q = quantize(y)    (output: [M, inter] FP8 + [M] FP32 scale)

Saves one kernel launch and one HBM round-trip compared to separate kernels.

Usage:
    python generate_triton_silu_mul_quant.py --n 512 --output_dir /path/to/triton_kernels
"""

import argparse
import os
import sys

import torch
import triton
import triton.language as tl

# Allow import from parent moe/ directory (for common.py)
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _parent_dir)

from common import dump_kernel, performance_kernel  # noqa: E402


@triton.jit
def _silu_and_mul_quant_kernel(
    x_ptr,
    xq_ptr,
    scale_ptr,
    stride_x,
    num_blocks,
    is_dynamic: tl.constexpr,
    N: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
    GROUPS_PER_BLOCK: tl.constexpr,
    dtype: tl.constexpr,
    SWIGLU_LIMIT: tl.constexpr,
):
    block_id = tl.program_id(0).to(tl.int64)

    row_num_blocks = N // GROUP_SIZE

    for g in tl.static_range(GROUPS_PER_BLOCK):
        group_id = block_id * GROUPS_PER_BLOCK + g
        in_range = group_id < num_blocks
        row_id = group_id // row_num_blocks
        col_block_id = group_id % row_num_blocks

        # Input offset: each row has stride_x = inter*2 elements (gate+up)
        offset = row_id * stride_x + col_block_id * GROUP_SIZE

        cols = tl.arange(0, BLOCK)
        mask = (cols < GROUP_SIZE) & in_range

        # Load gate and up halves
        a = tl.load(x_ptr + offset + cols, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(x_ptr + offset + GROUP_SIZE + cols, mask=mask, other=0.0).to(tl.float32)

        # Apply swiglu limit
        a = tl.minimum(a, SWIGLU_LIMIT)
        b = tl.minimum(tl.maximum(b, -SWIGLU_LIMIT), SWIGLU_LIMIT)

        # SiLU(a) * b
        y = a / (1.0 + tl.exp(-a)) * b

        # Per-token dynamic quantization to FP8 E4M3
        absmax = tl.maximum(tl.max(tl.abs(y)), 1e-30)
        scale = absmax / 448.0
        inv_scale = 1.0 / scale
        y_q = (y * inv_scale).to(tl.float8e4nv)

        tl.store(xq_ptr + group_id * GROUP_SIZE + cols, y_q, mask=mask)

        if is_dynamic:
            tl.store(scale_ptr + row_id * row_num_blocks + col_block_id, scale, mask=in_range)


@triton.jit
def _silu_and_mul_quant_ep_kernel(
    topk_ids_ptr,
    x_ptr,
    xq_ptr,
    scale_ptr,
    stride_x,
    num_blocks,
    is_dynamic: tl.constexpr,
    N: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
    GROUPS_PER_BLOCK: tl.constexpr,
    dtype: tl.constexpr,
    SWIGLU_LIMIT: tl.constexpr,
):
    block_id = tl.program_id(0).to(tl.int64)

    if tl.load(topk_ids_ptr + block_id) < 0:
        return

    row_num_blocks = N // GROUP_SIZE

    for g in tl.static_range(GROUPS_PER_BLOCK):
        group_id = block_id * GROUPS_PER_BLOCK + g
        in_range = group_id < num_blocks
        row_id = group_id // row_num_blocks
        col_block_id = group_id % row_num_blocks

        # Input offset: each row has stride_x = inter*2 elements (gate+up)
        offset = row_id * stride_x + col_block_id * GROUP_SIZE

        cols = tl.arange(0, BLOCK)
        mask = (cols < GROUP_SIZE) & in_range

        # Load gate and up halves
        a = tl.load(x_ptr + offset + cols, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(x_ptr + offset + GROUP_SIZE + cols, mask=mask, other=0.0).to(tl.float32)

        # Apply swiglu limit
        a = tl.minimum(a, SWIGLU_LIMIT)
        b = tl.minimum(tl.maximum(b, -SWIGLU_LIMIT), SWIGLU_LIMIT)

        # SiLU(a) * b
        y = a / (1.0 + tl.exp(-a)) * b

        # Per-token dynamic quantization to FP8 E4M3
        absmax = tl.maximum(tl.max(tl.abs(y)), 1e-30)
        scale = absmax / 448.0
        inv_scale = 1.0 / scale
        y_q = (y * inv_scale).to(tl.float8e4nv)

        tl.store(xq_ptr + group_id * GROUP_SIZE + cols, y_q, mask=mask)

        if is_dynamic:
            tl.store(scale_ptr + row_id * row_num_blocks + col_block_id, scale, mask=in_range)


def generate_silu_mul_quant_kernel(
    x: torch.Tensor,
    inter_size: int,
    swiglu_limit: float = 10.0,
) -> tuple:
    """Wrapper that calls _silu_and_mul_quant_kernel and returns (output, scales, kernel)."""
    assert x.is_cuda
    assert x.size(-1) == inter_size * 2, f"Input last dim {x.size(-1)} != inter_size*2 {inter_size*2}"

    inputs = x.view(-1, x.size(-1))
    num_tokens = inputs.size(0)
    group_size = inter_size

    output_dtype = torch.float8_e4m3fn
    outputs = torch.empty((num_tokens, inter_size), dtype=output_dtype, device=x.device)
    scales = torch.empty((num_tokens,), dtype=torch.float32, device=x.device)

    BLOCK = triton.next_power_of_2(group_size)
    groups_per_block = 1
    num_blocks = num_tokens
    grid_blocks = (num_blocks + groups_per_block - 1) // groups_per_block

    effective_block = BLOCK
    num_warps = min(max(effective_block // 256, 1), 8)
    num_stages = 1

    kernel = _silu_and_mul_quant_kernel[(grid_blocks,)](
        inputs,
        outputs,
        scales,
        inputs.stride(0),
        num_blocks,
        True,
        inter_size,
        group_size,
        BLOCK,
        groups_per_block,
        "float8e4m3",
        swiglu_limit,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return outputs, scales, kernel


def generate_silu_mul_quant_ep_kernel(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    inter_size: int,
    swiglu_limit: float = 10.0,
) -> tuple:
    """Wrapper that calls _silu_and_mul_quant_ep_kernel and returns (output, scales, kernel)."""
    assert x.is_cuda
    assert x.size(-1) == inter_size * 2, f"Input last dim {x.size(-1)} != inter_size*2 {inter_size*2}"

    inputs = x.view(-1, x.size(-1))
    num_tokens = inputs.size(0)
    group_size = inter_size

    output_dtype = torch.float8_e4m3fn
    outputs = torch.empty((num_tokens, inter_size), dtype=output_dtype, device=x.device)
    scales = torch.empty((num_tokens,), dtype=torch.float32, device=x.device)

    BLOCK = triton.next_power_of_2(group_size)
    groups_per_block = 1
    num_blocks = num_tokens
    grid_blocks = (num_blocks + groups_per_block - 1) // groups_per_block

    effective_block = BLOCK
    num_warps = min(max(effective_block // 256, 1), 8)
    num_stages = 1

    kernel = _silu_and_mul_quant_ep_kernel[(grid_blocks,)](
        topk_ids,
        inputs,
        outputs,
        scales,
        inputs.stride(0),
        num_blocks,
        True,
        inter_size,
        group_size,
        BLOCK,
        groups_per_block,
        "float8e4m3",
        swiglu_limit,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return outputs, scales, kernel


def generate_kernel_variant(n: int, output_dir: str) -> None:
    """Compile fused SiluAndMul+FP8 quant kernels (base + EP) for one N and dump artifacts.

    Shared by the CLI entry point (main) and the batch creator script so the
    Triton/torch runtime is imported only once per process.
    """
    assert n % 128 == 0, f"N must be divisible by 128, got {n}"
    os.makedirs(output_dir, exist_ok=True)

    max_tokens = 4096
    x = torch.randn(max_tokens, n * 2, dtype=torch.bfloat16, device="cuda")

    def fn():
        out, scale, k = generate_silu_mul_quant_kernel(x, n, swiglu_limit=10.0)
        return (out, k)

    compiled_kernel, kernel_time_us, _ = performance_kernel(fn)

    out, scale, _ = generate_silu_mul_quant_kernel(x, n, swiglu_limit=10.0)

    assert out.shape == (max_tokens, n), f"Output shape {out.shape} != ({max_tokens}, {n})"
    assert out.dtype == torch.float8_e4m3fn
    assert scale.numel() == max_tokens

    kernel_name = f"_silu_and_mul_quant_kernel_N_{n}"

    BLOCK = triton.next_power_of_2(n)
    num_warps = min(max(BLOCK // 256, 1), 8)
    num_stages = 1

    default_config = {
        "config": {
            "num_warps": num_warps,
            "num_stages": num_stages,
        }
    }

    dump_kernel(compiled_kernel, output_dir, kernel_name, default_config)
    print(f"Generated {kernel_name}.{{cubin,json,ptx}} in {output_dir} (kernel_time={kernel_time_us:.1f} us)")

    topk_ids = torch.zeros(max_tokens, dtype=torch.int32, device="cuda")

    def fn_ep():
        out, scale, k = generate_silu_mul_quant_ep_kernel(x, topk_ids, n, swiglu_limit=10.0)
        return (out, k)

    compiled_kernel_ep, kernel_time_ep, _ = performance_kernel(fn_ep)

    out_ep, scale_ep, _ = generate_silu_mul_quant_ep_kernel(x, topk_ids, n, swiglu_limit=10.0)

    assert out_ep.shape == (max_tokens, n), f"EP output shape {out_ep.shape} != ({max_tokens}, {n})"
    assert out_ep.dtype == torch.float8_e4m3fn
    assert scale_ep.numel() == max_tokens

    kernel_name_ep = f"_silu_and_mul_quant_ep_kernel_N_{n}"

    dump_kernel(compiled_kernel_ep, output_dir, kernel_name_ep, default_config)
    print(f"Generated {kernel_name_ep}.{{cubin,json,ptx}} in {output_dir} (kernel_time={kernel_time_ep:.1f} us)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate fused SiluAndMul+FP8 quant Triton kernel artifacts")
    parser.add_argument("--n", type=int, required=True,
                        help="Inter size (output dimension after SiluAndMul)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for .cubin, .json, .ptx files")
    args = parser.parse_args()

    generate_kernel_variant(n=args.n, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
