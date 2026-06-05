#!/usr/bin/env python3
"""
Generate Triton cubin/json/ptx for humming's FP8 activation quantization kernel.

Usage:
    python generate_triton_fp8_quant.py --n 7168 --output_dir /path/to/triton_kernels

Follows the same pattern as fused_moe_kernel_gptq_awq.py: wrap the Triton kernel
call in a function that returns (output, kernel), then use performance_kernel()
to obtain the compiled kernel handle, whose .asm/.metadata carry the artifacts
for dump_kernel().
"""

import argparse
import os
import sys

import torch
import triton

# Allow import from parent moe/ directory (for common.py) and the humming pip package.
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _parent_dir)

from common import dump_kernel, performance_kernel  # noqa: E402

# humming must be installed (done by CMake in preprocess/CMakeLists.txt).
from humming.ops.input import _quant_tensor_kernel  # noqa: E402


def generate_quant_kernel(
    x: torch.Tensor,
    dtype: str = "float8e4m3",
    group_size: int = 0,
) -> tuple:
    """Wrapper that calls _quant_tensor_kernel directly and returns (output, scales, kernel).

    Mirrors humming.ops.input.quant_input() but also captures the compiled kernel handle
    returned by the JITFunction[grid](...) call, which carries .asm / .metadata.
    """
    if group_size is None or group_size == 0:
        group_size = x.size(1)

    assert x.is_cuda
    output_dtype = torch.float8_e4m3fn
    outputs = torch.empty(x.shape, dtype=output_dtype, device=x.device)

    inputs = x.view(-1, x.size(-1))
    num_blocks = inputs.nelement() // group_size

    scales = torch.empty(
        (inputs.size(0), inputs.size(1) // group_size),
        dtype=torch.float32,
        device=x.device,
    )

    BLOCK = triton.next_power_of_2(group_size)
    groups_per_block = 1
    if group_size <= 256 and num_blocks >= 131072:
        groups_per_block = min(1024 // group_size, num_blocks)
    grid_blocks = (num_blocks + groups_per_block - 1) // groups_per_block

    effective_block = BLOCK  # float8e4m3 uses full BLOCK (not halved)
    num_warps = min(max(effective_block // 256, 1), 8)
    num_stages = 1

    kernel = _quant_tensor_kernel[(grid_blocks,)](
        inputs,
        outputs,
        scales,
        inputs.stride(0),
        num_blocks,
        True,                     # is_dynamic
        inputs.size(1),           # N
        group_size,               # GROUP_SIZE
        BLOCK,
        groups_per_block,
        dtype,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    scales = scales.view(*outputs.shape[:-1], scales.size(-1))
    return outputs, scales, kernel


def main():
    parser = argparse.ArgumentParser(
        description="Generate humming FP8 quant Triton kernel artifacts")
    parser.add_argument("--n", type=int, required=True,
                        help="Hidden size (N dimension, also used as GROUP_SIZE for per-token quant)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for .cubin, .json, .ptx files")
    args = parser.parse_args()

    assert args.n % 128 == 0, f"N must be divisible by 128, got {args.n}"
    os.makedirs(args.output_dir, exist_ok=True)

    # Create dummy input with enough tokens to force Triton to bake a large
    # num_blocks constant.  The kernel hardcodes `in_range = block_id < num_blocks`
    # at JIT time, so if we compile with num_blocks=1, only block 0 is valid and
    # multi-token launches produce all-zero rows.  4096 tokens covers realistic
    # batch sizes (~118 MB for N=7168 BF16 i/o).
    max_tokens = 4096
    x = torch.randn(max_tokens, args.n, dtype=torch.bfloat16, device="cuda")

    def fn():
        out, scale, k = generate_quant_kernel(x, "float8e4m3", group_size=args.n)
        return (out, k)  # performance_kernel expects (output, kernel)

    # performance_kernel triggers the JIT compile and returns the handle.
    compiled_kernel, kernel_time_us, _ = performance_kernel(fn)
    out, scale, _ = generate_quant_kernel(x, "float8e4m3", group_size=args.n)

    # Sanity checks.
    assert out.shape == x.shape, f"Output shape {out.shape} != input shape {x.shape}"
    assert out.dtype == torch.float8_e4m3fn, f"Output dtype {out.dtype} != float8_e4m3fn"
    assert scale.numel() == max_tokens, f"Expected {max_tokens} scales, got {scale.numel()}"

    kernel_name = f"_quant_tensor_kernel_N_{args.n}"

    # num_warps / num_stages mirror quant_input's auto-selection (capped at 8).
    BLOCK = triton.next_power_of_2(args.n)
    num_warps = min(max(BLOCK // 256, 1), 8)
    num_stages = 1

    default_config = {
        "config": {
            "num_warps": num_warps,
            "num_stages": num_stages,
        }
    }

    dump_kernel(compiled_kernel, args.output_dir, kernel_name, default_config)
    print(f"Generated {kernel_name}.{{cubin,json,ptx}} in {args.output_dir} (kernel_time={kernel_time_us:.1f} us)")


if __name__ == "__main__":
    main()