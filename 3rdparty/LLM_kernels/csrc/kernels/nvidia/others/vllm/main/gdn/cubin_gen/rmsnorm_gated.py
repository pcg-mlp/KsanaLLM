# Copyright 2026 Tencent Inc. All rights reserved.
#
# Wraps `gdn.ops.layernorm_guard.rmsnorm_gated_kernel` (vendored from vllm
# v0.21.0); used by Qwen3.5/3.6 GatedDeltaNet RMSNormGated.
# ============================================================================
"""Offline cubin generator for `rmsnorm_gated_kernel`."""

from __future__ import annotations

import argparse
import os
import sys

# Make `gdn` importable when this file is copied into CMAKE_BINARY_DIR.
_HERE = os.path.dirname(os.path.abspath(__file__))
_GDN_PARENT = os.path.dirname(os.path.dirname(_HERE))  # .../others/vllm/main/
sys.path.insert(0, _GDN_PARENT)

import torch  # noqa: E402

# Resolve `common.dump_kernel` even when this script is run from a flat copy.
_MOE_DIR = os.path.normpath(os.path.join(_GDN_PARENT, "../../../moe"))
if _MOE_DIR not in sys.path:
    sys.path.insert(0, _MOE_DIR)
from common import dump_kernel, set_global_seed, str_to_bool  # noqa: E402

from gdn.ops.layernorm_guard import rmsnorm_gated_kernel  # noqa: E402
from gdn._compat import cdiv, next_power_of_2  # noqa: E402
from gdn.dtype_contract import resolve  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, required=True,
                   help="hidden / group size (e.g. head_v_dim=128)")
    p.add_argument("--BLOCK_N", type=int, required=True,
                   help="must be next_power_of_2(N); inner-loop tile width")
    p.add_argument("--ROWS_PER_BLOCK", type=int, default=4)
    p.add_argument("--num_warps", type=int, default=4)
    p.add_argument("--num_stages", type=int, default=3)
    p.add_argument("--HAS_BIAS", type=str_to_bool, default=False)
    p.add_argument("--HAS_Z", type=str_to_bool, default=True)
    p.add_argument("--NORM_BEFORE_GATE", type=str_to_bool, default=True)
    p.add_argument("--IS_RMS_NORM", type=str_to_bool, default=True)
    p.add_argument("--ACTIVATION", type=str, default="swish",
                   choices=["swish", "silu", "sigmoid", "none"])
    p.add_argument("--compute_type", type=str, required=True,
                   choices=["BF16"])
    # M only matters for the dummy launch — the cubin is independent of M.
    p.add_argument("--M", type=int, default=512)
    p.add_argument("--output_dir", type=str, required=True)
    return p.parse_args()


def torch_dtype_from_str(s: str) -> torch.dtype:
    return torch.bfloat16


def main() -> int:
    args = parse_args()
    set_global_seed(42)
    dtype = torch_dtype_from_str(args.compute_type)
    device = torch.device("cuda")

    M, N = args.M, args.N
    assert args.BLOCK_N >= N, "BLOCK_N must be >= N"
    assert args.BLOCK_N == next_power_of_2(N), \
        f"BLOCK_N={args.BLOCK_N} must equal next_power_of_2(N={N})={next_power_of_2(N)}"

    # Allocate dummy tensors with shapes/strides matching layer_norm_fwd.
    # Dead 参数（bias/mean/z）用 tensor 占位、绝不传 None：传 None 会被 Triton
    # 剥出运行时签名，C++ 按源码序传参即错位；dead 参数被 constexpr 分支
    # 消除，dummy 内容不会被读取。
    x = torch.randn((M, N), dtype=dtype, device=device)
    y = torch.empty_like(x)
    w = torch.randn((N,), dtype=dtype, device=device)
    b = torch.randn((N,), dtype=dtype, device=device)  # dummy bias（HAS_BIAS=False 时不读）
    z = torch.randn((M, N), dtype=dtype, device=device)  # dummy gate（HAS_Z=False 时不读）
    rstd_dtype = resolve("rmsnorm_gated", "rstd", dtype)
    rstd = torch.empty((M,), dtype=rstd_dtype, device=device)
    mean = torch.empty((M,), dtype=torch.float32, device=device)  # dummy mean（IS_RMS_NORM=True 时不读）

    # ngroups = N / group_size, vendor sets group_size = N → ngroups = 1.
    ngroups = 1
    grid = (cdiv(M, args.ROWS_PER_BLOCK), ngroups)

    kernel = rmsnorm_gated_kernel[grid](
        x,
        y,
        w,
        b,
        z,
        mean,
        rstd,
        x.stride(0),
        y.stride(0),
        z.stride(0),
        M,
        N=N,                            # tl.constexpr (group_size)
        eps=1e-6,
        BLOCK_N=args.BLOCK_N,
        ROWS_PER_BLOCK=args.ROWS_PER_BLOCK,
        HAS_BIAS=args.HAS_BIAS,
        HAS_Z=args.HAS_Z,
        NORM_BEFORE_GATE=args.NORM_BEFORE_GATE,
        IS_RMS_NORM=args.IS_RMS_NORM,
        ACTIVATION=args.ACTIVATION,
        num_warps=args.num_warps,
        num_stages=args.num_stages,
    )
    torch.cuda.synchronize()

    # Build a stable cubin filename that fully identifies the specialization.
    kernel_name = (
        f"rmsnorm_gated_kernel"
        f"_N_{N}"
        f"_BN_{args.BLOCK_N}"
        f"_R_{args.ROWS_PER_BLOCK}"
        f"_W_{args.num_warps}"
        f"_S_{args.num_stages}"
        f"_HB_{int(args.HAS_BIAS)}"
        f"_HZ_{int(args.HAS_Z)}"
        f"_NBG_{int(args.NORM_BEFORE_GATE)}"
        f"_RMS_{int(args.IS_RMS_NORM)}"
        f"_ACT_{args.ACTIVATION}"
        f"_compute_type_{args.compute_type}"
    )

    os.makedirs(args.output_dir, exist_ok=True)
    config = {
        "config": {
            "num_warps": args.num_warps,
            "num_stages": args.num_stages,
        }
    }
    dump_kernel(kernel, args.output_dir, kernel_name, config)
    print(f"[rmsnorm_gated] dumped {kernel_name}.cubin "
          f"-> {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
