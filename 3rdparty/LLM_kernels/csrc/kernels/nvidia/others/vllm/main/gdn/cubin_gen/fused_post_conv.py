# Copyright 2026 Tencent Inc. All rights reserved.
#
# Prefill: Q/K l2norm + V gating in one fused kernel.
# ============================================================================
"""Offline cubin generator for `fused_post_conv_kernel`."""

from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_GDN_PARENT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _GDN_PARENT)

import torch  # noqa: E402

_MOE_DIR = os.path.normpath(os.path.join(_GDN_PARENT, "../../../moe"))
if _MOE_DIR not in sys.path:
    sys.path.insert(0, _MOE_DIR)
from common import dump_kernel, set_global_seed, str_to_bool  # noqa: E402

import triton  # noqa: E402

from gdn.dtype_contract import resolve  # noqa: E402

from gdn.ops.fused_gdn_prefill_post_conv import (  # noqa: E402
    fused_post_conv_kernel,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # Qwen3.5/3.6 numerics: head_k_dim=128 head_v_dim=128, H=Hg=numH_q,
    # HV=numH_v. K=128, V=128, BK=BV=128, BLOCK_T=16.
    p.add_argument("--H", type=int, default=16)
    p.add_argument("--HV", type=int, default=32)
    p.add_argument("--K", type=int, default=128)
    p.add_argument("--V", type=int, default=128)
    p.add_argument("--BK", type=int, default=128)
    p.add_argument("--BV", type=int, default=128)
    p.add_argument("--BLOCK_T", type=int, default=16)
    p.add_argument("--APPLY_L2NORM", type=str_to_bool, default=True)
    p.add_argument("--L2NORM_EPS", type=float, default=1e-6)
    p.add_argument("--OUTPUT_G_EXP", type=str_to_bool, default=False)
    p.add_argument("--SOFTPLUS_THRESHOLD", type=float, default=20.0)
    p.add_argument("--num_warps", type=int, default=4)
    p.add_argument("--num_stages", type=int, default=2)
    p.add_argument("--compute_type", type=str, required=True,
                   choices=["BF16"])
    p.add_argument("--output_dir", type=str, required=True)
    return p.parse_args()


def torch_dtype_from_str(s: str) -> torch.dtype:
    return torch.bfloat16


def main() -> int:
    args = parse_args()
    set_global_seed(42)
    dtype = torch_dtype_from_str(args.compute_type)
    device = torch.device("cuda")

    L = 256
    H, HV, K, V = args.H, args.HV, args.K, args.V
    qkv_dim = H * K * 2 + HV * V
    conv_output = torch.randn((L, qkv_dim), dtype=dtype, device=device)
    a = torch.randn((L, HV), dtype=dtype, device=device)
    b = torch.randn((L, HV), dtype=dtype, device=device)
    A_log = torch.randn((HV,), dtype=torch.float32, device=device)
    # dt_bias stays FP32 to match A_log (see packed_decode.py): the cubin loads
    # it with a fp32 ld.global for softplus(a + dt_bias) in fp32. HF keeps
    # dt_bias as a fp32 buffer though the checkpoint stores BF16; a BF16 ptr
    # would lose precision and mismatch the fp32-load contract.
    dt_bias = torch.randn((HV,), dtype=torch.float32, device=device)
    q = torch.empty((L, H, K), dtype=dtype, device=device)
    k = torch.empty((L, H, K), dtype=dtype, device=device)
    v = torch.empty((L, HV, V), dtype=dtype, device=device)
    g_dtype = resolve("fused_post_conv", "g", dtype)
    beta_dtype = resolve("fused_post_conv", "beta", dtype)
    g = torch.empty((L, HV), dtype=g_dtype, device=device)
    beta = torch.empty((L, HV), dtype=beta_dtype, device=device)

    grid = (triton.cdiv(L, args.BLOCK_T), H + HV)

    kernel = fused_post_conv_kernel[grid](
        mixed_qkv_ptr=conv_output,
        a_ptr=a,
        b_ptr=b,
        A_log_ptr=A_log,
        dt_bias_ptr=dt_bias,
        q_ptr=q,
        k_ptr=k,
        v_ptr=v,
        g_ptr=g,
        beta_ptr=beta,
        stride_x_tok=conv_output.stride(0),
        stride_a_tok=a.stride(0),
        stride_b_tok=b.stride(0),
        stride_q_tok=q.stride(0),
        stride_k_tok=k.stride(0),
        stride_v_tok=v.stride(0),
        L=L,
        H=H,
        HV=HV,
        K=K,
        V=V,
        APPLY_L2NORM=args.APPLY_L2NORM,
        L2NORM_EPS=args.L2NORM_EPS,
        OUTPUT_G_EXP=args.OUTPUT_G_EXP,
        SOFTPLUS_THRESHOLD=args.SOFTPLUS_THRESHOLD,
        BLOCK_T=args.BLOCK_T,
        BK=args.BK,
        BV=args.BV,
        num_warps=args.num_warps,
        num_stages=args.num_stages,
    )
    torch.cuda.synchronize()

    kernel_name = (
        f"fused_post_conv_kernel"
        f"_H_{H}_HV_{HV}_K_{K}_V_{V}"
        f"_BK_{args.BK}_BV_{args.BV}_BT_{args.BLOCK_T}"
        f"_L2_{int(args.APPLY_L2NORM)}"
        f"_GEXP_{int(args.OUTPUT_G_EXP)}"
        f"_W_{args.num_warps}_S_{args.num_stages}"
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
    print(f"[fused_post_conv] dumped {kernel_name}.cubin "
          f"-> {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
