# Copyright 2026 Tencent Inc. All rights reserved.
#
# chunk-prefill main path 5/5; gdn_ prefix isolates from the KDA family's
# same-named kernel. Strip the vendor autotune `.fn`; caller pins BK=BV=64.
# ============================================================================
"""Offline cubin generator for `gdn_recompute_w_u_fwd_kernel`."""

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
from gdn.ops.wy_fast import gdn_recompute_w_u_fwd_kernel as _wrapped  # noqa: E402


def _strip_to_jit(k):
    while type(k).__name__ != "JITFunction" and hasattr(k, "fn"):
        k = k.fn
    return k


_jit_kernel = _strip_to_jit(_wrapped)
assert type(_jit_kernel).__name__ == "JITFunction", \
    f"unexpected wrapper chain: {type(_jit_kernel).__name__}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--H", type=int, default=16)
    p.add_argument("--Hg", type=int, default=16)
    p.add_argument("--K", type=int, default=128)
    p.add_argument("--V", type=int, default=128)
    p.add_argument("--BT", type=int, default=64)
    p.add_argument("--BK", type=int, default=64,
                   help="caller pins BK=64")
    p.add_argument("--BV", type=int, default=64,
                   help="caller pins BV=64")
    p.add_argument("--IS_VARLEN", type=str_to_bool, default=True)
    p.add_argument("--num_warps", type=int, default=4)
    p.add_argument("--num_stages", type=int, default=3)
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

    B, T = 1, 128
    BT = args.BT
    NT = triton.cdiv(T, BT) if not args.IS_VARLEN else 2

    k = torch.randn((B, T, args.Hg, args.K), dtype=dtype, device=device)
    v = torch.randn((B, T, args.H, args.V), dtype=dtype, device=device)
    # beta: fp32 (contract) — d_beta is now float* in C++ workspace.
    beta_dtype = resolve("wy_fast_recompute", "beta", dtype)
    beta = torch.randn((B, T, args.H), dtype=beta_dtype, device=device)
    # A = solve_tril 输出（output_dtype=k.dtype → model dtype）；经 contract 路由，
    # solve_tril 输出改 fp32 时在 FP32_PARAMS 加一条即同步。
    A_dtype = resolve("wy_fast_recompute", "A", dtype)
    A = torch.randn((B, T, args.H, BT), dtype=A_dtype, device=device)
    g_dtype = resolve("wy_fast_recompute", "g", dtype)
    g = torch.randn((B, T, args.H), dtype=g_dtype, device=device)
    u = torch.empty_like(v)
    w = k.new_empty(B, T, args.H, args.K)

    if args.IS_VARLEN:
        cu_seqlens = torch.tensor([0, 64, 128], dtype=torch.int32, device=device)
        chunk_indices = torch.tensor([[0, 0], [1, 0]], dtype=torch.int32, device=device)
    else:
        cu_seqlens = None
        chunk_indices = None

    grid = (NT, B * args.H)

    kernel = _jit_kernel[grid](
        k,
        v,
        beta,
        w,
        u,
        A,
        g,
        cu_seqlens,
        chunk_indices,
        T,
        H=args.H,
        Hg=args.Hg,
        K=args.K,
        V=args.V,
        BT=args.BT,
        BK=args.BK,
        BV=args.BV,
        IS_VARLEN=args.IS_VARLEN,
        num_warps=args.num_warps,
        num_stages=args.num_stages,
    )
    torch.cuda.synchronize()

    kernel_name = (
        f"gdn_recompute_w_u_fwd_kernel"
        f"_H_{args.H}_Hg_{args.Hg}_K_{args.K}_V_{args.V}"
        f"_BT_{args.BT}_BK_{args.BK}_BV_{args.BV}"
        f"_VL_{int(bool(args.IS_VARLEN))}"
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
    print(f"[wy_fast_recompute] dumped {kernel_name}.cubin -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
