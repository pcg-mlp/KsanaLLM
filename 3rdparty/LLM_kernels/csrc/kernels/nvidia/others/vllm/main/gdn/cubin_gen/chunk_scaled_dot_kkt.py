# Copyright 2026 Tencent Inc. All rights reserved.
#
# chunk-prefill main path 1/5. Strip the vendor heuristics/autotune `.fn`
# → JITFunction; provide all constexpr meta-params explicitly (BK=64).
# ============================================================================
"""Offline cubin generator for `chunk_scaled_dot_kkt_fwd_kernel`."""

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
from gdn.ops.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd_kernel as _wrapped  # noqa: E402


def _strip_to_jit(k):
    while type(k).__name__ != "JITFunction" and hasattr(k, "fn"):
        k = k.fn
    return k


_jit_kernel = _strip_to_jit(_wrapped)
assert type(_jit_kernel).__name__ == "JITFunction", \
    f"unexpected wrapper chain: {type(_jit_kernel).__name__}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--H", type=int, default=16, help="num attn heads (output)")
    p.add_argument("--Hg", type=int, default=16, help="num KV/group heads")
    p.add_argument("--K", type=int, default=128, help="head_k_dim, 128 for Qwen3.5/3.6")
    p.add_argument("--BT", type=int, default=64, help="chunk_size = 64")
    p.add_argument("--BK", type=int, default=64,
                   help="vendor autotune in [32,64,128]; pin 64")
    p.add_argument("--USE_G", type=str_to_bool, default=False,
                   help="GDN gate path. False = no g tensor")
    p.add_argument("--IS_VARLEN", type=str_to_bool, default=True,
                   help="varlen via cu_seqlens, paged batching")
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

    # dummy varlen: B=1, T=128 (split as [0,64,128]), H/Hg=16, K=128
    B, T = 1, 128
    BT = args.BT
    k = torch.randn((B, T, args.Hg, args.K), dtype=dtype, device=device)
    beta = torch.randn((B, T, args.H), dtype=torch.float32, device=device)  # fp32 (contract)
    g = torch.randn((B, T, args.H), dtype=torch.float32, device=device) if args.USE_G else None
    A_dtype = resolve("chunk_scaled_dot_kkt", "A", dtype)
    A = torch.empty((B, T, args.H, BT), dtype=A_dtype, device=device)
    cu_seqlens = torch.tensor([0, 64, 128], dtype=torch.int32, device=device) \
        if args.IS_VARLEN else None
    # chunk_indices: shape [NT, 2] -> (i_n, local_i_t)
    if args.IS_VARLEN:
        chunk_indices = torch.tensor([[0, 0], [1, 0]], dtype=torch.int32, device=device)
        NT = chunk_indices.shape[0]
    else:
        chunk_indices = None
        NT = triton.cdiv(T, BT)

    grid = (NT, B * args.H)

    kernel = _jit_kernel[grid](
        k,
        beta,
        g,
        A,
        cu_seqlens,
        chunk_indices,
        T,
        H=args.H,
        Hg=args.Hg,
        K=args.K,
        BT=args.BT,
        BK=args.BK,
        IS_VARLEN=args.IS_VARLEN,
        USE_G=args.USE_G,
        num_warps=args.num_warps,
        num_stages=args.num_stages,
    )
    torch.cuda.synchronize()

    kernel_name = (
        f"chunk_scaled_dot_kkt_fwd_kernel"
        f"_H_{args.H}_Hg_{args.Hg}_K_{args.K}_BT_{args.BT}_BK_{args.BK}"
        f"_VL_{int(bool(args.IS_VARLEN))}_G_{int(bool(args.USE_G))}"
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
    print(f"[chunk_scaled_dot_kkt] dumped {kernel_name}.cubin -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
