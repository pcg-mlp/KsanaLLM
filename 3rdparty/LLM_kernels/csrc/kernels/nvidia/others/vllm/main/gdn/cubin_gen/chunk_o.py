# Copyright 2026 Tencent Inc. All rights reserved.
#
# chunk-prefill main path 3/5. Strip the vendor heuristics/autotune `.fn`
# → JITFunction; pin BK=64, BV=64.
# ============================================================================
"""Offline cubin generator for `chunk_fwd_kernel_o`."""

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

from gdn.ops.chunk_o import chunk_fwd_kernel_o as _wrapped  # noqa: E402


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
                   help="vendor autotune; pin 64")
    p.add_argument("--BV", type=int, default=64,
                   help="vendor autotune; pin 64")
    p.add_argument("--USE_G", type=str_to_bool, default=True,
                   help="GDN gate path always True for qwen3.5/3.6")
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

    q = torch.randn((B, T, args.Hg, args.K), dtype=dtype, device=device)
    k = torch.randn((B, T, args.Hg, args.K), dtype=dtype, device=device)
    v = torch.randn((B, T, args.H, args.V), dtype=dtype, device=device)
    h = torch.randn((B, NT, args.H, args.V, args.K), dtype=dtype, device=device)
    g_dtype = resolve("chunk_o", "g", dtype)
    g = torch.randn((B, T, args.H), dtype=g_dtype, device=device) if args.USE_G else None
    o = torch.empty_like(v)
    scale = float(args.K) ** -0.5

    if args.IS_VARLEN:
        cu_seqlens = torch.tensor([0, 64, 128], dtype=torch.int32, device=device)
        chunk_indices = torch.tensor([[0, 0], [1, 0]], dtype=torch.int32, device=device)
    else:
        cu_seqlens = None
        chunk_indices = None

    grid = (triton.cdiv(args.V, args.BV), NT, B * args.H)

    kernel = _jit_kernel[grid](
        q,
        k,
        v,
        h,
        g,
        o,
        cu_seqlens,
        chunk_indices,
        scale,
        T,
        H=args.H,
        Hg=args.Hg,
        K=args.K,
        V=args.V,
        BT=args.BT,
        BK=args.BK,
        BV=args.BV,
        USE_G=args.USE_G,
        IS_VARLEN=args.IS_VARLEN,
        num_warps=args.num_warps,
        num_stages=args.num_stages,
    )
    torch.cuda.synchronize()

    kernel_name = (
        f"chunk_fwd_kernel_o"
        f"_H_{args.H}_Hg_{args.Hg}_K_{args.K}_V_{args.V}"
        f"_BT_{args.BT}_BK_{args.BK}_BV_{args.BV}"
        f"_G_{int(bool(args.USE_G))}_VL_{int(bool(args.IS_VARLEN))}"
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
    print(f"[chunk_o] dumped {kernel_name}.cubin -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
