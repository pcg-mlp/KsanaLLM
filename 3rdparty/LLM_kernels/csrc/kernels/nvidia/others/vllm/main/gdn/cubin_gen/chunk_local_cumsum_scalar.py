# Copyright 2026 Tencent Inc. All rights reserved.
#
# Prefill per-chunk cumulative log-decay. Strip the vendor autotune `.fn`.
# ============================================================================
"""Offline cubin generator for `chunk_local_cumsum_scalar_kernel`."""

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

from gdn.ops.cumsum import (  # noqa: E402
    chunk_local_cumsum_scalar_kernel as _wrapped,
)


def _strip_to_jit(k):
    while type(k).__name__ != "JITFunction" and hasattr(k, "fn"):
        k = k.fn
    return k


_jit_kernel = _strip_to_jit(_wrapped)
assert type(_jit_kernel).__name__ == "JITFunction", \
    f"unexpected wrapper chain: {type(_jit_kernel).__name__}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--B", type=int, default=1)
    p.add_argument("--H", type=int, default=32)
    p.add_argument("--BT", type=int, default=64,
                   help="chunk_size; Qwen3.5/3.6 uses 64")
    p.add_argument("--HEAD_FIRST", type=str_to_bool, default=False)
    p.add_argument("--REVERSE", type=str_to_bool, default=False)
    p.add_argument("--IS_VARLEN", type=str_to_bool, default=True)
    p.add_argument("--num_warps", type=int, default=4)
    p.add_argument("--num_stages", type=int, default=1)
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

    B, T, H = args.B, 256, args.H
    s_dtype = resolve("chunk_local_cumsum_scalar", "s", dtype)
    s = torch.randn((B, T, H), dtype=s_dtype, device=device)
    o = torch.empty_like(s, dtype=torch.float32)

    cu_seqlens = (
        torch.tensor([0, 128, 256], dtype=torch.int32, device=device)
        if args.IS_VARLEN else None
    )
    chunk_indices = (
        torch.zeros((triton.cdiv(T, args.BT) * 2,),
                    dtype=torch.int32, device=device)
        if args.IS_VARLEN else None
    )

    NT = triton.cdiv(T, args.BT)
    grid = (NT, B * H)

    kernel = _jit_kernel[grid](
        s,
        o,
        cu_seqlens,
        chunk_indices,
        T,
        B=args.B,
        H=args.H,
        BT=args.BT,
        REVERSE=args.REVERSE,
        IS_VARLEN=args.IS_VARLEN,
        HEAD_FIRST=args.HEAD_FIRST,
        num_warps=args.num_warps,
        num_stages=args.num_stages,
    )
    torch.cuda.synchronize()

    kernel_name = (
        f"chunk_local_cumsum_scalar_kernel"
        f"_B_{args.B}_H_{args.H}_BT_{args.BT}"
        f"_REV_{int(args.REVERSE)}"
        f"_VL_{int(args.IS_VARLEN)}"
        f"_HF_{int(args.HEAD_FIRST)}"
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
    print(f"[chunk_local_cumsum_scalar] dumped {kernel_name}.cubin "
          f"-> {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
