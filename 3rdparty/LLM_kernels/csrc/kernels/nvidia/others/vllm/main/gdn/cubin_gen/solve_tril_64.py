# Copyright 2026 Tencent Inc. All rights reserved.
#
# chunk-prefill main path 4/5 (BT=64). Strip the vendor autotune `.fn`.
# USE_TMA=False (no TMA below sm_90); DOT_PRECISION="ieee" = vendor default.
# ============================================================================
"""Offline cubin generator for `merge_16x16_to_64x64_inverse_kernel` (BT=64)."""

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

from gdn.ops.solve_tril import (  # noqa: E402
    merge_16x16_to_64x64_inverse_kernel as _wrapped,
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
    p.add_argument("--H", type=int, default=16)
    p.add_argument("--BT", type=int, default=64,
                   help="must be 64 for this kernel variant")
    p.add_argument("--USE_TMA", type=str_to_bool, default=False,
                   help="L40 sm_89 has no TMA; force False")
    p.add_argument("--IS_VARLEN", type=str_to_bool, default=True)
    p.add_argument("--DOT_PRECISION", type=str, default="ieee",
                   choices=["ieee", "tf32", "tf32x3"])
    p.add_argument("--num_warps", type=int, default=4)
    p.add_argument("--num_stages", type=int, default=3)
    # A 输入按 contract 解析（fp32）；Ai 输出跟随 k.dtype（model dtype，bf16）。
    # compute_type 仅用于标注 cubin 文件名。
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

    A_dtype = resolve("solve_tril_64", "A", dtype)
    A = torch.randn((B, T, args.H, BT), dtype=A_dtype, device=device)
    # vLLM: output_dtype=k.dtype → Ai stays model dtype (bf16)
    Ai = torch.zeros((B, T, args.H, BT), dtype=dtype, device=device)

    if args.IS_VARLEN:
        cu_seqlens = torch.tensor([0, 64, 128], dtype=torch.int32, device=device)
        chunk_indices = torch.tensor([[0, 0], [1, 0]], dtype=torch.int32, device=device)
    else:
        cu_seqlens = None
        chunk_indices = None

    grid = (NT, B * args.H)

    kernel = _jit_kernel[grid](
        A,
        Ai,
        cu_seqlens,
        chunk_indices,
        T,
        H=args.H,
        BT=args.BT,
        USE_TMA=args.USE_TMA,
        IS_VARLEN=args.IS_VARLEN,
        DOT_PRECISION=args.DOT_PRECISION,
        num_warps=args.num_warps,
        num_stages=args.num_stages,
    )
    torch.cuda.synchronize()

    kernel_name = (
        f"merge_16x16_to_64x64_inverse_kernel"
        f"_H_{args.H}_BT_{args.BT}_TMA_{int(bool(args.USE_TMA))}"
        f"_VL_{int(bool(args.IS_VARLEN))}_DP_{args.DOT_PRECISION}"
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
    print(f"[solve_tril_64] dumped {kernel_name}.cubin -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
