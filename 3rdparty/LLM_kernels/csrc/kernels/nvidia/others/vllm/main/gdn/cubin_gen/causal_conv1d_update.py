# Copyright 2026 Tencent Inc. All rights reserved.
#
# Wraps `gdn.ops.causal_conv1d.gdn_causal_conv1d_update_kernel` (vendored from
# vllm v0.21.0). Used during decode for state-cache update.
# ============================================================================
"""Offline cubin generator for `gdn_causal_conv1d_update_kernel` (decode)."""

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

from gdn.ops.causal_conv1d import gdn_causal_conv1d_update_kernel  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--KERNEL_WIDTH", type=int, default=4)
    p.add_argument("--HAS_BIAS", type=str_to_bool, default=True)
    p.add_argument("--SILU_ACTIVATION", type=str_to_bool, default=True)
    p.add_argument("--IS_VARLEN", type=str_to_bool, default=True,
                   help="continuous batching uses varlen=True")
    p.add_argument("--IS_APC_ENABLED", type=str_to_bool, default=False)
    p.add_argument("--IS_SPEC_DECODING", type=str_to_bool, default=False,
                   help="spec decoding off in MVP — task #34 covers it")
    p.add_argument("--HAS_NULL_BLOCK", type=str_to_bool, default=True)
    p.add_argument("--NP2_STATELEN", type=int, default=4)
    p.add_argument("--STATE_LEN", type=int, default=0,
                   help="conv state_len; 0 = KW-1 (legacy), MTP = KW-1+mtp. Baked "
                        "into cubin name to disambiguate NP2 collisions.")
    p.add_argument("--BLOCK_N", type=int, default=256)
    p.add_argument("--num_warps", type=int, default=4)
    p.add_argument("--num_stages", type=int, default=2)
    p.add_argument("--compute_type", type=str, required=True,
                   choices=["BF16"])
    p.add_argument("--dim", type=int, required=True,
                   help="conv_dim: 6144 for 0.8B, 10240 for 27B")
    p.add_argument("--output_dir", type=str, required=True)
    return p.parse_args()


def torch_dtype_from_str(s: str) -> torch.dtype:
    return torch.bfloat16


def main() -> int:
    args = parse_args()
    set_global_seed(42)
    dtype = torch_dtype_from_str(args.compute_type)
    device = torch.device("cuda")

    batch = 4
    dim = args.dim
    # state_len = KW-1 (legacy SL3) or explicit SL4 (MTP: +1 draft slot). NP2
    # rounds both to 4, so SL is baked into the filename to disambiguate.
    state_len = args.STATE_LEN if args.STATE_LEN > 0 else args.KERNEL_WIDTH - 1
    assert args.NP2_STATELEN == triton.next_power_of_2(state_len), \
        f"NP2_STATELEN ({args.NP2_STATELEN}) != next_power_of_2(STATE_LEN={state_len})"
    # Spec dummy: 1+mtp tokens/seq (real + mtp drafts) so the IS_SPEC_DECODING
    # branch compiles against a representative varlen shape. mtp = SL-(KW-1).
    seqlen = (state_len - (args.KERNEL_WIDTH - 1)) + 1 if args.IS_SPEC_DECODING else 1
    # NCL 是运行时参数（do_not_specialize），dummy 仅类型检查。
    num_cache_lines = 4

    w = torch.randn((dim, args.KERNEL_WIDTH), dtype=dtype, device=device)
    b = torch.randn((dim,), dtype=dtype, device=device) if args.HAS_BIAS else None
    # conv_state dtype MUST match x/w dtype (bf16). The FLA causal_conv1d_update kernel
    # loads conv_state and re-assigns from x in-loop, requiring consistent dtype
    # (fp32 conv_state + bf16 x -> AssertionError "col2 fp32 re-assigned bf16").
    # So conv_state = bf16 (== x/w). C++ decode must use bf16 conv_state to match
    # (the hand-written fp32 was a float*-OOB workaround; FLA cubin reads bf16, no OOB).
    conv_state = torch.zeros(
        (num_cache_lines, dim, state_len), dtype=dtype, device=device,
    )
    # conv_state_indices (batch,) int32: stride_state_indices is tl.constexpr (baked
    # into the cubin). (batch,) bakes stride=1 so batch>1 indexes indices[idx_seq]
    # correctly; the old (batch,2) dummy baked stride=2 -> OOB at batch>1 (B1).
    conv_state_indices = torch.zeros((batch,), dtype=torch.int32, device=device)
    num_accepted_tokens = (
        torch.ones((batch,), dtype=torch.int32, device=device)  # >=1: kernel does num_accepted-1
        if args.IS_SPEC_DECODING else None
    )
    if args.IS_SPEC_DECODING:
        # Token-major flat dummy [total_tokens, dim] matches production layout.
        # seqlen=2 (real+draft): query_start_loc = [0, 2, 4, ...].
        total_tokens = batch * seqlen
        x = torch.randn((total_tokens, dim), dtype=dtype, device=device)
        block_idx_last_scheduled_token = torch.zeros((batch,), dtype=torch.int32, device=device)
        initial_state_idx = torch.zeros((batch,), dtype=torch.int32, device=device)
        out = torch.empty_like(x)
        query_start_loc = (
            torch.tensor(list(range(0, (batch + 1) * seqlen, seqlen)), dtype=torch.int32, device=device)
            if args.IS_VARLEN else None
        )
        # Flat 2D layout has no .stride(2); pass production strides explicitly.
        # stride_*_seq unused under IS_VARLEN (kept = seqlen*dim for correctness).
        stride_x_seq = seqlen * dim
        stride_x_dim = 1
        stride_x_token = dim
        stride_o_seq = seqlen * dim
        stride_o_dim = 1
        stride_o_token = dim
    else:
        x = torch.randn((batch, dim, seqlen), dtype=dtype, device=device)
        block_idx_last_scheduled_token = torch.zeros((batch,), dtype=torch.int32, device=device)
        initial_state_idx = torch.zeros((batch,), dtype=torch.int32, device=device)
        out = torch.empty_like(x)
        query_start_loc = (
            torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32, device=device)
            if args.IS_VARLEN else None
        )
        stride_x_seq = x.stride(0)
        stride_x_dim = x.stride(1)
        stride_x_token = x.stride(2)
        stride_o_seq = out.stride(0)
        stride_o_dim = out.stride(1)
        stride_o_token = out.stride(2)

    grid = (batch, triton.cdiv(dim, args.BLOCK_N))

    kernel = gdn_causal_conv1d_update_kernel[grid](
        x_ptr=x,
        w_ptr=w,
        bias_ptr=b,
        conv_state_ptr=conv_state,
        conv_state_indices_ptr=conv_state_indices,
        num_accepted_tokens_ptr=num_accepted_tokens,
        query_start_loc_ptr=query_start_loc,
        block_idx_last_scheduled_token=block_idx_last_scheduled_token,
        initial_state_idx=initial_state_idx,
        o_ptr=out,
        # dims
        batch=batch,
        dim=dim,
        seqlen=seqlen,
        state_len=state_len,
        num_cache_lines=num_cache_lines,
        # strides
        stride_x_seq=stride_x_seq,
        stride_x_dim=stride_x_dim,
        stride_x_token=stride_x_token,
        stride_w_dim=w.stride(0),
        stride_w_width=w.stride(1),
        stride_conv_state_seq=conv_state.stride(0),
        stride_conv_state_dim=conv_state.stride(1),
        stride_conv_state_tok=conv_state.stride(2),
        stride_state_indices=conv_state_indices.stride(0),
        stride_o_seq=stride_o_seq,
        stride_o_dim=stride_o_dim,
        stride_o_token=stride_o_token,
        # others
        null_block_id=0,
        # constexpr
        HAS_BIAS=args.HAS_BIAS,
        KERNEL_WIDTH=args.KERNEL_WIDTH,
        SILU_ACTIVATION=args.SILU_ACTIVATION,
        IS_VARLEN=args.IS_VARLEN,
        IS_APC_ENABLED=args.IS_APC_ENABLED,
        IS_SPEC_DECODING=args.IS_SPEC_DECODING,
        NP2_STATELEN=args.NP2_STATELEN,
        HAS_NULL_BLOCK=args.HAS_NULL_BLOCK,
        BLOCK_N=args.BLOCK_N,
        num_warps=args.num_warps,
        num_stages=args.num_stages,
    )
    torch.cuda.synchronize()

    kernel_name = (
        f"gdn_causal_conv1d_update_kernel"
        f"_D_{args.dim}"
        f"_KW_{args.KERNEL_WIDTH}"
        f"_BN_{args.BLOCK_N}"
        f"_NP2_{args.NP2_STATELEN}"
        f"_SL_{state_len}"
        f"_HB_{int(args.HAS_BIAS)}"
        f"_SILU_{int(args.SILU_ACTIVATION)}"
        f"_VL_{int(args.IS_VARLEN)}"
        f"_APC_{int(args.IS_APC_ENABLED)}"
        f"_SPEC_{int(args.IS_SPEC_DECODING)}"
        f"_NULL_{int(args.HAS_NULL_BLOCK)}"
        f"_W_{args.num_warps}"
        f"_S_{args.num_stages}"
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
    print(f"[causal_conv1d_update] dumped {kernel_name}.cubin "
          f"-> {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
