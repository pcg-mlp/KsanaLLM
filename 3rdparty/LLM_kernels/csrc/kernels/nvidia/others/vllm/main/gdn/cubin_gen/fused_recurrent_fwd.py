# Copyright 2026 Tencent Inc. All rights reserved.
#
# GDN spec-decode recurrent path (Qwen3.5/3.6 MTP).
# ============================================================================
"""Offline cubin generator for `fused_recurrent_gated_delta_rule_fwd_kernel`.

GDN spec-decode recurrent path (varlen batch + per-seq drafts; the varlen+spec
sibling of packed_decode). State indices are 2D [nd, 1+mtp], row selected via
num_accepted_tokens-1. SISQ (stride_indices_seq = 1+mtp, --SISQ) is the only
constexpr varying with MTP depth; state-row strides are runtime
(do_not_specialize), so one cubin serves any pool geometry.

Constexprs pinned: VAR=CB=SPEC=INP=UIS=1, L2N=0 (q/k pre-L2norm'd), IBH=0
(beta=[T,HV]), KDA=0.
"""

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
from common import dump_kernel, set_global_seed  # noqa: E402

import triton  # noqa: E402

from gdn.dtype_contract import resolve  # noqa: E402

from gdn.ops.fused_recurrent import (  # noqa: E402
    fused_recurrent_gated_delta_rule_fwd_kernel as _wrapped,
)


def _unwrap_jit(kernel):
    # Peel @triton.heuristics wrapper layers, stopping at the JITFunction
    # (it also has .fn — the raw python def, not subscriptable).
    while type(kernel).__name__ != "JITFunction":
        kernel = kernel.fn
    return kernel


_jit_kernel = _unwrap_jit(_wrapped)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--B", type=int, default=1, help="batch (dummy; varlen uses N)")
    p.add_argument("--H", type=int, default=16)
    p.add_argument("--HV", type=int, default=16)
    p.add_argument("--K", type=int, default=128)
    p.add_argument("--V", type=int, default=128)
    p.add_argument("--BK", type=int, default=128, help="next_pow2(K); NK must be 1")
    p.add_argument("--BV", type=int, default=32, help="min(next_pow2(V),32)")
    p.add_argument("--SISQ", type=int, default=2,
                   help="stride_indices_seq (constexpr, 进 cubin 名): "
                        "state-indices 列数 = 1+mtp。")
    p.add_argument("--num_warps", type=int, default=1)
    p.add_argument("--num_stages", type=int, default=3)
    p.add_argument("--compute_type", type=str, required=True, choices=["BF16"])
    p.add_argument("--output_dir", type=str, required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    set_global_seed(42)
    dtype = torch_dtype_from_str(args.compute_type)
    device = torch.device("cuda")

    H, HV, K, V = args.H, args.HV, args.K, args.V
    BK, BV = args.BK, args.BV
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, "fused_recurrent_fwd requires NK==1 (BK>=K)"

    scale = 1.0 / (K ** 0.5)

    # Varlen spec dummy: N/T 是运行时参数（do_not_specialize），dummy 形状不影响
    # 产物；唯一随 MTP 深度变化的 constexpr 是 SISQ（--SISQ 参数化）。
    N = 2
    T = 4
    cu_seqlens = torch.tensor([0, 2, 4], dtype=torch.int32, device=device)

    # g/beta = [T, HV] fp32 (NOT headwise; matches fused_post_conv → IBH=0).
    q = torch.randn((T, H, K), dtype=dtype, device=device)
    k = torch.randn((T, H, K), dtype=dtype, device=device)
    v = torch.randn((T, HV, V), dtype=dtype, device=device)
    g_dtype = resolve("fused_recurrent_fwd", "g", dtype)
    beta_dtype = resolve("fused_recurrent_fwd", "beta", dtype)
    g = torch.randn((T, HV), dtype=g_dtype, device=device)
    beta = torch.randn((T, HV), dtype=beta_dtype, device=device)

    # Flat state pool [state_versions*num_slots, HV, V, K] fp32. 行距是运行时 ABI
    # （C++ 按 RecRowStride 传入），池形状不影响产物。step-major rows:
    # slotX/stepS = S*num_slots + slotX. slot 0 = null.
    num_slots = 3
    state_versions = 2
    initial_state = torch.zeros((state_versions * num_slots, HV, V, K), dtype=torch.float32, device=device)
    # per-seq [step0_row, ..., step_mtp_row] = [slot, num_slots+slot, ...]。
    # 列数 = args.SISQ（烘焙 stride_indices_seq），行值仅作占位。
    ssm_state_indices = torch.arange(
        1, 1 + args.SISQ, dtype=torch.int32, device=device,
    ).unsqueeze(0).repeat(2, 1)
    num_accepted_tokens = torch.ones((2,), dtype=torch.int32, device=device)  # >=1: kernel does n-1

    o = q.new_empty(NK, T, HV, V)

    stride_init_state_token = initial_state.stride(0)  # flat: HV*V*K (one row)
    stride_final_state_token = stride_init_state_token  # INPLACE
    stride_indices_seq, stride_indices_tok = ssm_state_indices.stride()

    # grid = (NK, NV, N * HV); NK=1.
    grid = (NK, NV, N * HV)

    kernel = _jit_kernel[grid](
        q, k, v, g, beta, o, initial_state, initial_state, cu_seqlens,
        ssm_state_indices, num_accepted_tokens, scale,
        N, T,
        B=args.B, H=H, HV=HV, K=K, V=V, BK=BK, BV=BV,
        stride_init_state_token=stride_init_state_token,
        stride_final_state_token=stride_final_state_token,
        stride_indices_seq=stride_indices_seq,
        stride_indices_tok=stride_indices_tok,
        USE_INITIAL_STATE=True,
        INPLACE_FINAL_STATE=True,
        IS_BETA_HEADWISE=False,  # beta=[T,HV] matches fused_post_conv layout
        USE_QK_L2NORM_IN_KERNEL=False,
        IS_VARLEN=True,
        IS_CONTINUOUS_BATCHING=True,
        IS_SPEC_DECODING=True,
        IS_KDA=False,
        num_warps=args.num_warps,
        num_stages=args.num_stages,
    )
    torch.cuda.synchronize()

    # Filename key order MUST match kernel_key_map_ in triton_wrapper.cpp.
    # fwd kernel has no SOFTPLUS_THRESHOLD (g/beta pre-computed by caller).
    # IBH=0 (beta=[T,HV], not headwise [T,HV,V]) — must match parity/production.
    kernel_name = (
        f"fused_recurrent_gated_delta_rule_fwd_kernel"
        f"_H_{H}_HV_{HV}_K_{K}_V_{V}_BK_{BK}_BV_{BV}"
        f"_SISQ_{stride_indices_seq}_SIT_{stride_indices_tok}"
        f"_UIS_1_INP_1_IBH_0_L2N_0_VAR_1_CB_1_SPEC_1_KDA_0"
        f"_W_{args.num_warps}_S_{args.num_stages}"
        f"_compute_type_{args.compute_type}"
    )

    os.makedirs(args.output_dir, exist_ok=True)
    config = {"config": {"num_warps": args.num_warps, "num_stages": args.num_stages}}
    dump_kernel(kernel, args.output_dir, kernel_name, config)
    print(f"[fused_recurrent_fwd] dumped {kernel_name}.cubin -> {args.output_dir}")
    print(f"  grid={grid} NV={NV} NK={NK}")
    print(f"  stride_init_state_token={stride_init_state_token} "
          f"stride_indices_seq={stride_indices_seq} stride_indices_tok={stride_indices_tok}")
    return 0


def torch_dtype_from_str(s: str) -> torch.dtype:
    return torch.bfloat16


if __name__ == "__main__":
    sys.exit(main())
