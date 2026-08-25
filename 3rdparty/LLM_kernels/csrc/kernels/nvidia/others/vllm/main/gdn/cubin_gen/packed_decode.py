# Copyright 2026 Tencent Inc. All rights reserved.
#
# GDN decode single-token recurrent update: folds q/k/v split + L2norm +
# g/beta prep + the delta-rule update into one plain-@triton.jit kernel (all
# constexprs pinned directly). State pool [2, HV, V, K] (slot 0 =
# NULL_BLOCK_ID); h0 == ht INPLACE (loads complete before stores). Config
# matches the spec fwd kernel: USE_QK_L2NORM_IN_KERNEL=True,
# SOFTPLUS_THRESHOLD=20.0, scale=1/sqrt(K).
# ============================================================================
"""Offline cubin generator for `fused_recurrent_gated_delta_rule_packed_decode_kernel`."""

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

from gdn.ops.fused_recurrent import (  # noqa: E402
    fused_recurrent_gated_delta_rule_packed_decode_kernel as _jit_kernel,
)

assert type(_jit_kernel).__name__ == "JITFunction", \
    f"unexpected wrapper chain: {type(_jit_kernel).__name__}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--B", type=int, default=1)
    p.add_argument("--H", type=int, default=16)
    p.add_argument("--HV", type=int, default=16)
    p.add_argument("--K", type=int, default=128)
    p.add_argument("--V", type=int, default=128)
    p.add_argument("--BK", type=int, default=128, help="next_pow2(K); NK must be 1")
    p.add_argument("--BV", type=int, default=32, help="min(next_pow2(V),32)")
    p.add_argument("--SOFTPLUS_THRESHOLD", type=float, default=20.0)
    p.add_argument("--interleaved_ba", type=str_to_bool, default=False,
                   help="a/b live interleaved in one [B, 2*HV] buffer (in_proj ba merge)")
    p.add_argument("--USE_QK_L2NORM_IN_KERNEL", type=str_to_bool, default=True,
                   help="feed raw q/k (kernel does L2norm); matches fwd kernel")
    p.add_argument("--num_warps", type=int, default=1)
    p.add_argument("--num_stages", type=int, default=3)
    p.add_argument("--compute_type", type=str, required=True, choices=["BF16"])
    p.add_argument("--output_dir", type=str, required=True)
    return p.parse_args()


def torch_dtype_from_str(s: str) -> torch.dtype:
    return torch.bfloat16


def main() -> int:
    args = parse_args()
    set_global_seed(42)
    dtype = torch_dtype_from_str(args.compute_type)
    device = torch.device("cuda")

    B = args.B
    H, HV, K, V = args.H, args.HV, args.K, args.V
    BK, BV = args.BK, args.BV
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, "packed_decode requires NK==1 (BK>=K)"

    scale = 1.0 / (K ** 0.5)

    # packed mixed_qkv layout: [q(H*K) | k(H*K) | v(HV*V)] per token.
    conv_dim = 2 * H * K + HV * V
    mixed_qkv = torch.randn((B, conv_dim), dtype=dtype, device=device)
    if args.interleaved_ba:
        # a/b share one [B, 2*HV] buffer: b = ba[:, :HV], a = ba[:, HV:].
        # Slicing inherits the parent stride, so stride_a_tok = stride_b_tok = 2*HV
        # (matches the production ba GEMM output layout).
        ba = torch.randn((B, 2 * HV), dtype=dtype, device=device)
        a = ba[:, HV:]
        b = ba[:, :HV]
    else:
        a = torch.randn((B, HV), dtype=dtype, device=device)
        b = torch.randn((B, HV), dtype=dtype, device=device)
    A_log_dtype = resolve("packed_decode", "A_log", dtype)
    A_log = torch.randn((HV,), dtype=A_log_dtype, device=device)
    # dt_bias stays FP32 to match A_log: the cubin loads it with a fp32
    # ld.global and computes softplus(a + dt_bias) in fp32 (matches HF, which
    # keeps dt_bias as a fp32 buffer even though the 0.8B checkpoint stores it
    # as BF16). A BF16 dt_bias ptr loses gate precision and mismatches the
    # fp32-load cubin contract. weight_loader force-casts dt_bias to FP32 too.
    dt_bias_dtype = resolve("packed_decode", "dt_bias", dtype)
    dt_bias = torch.randn((HV,), dtype=dt_bias_dtype, device=device)
    out = torch.empty((B, 1, HV, V), dtype=dtype, device=device)

    # fp32 state pool [2, HV, V, K]：slot 0 = NULL_BLOCK_ID，slot 1 有效；
    # h0 == ht（INPLACE）。slot 行距运行时传入，dummy 用小形状即可。
    state_pool = torch.randn((2, HV, V, K), dtype=torch.float32, device=device)
    h0 = state_pool
    ht = state_pool

    # ssm_state_indices [B] int32 = {1} -> use slot 1 (kernel: state_idx>0).
    ssm_state_indices = torch.tensor([1] * B, dtype=torch.int32, device=device)

    stride_mixed_qkv_tok = mixed_qkv.stride(0)
    stride_a_tok = a.stride(0)
    stride_b_tok = b.stride(0)
    stride_init_state_token = h0.stride(0)
    stride_final_state_token = ht.stride(0)
    stride_indices_seq = ssm_state_indices.stride(0)

    # grid = (NV, B * HV); i_v = program_id(0), i_nh = program_id(1).
    grid = (NV, B * HV)

    kernel = _jit_kernel[grid](
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        o=out,
        h0=h0,
        ht=ht,
        ssm_state_indices=ssm_state_indices,
        scale=scale,
        stride_mixed_qkv_tok=stride_mixed_qkv_tok,
        stride_a_tok=stride_a_tok,
        stride_b_tok=stride_b_tok,
        stride_init_state_token=stride_init_state_token,
        stride_final_state_token=stride_final_state_token,
        stride_indices_seq=stride_indices_seq,
        H=H, HV=HV, K=K, V=V, BK=BK, BV=BV,
        SOFTPLUS_THRESHOLD=args.SOFTPLUS_THRESHOLD,
        USE_QK_L2NORM_IN_KERNEL=args.USE_QK_L2NORM_IN_KERNEL,
        num_warps=args.num_warps,
        num_stages=args.num_stages,
    )
    torch.cuda.synchronize()

    kernel_name = (
        f"fused_recurrent_gated_delta_rule_packed_decode_kernel"
        f"_SMQ_{stride_mixed_qkv_tok}"
        f"_SA_{stride_a_tok}"
        f"_SB_{stride_b_tok}"
        f"_SISQ_{stride_indices_seq}"
        f"_H_{H}_HV_{HV}_K_{K}_V_{V}_BK_{BK}_BV_{BV}"
        f"_SPT_{args.SOFTPLUS_THRESHOLD}"
        f"_L2N_{int(bool(args.USE_QK_L2NORM_IN_KERNEL))}"
        f"_W_{args.num_warps}_S_{args.num_stages}"
        f"_compute_type_{args.compute_type}"
    )

    os.makedirs(args.output_dir, exist_ok=True)
    config = {"config": {"num_warps": args.num_warps, "num_stages": args.num_stages}}
    dump_kernel(kernel, args.output_dir, kernel_name, config)
    print(f"[packed_decode] dumped {kernel_name}.cubin -> {args.output_dir}")
    print(f"  shm_size={kernel.metadata.shared} grid={grid} warps={args.num_warps} stages={args.num_stages}")
    print(f"  conv_dim={conv_dim} stride_init_state_token={stride_init_state_token} NV={NV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
