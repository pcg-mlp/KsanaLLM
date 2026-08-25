# Copyright 2026 Tencent Inc. All rights reserved.
#
# Copied-code copyright (kernel bodies below, vllm v0.21.0, Apache-2.0):
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Fused conv+post prefill kernels: causal-conv1d prefill + fused_post_conv in
# one launch (main tile + boundary correction). Kernel bodies adapted from
# these vllm v0.21.0 files:
#   https://github.com/vllm-project/vllm/blob/v0.21.0/vllm/model_executor/layers/mamba/ops/causal_conv1d.py
#   https://github.com/vllm-project/vllm/blob/v0.21.0/vllm/model_executor/layers/fla/ops/fused_gdn_prefill_post_conv.py
# (the causal-conv1d part is itself derived from Tri Dao's causal-conv1d:
#   https://github.com/Dao-AILab/causal-conv1d, Copyright (c) 2024, Tri Dao.)
#
# Baked-by-specialization args MUST match production values, else the cubin's
# param slot count changes and the C++ num_args (25/15) misaligns:
#   - debug_yq_ptr = None
#   - stride_x_dim / stride_w_width / stride_istate_token = 1 (triton 缺省
#     对值=1 的 int 参数特化；其余 stride 值为非 1 → 运行时槽)
# ============================================================================
"""Offline cubin generator for `fused_conv_post_c1_kernel` + `fused_conv_post_c1_boundary_kernel` (C1)."""

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

from gdn._compat import tl, triton  # noqa: E402
from gdn.dtype_contract import resolve  # noqa: E402

NUM_CACHE_LINES = 2049  # kGdnNumCacheLines = 2048 live slots + slot 0


@triton.jit
def _conv_tile(
    x_ptr, w_ptr, conv_states_ptr, cache_indices_ptr, has_initial_states_ptr,
    query_start_loc_ptr,
    offs_t, tile_start, d0, offs_k, mask_dim,
    stride_x_dim, stride_x_token, stride_w_dim, stride_w_width,
    stride_istate_seq, stride_istate_dim, stride_istate_token,
    stride_cache_indices,
    seqlen,
    KW: tl.constexpr, SL: tl.constexpr, BLOCK_T: tl.constexpr, NSEQ: tl.constexpr,
):
    """4-tap conv over [BLOCK_T, BK] tile at dim offset d0, returns fp32 acc (pre-silu).

    j 显式展开：x0 = x[t-3], x1 = x[t-2], x2 = x[t-1], x3 = x[t]（KW=4）。
    """
    # ---- 4 次 [BT, BK] load（j 显式展开，prior 修正只需 1 遍 s 循环）----
    m0 = (offs_t[:, None] - 3 >= 0) & (offs_t[:, None] - 3 < seqlen) & mask_dim[None, :]
    m1 = (offs_t[:, None] - 2 >= 0) & (offs_t[:, None] - 2 < seqlen) & mask_dim[None, :]
    m2 = (offs_t[:, None] - 1 >= 0) & (offs_t[:, None] - 1 < seqlen) & mask_dim[None, :]
    m3 = (offs_t[:, None] < seqlen) & mask_dim[None, :]  # x[t]：t 边界
    x0 = tl.load(x_ptr + (d0 + offs_k)[None, :] * stride_x_dim
                 + (offs_t - 3)[:, None] * stride_x_token, mask=m0, other=0.0)
    x1 = tl.load(x_ptr + (d0 + offs_k)[None, :] * stride_x_dim
                 + (offs_t - 2)[:, None] * stride_x_token, mask=m1, other=0.0)
    x2 = tl.load(x_ptr + (d0 + offs_k)[None, :] * stride_x_dim
                 + (offs_t - 1)[:, None] * stride_x_token, mask=m2, other=0.0)
    x3 = tl.load(x_ptr + (d0 + offs_k)[None, :] * stride_x_dim
                 + offs_t[:, None] * stride_x_token, mask=m3, other=0.0)
    w0 = tl.load(w_ptr + (d0 + offs_k) * stride_w_dim, mask=mask_dim, other=0.0)
    w1 = tl.load(w_ptr + (d0 + offs_k) * stride_w_dim + 1 * stride_w_width, mask=mask_dim, other=0.0)
    w2 = tl.load(w_ptr + (d0 + offs_k) * stride_w_dim + 2 * stride_w_width, mask=mask_dim, other=0.0)
    w3 = tl.load(w_ptr + (d0 + offs_k) * stride_w_dim + 3 * stride_w_width, mask=mask_dim, other=0.0)

    # ---- prior 修正：仅 NSEQ>0 编译期启用（v4 主 kernel NSEQ=0，无此逻辑）----
    if NSEQ > 0:
        idx16 = tl.arange(0, 16)
        qsl16 = tl.load(query_start_loc_ptr + idx16, mask=idx16 <= NSEQ, other=0)
        any_hit = tl.max(tl.where((qsl16 >= tile_start) & (qsl16 < tile_start + BLOCK_T), 1, 0)) > 0
        if any_hit:
            for s in tl.static_range(NSEQ):
                start_s = tl.sum(tl.where(idx16 == s, qsl16, 0))
                end_s = tl.sum(tl.where(idx16 == s + 1, qsl16, 0))
                if (start_s >= tile_start) & (start_s < tile_start + BLOCK_T):
                    slot_s = tl.load(cache_indices_ptr + s * stride_cache_indices).to(tl.int64)
                    has_init = tl.load(has_initial_states_ptr + s).to(tl.int1)
                    i_rel = offs_t - start_s  # [BT]
                    # j=0 (lag 3): tok = 2 - i_rel；j=1 (lag 2): tok = 1 - i_rel；j=2 (lag 1): tok = 0 - i_rel
                    for j in tl.static_range(3):
                        lag = 3 - j  # KW-1-j（KW=4）
                        need_prior = (i_rel < lag) & (i_rel >= 0) & (offs_t < end_s)  # [BT]
                        tok_idx = lag - 1 - i_rel  # i_rel<lag 时 ∈ {0..KW-2}
                        pool_val = tl.load(
                            conv_states_ptr + slot_s * stride_istate_seq
                            + (d0 + offs_k)[None, :] * stride_istate_dim
                            + tok_idx[:, None] * stride_istate_token,
                            mask=need_prior[:, None] & (tok_idx[:, None] >= 0) & mask_dim[None, :],
                            other=0.0)
                        prior = tl.where(has_init, pool_val, 0.0)
                        if j == 0:
                            x0 = tl.where(need_prior[:, None], prior, x0)
                        elif j == 1:
                            x1 = tl.where(need_prior[:, None], prior, x1)
                        else:
                            x2 = tl.where(need_prior[:, None], prior, x2)

    # bf16 乘积（有损，同现链 bf16×bf16）+ fp32 累加（同现链 acc fp32 += 逐项）
    acc = ((x0 * w0[None, :]).to(tl.float32) + (x1 * w1[None, :]).to(tl.float32)
           + (x2 * w2[None, :]).to(tl.float32) + (x3 * w3[None, :]).to(tl.float32))
    return acc


@triton.jit
def _write_seq_state(
    x_ptr, conv_states_ptr,
    d0, offs_k, mask_dim,
    stride_x_dim, stride_x_token, stride_istate_seq, stride_istate_dim,
    stride_istate_token, start_s, end_s, slot_s,
    SL: tl.constexpr,
):
    """seq 起点 tile 的 pool 写回：pool[slot_s, d0+offs_k, 0..SL-1] = seq 末 SL 个 pre-conv token。

    NP2=4 行 tile（同生产 NP2_STATELEN），store 只写 idx < SL 行；越界行写 0（同生产
    load mask 语义）。"""
    idx_c = tl.arange(0, 4)  # NP2=4
    len_s = end_s - start_s
    rel_last = len_s - SL + idx_c  # 4 行（seq 内相对）
    abs_last = start_s + rel_last
    m_load = (rel_last[:, None] >= 0) & (rel_last[:, None] < len_s) & mask_dim[None, :]
    x_last = tl.load(x_ptr + (d0 + offs_k)[None, :] * stride_x_dim
                     + abs_last[:, None] * stride_x_token, mask=m_load, other=0.0)
    tl.store(conv_states_ptr + slot_s * stride_istate_seq
             + (d0 + offs_k)[None, :] * stride_istate_dim
             + idx_c[:, None] * stride_istate_token, x_last,
             mask=(idx_c[:, None] < SL) & mask_dim[None, :])


@triton.jit
def fused_conv_post_c1_boundary_kernel(
    x_ptr, w_ptr, conv_states_ptr, cache_indices_ptr, has_initial_states_ptr,
    query_start_loc_ptr,
    q_ptr, k_ptr, v_ptr,
    stride_x_dim, stride_x_token, stride_w_dim, stride_w_width,
    stride_istate_seq, stride_istate_dim, stride_istate_token,
    stride_cache_indices,
    seqlen,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    KW: tl.constexpr, SL: tl.constexpr,
    APPLY_L2NORM: tl.constexpr, L2NORM_EPS: tl.constexpr,
    BLOCK_T: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
):
    """C1-v4 boundary：每 (seq s, 段) program 重算 [start_s, end_s) 行的 conv（含 prior 修正）
    并覆写 q/k/v；同时写回该 seq 的 pool。主 kernel 无 s-loop（NSEQ=0），两者串行，
    覆写区间 [start_s, end_s) 跨 seq 不相交 → 无竞态。"""
    i_s = tl.program_id(0)
    i_prog = tl.program_id(1)
    start_s = tl.load(query_start_loc_ptr + i_s)
    end_s = tl.load(query_start_loc_ptr + i_s + 1)
    slot_s = tl.load(cache_indices_ptr + i_s * stride_cache_indices).to(tl.int64)
    has_init = tl.load(has_initial_states_ptr + i_s).to(tl.int1)
    tile_start = (start_s // BLOCK_T) * BLOCK_T
    offs_t = tile_start + tl.arange(0, BLOCK_T)
    mask_t = (offs_t >= start_s) & (offs_t < end_s)  # 本 seq 行区间
    i_rel = offs_t - start_s
    HK: tl.constexpr = H * K
    V_OFFSET: tl.constexpr = 2 * H * K

    if i_prog < 2 * H:
        # ---- Q/K 单段 ----
        i_h = i_prog // 2
        is_k = i_prog % 2
        d0 = i_h * K + is_k * HK
        offs_k = tl.arange(0, BK)
        mask_dim = offs_k < K
        mask_2d = mask_t[:, None] & mask_dim[None, :]
        m0 = (offs_t[:, None] - 3 >= 0) & mask_dim[None, :]
        m1 = (offs_t[:, None] - 2 >= 0) & mask_dim[None, :]
        m2 = (offs_t[:, None] - 1 >= 0) & mask_dim[None, :]
        m3 = mask_dim[None, :]
        x0 = tl.load(x_ptr + (d0 + offs_k)[None, :] * stride_x_dim + (offs_t - 3)[:, None] * stride_x_token,
                     mask=m0, other=0.0)
        x1 = tl.load(x_ptr + (d0 + offs_k)[None, :] * stride_x_dim + (offs_t - 2)[:, None] * stride_x_token,
                     mask=m1, other=0.0)
        x2 = tl.load(x_ptr + (d0 + offs_k)[None, :] * stride_x_dim + (offs_t - 1)[:, None] * stride_x_token,
                     mask=m2, other=0.0)
        x3 = tl.load(x_ptr + (d0 + offs_k)[None, :] * stride_x_dim + offs_t[:, None] * stride_x_token,
                     mask=m3, other=0.0)
        w0 = tl.load(w_ptr + (d0 + offs_k) * stride_w_dim, mask=mask_dim, other=0.0)
        w1 = tl.load(w_ptr + (d0 + offs_k) * stride_w_dim + stride_w_width, mask=mask_dim, other=0.0)
        w2 = tl.load(w_ptr + (d0 + offs_k) * stride_w_dim + 2 * stride_w_width, mask=mask_dim, other=0.0)
        w3 = tl.load(w_ptr + (d0 + offs_k) * stride_w_dim + 3 * stride_w_width, mask=mask_dim, other=0.0)
        # prior 修正（单 seq，起点已知，无 s 循环）
        for j in tl.static_range(3):
            lag = 3 - j  # KW-1-j
            need_prior = (i_rel < lag) & (i_rel >= 0) & (offs_t < end_s)  # [BT]
            tok_idx = i_rel + (3 - lag)  # 现链语义: tap j 用 pool[j+i_rel]（j=3-lag）
            pool_val = tl.load(
                conv_states_ptr + slot_s * stride_istate_seq
                + (d0 + offs_k)[None, :] * stride_istate_dim
                + tok_idx[:, None] * stride_istate_token,
                mask=need_prior[:, None] & (tok_idx[:, None] >= 0) & mask_dim[None, :],
                other=0.0)
            prior = tl.where(has_init, pool_val, 0.0)
            if j == 0:
                x0 = tl.where(need_prior[:, None], prior, x0)
            elif j == 1:
                x1 = tl.where(need_prior[:, None], prior, x1)
            else:
                x2 = tl.where(need_prior[:, None], prior, x2)
        acc = ((x0 * w0[None, :]).to(tl.float32) + (x1 * w1[None, :]).to(tl.float32)
               + (x2 * w2[None, :]).to(tl.float32) + (x3 * w3[None, :]).to(tl.float32))
        y = (acc / (1.0 + tl.exp(-acc))).to(tl.bfloat16).to(tl.float32)
        if APPLY_L2NORM:
            sq_sum = tl.sum(y * y, axis=1)
            y = y * (1.0 / tl.sqrt(sq_sum + L2NORM_EPS))[:, None]
        if is_k:
            tl.store(k_ptr + offs_t[:, None] * HK + i_h * K + offs_k[None, :],
                     y.to(tl.bfloat16), mask=mask_2d)
        else:
            tl.store(q_ptr + offs_t[:, None] * HK + i_h * K + offs_k[None, :],
                     y.to(tl.bfloat16), mask=mask_2d)
        _write_seq_state(x_ptr, conv_states_ptr, d0, offs_k, mask_dim,
                         stride_x_dim, stride_x_token, stride_istate_seq, stride_istate_dim,
                         stride_istate_token, start_s, end_s, slot_s, SL)
    else:
        # ---- V 段 ----
        i_hv = i_prog - 2 * H
        d0 = V_OFFSET + i_hv * V
        offs_v = tl.arange(0, BV)
        mask_v = offs_v < V
        mask_2d = mask_t[:, None] & mask_v[None, :]
        m0 = (offs_t[:, None] - 3 >= 0) & mask_v[None, :]
        m1 = (offs_t[:, None] - 2 >= 0) & mask_v[None, :]
        m2 = (offs_t[:, None] - 1 >= 0) & mask_v[None, :]
        m3 = mask_v[None, :]
        x0 = tl.load(x_ptr + (d0 + offs_v)[None, :] * stride_x_dim + (offs_t - 3)[:, None] * stride_x_token,
                     mask=m0, other=0.0)
        x1 = tl.load(x_ptr + (d0 + offs_v)[None, :] * stride_x_dim + (offs_t - 2)[:, None] * stride_x_token,
                     mask=m1, other=0.0)
        x2 = tl.load(x_ptr + (d0 + offs_v)[None, :] * stride_x_dim + (offs_t - 1)[:, None] * stride_x_token,
                     mask=m2, other=0.0)
        x3 = tl.load(x_ptr + (d0 + offs_v)[None, :] * stride_x_dim + offs_t[:, None] * stride_x_token,
                     mask=m3, other=0.0)
        w0 = tl.load(w_ptr + (d0 + offs_v) * stride_w_dim, mask=mask_v, other=0.0)
        w1 = tl.load(w_ptr + (d0 + offs_v) * stride_w_dim + stride_w_width, mask=mask_v, other=0.0)
        w2 = tl.load(w_ptr + (d0 + offs_v) * stride_w_dim + 2 * stride_w_width, mask=mask_v, other=0.0)
        w3 = tl.load(w_ptr + (d0 + offs_v) * stride_w_dim + 3 * stride_w_width, mask=mask_v, other=0.0)
        for j in tl.static_range(3):
            lag = 3 - j
            need_prior = (i_rel < lag) & (i_rel >= 0) & (offs_t < end_s)
            tok_idx = i_rel + (3 - lag)
            pool_val = tl.load(
                conv_states_ptr + slot_s * stride_istate_seq
                + (d0 + offs_v)[None, :] * stride_istate_dim
                + tok_idx[:, None] * stride_istate_token,
                mask=need_prior[:, None] & (tok_idx[:, None] >= 0) & mask_v[None, :],
                other=0.0)
            prior = tl.where(has_init, pool_val, 0.0)
            if j == 0:
                x0 = tl.where(need_prior[:, None], prior, x0)
            elif j == 1:
                x1 = tl.where(need_prior[:, None], prior, x1)
            else:
                x2 = tl.where(need_prior[:, None], prior, x2)
        acc = ((x0 * w0[None, :]).to(tl.float32) + (x1 * w1[None, :]).to(tl.float32)
               + (x2 * w2[None, :]).to(tl.float32) + (x3 * w3[None, :]).to(tl.float32))
        yv = (acc / (1.0 + tl.exp(-acc))).to(tl.bfloat16)
        tl.store(v_ptr + offs_t[:, None] * (HV * V) + i_hv * V + offs_v[None, :],
                 yv, mask=mask_2d)
        _write_seq_state(x_ptr, conv_states_ptr, d0, offs_v, mask_v,
                         stride_x_dim, stride_x_token, stride_istate_seq, stride_istate_dim,
                         stride_istate_token, start_s, end_s, slot_s, SL)


@triton.jit
def fused_conv_post_c1_kernel(
    # ---- inputs ----
    x_ptr,  # [D, L] conv raw input (dim-major view of mixed_qkv)
    w_ptr,  # [D, KW]
    conv_states_ptr,  # pool [num_cache_lines, D, SL]
    cache_indices_ptr,  # [N, 8], [s,0] = slot_s
    has_initial_states_ptr,  # [N] bool
    query_start_loc_ptr,  # [N+1] int32
    a_ptr,  # [L, HV]
    b_ptr,  # [L, HV]
    A_log_ptr,  # [HV]
    dt_bias_ptr,  # [HV]
    # ---- outputs ----
    q_ptr,  # [L, H, K] bf16
    k_ptr,  # [L, H, K] bf16
    v_ptr,  # [L, HV, V] bf16
    g_ptr,  # [L, HV] fp32
    beta_ptr,  # [L, HV] fp32
    debug_yq_ptr,  # [L, H*?+V?] fp32 (L2norm 前，debug 用；NULL 时跳过)
    # ---- strides ----
    stride_x_dim,  # 1
    stride_x_token,  # D (int64)
    stride_w_dim,  # 1
    stride_w_width,  # D
    stride_istate_seq,  # pool slot stride (D*SL bf16 elems)
    stride_istate_dim,  # SL
    stride_istate_token,  # 1
    stride_cache_indices,  # 8
    stride_a_tok,  # HV
    stride_b_tok,  # HV
    # ---- dims ----
    dim,  # conv_dim
    seqlen,  # total tokens
    num_cache_lines,
    # ---- constexpr ----
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    KW: tl.constexpr,
    SL: tl.constexpr,
    APPLY_L2NORM: tl.constexpr,
    L2NORM_EPS: tl.constexpr,
    OUTPUT_G_EXP: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NSEQ: tl.constexpr,
):
    i_tb = tl.program_id(0)
    i_prog = tl.program_id(1)
    tile_start = i_tb * BLOCK_T
    offs_t = tile_start + tl.arange(0, BLOCK_T)
    mask_t = offs_t < seqlen
    HK: tl.constexpr = H * K
    V_OFFSET: tl.constexpr = 2 * H * K

    if i_prog < 2 * H:
        # ================= Q/K single segment =================
        i_h = i_prog // 2
        is_k = i_prog % 2  # 0 → Q 段，1 → K 段
        d0 = i_h * K + is_k * HK
        offs_k = tl.arange(0, BK)
        mask_dim = offs_k < K
        mask_2d = mask_t[:, None] & mask_dim[None, :]

        acc = _conv_tile(x_ptr, w_ptr, conv_states_ptr, cache_indices_ptr,
                         has_initial_states_ptr, query_start_loc_ptr,
                         offs_t, tile_start, d0, offs_k, mask_dim,
                         stride_x_dim, stride_x_token, stride_w_dim, stride_w_width,
                         stride_istate_seq, stride_istate_dim, stride_istate_token,
                         stride_cache_indices, seqlen, KW, SL, BLOCK_T, NSEQ)

        # silu（同 causal_conv1d.py:473 算式）+ bf16 量化点（约束 1）→ fp32
        y = (acc / (1.0 + tl.exp(-acc))).to(tl.bfloat16).to(tl.float32)

        if debug_yq_ptr is not None:
            if is_k == 0:
                tl.store(debug_yq_ptr + offs_t[:, None] * (H * K) + i_h * K + offs_k[None, :],
                         y, mask=mask_2d)

        if APPLY_L2NORM:
            sq_sum = tl.sum(y * y, axis=1)
            inv = 1.0 / tl.sqrt(sq_sum + L2NORM_EPS)
            y = y * inv[:, None]

        if is_k:
            tl.store(k_ptr + offs_t[:, None] * (H * K) + i_h * K + offs_k[None, :],
                     y.to(tl.bfloat16), mask=mask_2d)
        else:
            tl.store(q_ptr + offs_t[:, None] * (H * K) + i_h * K + offs_k[None, :],
                     y.to(tl.bfloat16), mask=mask_2d)

        # ---- pool 写回：seq 起点 tile（该段列；qsl 向量 load + any_hit 门控）----
        idx16 = tl.arange(0, 16)
        qsl16 = tl.load(query_start_loc_ptr + idx16, mask=idx16 <= NSEQ, other=0)
        any_hit = tl.max(tl.where((qsl16 >= tile_start) & (qsl16 < tile_start + BLOCK_T), 1, 0)) > 0
        if any_hit:
            for s in tl.static_range(NSEQ):
                start_s = tl.sum(tl.where(idx16 == s, qsl16, 0))
                end_s = tl.sum(tl.where(idx16 == s + 1, qsl16, 0))
                if (start_s >= tile_start) & (start_s < tile_start + BLOCK_T):
                    slot_s = tl.load(cache_indices_ptr + s * stride_cache_indices).to(tl.int64)
                    _write_seq_state(x_ptr, conv_states_ptr, d0, offs_k, mask_dim,
                                     stride_x_dim, stride_x_token, stride_istate_seq,
                                     stride_istate_dim, stride_istate_token,
                                     start_s, end_s, slot_s, SL)
    else:
        # ================= V head + gating =================
        i_hv = i_prog - 2 * H
        offs_v = tl.arange(0, BV)
        mask_v = offs_v < V
        mask_2d = mask_t[:, None] & mask_v[None, :]

        acc_v = _conv_tile(x_ptr, w_ptr, conv_states_ptr, cache_indices_ptr,
                           has_initial_states_ptr, query_start_loc_ptr,
                           offs_t, tile_start, V_OFFSET + i_hv * V, offs_v, mask_v,
                           stride_x_dim, stride_x_token, stride_w_dim, stride_w_width,
                           stride_istate_seq, stride_istate_dim, stride_istate_token,
                           stride_cache_indices, seqlen, KW, SL, BLOCK_T, NSEQ)
        yv = (acc_v / (1.0 + tl.exp(-acc_v))).to(tl.bfloat16)
        tl.store(v_ptr + offs_t[:, None] * (HV * V) + i_hv * V + offs_v[None, :],
                 yv, mask=mask_2d)

        # gating（与 fused_gdn_prefill_post_conv.py:138-159 逐符号一致）
        A_log_val = tl.load(A_log_ptr + i_hv).to(tl.float32)
        dt_bias_val = tl.load(dt_bias_ptr + i_hv).to(tl.float32)
        a_vals = tl.load(a_ptr + offs_t * stride_a_tok + i_hv, mask=mask_t, other=0.0).to(tl.float32)
        b_vals = tl.load(b_ptr + offs_t * stride_b_tok + i_hv, mask=mask_t, other=0.0).to(tl.float32)
        x = a_vals + dt_bias_val
        sp = tl.where(x > 0, x + tl.log(1.0 + tl.exp(-x)), tl.log(1.0 + tl.exp(x)))
        sp = tl.where(x <= SOFTPLUS_THRESHOLD, sp, x)
        g_vals = -tl.exp(A_log_val) * sp
        if OUTPUT_G_EXP:
            g_vals = tl.exp(g_vals)
        beta_vals = tl.sigmoid(b_vals)
        gb_offsets = offs_t * HV + i_hv
        tl.store(g_ptr + gb_offsets, g_vals, mask=mask_t)
        tl.store(beta_ptr + gb_offsets, beta_vals, mask=mask_t)

        # ---- pool 写回：seq 起点 tile（V 段）----
        idx16 = tl.arange(0, 16)
        qsl16 = tl.load(query_start_loc_ptr + idx16, mask=idx16 <= NSEQ, other=0)
        any_hit = tl.max(tl.where((qsl16 >= tile_start) & (qsl16 < tile_start + BLOCK_T), 1, 0)) > 0
        if any_hit:
            for s in tl.static_range(NSEQ):
                start_s = tl.sum(tl.where(idx16 == s, qsl16, 0))
                end_s = tl.sum(tl.where(idx16 == s + 1, qsl16, 0))
                if (start_s >= tile_start) & (start_s < tile_start + BLOCK_T):
                    slot_s = tl.load(cache_indices_ptr + s * stride_cache_indices).to(tl.int64)
                    _write_seq_state(x_ptr, conv_states_ptr, V_OFFSET + i_hv * V, offs_v, mask_v,
                                     stride_x_dim, stride_x_token, stride_istate_seq,
                                     stride_istate_dim, stride_istate_token,
                                     start_s, end_s, slot_s, SL)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # Qwen3.5/3.6 numerics: head_k_dim=128 head_v_dim=128, H=num_key_heads,
    # HV=num_value_heads. KW=4/SL=3 固定（非 MTP 生产几何）。
    p.add_argument("--H", type=int, required=True)
    p.add_argument("--HV", type=int, required=True)
    p.add_argument("--K", type=int, default=128)
    p.add_argument("--V", type=int, default=128)
    p.add_argument("--BLOCK_T", type=int, required=True)
    p.add_argument("--NSEQ", type=int, default=0)
    p.add_argument("--boundary", action="store_true",
                   help="dump boundary kernel instead of the main kernel")
    p.add_argument("--num_warps", type=int, default=4)
    p.add_argument("--num_stages", type=int, default=2)
    p.add_argument("--compute_type", type=str, required=True,
                   choices=["BF16"])
    p.add_argument("--output_dir", type=str, required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    set_global_seed(42)
    dtype = torch.bfloat16
    device = torch.device("cuda")

    H, HV, K, V = args.H, args.HV, args.K, args.V
    D = 2 * H * K + HV * V  # mixed_qkv 宽度 = 生产 conv_dim_（27B 5120 / 35B 4096）
    L = 256
    N = 1
    SL = 3  # KW - 1（非 MTP）

    # dummy 布局与生产逐项一致（烘焙特化依赖值=1 的 stride，见文件头注释）：
    #   x: 物理连续 [L, D] 的 [D, L] 视图（stride (1, D)），同生产 mixed_qkv
    #   w: [D, KW] 连续；conv_states: [num_cache_lines, D, SL] 连续
    #   cache_indices: [N, 8] int32；has_initial: [N] bool；qsl: [N+1] int32
    x_buf = torch.randn((L, D), dtype=dtype, device=device) * 0.1
    x = x_buf.t()
    w = torch.randn((D, 4), dtype=dtype, device=device) * 0.2
    conv_states = torch.zeros((NUM_CACHE_LINES, D, SL), dtype=dtype, device=device)
    cache_indices = torch.zeros((N, 8), dtype=torch.int32, device=device)
    cache_indices[0, 0] = 1  # slot 0 = NULL block，slots 从 1 起
    has_initial = torch.zeros((N,), dtype=torch.bool, device=device)
    query_start_loc = torch.tensor([0, L], dtype=torch.int32, device=device)
    a = torch.randn((L, HV), dtype=dtype, device=device) * 0.1
    b = torch.randn((L, HV), dtype=dtype, device=device) * 0.1
    A_log_dtype = resolve("fused_conv_post_c1", "A_log", dtype)
    dt_bias_dtype = resolve("fused_conv_post_c1", "dt_bias", dtype)
    A_log = torch.randn((HV,), dtype=A_log_dtype, device=device) * 0.1
    dt_bias = torch.randn((HV,), dtype=dt_bias_dtype, device=device) * 0.1
    q = torch.empty((L, H, K), dtype=dtype, device=device)
    k = torch.empty((L, H, K), dtype=dtype, device=device)
    v = torch.empty((L, HV, V), dtype=dtype, device=device)
    g_dtype = resolve("fused_conv_post_c1", "g", dtype)
    beta_dtype = resolve("fused_conv_post_c1", "beta", dtype)
    g = torch.empty((L, HV), dtype=g_dtype, device=device)
    beta = torch.empty((L, HV), dtype=beta_dtype, device=device)

    if args.boundary:
        kernel = fused_conv_post_c1_boundary_kernel[(N, 2 * H + HV)](
            x, w, conv_states, cache_indices, has_initial, query_start_loc,
            q, k, v,
            stride_x_dim=x.stride(0), stride_x_token=x.stride(1),
            stride_w_dim=w.stride(0), stride_w_width=w.stride(1),
            stride_istate_seq=conv_states.stride(0),
            stride_istate_dim=conv_states.stride(1),
            stride_istate_token=conv_states.stride(2),
            stride_cache_indices=cache_indices.stride(0),
            seqlen=L,
            H=H, HV=HV, K=K, V=V, KW=4, SL=SL,
            APPLY_L2NORM=True, L2NORM_EPS=1e-6,
            BLOCK_T=args.BLOCK_T, BK=128, BV=128,
            num_warps=args.num_warps, num_stages=args.num_stages,
        )
        torch.cuda.synchronize()
        kernel_name = (
            f"fused_conv_post_c1_boundary_kernel"
            f"_H_{H}_HV_{HV}_K_{K}_V_{V}"
            f"_BT_{args.BLOCK_T}_L2_1"
            f"_W_{args.num_warps}_S_{args.num_stages}"
            f"_compute_type_{args.compute_type}"
        )
    else:
        grid = (triton.cdiv(L, args.BLOCK_T), 2 * H + HV)
        kernel = fused_conv_post_c1_kernel[grid](
            x, w, conv_states, cache_indices, has_initial, query_start_loc,
            a, b, A_log, dt_bias,
            q, k, v, g, beta, None,
            stride_x_dim=x.stride(0), stride_x_token=x.stride(1),
            stride_w_dim=w.stride(0), stride_w_width=w.stride(1),
            stride_istate_seq=conv_states.stride(0),
            stride_istate_dim=conv_states.stride(1),
            stride_istate_token=conv_states.stride(2),
            stride_cache_indices=cache_indices.stride(0),
            stride_a_tok=a.stride(0), stride_b_tok=b.stride(0),
            dim=D, seqlen=L, num_cache_lines=NUM_CACHE_LINES,
            H=H, HV=HV, K=K, V=V, KW=4, SL=SL,
            APPLY_L2NORM=True, L2NORM_EPS=1e-6,
            OUTPUT_G_EXP=False, SOFTPLUS_THRESHOLD=20.0,
            BLOCK_T=args.BLOCK_T, BK=128, BV=128, NSEQ=args.NSEQ,
            num_warps=args.num_warps, num_stages=args.num_stages,
            maxnreg=None,
        )
        torch.cuda.synchronize()
        kernel_name = (
            f"fused_conv_post_c1_kernel"
            f"_H_{H}_HV_{HV}_K_{K}_V_{V}"
            f"_BT_{args.BLOCK_T}_L2_1_GEXP_0_NSEQ_{args.NSEQ}"
            f"_W_{args.num_warps}_S_{args.num_stages}"
            f"_compute_type_{args.compute_type}"
        )

    # 生成期实参数量自检：编译产物 signature 中非 constexpr 条目数即 C++
    # 调用侧必须传的槽位数（文件头 25/15 契约的可执行版本）。
    def _runtime_param_count(kernel):
        try:
            sig = kernel.src.signature
        except AttributeError:
            sig = kernel["src"].signature
        return sum(1 for t in sig.values() if t != "constexpr")

    expected = 15 if args.boundary else 25
    assert _runtime_param_count(kernel) == expected, \
        f"runtime param count mismatch: expected {expected}, " \
        f"got {_runtime_param_count(kernel)}"

    os.makedirs(args.output_dir, exist_ok=True)
    config = {
        "config": {
            "num_warps": args.num_warps,
            "num_stages": args.num_stages,
        }
    }
    dump_kernel(kernel, args.output_dir, kernel_name, config)
    print(f"[fused_conv_post_c1] dumped {kernel_name}.cubin "
          f"-> {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
