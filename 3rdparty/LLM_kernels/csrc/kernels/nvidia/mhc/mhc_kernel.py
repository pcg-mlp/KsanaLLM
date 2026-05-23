# Adapted from
# https://github.com/deepseek-ai/TileKernels/tree/main/tile_kernels/mhc
#
# Copyright (c) 2026 DeepSeek
# Copyright 2026 Tencent Inc.  All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import math

import tilelang
import torch
from tilelang import language as T


'''@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
        tilelang.PassConfigKey.TL_DISABLE_VECTORIZE_256: True,
    },
)'''
def _mhc_pre_big_fuse(
    hidden_size: int,
    rms_eps: float,
    mhc_pre_eps: float,
    mhc_sinkhorn_eps: float,
    mhc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 16,
    mhc_mult: int = 4,
):
    num_tokens = T.dynamic('num_tokens')
    mhc_mult3 = mhc_mult * (2 + mhc_mult)
    hidden_block = math.gcd(512, hidden_size)

    @T.prim_func
    def mhc_pre_big_fuse(
        gemm_out_mul: T.Tensor[(n_splits, num_tokens, mhc_mult3), T.float32],
        gemm_out_sqrsum: T.Tensor[(n_splits, num_tokens), T.float32],
        mhc_scale: T.Tensor[(3,), T.float32],
        mhc_base: T.Tensor[(mhc_mult3,), T.float32],
        residual: T.Tensor[(num_tokens, mhc_mult, hidden_size), T.bfloat16],
        # outputs
        post_mix: T.Tensor[(num_tokens, mhc_mult), T.float32],
        comb_mix: T.Tensor[(num_tokens, mhc_mult * mhc_mult), T.float32],
        layer_input: T.Tensor[(num_tokens, hidden_size), T.bfloat16],
    ) -> None:
        with T.Kernel(num_tokens, threads=96) as pid:
            ##################################################################
            # _mhc_pre_norm_fn_fwd_norm
            mixes_shared = T.alloc_shared(mhc_mult3, T.float32)
            T.pdl_sync()
            if T.get_thread_binding() < 32:
                rms = T.alloc_fragment(1, T.float32)
                mixes = T.alloc_fragment(mhc_mult3, T.float32)
                T.clear(mixes)
                rms[0] = 0
                for i_split in T.serial(n_splits):
                    rms[0] += gemm_out_sqrsum[i_split, pid]
                rms[0] = T.rsqrt(rms[0] / (mhc_mult * hidden_size) + rms_eps)
                for j in T.Parallel(mhc_mult3):
                    mixes[j] = 0
                    for i_split in T.serial(n_splits):
                        mixes[j] += gemm_out_mul[i_split, pid, j]
                    mixes[j] *= rms[0]
                T.copy(mixes, mixes_shared, disable_tma=True)

            if T.get_thread_binding() < 32:
                ##################################################################
                # _mhc_pre_split_mixes_fwd (post & comb)
                cm = T.alloc_fragment((mhc_mult, mhc_mult), T.float32)
                for j in T.Parallel(mhc_mult):
                    post_mix[pid, j] = (
                        T.sigmoid(mixes_shared[j + mhc_mult] * mhc_scale[1] + mhc_base[j + mhc_mult])
                        * mhc_post_mult_value
                    )
                for j, k in T.Parallel(mhc_mult, mhc_mult):
                    cm[j, k] = (
                        mixes_shared[j * mhc_mult + k + mhc_mult * 2] * mhc_scale[2]
                        + mhc_base[j * mhc_mult + k + mhc_mult * 2]
                    )

                ##################################################################
                # _mhc_sinkhorn_fwd
                row_sum = T.alloc_fragment(mhc_mult, T.float32)
                col_sum = T.alloc_fragment(mhc_mult, T.float32)

                # comb = comb.softmax(-1) + eps
                row_max = T.alloc_fragment(mhc_mult, T.float32)
                T.reduce_max(cm, row_max, dim=1)
                for j, k in T.Parallel(mhc_mult, mhc_mult):
                    cm[j, k] = T.exp(cm[j, k] - row_max[j])
                T.reduce_sum(cm, row_sum, dim=1)
                for j, k in T.Parallel(mhc_mult, mhc_mult):
                    cm[j, k] = cm[j, k] / row_sum[j] + mhc_sinkhorn_eps

                # comb = comb / (comb.sum(-2) + eps)
                T.reduce_sum(cm, col_sum, dim=0)
                for j, k in T.Parallel(mhc_mult, mhc_mult):
                    cm[j, k] = cm[j, k] / (col_sum[k] + mhc_sinkhorn_eps)

                for _ in T.serial(sinkhorn_repeat - 1):
                    # comb = comb / (comb.sum(-1) + eps)
                    T.reduce_sum(cm, row_sum, dim=1)
                    for j, k in T.Parallel(mhc_mult, mhc_mult):
                        cm[j, k] = cm[j, k] / (row_sum[j] + mhc_sinkhorn_eps)

                    # comb = comb / (comb.sum(-2) + eps)
                    T.reduce_sum(cm, col_sum, dim=0)
                    for j, k in T.Parallel(mhc_mult, mhc_mult):
                        cm[j, k] = cm[j, k] / (col_sum[k] + mhc_sinkhorn_eps)

                # save comb_mix to global memory
                for j, k in T.Parallel(mhc_mult, mhc_mult):
                    comb_mix[pid, j * mhc_mult + k] = cm[j, k]
            else:
                ##################################################################
                # _mhc_pre_split_mixes_fwd (pre)
                pre_mix_shared = T.alloc_shared(mhc_mult, T.float32)
                for j in T.Parallel(mhc_mult):
                    pre_mix_shared[j] = (
                        T.sigmoid(
                            mixes_shared[j] * mhc_scale[0] + mhc_base[j],
                        )
                        + mhc_pre_eps
                    )
                ###################################################################
                # _mhc_pre_apply_mix_fwd
                for i0_h in T.Pipelined(hidden_size // hidden_block, num_stages=2):
                    xs = T.alloc_shared((mhc_mult, hidden_block), T.bfloat16)
                    xl = T.alloc_fragment((mhc_mult, hidden_block), T.float32)
                    T.copy(residual[pid, 0, i0_h * hidden_block], xs, disable_tma=True)
                    T.copy(xs, xl, disable_tma=True)

                    ol = T.alloc_fragment(hidden_block, T.float32)
                    T.clear(ol)

                    for i_mhc in T.serial(mhc_mult):
                        pre = pre_mix_shared[i_mhc]
                        for i1_h in T.Parallel(hidden_block):
                            ol[i1_h] += pre * xl[i_mhc, i1_h]

                    T.copy(ol, layer_input[pid, i0_h * hidden_block], disable_tma=True)

    return mhc_pre_big_fuse


'''@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
        tilelang.PassConfigKey.TL_DISABLE_VECTORIZE_256: True,
    },
)'''
def _mhc_pre_big_fuse_with_norm(
    hidden_size: int,
    rms_eps: float,
    mhc_pre_eps: float,
    mhc_sinkhorn_eps: float,
    mhc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 16,
    mhc_mult: int = 4,
):
    """Fused mhc_pre big_fuse + RMSNorm of layer_input.

    Identical to _mhc_pre_big_fuse for the (post_mix, comb_mix) path.
    For the layer_input path, the weighted-sum result is stashed in shared
    memory while accumulating sum_sq, then a second pipelined sweep applies
    rsqrt(sum_sq/D + rms_eps) * norm_weight before writing to HBM.
    """
    num_tokens = T.dynamic('num_tokens')
    mhc_mult3 = mhc_mult * (2 + mhc_mult)
    hidden_block = math.gcd(1024, hidden_size)

    @T.prim_func
    def mhc_pre_big_fuse_with_norm(
        gemm_out_mul: T.Tensor[(n_splits, num_tokens, mhc_mult3), T.float32],
        gemm_out_sqrsum: T.Tensor[(n_splits, num_tokens), T.float32],
        mhc_scale: T.Tensor[(3,), T.float32],
        mhc_base: T.Tensor[(mhc_mult3,), T.float32],
        residual: T.Tensor[(num_tokens, mhc_mult, hidden_size), T.bfloat16],
        norm_weight: T.Tensor[(hidden_size,), T.bfloat16],
        # outputs
        post_mix: T.Tensor[(num_tokens, mhc_mult), T.float32],
        comb_mix: T.Tensor[(num_tokens, mhc_mult * mhc_mult), T.float32],
        layer_input: T.Tensor[(num_tokens, hidden_size), T.bfloat16],
    ) -> None:
        with T.Kernel(num_tokens, threads=96) as pid:
            ##################################################################
            # _mhc_pre_norm_fn_fwd_norm
            mixes_shared = T.alloc_shared(mhc_mult3, T.float32)
            T.pdl_sync()
            if T.get_thread_binding() < 32:
                rms = T.alloc_fragment(1, T.float32)
                mixes = T.alloc_fragment(mhc_mult3, T.float32)
                T.clear(mixes)
                rms[0] = 0
                for i_split in T.serial(n_splits):
                    rms[0] += gemm_out_sqrsum[i_split, pid]
                rms[0] = T.rsqrt(rms[0] / (mhc_mult * hidden_size) + rms_eps)
                for j in T.Parallel(mhc_mult3):
                    mixes[j] = 0
                    for i_split in T.serial(n_splits):
                        mixes[j] += gemm_out_mul[i_split, pid, j]
                    mixes[j] *= rms[0]
                T.copy(mixes, mixes_shared, disable_tma=True)

            if T.get_thread_binding() < 32:
                ##################################################################
                # _mhc_pre_split_mixes_fwd (post & comb)
                cm = T.alloc_fragment((mhc_mult, mhc_mult), T.float32)
                for j in T.Parallel(mhc_mult):
                    post_mix[pid, j] = (
                        T.sigmoid(mixes_shared[j + mhc_mult] * mhc_scale[1] + mhc_base[j + mhc_mult])
                        * mhc_post_mult_value
                    )
                for j, k in T.Parallel(mhc_mult, mhc_mult):
                    cm[j, k] = (
                        mixes_shared[j * mhc_mult + k + mhc_mult * 2] * mhc_scale[2]
                        + mhc_base[j * mhc_mult + k + mhc_mult * 2]
                    )

                ##################################################################
                # _mhc_sinkhorn_fwd
                row_sum = T.alloc_fragment(mhc_mult, T.float32)
                col_sum = T.alloc_fragment(mhc_mult, T.float32)

                # comb = comb.softmax(-1) + eps
                row_max = T.alloc_fragment(mhc_mult, T.float32)
                T.reduce_max(cm, row_max, dim=1)
                for j, k in T.Parallel(mhc_mult, mhc_mult):
                    cm[j, k] = T.exp(cm[j, k] - row_max[j])
                T.reduce_sum(cm, row_sum, dim=1)
                for j, k in T.Parallel(mhc_mult, mhc_mult):
                    cm[j, k] = cm[j, k] / row_sum[j] + mhc_sinkhorn_eps

                # comb = comb / (comb.sum(-2) + eps)
                T.reduce_sum(cm, col_sum, dim=0)
                for j, k in T.Parallel(mhc_mult, mhc_mult):
                    cm[j, k] = cm[j, k] / (col_sum[k] + mhc_sinkhorn_eps)

                for _ in T.serial(sinkhorn_repeat - 1):
                    # comb = comb / (comb.sum(-1) + eps)
                    T.reduce_sum(cm, row_sum, dim=1)
                    for j, k in T.Parallel(mhc_mult, mhc_mult):
                        cm[j, k] = cm[j, k] / (row_sum[j] + mhc_sinkhorn_eps)

                    # comb = comb / (comb.sum(-2) + eps)
                    T.reduce_sum(cm, col_sum, dim=0)
                    for j, k in T.Parallel(mhc_mult, mhc_mult):
                        cm[j, k] = cm[j, k] / (col_sum[k] + mhc_sinkhorn_eps)

                # save comb_mix to global memory
                for j, k in T.Parallel(mhc_mult, mhc_mult):
                    comb_mix[pid, j * mhc_mult + k] = cm[j, k]
            else:
                ##################################################################
                # _mhc_pre_split_mixes_fwd (pre) + fused RMSNorm
                pre_mix_shared = T.alloc_shared(mhc_mult, T.float32)
                for j in T.Parallel(mhc_mult):
                    pre_mix_shared[j] = (
                        T.sigmoid(
                            mixes_shared[j] * mhc_scale[0] + mhc_base[j],
                        )
                        + mhc_pre_eps
                    )

                # Stash unnormalized weighted-sum output in shared memory as bf16
                # (matches the rounding the reference path does when RMSNorm reads bf16).
                output_shared = T.alloc_shared(hidden_size, T.bfloat16)
                sumsq_per_pos = T.alloc_fragment(hidden_block, T.float32)
                T.clear(sumsq_per_pos)

                ###################################################################
                # Pass 1: compute weighted sum and accumulate sum-of-squares
                for i0_h in T.Pipelined(hidden_size // hidden_block, num_stages=3):
                    xs = T.alloc_shared((mhc_mult, hidden_block), T.bfloat16)
                    xl = T.alloc_fragment((mhc_mult, hidden_block), T.float32)
                    T.copy(residual[pid, 0, i0_h * hidden_block], xs, disable_tma=True)
                    T.copy(xs, xl, disable_tma=True)

                    ol = T.alloc_fragment(hidden_block, T.float32)
                    T.clear(ol)

                    for i_mhc in T.serial(mhc_mult):
                        pre = pre_mix_shared[i_mhc]
                        for i1_h in T.Parallel(hidden_block):
                            ol[i1_h] += pre * xl[i_mhc, i1_h]

                    for i1_h in T.Parallel(hidden_block):
                        sumsq_per_pos[i1_h] += ol[i1_h] * ol[i1_h]
                        output_shared[i0_h * hidden_block + i1_h] = T.bfloat16(ol[i1_h])

                # Compute RMSNorm rsqrt
                sumsq = T.alloc_fragment(1, T.float32)
                T.reduce_sum(sumsq_per_pos, sumsq, dim=0)
                rsqrt_norm = T.alloc_fragment(1, T.float32)
                rsqrt_norm[0] = T.rsqrt(sumsq[0] / hidden_size + rms_eps)

                ###################################################################
                # Pass 2: apply norm_weight * rsqrt and write to HBM
                for i0_h in T.Pipelined(hidden_size // hidden_block, num_stages=2):
                    w_shared = T.alloc_shared(hidden_block, T.bfloat16)
                    w_local = T.alloc_fragment(hidden_block, T.float32)
                    T.copy(norm_weight[i0_h * hidden_block], w_shared, disable_tma=True)
                    T.copy(w_shared, w_local, disable_tma=True)

                    ol = T.alloc_fragment(hidden_block, T.float32)
                    for i1_h in T.Parallel(hidden_block):
                        ol[i1_h] = (
                            output_shared[i0_h * hidden_block + i1_h]
                            * rsqrt_norm[0]
                            * w_local[i1_h]
                        )

                    T.copy(ol, layer_input[pid, i0_h * hidden_block], disable_tma=True)

    return mhc_pre_big_fuse_with_norm


'''@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
        tilelang.PassConfigKey.TL_DISABLE_VECTORIZE_256: True,
    },
)'''
def _mhc_post_fwd(mhc: int, hidden: int, n_thr: int = 128, h_blk: int = 1024) -> tilelang.JITKernel:
    n = T.dynamic('num_tokens')
    h = hidden

    h_blk = math.gcd(hidden, h_blk)

    @T.prim_func
    def mhc_post_fwd(
        a: T.Tensor[(n, mhc, mhc), T.float32],
        b: T.Tensor[(n, mhc, h), T.bfloat16],
        c: T.Tensor[(n, mhc), T.float32],
        d: T.Tensor[(n, h), T.bfloat16],
        x: T.Tensor[(n, mhc, h), T.bfloat16],
    ) -> None:
        with T.Kernel(n, threads=n_thr) as pid_n:
            x_shared = T.alloc_shared((mhc, h_blk), T.bfloat16)
            b_shared = T.alloc_shared((mhc, h_blk), T.bfloat16)
            d_shared = T.alloc_shared(h_blk, T.bfloat16)

            x_local = T.alloc_fragment((mhc, h_blk), T.float32)
            b_local = T.alloc_fragment((mhc, h_blk), T.float32)
            d_local = T.alloc_fragment(h_blk, T.float32)

            a_local = T.alloc_fragment((mhc, mhc), T.float32)
            c_local = T.alloc_fragment(mhc, T.float32)
            T.pdl_sync()
            T.copy(a[pid_n, 0, 0], a_local)
            T.copy(c[pid_n, 0], c_local)

            for i0_h in T.Pipelined(T.ceildiv(h, h_blk), num_stages=2):
                T.copy(b[pid_n, 0, i0_h * h_blk], b_shared, disable_tma=True)
                T.copy(d[pid_n, i0_h * h_blk], d_shared, disable_tma=True)

                T.copy(b_shared, b_local)
                T.copy(d_shared, d_local)
                for i_mhco, i1_h in T.Parallel(mhc, h_blk):
                    x_local[i_mhco, i1_h] = c_local[i_mhco] * d_local[i1_h]
                    for i_mhci in T.serial(mhc):
                        x_local[i_mhco, i1_h] += a_local[i_mhci, i_mhco] * b_local[i_mhci, i1_h]
                T.copy(x_local, x_shared)

                T.copy(x_shared, x[pid_n, 0, i0_h * h_blk], disable_tma=True)

    return mhc_post_fwd


'''@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
        tilelang.PassConfigKey.TL_DISABLE_VECTORIZE_256: True,
    },
)'''
def _mhc_head_big_fuse(
    hidden_size: int,
    rms_eps: float,
    mhc_pre_eps: float,
    n_splits: int = 16,
    mhc_mult: int = 4,
):
    num_tokens = T.symbolic("num_tokens")
    hidden_block = math.gcd(512, hidden_size)

    @T.prim_func
    def mhc_head_big_fuse(
        gemm_out_mul: T.Tensor[(n_splits, num_tokens, 2 * mhc_mult), T.float32],
        gemm_out_sqrsum: T.Tensor[(n_splits, num_tokens), T.float32],
        mhc_scale: T.Tensor[(1,), T.float32],
        mhc_base: T.Tensor[(mhc_mult,), T.float32],
        residual: T.Tensor[(num_tokens, mhc_mult, hidden_size), T.bfloat16],
        # outputs
        layer_input: T.Tensor[(num_tokens, hidden_size), T.bfloat16],
    ):
        with T.Kernel(num_tokens, threads=64) as pid:
            ##################################################################
            # _mhc_pre_norm_fn_fwd_norm
            mixes_shared = T.alloc_shared(mhc_mult, T.float32)
            T.pdl_sync()
            if T.get_thread_binding() < 32:
                rms = T.alloc_fragment(1, T.float32)
                mixes = T.alloc_fragment(mhc_mult, T.float32)
                T.clear(mixes)
                rms[0] = 0
                for i_split in T.serial(n_splits):
                    rms[0] += gemm_out_sqrsum[i_split, pid]
                rms[0] = T.rsqrt(rms[0] / (mhc_mult * hidden_size) + rms_eps)
                for j in T.Parallel(mhc_mult):
                    mixes[j] = 0
                    for i_split in T.serial(n_splits):
                        mixes[j] += gemm_out_mul[i_split, pid, j]
                    mixes[j] *= rms[0]
                T.copy(mixes, mixes_shared, disable_tma=True)

            ##################################################################
            # _mhc_pre_split_mixes_fwd (pre)
            pre_mix_shared = T.alloc_shared(mhc_mult, T.float32)
            for j in T.Parallel(mhc_mult):
                pre_mix_shared[j] = (
                    T.sigmoid(
                        mixes_shared[j] * mhc_scale[0] + mhc_base[j],
                    )
                    + mhc_pre_eps
                )
            ###################################################################
            # _mhc_pre_apply_mix_fwd
            for i0_h in T.Pipelined(hidden_size // hidden_block, num_stages=2):
                xs = T.alloc_shared((mhc_mult, hidden_block), T.bfloat16)
                xl = T.alloc_fragment((mhc_mult, hidden_block), T.float32)
                T.copy(residual[pid, 0, i0_h * hidden_block], xs, disable_tma=True)
                T.copy(xs, xl, disable_tma=True)

                ol = T.alloc_fragment(hidden_block, T.float32)
                T.clear(ol)

                for i_mhc in T.serial(mhc_mult):
                    pre = pre_mix_shared[i_mhc]
                    for i1_h in T.Parallel(hidden_block):
                        ol[i1_h] += pre * xl[i_mhc, i1_h]

                T.copy(ol, layer_input[pid, i0_h * hidden_block], disable_tma=True)

    return mhc_head_big_fuse
