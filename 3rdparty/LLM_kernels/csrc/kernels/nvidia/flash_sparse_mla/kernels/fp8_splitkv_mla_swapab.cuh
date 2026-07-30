/*
 * Adapted from
 * https://github.com/deepseek-ai/FlashMLA/blob/main/csrc/sm90/decode/sparse_fp8/splitkv_mla.cuh
 *
 * The swap-AB implementation references the following sources:
 * https://github.com/meituan-longcat/FlashMLA/commit/b42c48954c7b7bac03644723dad83b4d790147ad
 * https://github.com/pengcuo/FlashMLA-ETAP/blob/main/csrc/flash_fwd_mla_kernel.h
 * https://github.com/lightseekorg/tokenspeed/tree/main/tokenspeed-mla
 *
 * Copyright (c) 2025 DeepSeek
 * Copyright (c) 2026, Tencent Inc.
 *
 * MIT License
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#pragma once

#include "csrc/kernels/nvidia/flash_sparse_mla/kernels/fp8_splitkv_mla.h"

#include <cuda_fp8.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/barrier.h>
#include <math_constants.h>
#include <cutlass/cluster_launch.hpp>

#include <kerutils/kerutils.cuh>

#include "components/helpers.h"
#include "csrc/kernels/nvidia/flash_mla/kernels/fp8_transpose_v.h"
#include "csrc/kernels/nvidia/flash_mla/kernels/fp8_utils.h"
#include "csrc/kernels/nvidia/flash_sparse_mla/kernels/fp8_config.h"
#include "helpers.h"
#include "utils.h"

namespace llm_kernels {
namespace nvidia {

using namespace cute;
using namespace sm90::decode::sparse_fp8;

using cutlass::arch::fence_view_async_shared;
using cutlass::arch::NamedBarrier;

static constexpr float MAX_INIT_VAL_SM = -1e30f;
static constexpr float MAX_INIT_VAL = -1e33f;

struct ColMaxOp {
  __device__ __forceinline__ float operator()(float const x, float const y) { return fmax(x, y); }
};
struct ColSumOp {
  __device__ __forceinline__ float operator()(float const x, float const y) { return x + y; }
};

template <typename Op>
struct WarpColOp {
  __device__ __forceinline__ float operator()(float val) {
    const unsigned int lane_id = threadIdx.x % 32;
    const int group_id = lane_id % 4;
    const unsigned mask_group = 0x11111111 << group_id;
    val = Op()(val, __shfl_xor_sync(mask_group, val, 16));
    val = Op()(val, __shfl_xor_sync(mask_group, val, 8));
    val = Op()(val, __shfl_xor_sync(mask_group, val, 4));
    return val;
  }
};

template <int NUM_LOCAL_COLS, typename Op, typename Engine0, typename Layout0>
__device__ __forceinline__ void warp_group_col_reduce(float* r, int idx_in_warpgroup, uint32_t barrier_id,
                                                      Tensor<Engine0, Layout0>& wksp) {
  CUTLASS_PRAGMA_UNROLL
  for (int c = 0; c < NUM_LOCAL_COLS; ++c) r[c] = WarpColOp<Op>()(r[c]);
  int lane = idx_in_warpgroup % 32;
  int my_warp = idx_in_warpgroup / 32;
  int thr_col = lane % 4;
  if ((lane / 4) == 0) {
    CUTLASS_PRAGMA_UNROLL
    for (int c = 0; c < NUM_LOCAL_COLS; ++c) wksp(my_warp, thr_col * NUM_LOCAL_COLS + c) = r[c];
  }
  NamedBarrier::arrive_and_wait(128, barrier_id);
  CUTLASS_PRAGMA_UNROLL
  for (int c = 0; c < NUM_LOCAL_COLS; ++c) {
    CUTLASS_PRAGMA_UNROLL
    for (int w = 0; w < 4; ++w) {
      if (my_warp != w) r[c] = Op()(wksp(w, thr_col * NUM_LOCAL_COLS + c), r[c]);
    }
  }
}

__forceinline__ __device__ int get_col_idx(int local_col_idx, int idx_in_warpgroup) {
  return idx_in_warpgroup % 4 * 2 + local_col_idx / 2 * 8 + local_col_idx % 2;
}

template <typename Register>
__device__ __forceinline__ int get_i_rO(const Register& r, int local_col_idx, int local_row_idx) {
  constexpr int N8 = decltype(size<0, 2>(r))::value;
  return local_col_idx / 2 % N8 * 4 + local_row_idx * 2 + local_col_idx % 2;
}

template <typename Register>
__device__ __forceinline__ int get_k_rO(const Register& r, int local_col_idx) {
  constexpr int N8 = decltype(size<0, 2>(r))::value;
  return local_col_idx / 2 / N8;
}

template <ModelType MODEL_TYPE, int NUM_HEADS, int BLOCK_SIZE_N>
template <typename TMAParams>
__device__ void KernelTemplateSwapAB<MODEL_TYPE, NUM_HEADS, BLOCK_SIZE_N>::devfunc(const SparseAttnDecodeParams& params,
                                                                                   const TMAParams& tma_params) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900))
  const int s_q_idx = blockIdx.x;
  const int head_block_idx = NUM_M_BLOCKS == 1 ? 0 : blockIdx.y;
  const int partition_idx = blockIdx.z;
  // For FP8 version: CLUSTER_SIZE is 1
  static_assert(CLUSTER_SIZE == 1);
  const int warpgroup_idx = cutlass::canonical_warp_group_idx();
  const int idx_in_warpgroup = threadIdx.x % 128;
  const int warp_idx = cutlass::canonical_warp_idx_sync();

  // Define shared tensors
  extern __shared__ char wksp_buf[];
  SharedMemoryPlan& plan = *reinterpret_cast<SharedMemoryPlan*>(wksp_buf);
  Tensor sQ = make_tensor(make_smem_ptr(plan.q.data()), SmemLayoutQ{});
  Tensor sOBuf = make_tensor(make_smem_ptr(plan.u.oBuf.data()), SmemLayoutOBuf{});
  Tensor sOAccumBuf = make_tensor(make_smem_ptr(plan.u.oAccumBuf.data()), SmemLayoutOAccumBuf{});
  Tensor sS = make_tensor(make_smem_ptr(plan.s.data()), SmemLayoutS{});
  float* sM = plan.sM;
  float* sL = plan.sL;
  float* sScale = plan.sScale;

  // Prefetch TMA descriptors
  if (warp_idx == 0 && elect_one_sync()) {
    cute::prefetch_tma_descriptor(tma_params.tma_Q.get_tma_descriptor());
    cute::prefetch_tma_descriptor(&tma_params.tensor_map_o);
  }

  // Initialize TMA barriers
  if (warp_idx == 0 && elect_one_sync()) {
    plan.bar_q.init(1);
    if constexpr (CLUSTER_SIZE == 2) {
      CUTE_UNROLL
      for (int i = 0; i < NUM_K_BUFS; ++i) {
        plan.bar_k_local_ready[i].init(128);
        plan.bar_k_remote_ready[i].init(1);
        plan.bar_k_avail[i].init(4);
      }
    } else {
      CUTE_UNROLL
      for (int i = 0; i < NUM_K_BUFS; ++i) {
        plan.bar_k_local_ready[i].init(128);
        plan.bar_k_avail[i].init(256);
        // For FP8 version
        plan.bar_v_local_ready[i].init(128);
      }
    }
    cutlass::arch::fence_barrier_init();
  }
  // For FP8 version: Do not require cluster-level synchronization
  __syncthreads();

  int bar_phase_k = 0;  // Don't use array here to prevent using local memory

  // Programmatic Dependent Launch: Wait for the previous kernel to finish
  // Don't use PDL because of compiler bugs!

  DecodingSchedMeta sched_meta = params.tile_scheduler_metadata_ptr[partition_idx];

  if (sched_meta.begin_req_idx >= params.b) return;

  if (warp_idx == 0 && elect_one_sync()) {
    Tensor gQ =
        flat_divide(tma_params.tma_Q.get_tma_tensor(tma_params.shape_Q)(_, _, s_q_idx, sched_meta.begin_req_idx),
                    Tile<Int<BLOCK_SIZE_N>, Int<HEAD_DIM_K>>{})(_, _, head_block_idx, _0{});
    launch_tma_copy(tma_params.tma_Q, gQ, sQ, plan.bar_q, TMA::CacheHintSm90::EVICT_FIRST);
    plan.bar_q.arrive_and_expect_tx(BLOCK_SIZE_N * HEAD_DIM_K * sizeof(fp8));
  }

  // For FP8 version: Do not require cluster-level synchronization

  struct MainloopArgs {
    int start_block_idx, end_block_idx;
    bool is_no_split;

    // The following fields are only valid for MODEL1
    int topk_length, extra_topk_length, num_orig_kv_blocks;
  };
  auto get_cur_req_info = [&](int batch_idx) -> MainloopArgs {
    MainloopArgs args;
    int total_topk_padded;
    if constexpr (MODEL_TYPE == ModelType::V32) {
      total_topk_padded = params.topk;
    } else {
      int topk_length = params.topk_length ? __ldg(params.topk_length + batch_idx) : params.topk;
      int orig_topk_padded = max(ku::ceil(topk_length, (int)TOPK_BLOCK_SIZE), (int)TOPK_BLOCK_SIZE);
      int extra_topk_length =
          params.extra_topk_length ? __ldg(params.extra_topk_length + batch_idx) : params.extra_topk;
      total_topk_padded = orig_topk_padded + ku::ceil(extra_topk_length, (int)TOPK_BLOCK_SIZE);
      args.topk_length = topk_length;
      args.extra_topk_length = extra_topk_length;
      args.num_orig_kv_blocks = orig_topk_padded / TOPK_BLOCK_SIZE;
    }

    args.start_block_idx = batch_idx == sched_meta.begin_req_idx ? sched_meta.begin_block_idx : 0;
    args.end_block_idx =
        batch_idx == sched_meta.end_req_idx ? sched_meta.end_block_idx : total_topk_padded / TOPK_BLOCK_SIZE;
    args.is_no_split = batch_idx == sched_meta.begin_req_idx
                           ? !sched_meta.is_first_req_splitted
                           : (batch_idx == sched_meta.end_req_idx ? !sched_meta.is_last_req_splitted : true);

    return args;
  };

  // For FP8 version: Used to transpose V
  using SmemFp8Tranpose = SmemTransposeFp8_64x64<TOPK_BLOCK_SIZE, HEAD_DIM_V, /*unused*/ SmemLayoutK>;
  SmemFp8Tranpose smem_transpose_V;
  auto trans_v = [&](int buf_idx, int start_j, int end_j) -> void {
    Tensor sV_divide = as_position_independent_swizzle_tensor(
        make_tensor(make_smem_ptr(plan.u.kv.k[buf_idx].data()), typename SmemFp8Tranpose::SmemLayoutTransposeV{}));
    Tensor sVt_divide = as_position_independent_swizzle_tensor(
        make_tensor(make_smem_ptr(plan.u.kv.vt[buf_idx].data()), typename SmemFp8Tranpose::SmemLayoutTransposeVt{}));

    CUTE_UNROLL
    for (int j = start_j; j < end_j; ++j) {
      CUTE_UNROLL
      for (int i = 0; i < shape<1>(typename SmemFp8Tranpose::SmemLayoutTransposeV{}); ++i) {
        smem_transpose_V.transpose(flatten(sV_divide(_, i, j)), flatten(sVt_divide(_, i, j)));
      }
    }
  };

  if (warpgroup_idx == 0) {
    cutlass::arch::warpgroup_reg_alloc<216>();

    TiledMMA tiled_mma_QK = TiledMMA_QK{};
    ThrMMA thr_mma_QK = tiled_mma_QK.get_slice(idx_in_warpgroup);
    TiledMMA tiled_mma_PV = TiledMMA_PV{};
    ThrMMA thr_mma_PV = tiled_mma_PV.get_slice(idx_in_warpgroup);

    Tensor rO = partition_fragment_C(TiledMMA_PV{}, Shape<Int<HEAD_DIM_V / 2>, Int<BLOCK_SIZE_N>>{});
    Tensor rP = partition_fragment_C(TiledMMA_QK{}, Shape<Int<TOPK_BLOCK_SIZE>, Int<BLOCK_SIZE_N>>{});

    float rL[NUM_LOCAL_COLS], rM[NUM_LOCAL_COLS];
    float rAttn_sink[NUM_LOCAL_COLS];
    CUTE_UNROLL
    for (int c = 0; c < NUM_LOCAL_COLS; ++c) rAttn_sink[c] = -CUDART_INF_F;
    if (params.attn_sink != nullptr) {
      CUTE_UNROLL
      for (int c = 0; c < NUM_LOCAL_COLS; ++c) {
        int col_idx = get_col_idx(c, idx_in_warpgroup);
        rAttn_sink[c] = __ldg((float*)params.attn_sink + col_idx) * CUDART_L2E_F;
      }
    }

    Tensor wksp_0 = make_tensor(make_smem_ptr(plan.warp_group_reduction_wksp_0.data()),
                                make_shape(Int<4>{}, Int<4 * NUM_LOCAL_COLS>{}));

#  pragma unroll 1
    for (int batch_idx = sched_meta.begin_req_idx; batch_idx <= sched_meta.end_req_idx; ++batch_idx) {
      MainloopArgs args = get_cur_req_info(batch_idx);

      CUTE_UNROLL
      for (int c = 0; c < NUM_LOCAL_COLS; ++c) {
        rL[c] = 0.0f;
        rM[c] = MAX_INIT_VAL_SM;
      }
      cute::fill(rO, 0.);

      if (idx_in_warpgroup < BLOCK_SIZE_N) sM[idx_in_warpgroup] = MAX_INIT_VAL_SM;

      plan.bar_q.wait((sched_meta.begin_req_idx - batch_idx) & 1);

      CUTE_NO_UNROLL
      for (int block_idx = args.start_block_idx; block_idx < args.end_block_idx; block_idx++) {
        int buf_idx = (block_idx - args.start_block_idx) % NUM_K_BUFS;
        Tensor sK = make_tensor(make_smem_ptr(plan.u.kv.k[buf_idx].data()), SmemLayoutK{});
        Tensor sV = make_tensor(make_smem_ptr(plan.u.kv.vt[buf_idx].data()), SmemLayoutHalfV{});

        plan.bar_v_local_ready[buf_idx].wait(bar_phase_k >> buf_idx & 1);
        trans_v(buf_idx, 0, shape<2>(typename SmemFp8Tranpose::SmemLayoutTransposeV{}) / 2);

        plan.bar_k_local_ready[buf_idx].wait(bar_phase_k >> buf_idx & 1);

        gemm<true, -1>(tiled_mma_QK, thr_mma_QK.partition_fragment_A(sK), thr_mma_QK.partition_fragment_B(sQ), rP);

        bar_phase_k ^= 1 << buf_idx;

        if (block_idx != args.start_block_idx) {
          cute::warpgroup_wait<1>();
          NamedBarrier::arrive_and_wait(256, NamedBarriers::sScale_and_sS_free);
        }

        cute::warpgroup_wait<0>();

        float rmax[NUM_LOCAL_COLS];
        CUTE_UNROLL
        for (int c = 0; c < NUM_LOCAL_COLS; ++c) {
          float cur_max = MAX_INIT_VAL;
          CUTE_UNROLL
          for (int r = 0; r < 2; ++r) {
            int i = c / 2 * 4 + r * 2 + c % 2;
            int kv_row = (idx_in_warpgroup / 32) * 16 + (idx_in_warpgroup % 32 / 4) + r * 8;
            if (!plan.is_kv_valid[buf_idx][kv_row]) rP(i) = MAX_INIT_VAL;
            cur_max = max(cur_max, rP(i));
          }
          rmax[c] = cur_max;
        }

        warp_group_col_reduce<NUM_LOCAL_COLS, ColMaxOp>(rmax, idx_in_warpgroup, NamedBarriers::sColMaxReady, wksp_0);

        float r_new_max[NUM_LOCAL_COLS];
        float r_scale_for_old[NUM_LOCAL_COLS];
        CUTE_UNROLL
        for (int c = 0; c < NUM_LOCAL_COLS; ++c) {
          float cur_max = rmax[c] * params.sm_scale_div_log2;
          float new_max = max(rM[c], cur_max);
          float scale_for_old = exp2f(rM[c] - new_max);
          r_new_max[c] = new_max;
          r_scale_for_old[c] = scale_for_old;
        }

        if (idx_in_warpgroup < 4) {
          CUTE_UNROLL
          for (int c = 0; c < NUM_LOCAL_COLS; ++c) {
            int col_idx = get_col_idx(c, idx_in_warpgroup);
            sM[col_idx] = r_new_max[c];
            sScale[col_idx] = r_scale_for_old[c];
          }
        }

        CUTE_UNROLL
        for (int c = 0; c < NUM_LOCAL_COLS; ++c) {
          CUTE_UNROLL
          for (int r = 0; r < 2; ++r) {
            int i_rO = get_i_rO(rO, c, r);
            int k_rO = get_k_rO(rO, c);
            CUTE_UNROLL
            for (int j = 0; j < size<1>(rO); ++j) {
              rO(i_rO, j, k_rO) *= r_scale_for_old[c];
            }
          }
          float cur_sum = 0;
          CUTE_UNROLL
          for (int r = 0; r < 2; ++r) {
            int i = c / 2 * 4 + r * 2 + c % 2;
            rP(i) = exp2f(rP(i) * params.sm_scale_div_log2 - r_new_max[c]);
            cur_sum += rP(i);
          }
          rL[c] = rL[c] * r_scale_for_old[c] + cur_sum;
          rM[c] = r_new_max[c];
        }

        Tensor rS = make_tensor_like<fp8>(rP);
        convert_type_out(rP, rS);

        CUTE_UNROLL
        for (int c = 0; c < NUM_LOCAL_COLS; ++c) {
          int n = get_col_idx(c, idx_in_warpgroup);
          CUTE_UNROLL
          for (int r = 0; r < 2; ++r) {
            int m = (idx_in_warpgroup / 32) * 16 + (idx_in_warpgroup % 32 / 4) + r * 8;
            int i = c / 2 * 4 + r * 2 + c % 2;
            sS(n, m) = rS(i);
          }
        }
        fence_view_async_shared();

        gemm<false, -1>(tiled_mma_PV, thr_mma_PV.partition_fragment_A(sV), thr_mma_PV.partition_fragment_B(sS), rO);

        NamedBarrier::arrive(256, NamedBarriers::sScale_and_sS_ready);

        plan.bar_k_avail[buf_idx].arrive();
      }

      if (threadIdx.x / 32 == 0 && elect_one_sync()) {
        if (batch_idx != sched_meta.end_req_idx) {
          Tensor gQ = flat_divide(tma_params.tma_Q.get_tma_tensor(tma_params.shape_Q)(_, _, s_q_idx, batch_idx + 1),
                                  Tile<Int<BLOCK_SIZE_N>, Int<HEAD_DIM_K>>{})(_, _, head_block_idx, _0{});
          launch_tma_copy(tma_params.tma_Q, gQ, sQ, plan.bar_q, TMA::CacheHintSm90::EVICT_FIRST);
          plan.bar_q.arrive_and_expect_tx(BLOCK_SIZE_N * HEAD_DIM_K * sizeof(fp8));
        } else {
          cudaTriggerProgrammaticLaunchCompletion();
        }
      }

      warp_group_col_reduce<NUM_LOCAL_COLS, ColSumOp>(rL, idx_in_warpgroup, NamedBarriers::sColSumReady, wksp_0);

      float o_scales[NUM_LOCAL_COLS];
      CUTE_UNROLL
      for (int c = 0; c < NUM_LOCAL_COLS; ++c) {
        int col_idx = get_col_idx(c, idx_in_warpgroup);
        if (args.is_no_split) {
          o_scales[c] = rL[c] == 0.0f ? 0.0f : __fdividef(1.0f, rL[c] + exp2f(rAttn_sink[c] - rM[c]));
        } else {
          o_scales[c] = rL[c] == 0.0f ? 0.0f : __fdividef(1.0f, rL[c]);
        }
        if (idx_in_warpgroup < 4) {
          sL[col_idx] = rL[c];
          plan.sOScale[col_idx] = o_scales[c];
        }
      }

      NamedBarrier::arrive_and_wait(256, NamedBarriers::oBuf_free_and_sL_ready);

      cute::warpgroup_wait<0>();

      CUTE_UNROLL
      for (int c = 0; c < NUM_LOCAL_COLS; ++c) rL[c] = rL[c] == 0.0f ? 1.0f : rL[c];

      if (args.is_no_split) {
        bf16* o_ptr = (bf16*)params.out + batch_idx * params.stride_o_b + s_q_idx * params.stride_o_s_q;
        Tensor gO = make_tensor(make_gmem_ptr(o_ptr), make_layout(Shape<Int<BLOCK_SIZE_N>, Int<HEAD_DIM_V>>{},
                                                                  make_stride(params.stride_o_h_q, _1{})));
        float* gSoftmaxLse = (float*)params.lse + batch_idx * params.stride_lse_b + s_q_idx * params.stride_lse_s_q;

        CUTE_UNROLL
        for (int c = 0; c < NUM_LOCAL_COLS; ++c) {
          int n = get_col_idx(c, idx_in_warpgroup);
          CUTE_UNROLL
          for (int r = 0; r < 2; ++r) {
            int i_rO = get_i_rO(rO, c, r);
            int k_rO = get_k_rO(rO, c);
            CUTE_UNROLL
            for (int j = 0; j < size<1>(rO); ++j) {
              int m = warpgroup_idx * (HEAD_DIM_V / 2) + j * 64 + (idx_in_warpgroup / 32) * 16 +
                      (idx_in_warpgroup % 32 / 4) + r * 8;
              sOBuf(n, m) = (bf16)(rO(i_rO, j, k_rO) * o_scales[c]);
            }
          }
        }
        cutlass::arch::fence_view_async_shared();
        NamedBarrier::arrive_and_wait(256, NamedBarriers::epilogue_r2s_ready);

        if (threadIdx.x == 0) {
          SM90_TMA_STORE_5D::copy(&tma_params.tensor_map_o, plan.u.oBuf.data(), 0, head_block_idx * 64, 0, s_q_idx,
                                  batch_idx);
          cute::tma_store_arrive();
        }

        if (idx_in_warpgroup < BLOCK_SIZE_N) {
          float cur_L = sL[idx_in_warpgroup];
          gSoftmaxLse[idx_in_warpgroup] =
              cur_L == 0.0f ? INFINITY : logf(cur_L) + sM[idx_in_warpgroup] / (float)M_LOG2E;
        }

        cute::tma_store_wait<0>();
      } else {
        int n_split_idx = batch_idx == sched_meta.begin_req_idx ? sched_meta.begin_split_idx : 0;
        int split_idx = __ldg(params.num_splits_ptr + batch_idx) + n_split_idx;
        float* oaccum_ptr =
            (float*)params.o_accum + split_idx * params.stride_o_accum_split + s_q_idx * params.stride_o_accum_s_q;
        float* gSoftmaxLseAccum = (float*)params.lse_accum + split_idx * params.stride_lse_accum_split +
                                  s_q_idx * params.stride_lse_accum_s_q;
        Tensor gOAccum = make_tensor(
            make_gmem_ptr(oaccum_ptr),
            make_layout(Shape<Int<BLOCK_SIZE_N>, Int<HEAD_DIM_V>>{}, make_stride(params.stride_o_accum_h_q, _1{})));

        CUTE_UNROLL
        for (int c = 0; c < NUM_LOCAL_COLS; ++c) {
          int n = get_col_idx(c, idx_in_warpgroup);
          CUTE_UNROLL
          for (int r = 0; r < 2; ++r) {
            int i_rO = get_i_rO(rO, c, r);
            int k_rO = get_k_rO(rO, c);
            CUTE_UNROLL
            for (int j = 0; j < size<1>(rO); ++j) {
              int m = warpgroup_idx * (HEAD_DIM_V / 2) + j * 64 + (idx_in_warpgroup / 32) * 16 +
                      (idx_in_warpgroup % 32 / 4) + r * 8;
              sOAccumBuf(n, m) = rO(i_rO, j, k_rO) * o_scales[c];
            }
          }
        }
        cutlass::arch::fence_view_async_shared();
        NamedBarrier::arrive_and_wait(256, NamedBarriers::epilogue_r2s_ready);

        if (idx_in_warpgroup < BLOCK_SIZE_N) {
          SM90_BULK_COPY_S2G::copy(&sOAccumBuf(idx_in_warpgroup, _0{}), &gOAccum(idx_in_warpgroup, _0{}),
                                   HEAD_DIM_V * sizeof(float));
          cute::tma_store_arrive();
        }

        if (idx_in_warpgroup < BLOCK_SIZE_N) {
          float cur_L = sL[idx_in_warpgroup];
          gSoftmaxLseAccum[idx_in_warpgroup] = cur_L == 0.0f ? -INFINITY : log2f(cur_L) + sM[idx_in_warpgroup];
        }

        cute::tma_store_wait<0>();
      }

      if (batch_idx != sched_meta.end_req_idx) sync_all_threads_in_cluster();
    }
  } else if (warpgroup_idx == 1) {
    cutlass::arch::warpgroup_reg_alloc<192>();

    TiledMMA tiled_mma_PV = TiledMMA_PV{};
    ThrMMA thr_mma_PV = tiled_mma_PV.get_slice(idx_in_warpgroup);
    Tensor rO = partition_fragment_C(tiled_mma_PV, Shape<Int<HEAD_DIM_V / 2>, Int<BLOCK_SIZE_N>>{});

#  pragma unroll 1
    for (int batch_idx = sched_meta.begin_req_idx; batch_idx <= sched_meta.end_req_idx; ++batch_idx) {
      MainloopArgs args = get_cur_req_info(batch_idx);
      cute::fill(rO, 0.);

      CUTE_NO_UNROLL
      for (int block_idx = args.start_block_idx; block_idx < args.end_block_idx; block_idx++) {
        int buf_idx = (block_idx - args.start_block_idx) % NUM_K_BUFS;
        Tensor sV =
            make_tensor(make_smem_ptr(plan.u.kv.vt[buf_idx].data() + (SmemLayoutV{})(_256{}, _0{})), SmemLayoutHalfV{});

        plan.bar_k_local_ready[buf_idx].wait(bar_phase_k >> buf_idx & 1);
        trans_v(buf_idx, shape<2>(typename SmemFp8Tranpose::SmemLayoutTransposeV{}) / 2,
                shape<2>(typename SmemFp8Tranpose::SmemLayoutTransposeV{}));
        bar_phase_k ^= 1 << buf_idx;

        NamedBarrier::arrive_and_wait(256, NamedBarriers::sScale_and_sS_ready);

        float cur_scales[NUM_LOCAL_COLS];
        CUTE_UNROLL
        for (int c = 0; c < NUM_LOCAL_COLS; ++c) {
          int col_idx = get_col_idx(c, idx_in_warpgroup);
          cur_scales[c] = sScale[col_idx];
        }
        CUTE_UNROLL
        for (int c = 0; c < NUM_LOCAL_COLS; ++c) {
          CUTE_UNROLL
          for (int r = 0; r < 2; ++r) {
            int i_rO = get_i_rO(rO, c, r);
            int k_rO = get_k_rO(rO, c);
            CUTE_UNROLL
            for (int j = 0; j < size<1>(rO); ++j) {
              rO(i_rO, j, k_rO) *= cur_scales[c];
            }
          }
        }

        gemm<false, -1>(tiled_mma_PV, thr_mma_PV.partition_fragment_A(sV), thr_mma_PV.partition_fragment_B(sS), rO);
        cute::warpgroup_wait<0>();

        plan.bar_k_avail[buf_idx].arrive();

        if (block_idx != args.end_block_idx - 1) NamedBarrier::arrive(256, NamedBarriers::sScale_and_sS_free);
      }

      NamedBarrier::arrive_and_wait(256, NamedBarriers::oBuf_free_and_sL_ready);

      float o_scales[NUM_LOCAL_COLS];
      CUTE_UNROLL
      for (int c = 0; c < NUM_LOCAL_COLS; ++c) {
        int col_idx = get_col_idx(c, idx_in_warpgroup);
        o_scales[c] = plan.sOScale[col_idx];
      }

      if (args.is_no_split) {
        bf16* o_ptr = (bf16*)params.out + batch_idx * params.stride_o_b + s_q_idx * params.stride_o_s_q;
        Tensor gO = make_tensor(make_gmem_ptr(o_ptr), make_layout(Shape<Int<BLOCK_SIZE_N>, Int<HEAD_DIM_V>>{},
                                                                  make_stride(params.stride_o_h_q, _1{})));

        CUTE_UNROLL
        for (int c = 0; c < NUM_LOCAL_COLS; ++c) {
          int n = get_col_idx(c, idx_in_warpgroup);
          CUTE_UNROLL
          for (int r = 0; r < 2; ++r) {
            int i_rO = get_i_rO(rO, c, r);
            int k_rO = get_k_rO(rO, c);
            CUTE_UNROLL
            for (int j = 0; j < size<1>(rO); ++j) {
              int m = warpgroup_idx * (HEAD_DIM_V / 2) + j * 64 + (idx_in_warpgroup / 32) * 16 +
                      (idx_in_warpgroup % 32 / 4) + r * 8;
              sOBuf(n, m) = (bf16)(rO(i_rO, j, k_rO) * o_scales[c]);
            }
          }
        }
        cutlass::arch::fence_view_async_shared();
        NamedBarrier::arrive_and_wait(256, NamedBarriers::epilogue_r2s_ready);

        cute::tma_store_wait<0>();
      } else {
        int n_split_idx = batch_idx == sched_meta.begin_req_idx ? sched_meta.begin_split_idx : 0;
        int split_idx = __ldg(params.num_splits_ptr + batch_idx) + n_split_idx;
        float* oaccum_ptr =
            (float*)params.o_accum + split_idx * params.stride_o_accum_split + s_q_idx * params.stride_o_accum_s_q;
        Tensor gOAccum = make_tensor(
            make_gmem_ptr(oaccum_ptr),
            make_layout(Shape<Int<BLOCK_SIZE_N>, Int<HEAD_DIM_V>>{}, make_stride(params.stride_o_accum_h_q, _1{})));

        CUTE_UNROLL
        for (int c = 0; c < NUM_LOCAL_COLS; ++c) {
          int n = get_col_idx(c, idx_in_warpgroup);
          CUTE_UNROLL
          for (int r = 0; r < 2; ++r) {
            int i_rO = get_i_rO(rO, c, r);
            int k_rO = get_k_rO(rO, c);
            CUTE_UNROLL
            for (int j = 0; j < size<1>(rO); ++j) {
              int m = warpgroup_idx * (HEAD_DIM_V / 2) + j * 64 + (idx_in_warpgroup / 32) * 16 +
                      (idx_in_warpgroup % 32 / 4) + r * 8;
              sOAccumBuf(n, m) = rO(i_rO, j, k_rO) * o_scales[c];
            }
          }
        }
        cutlass::arch::fence_view_async_shared();
        NamedBarrier::arrive_and_wait(256, NamedBarriers::epilogue_r2s_ready);

        cute::tma_store_wait<0>();
      }

      if (batch_idx != sched_meta.end_req_idx) sync_all_threads_in_cluster();
    }
  } else {
    // Producer warpgroup
    // For FP8 version: Less registers for fetching
    cutlass::arch::warpgroup_reg_dealloc<72>();

    static_assert(CLUSTER_SIZE == 1 || CLUSTER_SIZE == 2);
    static constexpr int NUM_TOKENS_PER_THREAD = 1;
    static constexpr int NUM_TOKENS_PER_ROUND =
        64;  // For FP8 version: each CTA is responsible for fetching 64 tokens (1 round), no exchange
    int warp_idx = __shfl_sync(0xffffffff, idx_in_warpgroup / 32, 0);
    int lane_idx = idx_in_warpgroup % 32;
    int my_token_idx_base = warp_idx * 16 + lane_idx % 16;

    int64_t cache_policy = sm90::createpolicy_evict_last();

    CUTE_NO_UNROLL
    for (int batch_idx = sched_meta.begin_req_idx; batch_idx <= sched_meta.end_req_idx; ++batch_idx) {
      MainloopArgs args = get_cur_req_info(batch_idx);
      int* gIndices =
          params.indices + batch_idx * params.stride_indices_b + s_q_idx * params.stride_indices_s_q;  // (topk) : (1)
      int* gExtraIndices = params.extra_indices + batch_idx * params.stride_extra_indices_b +
                           s_q_idx * params.stride_extra_indices_s_q;  // (extra_topk) : (1)

      int nxt_token_indexs[NUM_TOKENS_PER_THREAD];
      CUTE_UNROLL
      for (int round = 0; round < NUM_TOKENS_PER_THREAD; ++round) {
        if (MODEL_TYPE == ModelType::V32 || args.start_block_idx < args.num_orig_kv_blocks)
          nxt_token_indexs[round] = __ldg(gIndices + args.start_block_idx * TOPK_BLOCK_SIZE + my_token_idx_base);
      }

      struct IsOrigBlock {};
      struct IsExtraBlock {};

      struct IsFirstExtraBlock {};
      struct IsNotFirstExtraBlock {};
      auto process_one_block = [&](int block_idx, auto is_extra_block_t, auto is_first_extra_block_t) {
        static constexpr bool IS_EXTRA_BLOCK = std::is_same_v<decltype(is_extra_block_t), IsExtraBlock>;
        static constexpr bool IS_FIRST_EXTRA_BLOCK =
            std::is_same_v<decltype(is_first_extra_block_t), IsFirstExtraBlock>;
        int buf_idx = (block_idx - args.start_block_idx) % NUM_K_BUFS;

        // For FP8 version: Used to get the real index after layout and swizzle transformation
        Tensor sK = make_tensor(make_smem_ptr(plan.u.kv.k[buf_idx].data()), SmemLayoutK{});

        int* indices_base;
        int page_block_size;
        int64_t k_block_stride, k_row_stride;
        fp8* k_ptr;
        if constexpr (!IS_EXTRA_BLOCK) {
          indices_base = gIndices + (block_idx)*TOPK_BLOCK_SIZE;
          page_block_size = params.page_block_size;
          k_block_stride = params.stride_kv_block;
          k_row_stride = params.stride_kv_row;
          k_ptr = (fp8*)params.kv;
        } else {
          indices_base = gExtraIndices + (block_idx - args.num_orig_kv_blocks) * TOPK_BLOCK_SIZE;
          page_block_size = params.extra_page_block_size;
          k_block_stride = params.stride_extra_kv_block;
          k_row_stride = params.stride_extra_kv_row;
          k_ptr = (fp8*)params.extra_kv;
        }
        [[maybe_unused]] int topk_length = IS_EXTRA_BLOCK ? args.extra_topk_length : args.topk_length;
        [[maybe_unused]] int rel_block_idx = IS_EXTRA_BLOCK ? (block_idx - args.num_orig_kv_blocks) : block_idx;

        CUTE_UNROLL
        for (int round = 0; round < NUM_TOKENS_PER_THREAD; ++round) {
          int my_token_idx = my_token_idx_base + round * NUM_TOKENS_PER_ROUND;

          // Get prefetched token index
          int token_index;
          if constexpr (!IS_EXTRA_BLOCK) {
            token_index = nxt_token_indexs[round];
            if (block_idx + 1 != (MODEL_TYPE == ModelType::V32 ? args.end_block_idx : args.num_orig_kv_blocks))
              nxt_token_indexs[round] = __ldg(gIndices + (block_idx + 1) * TOPK_BLOCK_SIZE + my_token_idx);
          } else {
            if constexpr (IS_FIRST_EXTRA_BLOCK) {
              token_index =
                  __ldg(gExtraIndices + (block_idx - args.num_orig_kv_blocks) * TOPK_BLOCK_SIZE + my_token_idx);
            } else {
              token_index = nxt_token_indexs[round];
            }
            if (block_idx + 1 != args.end_block_idx)
              nxt_token_indexs[round] =
                  __ldg(gExtraIndices + (block_idx + 1 - args.num_orig_kv_blocks) * TOPK_BLOCK_SIZE + my_token_idx);
          }

          if constexpr (MODEL_TYPE == ModelType::MODEL1) {
            // For MODEL1, we need to check whether the token_index is within topk_length
            if (rel_block_idx * TOPK_BLOCK_SIZE + my_token_idx >= topk_length) {
              token_index = -1;  // To prevent IMA when we have invalid (e.g. INT_MAX) topk indexes outside topk_length
            }
          }

          int block_index =
              token_index == -1
                  ? 0
                  : (int)((uint32_t)token_index /
                          (uint32_t)page_block_size);  // Use uint32_t division and mod to improve performance
          int rel_idx_in_block =
              (uint32_t)token_index %
              (uint32_t)page_block_size;  // NOTE When token_index is -1 (UINT_MAX), UINT_MAX%page_block_size <
                                          // page_block_size, so there will be no illegal-memory-access error

          fp8* const gK_base = k_ptr + block_index * k_block_stride + rel_idx_in_block * k_row_stride;

          // Wait for the nope buffer to be available
          if (round == 0) {
            plan.bar_k_avail[buf_idx].wait((bar_phase_k >> buf_idx & 1) ^ 1);
          }

          // For FP8 version: Load the first half of V from gmem to smem
          CUTE_UNROLL
          for (int dim_idx = 0, my_dim_idx = (lane_idx / 16) * 16; dim_idx < HEAD_DIM_V / 64;
               dim_idx += 1, my_dim_idx += 32) {
            sm90::cp_async_cacheglobal_l2_prefetch_256B(gK_base + my_dim_idx, &sK(my_token_idx, my_dim_idx),
                                                        token_index != -1, cache_policy);
          }
          // For FP8 version: Signal the half v barrier
          cutlass::arch::cpasync_barrier_arrive_noinc((uint64_t*)(&plan.bar_v_local_ready[buf_idx]));

          // For FP8 version: Load the remaining K from gmem to smem (the last half V and rope)
          CUTE_UNROLL
          for (int dim_idx = HEAD_DIM_V / 64, my_dim_idx = (lane_idx / 16) * 16 + HEAD_DIM_V / 2;
               dim_idx < HEAD_DIM_K / 32; dim_idx += 1, my_dim_idx += 32) {
            sm90::cp_async_cacheglobal_l2_prefetch_256B(gK_base + my_dim_idx, &sK(my_token_idx, my_dim_idx),
                                                        token_index != -1, cache_policy);
          }
        }

        fence_view_async_shared();

        if (idx_in_warpgroup < 32) {
          // We put this after fence_view_async_shared() since this won't be read by async proxy
          auto is_index_valid = [&](int index, int offset_within_thread) -> bool {
            if constexpr (MODEL_TYPE == ModelType::V32) {
              return index != -1;
            } else {
              return index != -1 && rel_block_idx * TOPK_BLOCK_SIZE + lane_idx * 2 + offset_within_thread < topk_length;
            }
          };
          int2 indices = __ldg((int2*)(indices_base + lane_idx * 2));
          *(char2*)(&plan.is_kv_valid[buf_idx][lane_idx * 2]) = {is_index_valid(indices.x, 0),
                                                                 is_index_valid(indices.y, 1)};
        }

        // Signal the barrier
        cutlass::arch::cpasync_barrier_arrive_noinc((uint64_t*)(&plan.bar_k_local_ready[buf_idx]));
        bar_phase_k ^= 1 << buf_idx;
      };

      if constexpr (MODEL_TYPE == ModelType::V32) {
        CUTE_NO_UNROLL
        for (int block_idx = args.start_block_idx; block_idx < args.end_block_idx; ++block_idx) {
          process_one_block(block_idx, IsOrigBlock{}, IsNotFirstExtraBlock{});
        }
      } else {
        CUTE_NO_UNROLL
        for (int block_idx = args.start_block_idx; block_idx < min(args.num_orig_kv_blocks, args.end_block_idx);
             ++block_idx) {
          process_one_block(block_idx, IsOrigBlock{}, IsNotFirstExtraBlock{});
        }

        if (args.num_orig_kv_blocks < args.end_block_idx) {
          process_one_block(max(args.start_block_idx, args.num_orig_kv_blocks), IsExtraBlock{}, IsFirstExtraBlock{});
        }
        CUTE_NO_UNROLL
        for (int block_idx = max(args.start_block_idx, args.num_orig_kv_blocks) + 1; block_idx < args.end_block_idx;
             ++block_idx) {
          process_one_block(block_idx, IsExtraBlock{}, IsNotFirstExtraBlock{});
        }
      }

      if (batch_idx != sched_meta.end_req_idx) sync_all_threads_in_cluster();
    }
  }
#else
  if (cute::thread0()) {
    CUTE_INVALID_CONTROL_PATH("This kernel only supports sm90");
  }
#endif
}

template <typename Kernel, typename TMAParams>
__global__ void __launch_bounds__(Kernel::NUM_THREADS, 1, Kernel::CLUSTER_SIZE)
    flash_fwd_splitkv_mla_fp8_sparse_swapab_kernel(__grid_constant__ const SparseAttnDecodeParams params,
                                                   __grid_constant__ const TMAParams tma_params) {
  Kernel::devfunc(params, tma_params);
}

template <ModelType MODEL_TYPE, int NUM_HEADS, int BLOCK_SIZE_N>
void KernelTemplateSwapAB<MODEL_TYPE, NUM_HEADS, BLOCK_SIZE_N>::run(const SparseAttnDecodeParams& params) {
  KU_ASSERT(params.h_kv == 1);
  KU_ASSERT(params.topk % TOPK_BLOCK_SIZE == 0);
  KU_ASSERT(params.d_qk == HEAD_DIM_K);
  KU_ASSERT(params.d_v == HEAD_DIM_V);
  KU_ASSERT(params.h_q == BLOCK_SIZE_N);
  constexpr int BYTES_PER_TOKEN = HEAD_DIM_NOPE + HEAD_DIM_ROPE;
  if constexpr (MODEL_TYPE == ModelType::MODEL1) {
    KU_ASSERT(params.stride_kv_row == BYTES_PER_TOKEN);
    if (params.extra_kv != nullptr) {
      KU_ASSERT(params.stride_extra_kv_row == BYTES_PER_TOKEN);
    }
  } else {
    KU_ASSERT(params.extra_kv == nullptr);
    KU_ASSERT(params.topk_length == nullptr);
    KU_ASSERT(params.stride_kv_row == BYTES_PER_TOKEN);
  }

  auto shape_Q = make_shape(params.h_q, params.d_qk, params.s_q, params.b);
  auto tma_Q = cute::make_tma_copy(
      SM90_TMA_LOAD{},
      make_tensor(make_gmem_ptr((fp8*)params.q),
                  make_layout(shape_Q, make_stride(params.stride_q_h_q, _1{}, params.stride_q_s_q, params.stride_q_b))),
      SmemLayoutQ{});

  CUtensorMap tensor_map_o;
  {
    uint64_t size[5] = {OBUF_SW, (unsigned long)params.h_q, HEAD_DIM_V / OBUF_SW, (unsigned long)params.s_q,
                        (unsigned long)params.b};
    uint64_t stride[4] = {params.stride_o_h_q * sizeof(bf16), OBUF_SW * sizeof(bf16),
                          params.stride_o_s_q * sizeof(bf16), params.stride_o_b * sizeof(bf16)};
    uint32_t box_size[5] = {OBUF_SW, BLOCK_SIZE_N, HEAD_DIM_V / OBUF_SW, 1, 1};
    uint32_t elem_stride[5] = {1, 1, 1, 1, 1};
    CUresult res = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
        &tensor_map_o, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 5, params.out, size, stride, box_size,
        elem_stride, CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        OBUF_SW == 64   ? CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B
        : OBUF_SW == 32 ? CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_64B
        : OBUF_SW == 16 ? CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_32B
                        : CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    KU_ASSERT(res == CUresult::CUDA_SUCCESS);
  }

  TmaParams<decltype(shape_Q), decltype(tma_Q)> tma_params = {shape_Q, tma_Q, tensor_map_o};
  auto mla_kernel =
      &flash_fwd_splitkv_mla_fp8_sparse_swapab_kernel<KernelTemplateSwapAB<MODEL_TYPE, NUM_HEADS, BLOCK_SIZE_N>,
                                                      decltype(tma_params)>;

  constexpr size_t smem_size = sizeof(SharedMemoryPlan);
  KU_CUDA_CHECK(cudaFuncSetAttribute(mla_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

  cudaLaunchAttribute mla_kernel_attributes[1];
  mla_kernel_attributes[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  mla_kernel_attributes[0].val.programmaticStreamSerializationAllowed = 0;
  cudaLaunchConfig_t mla_kernel_config = {dim3(params.s_q, NUM_M_BLOCKS, params.num_sm_parts),
                                          dim3(NUM_THREADS, 1, 1),
                                          smem_size,
                                          params.stream,
                                          mla_kernel_attributes,
                                          1};
  cudaLaunchKernelEx(&mla_kernel_config, mla_kernel, params, tma_params);
  KU_CHECK_KERNEL_LAUNCH();
}

template <ModelType MODEL_TYPE, int NUM_HEADS, int BLOCK_SIZE_N>
void run_flash_splitkv_mla_fp8_sparse_swapab_kernel(const SparseAttnDecodeParams& params) {
  KernelTemplateSwapAB<MODEL_TYPE, NUM_HEADS, BLOCK_SIZE_N>::run(params);
}

template void run_flash_splitkv_mla_fp8_sparse_swapab_kernel<ModelType::MODEL1, 16, 16>(const SparseAttnDecodeParams&);
template void run_flash_splitkv_mla_fp8_sparse_swapab_kernel<ModelType::V32, 16, 16>(const SparseAttnDecodeParams&);

}  // namespace nvidia
}  // namespace llm_kernels
