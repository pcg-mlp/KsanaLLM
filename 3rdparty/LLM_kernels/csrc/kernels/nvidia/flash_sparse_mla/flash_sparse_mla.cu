/*
 * Adapted from
 * https://github.com/deepseek-ai/FlashMLA/blob/main/csrc/pybind.cpp
 *
 * Copyright (c) 2025 DeepSeek
 * Copyright 2026 Tencent Inc.  All rights reserved.
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

#include "csrc/kernels/nvidia/flash_sparse_mla/flash_sparse_mla.h"

#include "csrc/utils/nvidia/cuda_utils.h"

#ifdef ENABLE_FLASH_MLA
#  include "api/sparse_decode.h"
#  include "common.h"
#  include "csrc/kernels/nvidia/flash_sparse_mla/kernels/fp8_splitkv_mla.h"
#  include "params.h"
#  include "sm90/decode/sparse_fp8/splitkv_mla.h"
#  include "smxx/decode/combine/combine.h"
#  include "smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.h"

namespace llm_kernels {
namespace nvidia {

using namespace llm_kernels::utils;

inline constexpr int kMinNumHeads = 64;

int GetNumSmParts(int num_heads_q, int seqlen_q) {
  return Decode_Sm90_Impl{}.get_meta(std::max(kMinNumHeads, num_heads_q), seqlen_q).num_sm_parts;
}

int GetDecodingSchedMetaSize() { return DecodingSchedMetaSize; }

// Helper function to get strides from shape
template <typename T>
std::vector<T> GetStride(const std::vector<int>& shape) {
  std::vector<T> stride(shape.size(), 1);
  if (shape.size() > 1) {
    for (int i = shape.size() - 2; i >= 0; i--) {
      stride[i] = stride[i + 1] * shape[i + 1];
    }
  }
  return stride;
}

// `cudaMalloc()` is guaranteed to be aligned to at least 256 bytes
constexpr size_t kAlignment = 256 / sizeof(float);

// Helper function to align size
inline size_t AlignSize(const size_t size) {
  static_assert((kAlignment & (kAlignment - 1)) == 0, "Expect kAlignment to be a power of 2");
  return (size + kAlignment - 1) & ~(kAlignment - 1);
};

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void InvokeFlashSparseMlaWithKVCache(void* q_ptr, CACHE_T* kcache_ptr, int* indices_ptr,
                                     int* tile_scheduler_metadata_ptr, int* num_splits_ptr, bool sched_meta_initialized,
                                     int batch_size, int seqlen_q, int num_heads_q, int head_size_k, int num_blocks,
                                     int block_size, int num_heads_k, int kcache_batch_stride, int topk,
                                     int head_size_v, float softmax_scale, cudaStream_t stream, float* workspace_ptr,
                                     SCALAR_T* out_ptr, int* topk_length_ptr, float* attn_sink_ptr,
                                     CACHE_T* extra_kcache_ptr, int* extra_indices_ptr, int* extra_topk_length_ptr,
                                     int extra_num_blocks, int extra_block_size, int extra_topk,
                                     int extra_kcache_batch_stride) {
  using bf16 = cutlass::bfloat16_t;

  // Check the architecture
  Arch arch = Arch();

  bool have_topk_length = topk_length_ptr != nullptr;
  bool have_extra_kcache = extra_kcache_ptr != nullptr;
  bool have_extra_topk_length = extra_topk_length_ptr != nullptr;
  bool have_attn_sink = attn_sink_ptr != nullptr;

  // metadata sanity check
  KLLM_KERNEL_CHECK(batch_size > 0);
  KLLM_KERNEL_CHECK(seqlen_q > 0);
  KLLM_KERNEL_CHECK(num_heads_q > 0);
  KLLM_KERNEL_CHECK_WITH_INFO(num_heads_k == 1, "Currently only MQA (i.e. h_kv == 1) is supported for sparse decoding");
  KLLM_KERNEL_CHECK_WITH_INFO(head_size_k == 576 || head_size_k == 512,
                              "Only head_size_k == 576 or 512 is supported for sparse decoding");
  KLLM_KERNEL_CHECK_WITH_INFO(head_size_v == 512, "Only head_size_v == 512 is supported for sparse decoding");
  KLLM_KERNEL_CHECK(topk > 0);

  if (have_extra_kcache) {
    KLLM_KERNEL_CHECK_WITH_INFO(
        extra_indices_ptr != nullptr,
        "extra_indices_in_kvcache must be provided when extra_kcache is provided for sparse attention");
  } else {
    KLLM_KERNEL_CHECK_WITH_INFO(extra_indices_ptr == nullptr,
                                "extra_indices_in_kvcache must not be provided when extra_k_cache is not provided");
    KLLM_KERNEL_CHECK_WITH_INFO(extra_topk_length_ptr == nullptr,
                                "extra_topk_length must not be provided when extra_k_cache is not provided");
  }

  // Check data type
  KLLM_KERNEL_CHECK_WITH_INFO((std::is_same_v<SCALAR_T, __nv_bfloat16>),
                              "Only bfloat16 is supported for sparse decoding");

  int bytes_per_token;
  if (head_size_k == 576 && head_size_v == 512) {
    // V3.2 style
    if constexpr (KV_DTYPE == llm_kernels::utils::KVCacheType::kFp8DsMla) {
      bytes_per_token = 512 + 64 * 2 + (512 / 128) * 4;
    } else {
      bytes_per_token = head_size_k;
    }
  } else if (head_size_k == 512 && head_size_v == 512) {
    // MODEL1 style
    if constexpr (KV_DTYPE != llm_kernels::utils::KVCacheType::kFp8DsMla) {
      KLLM_KERNEL_THROW("Only fp8_ds_mla is supported for model1 sparse decoding");
    }
    bytes_per_token = 448 + 64 * 2 + (448 / 64) * 1 + 1;
  } else {
    KLLM_KERNEL_THROW("Unsupported head sizes for is_fp8_kvcache == True");
  }

  ModelType model_type;
  if (head_size_k == 576) {
    model_type = ModelType::V32;
  } else if (head_size_k == 512) {
    model_type = ModelType::MODEL1;
  } else {
    KLLM_KERNEL_THROW("Unsupported head_size_k: %d", head_size_k);
  }

  std::vector<DecodeFeatures> features;
  if constexpr (KV_DTYPE == llm_kernels::utils::KVCacheType::kFp8DsMla) {
    if (num_heads_q == 64) {
      features.push_back(DecodeFeatures::HEAD_64);
    } else if (num_heads_q == 128) {
      features.push_back(DecodeFeatures::HEAD_128);
    } else {
      KLLM_KERNEL_THROW("Unsupported num_heads_q: %d", num_heads_q);
    }
    if (head_size_k == 576) {
      features.push_back(DecodeFeatures::HEAD_DIM_576);
    } else if (head_size_k == 512) {
      features.push_back(DecodeFeatures::HEAD_DIM_512);
    } else {
      KLLM_KERNEL_THROW("Unsupported head_size_k: %d", head_size_k);
    }
    if (model_type == ModelType::V32) {
      features.push_back(DecodeFeatures::V32_KVCACHE_FORMAT);
    } else if (model_type == ModelType::MODEL1) {
      features.push_back(DecodeFeatures::MODEL1_KVCACHE_FORMAT);
    } else {
      KLLM_KERNEL_THROW("Unsupported model type: %d", static_cast<int>(model_type));
    }
    if (have_attn_sink) {
      features.push_back(DecodeFeatures::ATTN_SINK);
    }
    if (have_topk_length) {
      features.push_back(DecodeFeatures::TOPK_LENGTH);
    }
    if (have_extra_kcache) {
      features.push_back(DecodeFeatures::EXTRA_KVCACHE);
    }
    if (have_extra_topk_length) {
      features.push_back(DecodeFeatures::EXTRA_TOPK_LENGTH);
    }
  }

  DecodeImplBase* impl;
  KLLM_KERNEL_CHECK_WITH_INFO(arch.is_sm90a(), "Only sm90 is supported");
  impl = new Decode_Sm90_Impl();

  DecodeImplMeta impl_meta = impl->get_meta(std::max(kMinNumHeads, num_heads_q), seqlen_q);

  const std::vector<int> q_shape{batch_size, seqlen_q, num_heads_q, head_size_k};
  const std::vector<int> kcache_shape{num_blocks, block_size, num_heads_k, bytes_per_token};
  const std::vector<int> indices_shape{batch_size, seqlen_q, topk};
  const std::vector<int> topk_length_shape{batch_size};
  const std::vector<int> attn_sink_shape{num_heads_q};
  const std::vector<int> extra_kcache_shape{extra_num_blocks, extra_block_size, num_heads_k, bytes_per_token};
  const std::vector<int> extra_indices_shape{batch_size, seqlen_q, extra_topk};
  const std::vector<int> extra_topk_length_shape{batch_size};
  const std::vector<int> out_shape{batch_size, seqlen_q, num_heads_q, head_size_v};
  const std::vector<int> lse_shape{batch_size, seqlen_q, num_heads_q};

  const auto q_stride = GetStride<int64_t>(q_shape);
  const auto kcache_stride = GetStride<int>(kcache_shape);
  const auto indices_stride = GetStride<int>(indices_shape);
  const auto topk_length_stride = GetStride<int>(topk_length_shape);
  const auto attn_sink_stride = GetStride<int>(attn_sink_shape);
  const auto extra_kcache_stride = GetStride<int>(extra_kcache_shape);
  const auto extra_indices_stride = GetStride<int>(extra_indices_shape);
  const auto extra_topk_length_stride = GetStride<int>(extra_topk_length_shape);
  const auto out_stride = GetStride<int64_t>(out_shape);
  const auto lse_stride = GetStride<int>(lse_shape);

  float* lse_ptr = workspace_ptr;
  workspace_ptr += AlignSize(lse_shape[0] * lse_stride[0]);

  SparseAttnDecodeParams params = {
      batch_size,
      seqlen_q,
      num_heads_q,
      num_heads_k,
      head_size_k,
      head_size_v,
      softmax_scale,
      softmax_scale * LOG_2_E,
      num_blocks,
      block_size,
      topk,
      model_type,

      reinterpret_cast<bf16*>(q_ptr),
      reinterpret_cast<bf16*>(kcache_ptr),
      indices_ptr,
      topk_length_ptr,
      attn_sink_ptr,
      lse_ptr,
      reinterpret_cast<bf16*>(out_ptr),

      extra_num_blocks,
      extra_block_size,
      extra_topk,
      reinterpret_cast<bf16*>(extra_kcache_ptr),
      extra_indices_ptr,
      extra_topk_length_ptr,

      q_stride[0],
      q_stride[1],
      q_stride[2],
      kcache_batch_stride > 0 ? kcache_batch_stride
                              : kcache_stride[0],  // kcache_batch_stride can be larger than kcache_stride[0]
      kcache_stride[1],
      indices_stride[0],
      indices_stride[1],
      lse_stride[0],
      lse_stride[1],
      out_stride[0],
      out_stride[1],
      out_stride[2],

      // extra_kcache 在 dim 0 上的 byte stride, 选择优先级:
      //   1) caller 显式传入的 extra_kcache_batch_stride (primary 与 extra pool 的 per-block 大小不同时必填);
      //   2) primary 的 kcache_batch_stride (primary 与 extra 共池布局);
      //   3) 按 shape 计算的 extra_kcache_stride[0] 兜底。
      have_extra_kcache ? (extra_kcache_batch_stride > 0
                               ? extra_kcache_batch_stride
                               : (kcache_batch_stride > 0 ? kcache_batch_stride : extra_kcache_stride[0]))
                        : 0,
      have_extra_kcache ? extra_kcache_stride[1] : 0,
      have_extra_kcache ? extra_indices_stride[0] : 0,
      have_extra_kcache ? extra_indices_stride[1] : 0,
      stream};

  // Get MLA metadata if necessary
  if (!sched_meta_initialized) {
    GetDecodeSchedMetaParams get_sched_meta_params = {batch_size,
                                                      seqlen_q,
                                                      impl_meta.block_size_topk,
                                                      impl_meta.fixed_overhead_num_blocks,
                                                      topk,
                                                      extra_topk,
                                                      topk_length_ptr,
                                                      extra_topk_length_ptr,
                                                      /*seqlens_k_ptr*/ nullptr,
                                                      reinterpret_cast<DecodingSchedMeta*>(tile_scheduler_metadata_ptr),
                                                      num_splits_ptr,
                                                      impl_meta.num_sm_parts,
                                                      stream};
    smxx::decode::run_get_decoding_sched_meta_kernel(get_sched_meta_params);
  }
  params.tile_scheduler_metadata_ptr = reinterpret_cast<DecodingSchedMeta*>(tile_scheduler_metadata_ptr);
  params.num_splits_ptr = num_splits_ptr;
  params.num_sm_parts = impl_meta.num_sm_parts;

  // Allocate intermediate buffers for split-KV
  const int total_num_splits = batch_size + impl_meta.num_sm_parts;

  const std::vector<int> lse_accum_shape{total_num_splits, seqlen_q, num_heads_q};
  const std::vector<int> o_accum_shape{total_num_splits, seqlen_q, num_heads_q, head_size_v};

  const auto lse_accum_stride = GetStride<int>(lse_accum_shape);
  const auto o_accum_stride = GetStride<int>(o_accum_shape);

  float* lse_accum_ptr = workspace_ptr;
  workspace_ptr += AlignSize(lse_accum_shape[0] * lse_accum_stride[0]);
  float* o_accum_ptr = workspace_ptr;

  params.lse_accum = lse_accum_ptr;
  params.o_accum = o_accum_ptr;
  params.stride_lse_accum_split = lse_accum_stride[0];
  params.stride_lse_accum_s_q = lse_accum_stride[1];
  params.stride_o_accum_split = o_accum_stride[0];
  params.stride_o_accum_s_q = o_accum_stride[1];
  params.stride_o_accum_h_q = o_accum_stride[2];

  if constexpr (KV_DTYPE == llm_kernels::utils::KVCacheType::kFp8DsMla) {
    impl->run(params, features);
  } else {
#  define LAUNCH_KERNEL_BY_NUM_HEADS(NUM_HEADS) \
    llm_kernels::nvidia::run_flash_splitkv_mla_fp8_sparse_kernel<ModelType::V32, NUM_HEADS>(params)
    // Supported num_heads must be instantiated under kernels/instantiations
    switch (params.h_q) {
      case 8:  // GLM-5 TP8
        LAUNCH_KERNEL_BY_NUM_HEADS(8);
        break;
      case 16:  // DSV-32 TP8
        LAUNCH_KERNEL_BY_NUM_HEADS(16);
        break;
      case 64:  // GLM-5 DP8
        LAUNCH_KERNEL_BY_NUM_HEADS(64);
        break;
      case 128:  // DSV-32 DP8
        LAUNCH_KERNEL_BY_NUM_HEADS(128);
        break;
      case 32:
        LAUNCH_KERNEL_BY_NUM_HEADS(32);
        break;
      default:
        KLLM_KERNEL_THROW("Unsupported num_heads_q: %d", num_heads_q);
    }
#  undef LAUNCH_KERNEL_BY_NUM_HEADS
  }

  CombineParams combine_params = {batch_size,
                                  seqlen_q,
                                  num_heads_q,
                                  head_size_v,

                                  params.lse,
                                  params.out,
                                  params.stride_lse_b,
                                  params.stride_lse_s_q,
                                  params.stride_o_b,
                                  params.stride_o_s_q,
                                  params.stride_o_h_q,

                                  params.lse_accum,
                                  params.o_accum,
                                  params.stride_lse_accum_split,
                                  params.stride_lse_accum_s_q,
                                  params.stride_o_accum_split,
                                  params.stride_o_accum_s_q,
                                  params.stride_o_accum_h_q,

                                  params.tile_scheduler_metadata_ptr,
                                  params.num_splits_ptr,
                                  params.num_sm_parts,

                                  attn_sink_ptr,
                                  stream};
  smxx::decode::run_flash_mla_combine_kernel<bf16>(combine_params);

  delete impl;
}
#  define INVOKE_FLASH_SPARSE_MLA_WITH_KVCACHE(SCALAR_T, CACHE_T, KV_DTYPE)                               \
    template void InvokeFlashSparseMlaWithKVCache<SCALAR_T, CACHE_T, KV_DTYPE>(                           \
        void*, CACHE_T*, int*, int*, int*, bool, int, int, int, int, int, int, int, int, int, int, float, \
        cudaStream_t stream, float*, SCALAR_T*, int*, float*, CACHE_T*, int*, int*, int, int, int, int)
INVOKE_FLASH_SPARSE_MLA_WITH_KVCACHE(__nv_bfloat16, uint8_t, llm_kernels::utils::KVCacheType::kFp8DsMla);
INVOKE_FLASH_SPARSE_MLA_WITH_KVCACHE(__nv_bfloat16, uint8_t, llm_kernels::utils::KVCacheType::kFp8E4M3);
#  undef INVOKE_FLASH_SPARSE_MLA_WITH_KVCACHE

}  // namespace nvidia
}  // namespace llm_kernels
#else
namespace llm_kernels {
namespace nvidia {

int GetNumSmParts(int num_heads_q, int seqlen_q) { return 0; }

int GetDecodingSchedMetaSize() { return 0; }

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void InvokeFlashSparseMlaWithKVCache(void* q_ptr, CACHE_T* kcache_ptr, int* indices_ptr,
                                     int* tile_scheduler_metadata_ptr, int* num_splits_ptr, bool sched_meta_initialized,
                                     int batch_size, int seqlen_q, int num_heads_q, int head_size_k, int num_blocks,
                                     int block_size, int num_heads_k, int kcache_batch_stride, int topk,
                                     int head_size_v, float softmax_scale, cudaStream_t stream, float* workspace_ptr,
                                     SCALAR_T* out_ptr, int* topk_length_ptr, float* attn_sink_ptr,
                                     CACHE_T* extra_kcache_ptr, int* extra_indices_ptr, int* extra_topk_length_ptr,
                                     int extra_num_blocks, int extra_block_size, int extra_topk,
                                     int extra_kcache_batch_stride) {}
#  define INVOKE_FLASH_SPARSE_MLA_WITH_KVCACHE(SCALAR_T, CACHE_T, KV_DTYPE)                               \
    template void InvokeFlashSparseMlaWithKVCache<SCALAR_T, CACHE_T, KV_DTYPE>(                           \
        void*, CACHE_T*, int*, int*, int*, bool, int, int, int, int, int, int, int, int, int, int, float, \
        cudaStream_t stream, float*, SCALAR_T*, int*, float*, CACHE_T*, int*, int*, int, int, int, int)
INVOKE_FLASH_SPARSE_MLA_WITH_KVCACHE(__nv_bfloat16, uint8_t, llm_kernels::utils::KVCacheType::kFp8DsMla);
INVOKE_FLASH_SPARSE_MLA_WITH_KVCACHE(__nv_bfloat16, uint8_t, llm_kernels::utils::KVCacheType::kFp8E4M3);
#  undef INVOKE_SPARSE_FLASH_MLA_WITH_KVCACHE

}  // namespace nvidia
}  // namespace llm_kernels
#endif
