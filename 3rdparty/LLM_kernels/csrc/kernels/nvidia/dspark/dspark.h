/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 */
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

#include "csrc/utils/quant_type.h"

namespace llm_kernels {
namespace nvidia {

// DSpark capture: reduce-mean over the mhc (hc_mult) axis and scatter into the
// concatenated dspark hidden buffer.
//
// dst[t, h] = (1 / hc_mult) * sum_{m=0..hc_mult-1} src[t, m, h]
template <typename T>
void InvokeDSparkCaptureMeanReduce(void* dst, const void* src, int num_tokens, int hc_mult, int hidden_dim,
                                   int64_t dst_pitch_bytes, cudaStream_t stream);

// DSpark KV injection: weighted RMSNorm (kv_norm.weight) -> RoPE -> FP8 cache store
template <llm_kernels::utils::KVCacheType KV_DTYPE>
void InvokeDSparkInjectKV(const __nv_bfloat16* kv_in, const __nv_bfloat16* kv_norm_weight, void* swa_kv_cache,
                          const int64_t* positions, const __nv_bfloat16* cos_sin_cache, const int32_t* raw_out_loc,
                          float norm_eps, int num_tokens, int64_t page_stride, int page_bits, cudaStream_t stream);

// DSpark Markov embedding lookup.
//
// output[i, :] = emb_table[token_ids[i], :]
template <typename T>
void InvokeDSparkMarkovEmbedLookup(void* output, const uint32_t* token_ids, const void* emb_table, int batch_size,
                                   int hidden_dim, cudaStream_t stream);

// DSpark logits += bias with mixed precision.
//
// logits (fp32, [B, V]) is updated in place:  logits[b, v] += static_cast<float>(bias[b, v]).
template <typename BiasT>
void InvokeDSparkLogitsAdd(float* logits, const void* bias, int batch_size, int vocab_size, cudaStream_t stream);

}  // namespace nvidia
}  // namespace llm_kernels
