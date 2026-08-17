/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "layers/dspark_layer.h"

#include <cuda_fp16.h>
#include <fmt/core.h>

#include "3rdparty/LLM_kernels/csrc/kernels/nvidia/dspark/dspark.h"
#include "csrc/utils/quant_type.h"
#include "utils/logger.h"

namespace ksana_llm {

// =============================================================================
// CaptureMeanReduce
// =============================================================================
template <typename T>
Status DSparkLayer::ForwardCaptureMeanReduceT(void* dst, const void* src, int num_tokens, int hc_mult, int hidden_dim,
                                              int64_t dst_pitch_bytes) {
  llm_kernels::nvidia::InvokeDSparkCaptureMeanReduce<T>(dst, src, num_tokens, hc_mult, hidden_dim, dst_pitch_bytes,
                                                        context_->device->GetPrimaryComputeStream().Get());
  return Status();
}

Status DSparkLayer::ForwardCaptureMeanReduce(void* dst, const void* src, int num_tokens, int hc_mult, int hidden_dim,
                                             int64_t dst_pitch_bytes, DataType dtype) {
  DISPATCH_BY_2_DTYPE(dtype, ForwardCaptureMeanReduceT, dst, src, num_tokens, hc_mult, hidden_dim, dst_pitch_bytes);
}

// =============================================================================
// InjectKV (bf16 input, KV cache dtype dispatched)
// =============================================================================
Status DSparkLayer::ForwardInjectKV(const void* kv_in, const void* kv_norm_weight, void* swa_kv_cache,
                                    const int64_t* positions, const void* cos_sin_cache, const int32_t* raw_out_loc,
                                    float norm_eps, int num_tokens, int64_t page_stride, int page_bits,
                                    DataType kv_cache_dtype) {
#define DISPATCH_INJECT_KV_BY_KVTYPE(kv_cache_dtype, ...)                                                   \
  switch (kv_cache_dtype) {                                                                                 \
    case TYPE_FP8_DS_MLA:                                                                                   \
      llm_kernels::nvidia::InvokeDSparkInjectKV<llm_kernels::utils::KVCacheType::kFp8DsMla>(__VA_ARGS__);   \
      break;                                                                                                \
    case TYPE_FP8_E4M3:                                                                                     \
      llm_kernels::nvidia::InvokeDSparkInjectKV<llm_kernels::utils::KVCacheType::kFp8E4M3>(__VA_ARGS__);    \
      break;                                                                                                \
    default:                                                                                                \
      KLLM_THROW(fmt::format("DSparkLayer: unsupported kv_cache_dtype {}", GetTypeString(kv_cache_dtype))); \
  }
  DISPATCH_INJECT_KV_BY_KVTYPE(kv_cache_dtype, reinterpret_cast<const __nv_bfloat16*>(kv_in),
                               reinterpret_cast<const __nv_bfloat16*>(kv_norm_weight), swa_kv_cache, positions,
                               reinterpret_cast<const __nv_bfloat16*>(cos_sin_cache), raw_out_loc, norm_eps, num_tokens,
                               page_stride, page_bits, context_->device->GetPrimaryComputeStream().Get());
#undef DISPATCH_INJECT_KV_BY_KVTYPE
  return Status();
}

// =============================================================================
// MarkovEmbedLookup
// =============================================================================
template <typename T>
Status DSparkLayer::ForwardMarkovEmbedLookupT(void* output, const uint32_t* token_ids, const void* emb_table,
                                              int batch_size, int hidden_dim) {
  llm_kernels::nvidia::InvokeDSparkMarkovEmbedLookup<T>(output, token_ids, emb_table, batch_size, hidden_dim,
                                                        context_->device->GetPrimaryComputeStream().Get());
  return Status();
}

Status DSparkLayer::ForwardMarkovEmbedLookup(void* output, const uint32_t* token_ids, const void* emb_table,
                                             int batch_size, int hidden_dim, DataType dtype) {
  DISPATCH_BY_2_DTYPE(dtype, ForwardMarkovEmbedLookupT, output, token_ids, emb_table, batch_size, hidden_dim);
}

// =============================================================================
// LogitsAdd (float logits += cast<float>(bias<BiasT>))
// =============================================================================
template <typename BiasT>
Status DSparkLayer::ForwardLogitsAddT(float* logits, const void* bias, int batch_size, int vocab_size) {
  llm_kernels::nvidia::InvokeDSparkLogitsAdd<BiasT>(logits, bias, batch_size, vocab_size,
                                                    context_->device->GetPrimaryComputeStream().Get());
  return Status();
}

Status DSparkLayer::ForwardLogitsAdd(float* logits, const void* bias, int batch_size, int vocab_size,
                                     DataType bias_dtype) {
  DISPATCH_BY_2_DTYPE(bias_dtype, ForwardLogitsAddT, logits, bias, batch_size, vocab_size);
}

}  // namespace ksana_llm
