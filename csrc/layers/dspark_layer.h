/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cstdint>

#include "layers/base_layer.h"

namespace ksana_llm {

// -----------------------------------------------------------------------------
// DSparkLayer
// -----------------------------------------------------------------------------
//   - CaptureMeanReduce: reduce-mean over hc_mult axis into dspark hidden buffer.
//   - InjectKV: fused weighted-RMSNorm + RoPE + FP8 paged-cache store.
//   - MarkovEmbedLookup: gather rows of the markov embedding table.
//   - LogitsAdd: fp32 logits += cast<float>(bias).
class DSparkLayer : public BaseLayer {
 public:
  Status ForwardCaptureMeanReduce(void* dst, const void* src, int num_tokens, int hc_mult, int hidden_dim,
                                  int64_t dst_pitch_bytes, DataType dtype);

  Status ForwardInjectKV(const void* kv_in, const void* kv_norm_weight, void* swa_kv_cache, const int64_t* positions,
                         const void* cos_sin_cache, const int32_t* raw_out_loc, float norm_eps, int num_tokens,
                         int64_t page_stride, int page_bits, DataType kv_cache_dtype);

  Status ForwardMarkovEmbedLookup(void* output, const uint32_t* token_ids, const void* emb_table, int batch_size,
                                  int hidden_dim, DataType dtype);

  Status ForwardLogitsAdd(float* logits, const void* bias, int batch_size, int vocab_size, DataType bias_dtype);

 private:
  template <typename T>
  Status ForwardCaptureMeanReduceT(void* dst, const void* src, int num_tokens, int hc_mult, int hidden_dim,
                                   int64_t dst_pitch_bytes);

  template <typename T>
  Status ForwardMarkovEmbedLookupT(void* output, const uint32_t* token_ids, const void* emb_table, int batch_size,
                                   int hidden_dim);

  template <typename BiasT>
  Status ForwardLogitsAddT(float* logits, const void* bias, int batch_size, int vocab_size);
};

}  // namespace ksana_llm
