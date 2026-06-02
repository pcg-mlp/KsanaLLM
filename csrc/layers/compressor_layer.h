/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "layers/base_layer.h"
#include "utils/tensor.h"

#ifdef ENABLE_CUDA
#  include <optional>
#  include "csrc/kernels/nvidia/rotary_embedding/rotary_embedding.h"
#endif

namespace ksana_llm {

// -----------------------------------------------------------------------------
// CompressorLayer
// -----------------------------------------------------------------------------
// Fused gated-pooling token compressor for Deepseek-V4
// One forward pass:
//   1. [Prefill] plan_prefill_compress → flash_c4/c128_prefill
//      [Decode ] flash_c4/c128_decode
//   2. compress_fused_norm_rope_inplace (RMSNorm + RoPE in-place)
class CompressorLayer : public BaseLayer {
 public:
  virtual Status Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                      std::shared_ptr<Context> context, int rank) override;

  virtual Status Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors,
                         const LayerExecContext& exec_ctx = {}) override;

 private:
  Status ForwardPrefillC4(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors);

  Status ForwardPrefillC128(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors);

  Status ForwardDecodeC4(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors);

  Status ForwardDecodeC128(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors);

 private:
  // Model/layer config
  uint32_t compress_ratio_;  // 4 or 128
  int head_dim_;             // head dimension (128 or 512)
  float rms_norm_eps_;
  int layer_idx_;              // local layer index
  bool is_overlap_;            // true when compress_ratio_ == 4
  DataType kv_cache_dtype_;    // for mla extra cache
  DataType compressor_dtype_;  // internal state dtype (TYPE_FP32 or TYPE_BF16)

  // Fused store config (only when head_dim_==512)
  uint32_t compressed_page_bits_{0};  // log2(block_token_num / compress_ratio)
  int64_t page_stride_{0};            // bytes per page in paged cache

  void* freqs_cis_{nullptr};  // for rotary embedding
#ifdef ENABLE_CUDA
  // RoPE embedding
  std::optional<llm_kernels::nvidia::RotaryEmbeddingCuda> rotary_embedding_cuda_;
#endif
};

}  // namespace ksana_llm
