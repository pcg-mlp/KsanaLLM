/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>

#include "layers/base_layer.h"
#include "models/base/forwarding_context.h"
#include "models/base/layer_creation_context.h"
#include "modules/basic/linear.h"

namespace ksana_llm {

// Compressor module for DeepSeek V4 token compression.
class Compressor {
 public:
  Compressor(int layer_idx, bool is_in_indexer, const LayerCreationContext& creation_context,
             const ModelCreationConfig& model_creation_config);

  ~Compressor() = default;

  // Full forward pass:
  //   1. kv_score_input = wkv_gate_proj_(x)
  //   2. fp32_kv_score_input = fp32_cast(kv_score_input)
  //   3. kv_compressed_output = compress_layer_(fp32_kv_score_input)
  // Inputs:
  //   - x: hidden states [total_tokens, hidden_dim]
  //   - workspace: intermediate buffer for kv_score_input and fp32_kv_score_input
  // Outputs:
  //   - kv_compressed_output: compressed kv [total_tokens, head_dim]
  Status Forward(const Tensor& x, const Tensor& workspace, Tensor& kv_compressed_output,
                 ForwardingContext& forwarding_context);

 private:
  const int layer_idx_;
  // Local layer index on the current node (= layer_idx_ - lower_layer_idx for main layers,
  // or main_layer_num_on_node + nextn_offset for MTP layers). Required for indexing
  // mqa_shared / compressor_shared per-layer tensors which are populated by local index.
  // See dsv4_mqa.cpp:34-39 for the same formula used by attn_config.idx.
  const int local_layer_idx_;
  const bool is_in_indexer_;

  uint32_t compress_ratio_;  // 4 or 128
  uint32_t head_dim_;
  uint32_t coff_;  // for ratio=4: coff=2, for ratio=128: coff=1.

  // Compute wkv and wgate proj
  std::unique_ptr<Linear> wkv_gate_proj_;

  // Cast layer for converting kv_score_input from bf16 to fp32
  std::unique_ptr<BaseLayer> cast_layer_;
  Tensor cast_shape_tensor_;  // Auxiliary shape tensor: {offset=0, vocab_size=1, vocab_size_pad=1}

  // CompressorLayer handles: plan + flash_c4/c128 compress + fused norm+rope
  std::unique_ptr<BaseLayer> compressor_layer_;

  Tensor ape_weight_;
  Tensor norm_weight_;
};

}  // namespace ksana_llm
