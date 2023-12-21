/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/layers/attention_layer.h"

namespace numerous_llm {
Status AttentionLayer::Init(const std::vector<std::any>& parameters, cudaStream_t stream) {
  stream_ = stream;
  int parameter_index = 0;
  max_position_embeddings_ = std::any_cast<const int>(parameters[parameter_index++]);
  NLLM_LOG_INFO << fmt::format("max_position_embeddings {}", max_position_embeddings_);
  return Status();
}

}  // namespace numerous_llm
