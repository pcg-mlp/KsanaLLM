/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/layers/attention_layer.h"

namespace numerous_llm {
Status AttentionLayer::Init(const std::vector<std::any>& parameters, std::shared_ptr<Context> context, int rank) {
  BaseLayer::Init(parameters, context, rank);
  int parameter_index = 0;
  layer_index_ = std::any_cast<const int>(parameters[parameter_index++]);
  max_position_embeddings_ = std::any_cast<const int>(parameters[parameter_index++]);
  BlockManagerConfig block_manager_config;
  Singleton<Environment>::GetInstance()->GetBlockManagerConfig(block_manager_config);
  block_size_ = block_manager_config.device_allocator_config.block_size;
  NLLM_LOG_INFO << fmt::format("layer_index_ {}; max_position_embeddings {}; block_size_ {}", layer_index_,
                               max_position_embeddings_, block_size_);
  return Status();
}

}  // namespace numerous_llm
