/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <vector>

#include "configure/model_config.h"
#include "configure/schedule_config_parser.h"
#include "models/base/attention_meta/attention_meta.h"

namespace ksana_llm {

// DSA-specific shared tensors (indexer KV, paged schedule, expanded batch views).
struct DSASharedAttentionMeta {
  Tensor store_out_loc;
  Tensor layer_indexer_kv_cache_ptr;
  std::vector<Tensor> paged_schedule_metas;
  Tensor expanded_block_table;

  // Allocates DSA tensors based on model/runtime config. No-op when use_dsa is false.
  void Init(const ModelConfig& model_config, const RuntimeConfig& runtime_config, int layer_num_on_node,
            size_t max_table_block_num, size_t max_token_num, size_t decode_token_num_threshold);
};

}  // namespace ksana_llm
