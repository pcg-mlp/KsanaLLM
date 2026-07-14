/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "cache_manager/block_allocator/block_allocator_interface.h"
#include "configure/environment.h"
#include "configure/model_config.h"
#include "configure/schedule_config_parser.h"
#include "utils/singleton.h"

namespace ksana_llm {

inline void BuildFlatKVCacheBlockIds(uint32_t layer_num, const std::vector<int>& device_block_ids,
                                     std::vector<std::vector<int32_t>>& atb_block_ids,
                                     std::shared_ptr<BlockAllocatorInterface> block_allocator) {
  atb_block_ids.resize(1);
  atb_block_ids[0].clear();
  size_t base_id = block_allocator->GetBlocksBaseId();
  atb_block_ids[0].reserve(device_block_ids.size());
  std::transform(device_block_ids.begin(), device_block_ids.end(), std::back_inserter(atb_block_ids[0]),
                 [base_id, layer_num](int block_id) {
                   size_t original_block_id = block_id - base_id;
                   return original_block_id * layer_num * 2;
                 });
}

inline uint32_t GetTotalLayerNum(const ModelConfig& model_config, const PipelineConfig& pipeline_config) {
  uint32_t layer_num = pipeline_config.upper_layer_idx - pipeline_config.lower_layer_idx + 1;
  if (pipeline_config.lower_nextn_layer_idx >= static_cast<int>(model_config.num_layer)) {
    layer_num += pipeline_config.upper_nextn_layer_idx - pipeline_config.lower_nextn_layer_idx + 1;
  }
  return layer_num;
}

}  // namespace ksana_llm
