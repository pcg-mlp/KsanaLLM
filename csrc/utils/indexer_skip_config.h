/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#pragma once

#include <unordered_set>

#include "configure/model_config.h"

namespace ksana_llm {

inline bool ShouldSkip(int layer, int num_layer, const DsaConfig& dsa_config) {
  if (layer < static_cast<int>(dsa_config.index_skip_topk_offset)) {  // Keep the earlier main layers
    return false;
  } else if (layer == num_layer) {  // Keep the first mtp layer
    return false;
  } else if (layer > num_layer) {
    return dsa_config.index_share_for_mtp_iteration;
  }
  return (layer + 1 - dsa_config.index_skip_topk_offset) % dsa_config.index_topk_freq != 0;
}

inline std::unordered_set<int> GetIndexerSkipLayers(int num_layer, int nextn_num_layer, const DsaConfig& dsa_config) {
  std::unordered_set<int> layers;
  for (int layer = 0; layer < num_layer + nextn_num_layer; ++layer) {
    if (ShouldSkip(layer, num_layer, dsa_config)) {
      layers.insert(layer);
    }
  }
  return layers;
}

}  // namespace ksana_llm
