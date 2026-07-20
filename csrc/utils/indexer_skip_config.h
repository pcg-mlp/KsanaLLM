/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#pragma once

#include <algorithm>
#include <cstdlib>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "configure/model_config.h"
#include "utils/logger.h"

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

inline std::unordered_set<int> ComputeIndexerSkipLayers(int num_layer, int nextn_num_layer, const DsaConfig& dsa_config,
                                                        const Dsv4AttnConfig& dsv4_attn_config) {
  std::unordered_set<int> layers;

  // Env override (highest priority): KLLM_INDEX_TOPK_PATTERN — comma-separated list of layer indices to skip.
  if (const char* pattern = std::getenv("KLLM_INDEX_TOPK_PATTERN"); pattern != nullptr && *pattern != '\0') {
    std::stringstream ss(pattern);
    std::string layer;
    while (std::getline(ss, layer, ',')) {
      if (!layer.empty()) {
        layers.insert(std::stoi(layer));
      }
    }
    return layers;
  }

  // Env overrides for offset/freq.
  DsaConfig overrode_dsa_config = dsa_config;
  if (const char* env_off = std::getenv("KLLM_INDEX_SKIP_TOPK_OFFSET"); env_off != nullptr && *env_off != '\0') {
    overrode_dsa_config.index_skip_topk_offset = static_cast<uint32_t>(std::stoul(env_off));
  }
  if (const char* env_freq = std::getenv("KLLM_INDEX_TOPK_FREQ"); env_freq != nullptr && *env_freq != '\0') {
    overrode_dsa_config.index_topk_freq = static_cast<uint32_t>(std::stoul(env_freq));
    if (overrode_dsa_config.index_topk_freq == 0) {
      overrode_dsa_config.index_topk_freq = 1;
    }
  }

  const bool has_compress = !dsv4_attn_config.compress_ratios.empty();
  int c4_layer = 0;  // Relative index within compress_ratio=4 main layers.
  for (int layer = 0; layer < num_layer + nextn_num_layer; ++layer) {
    if (layer < static_cast<int>(dsv4_attn_config.compress_ratios.size()) &&
        dsv4_attn_config.compress_ratios[layer] != 4) {
      continue;
    }

    // With compress_ratios, offset/freq only count over compress_ratio=4 main layers.
    if (has_compress) {
      if (ShouldSkip(c4_layer++, num_layer, overrode_dsa_config)) {
        layers.insert(layer);
      }
    } else if (ShouldSkip(layer, num_layer, overrode_dsa_config)) {
      layers.insert(layer);
    }
  }
  return layers;
}

inline const std::unordered_set<int>& GetIndexerSkipLayers(int num_layer, int nextn_num_layer,
                                                           const DsaConfig& dsa_config,
                                                           const Dsv4AttnConfig& dsv4_attn_config = {}) {
  static const std::unordered_set<int> cached_layers = [&]() {
    std::unordered_set<int> layers = ComputeIndexerSkipLayers(num_layer, nextn_num_layer, dsa_config, dsv4_attn_config);
    std::vector<int> sorted(layers.begin(), layers.end());
    std::sort(sorted.begin(), sorted.end());
    std::stringstream ss;
    ss << "[";
    bool first = true;
    for (int layer : sorted) {
      if (!first) {
        ss << ",";
      }
      ss << layer;
      first = false;
    }
    ss << "]";
    KLLM_LOG_INFO << "Indexer skip layers computed (num_layer=" << num_layer << ", nextn_num_layer=" << nextn_num_layer
                  << ", size=" << layers.size() << "): " << ss.str();
    return layers;
  }();
  return cached_layers;
}

}  // namespace ksana_llm
