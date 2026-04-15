/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cstdint>
#include <cstdio>
#include <string>

#include "global_cache_connector/sha256_lite.h"

namespace ksana_llm {

/**
 * @brief SHA256 hash of raw token prefix bytes, returned as 16-char hex string.
 *
 * Uses SHA256 for collision safety (aligned with vLLM v0.11 default).
 * Takes the first 8 bytes (64-bit equivalent) of the digest to keep keys compact.
 */
inline std::string HashTokenPrefix(const int* data, size_t len) {
  uint8_t digest[32];
  sha256_detail::Sha256(reinterpret_cast<const uint8_t*>(data), len * sizeof(int), digest);
  char hex[17];
  for (int i = 0; i < 8; ++i) {
    snprintf(hex + i * 2, 3, "%02x", digest[i]);
  }
  return std::string(hex, 16);
}

inline std::string MakeBlockKey(const std::string& prefix_hash, int tp_rank, int layer_index, int block_idx) {
  return "kv_" + prefix_hash + "_tp" + std::to_string(tp_rank) + "_l" + std::to_string(layer_index) + "_b" +
         std::to_string(block_idx);
}

inline std::string MakeFirstTokenKey(const std::string& prefix_hash) { return "ft_" + prefix_hash; }

}  // namespace ksana_llm
