/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <string>

#include "global_cache_connector/global_cache_connector_interface.h"

namespace ksana_llm {

/**
 * @brief 基于 mooncake::Client 的 GlobalCacheConnector 后端实现（stub）。
 *
 * 真实实现由 Mooncake 团队提供，当前仅保留接口占位。
 * 若在运行时选择 "mooncake" 后端，Initialize 会返回错误。
 */
class MooncakeConnector : public GlobalCacheConnectorInterface {
 public:
  MooncakeConnector() = default;
  ~MooncakeConnector() override = default;

  Status Initialize(const GlobalCacheConnectorConfig& config) override;

  Status QueryAndFetch(const std::vector<int>& token_ids, size_t local_cached_len, int tp_size, int tp_rank,
                       int layer_index, int block_size, size_t block_byte_size, const std::vector<void*>& block_ptrs,
                       std::vector<int>& matched_ranges, bool retry_until_full_hit = false,
                       int timeout_ms = 5000) override;

  Status PutGlobalCacheBlocks(const std::vector<int>& token_ids, int tp_size, int tp_rank, int layer_index,
                              int block_size, size_t block_byte_size, const std::vector<int>& block_ranges,
                              const std::vector<void*>& block_ptrs) override;

  Status PutFirstTokens(const std::vector<int>& token_ids, const std::vector<int>& tokens) override;

  Status QueryFirstTokens(const std::vector<int>& token_ids, std::vector<int>& tokens, bool& found) override;

  Status Shutdown() override;
};

}  // namespace ksana_llm
