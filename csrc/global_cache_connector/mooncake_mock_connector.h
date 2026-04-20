/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include <string>

#include "global_cache_connector/global_cache_connector_interface.h"
#include "global_cache_connector/global_cache_key_utils.h"
#include "mooncake/store.h"

namespace ksana_llm {

/**
 * @brief 基于 mooncake-mock-store 的 Mock GlobalCacheConnector 后端实现。
 *
 * 使用 mooncake::MockStore 作为全局 KV cache 存储后端（基于共享目录的本地 mock）。
 * config.service_endpoint 对应 MockStore 的 metadata_server（共享目录路径）。
 *
 * 存储 key 格式: "kv_{prefix_hash}_tp{tp_rank}_l{layer_index}_b{block_idx}"
 * 其中 prefix_hash 是 token_ids[0..(block_idx+1)*block_size] 的哈希值，
 * 保证相同前缀的请求共享相同的缓存条目。
 */
class MooncakeMockConnector : public GlobalCacheConnectorInterface {
 public:
  MooncakeMockConnector() = default;
  ~MooncakeMockConnector() override;

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

 protected:
  // 可在测试子类中重写，以模拟 Memcpy 异常。kind 对应 MemcpyKind 枚举值。
  virtual void DoMemcpy(void* dst, const void* src, size_t count, int kind);

 private:
  std::unique_ptr<mooncake::MockStore> store_;
  bool initialized_ = false;
};

}  // namespace ksana_llm
