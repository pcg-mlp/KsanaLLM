/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "client_service.h"
#include "global_cache_connector/global_cache_connector_interface.h"
#include "global_cache_connector/global_cache_key_utils.h"
#include "global_cache_connector/staging_buffer_pool.h"

namespace ksana_llm {

/**
 * @brief 基于 mooncake::Client 的 GlobalCacheConnector 后端实现。
 *
 * 通过 RDMA/TCP 与 Mooncake Store 通信，实现 KV cache 的全局存取。
 * 使用 BatchPut/BatchGet 批量操作提升传输效率。
 *
 * 该文件仅在 USE_MOONCAKE_REAL_STORE=ON 时编译为独立的 libmooncake_connector.so。
 */
class MooncakeConnector : public GlobalCacheConnectorInterface {
 public:
  MooncakeConnector() = default;
  ~MooncakeConnector() override;

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

 private:
  std::shared_ptr<mooncake::Client> client_;

  // Staging buffer pool: eagerly initialized in Initialize() if block config is provided.
  // Falls back to lazy initialization on first Put/Get call if block_byte_size was not in config.
  // Blocks are registered with Mooncake for RDMA after allocation.
  void InitStagingPool(int block_size, size_t block_byte_size);

  std::once_flag staging_pool_init_flag_;
  std::unique_ptr<StagingBufferPool> staging_pool_;

  size_t max_step_tokens_ = 0;

  // Segment buffer registered with MountSegment for storage allocation
  void* segment_buffer_ = nullptr;
  size_t global_segment_size_ = 0;

  // Local hostname:port used during Client::Create
  std::string local_hostname_;

  bool initialized_ = false;
};

}  // namespace ksana_llm
