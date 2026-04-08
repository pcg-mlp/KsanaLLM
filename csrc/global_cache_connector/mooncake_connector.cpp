/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "global_cache_connector/mooncake_connector.h"

#include "utils/logger.h"

namespace ksana_llm {

static const char* kNotImplementedMsg =
    "MooncakeConnector (real store) is not implemented yet. "
    "The real implementation will be provided by the Mooncake team. "
    "Please use 'mooncake-mock' backend for testing.";

Status MooncakeConnector::Initialize(const GlobalCacheConnectorConfig& config) {
  KLLM_LOG_ERROR << kNotImplementedMsg << ", backend_type=" << config.backend_type;
  return Status(RetCode::RET_INVALID_ARGUMENT, kNotImplementedMsg);
}

Status MooncakeConnector::QueryAndFetch(const std::vector<int>& token_ids, size_t local_cached_len, int tp_size,
                                        int tp_rank, int layer_index, int block_size, size_t block_byte_size,
                                        const std::vector<void*>& block_ptrs, std::vector<int>& matched_ranges,
                                        bool retry_until_full_hit, int timeout_ms) {
  matched_ranges.clear();
  return Status(RetCode::RET_INVALID_ARGUMENT, kNotImplementedMsg);
}

Status MooncakeConnector::PutGlobalCacheBlocks(const std::vector<int>& token_ids, int tp_size, int tp_rank,
                                               int layer_index, int block_size, size_t block_byte_size,
                                               const std::vector<int>& block_ranges,
                                               const std::vector<void*>& block_ptrs) {
  return Status(RetCode::RET_INVALID_ARGUMENT, kNotImplementedMsg);
}

Status MooncakeConnector::PutFirstTokens(const std::vector<int>& token_ids, const std::vector<int>& tokens) {
  return Status(RetCode::RET_INVALID_ARGUMENT, kNotImplementedMsg);
}

Status MooncakeConnector::QueryFirstTokens(const std::vector<int>& token_ids, std::vector<int>& tokens, bool& found) {
  found = false;
  tokens.clear();
  return Status(RetCode::RET_INVALID_ARGUMENT, kNotImplementedMsg);
}

Status MooncakeConnector::Shutdown() { return Status(); }

}  // namespace ksana_llm
