/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "global_cache_connector/mooncake_mock_connector.h"

#include <cuda_runtime.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <functional>
#include <string>
#include <thread>
#include <vector>

#include "utils/logger.h"

namespace ksana_llm {

MooncakeMockConnector::~MooncakeMockConnector() {
  if (initialized_) {
    Shutdown();
  }
}

Status MooncakeMockConnector::Initialize(const GlobalCacheConnectorConfig& config) {
  KLLM_LOG_DEBUG << "start, endpoint=" << config.service_endpoint << ", device_id=" << config.device_id
                 << ", device_count=" << config.device_count;

  if (initialized_) {
    KLLM_LOG_WARNING << "already initialized, skipped";
    return Status(RetCode::RET_INVALID_ARGUMENT, "MooncakeMockConnector already initialized.");
  }

  store_ = std::make_unique<mooncake::MockStore>();

  char hostname[256];
  gethostname(hostname, sizeof(hostname));
  KLLM_LOG_DEBUG << "hostname=" << hostname;

  int ret = store_->setup(hostname, config.service_endpoint.c_str());
  if (ret != 0) {
    KLLM_LOG_ERROR << "MockStore setup failed, ret=" << ret << ", endpoint=" << config.service_endpoint;
    store_.reset();
    return Status(RetCode::RET_RUNTIME_FAILED, "MockStore setup failed, endpoint: " + config.service_endpoint);
  }

  initialized_ = true;
  KLLM_LOG_INFO << "success, endpoint=" << config.service_endpoint;
  return Status();
}

Status MooncakeMockConnector::QueryAndFetch(const std::vector<int>& token_ids, size_t local_cached_len, int tp_size,
                                            int tp_rank, int layer_index, int block_size, size_t block_byte_size,
                                            const std::vector<void*>& block_ptrs, std::vector<int>& matched_ranges,
                                            bool retry_until_full_hit, int timeout_ms) {
  matched_ranges.clear();

  if (!initialized_ || block_size <= 0) {
    KLLM_LOG_DEBUG << "skipped: initialized=" << initialized_ << ", block_size=" << block_size;
    return Status();
  }

  int total_tokens = static_cast<int>(token_ids.size());
  int total_blocks = static_cast<int>(block_ptrs.size());
  int start_block = static_cast<int>(local_cached_len) / block_size;

  KLLM_LOG_DEBUG << "start: total_tokens=" << total_tokens << ", total_blocks=" << total_blocks
                 << ", local_cached_len=" << local_cached_len << ", start_block=" << start_block
                 << ", tp_size=" << tp_size << ", tp_rank=" << tp_rank << ", layer_index=" << layer_index
                 << ", block_size=" << block_size << ", retry=" << retry_until_full_hit
                 << ", timeout_ms=" << timeout_ms;

  std::vector<char> cpu_buffer(block_byte_size);

  // 跟踪每个 block 是否已成功拉取到 GPU，重试时跳过已拉取的 block
  std::vector<bool> block_fetched(total_blocks, false);

  auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  constexpr int kRetryIntervalMs = 10;
  int attempt = 0;

  while (true) {
    ++attempt;
    int hit_count = 0;
    int miss_count = 0;

    for (int bi = start_block; bi < total_blocks; ++bi) {
      if (block_fetched[bi]) {
        ++hit_count;
        continue;
      }

      size_t prefix_len = std::min(static_cast<size_t>(bi + 1) * block_size, static_cast<size_t>(total_tokens));
      std::string prefix_hash = HashTokenPrefix(token_ids.data(), prefix_len);
      std::string key = MakeBlockKey(prefix_hash, tp_rank, layer_index, bi);

      if (!store_->is_exist(key.c_str())) {
        KLLM_LOG_DEBUG << "block=" << bi << " key=" << key << " miss (attempt=" << attempt << ")";
        ++miss_count;
        continue;
      }

      int64_t fetched = store_->get_into(key.c_str(), reinterpret_cast<uintptr_t>(cpu_buffer.data()), block_byte_size);
      if (fetched != static_cast<int64_t>(block_byte_size)) {
        KLLM_LOG_WARNING << "block=" << bi << " key=" << key << " incomplete fetch: expected=" << block_byte_size
                         << ", fetched=" << fetched;
        ++miss_count;
        continue;
      }

      cudaError_t err = cudaMemcpy(block_ptrs[bi], cpu_buffer.data(), block_byte_size, cudaMemcpyHostToDevice);
      if (err != cudaSuccess) {
        KLLM_LOG_WARNING << "block=" << bi << " key=" << key << " cudaMemcpy H2D failed: " << cudaGetErrorString(err);
        ++miss_count;
        continue;
      }

      KLLM_LOG_DEBUG << "block=" << bi << " key=" << key << " hit, block_byte_size=" << block_byte_size;
      block_fetched[bi] = true;
      ++hit_count;
    }

    KLLM_LOG_DEBUG << "attempt=" << attempt << ", hit=" << hit_count << ", miss=" << miss_count;

    if (miss_count == 0) break;
    if (!retry_until_full_hit) break;
    if (std::chrono::steady_clock::now() >= deadline) {
      KLLM_LOG_WARNING << "retry timeout after " << attempt << " attempts, remaining miss=" << miss_count;
      break;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(kRetryIntervalMs));
  }

  // 根据 block_fetched 构建 matched_ranges
  int range_start = -1;
  int last_hit = -1;
  for (int bi = start_block; bi < total_blocks; ++bi) {
    if (block_fetched[bi]) {
      if (range_start < 0) range_start = bi;
      last_hit = bi;
    } else {
      if (range_start >= 0) {
        matched_ranges.push_back(range_start * block_size);
        matched_ranges.push_back(std::min((last_hit + 1) * block_size, total_tokens) - 1);
        range_start = -1;
      }
    }
  }
  if (range_start >= 0) {
    matched_ranges.push_back(range_start * block_size);
    matched_ranges.push_back(std::min((last_hit + 1) * block_size, total_tokens) - 1);
  }

  KLLM_LOG_DEBUG << "done: matched_ranges_size=" << matched_ranges.size() << ", attempts=" << attempt;
  for (size_t i = 0; i + 1 < matched_ranges.size(); i += 2) {
    KLLM_LOG_DEBUG << "matched_range[" << (i / 2) << "]: [" << matched_ranges[i] << ", " << matched_ranges[i + 1]
                   << "]";
  }

  return Status();
}

Status MooncakeMockConnector::PutGlobalCacheBlocks(const std::vector<int>& token_ids, int tp_size, int tp_rank,
                                                   int layer_index, int block_size, size_t block_byte_size,
                                                   const std::vector<int>& block_ranges,
                                                   const std::vector<void*>& block_ptrs) {
  if (!initialized_ || block_size <= 0) {
    KLLM_LOG_WARNING << "skipped: initialized=" << initialized_ << ", block_size=" << block_size;
    return Status(RetCode::RET_INVALID_ARGUMENT, "Connector not initialized or invalid block_size.");
  }

  if (block_ranges.size() % 2 != 0) {
    KLLM_LOG_ERROR << "block_ranges size is odd: " << block_ranges.size();
    return Status(RetCode::RET_INVALID_ARGUMENT, "block_ranges size must be a multiple of 2.");
  }

  if (block_byte_size == 0) {
    KLLM_LOG_ERROR << "block_byte_size=0";
    return Status(RetCode::RET_INVALID_ARGUMENT, "block_byte_size is 0.");
  }

  int total_tokens = static_cast<int>(token_ids.size());
  KLLM_LOG_DEBUG << "start: total_tokens=" << total_tokens << ", block_ptrs_size=" << block_ptrs.size()
                 << ", block_ranges_size=" << block_ranges.size() << ", tp_size=" << tp_size << ", tp_rank=" << tp_rank
                 << ", layer_index=" << layer_index << ", block_size=" << block_size
                 << ", block_byte_size=" << block_byte_size;

  std::vector<char> cpu_buffer(block_byte_size);

  int ptr_idx = 0;
  int put_count = 0;
  int skip_exist_count = 0;

  for (size_t i = 0; i < block_ranges.size(); i += 2) {
    int range_start_token = block_ranges[i];
    int range_end_token = block_ranges[i + 1];

    int start_block = range_start_token / block_size;
    int end_block = range_end_token / block_size;

    KLLM_LOG_DEBUG << "processing range [" << range_start_token << ", " << range_end_token << "] -> block ["
                   << start_block << ", " << end_block << "]";

    for (int bi = start_block; bi <= end_block; ++bi) {
      if (ptr_idx >= static_cast<int>(block_ptrs.size())) {
        KLLM_LOG_DEBUG << "ptr_idx=" << ptr_idx << " exceeds block_ptrs_size, stop current range";
        break;
      }

      // 允许不满一个 block 的尾部参与存储，以支持全命中直接进入 decode
      size_t prefix_len = std::min(static_cast<size_t>(bi + 1) * block_size, static_cast<size_t>(total_tokens));

      std::string prefix_hash = HashTokenPrefix(token_ids.data(), prefix_len);
      std::string key = MakeBlockKey(prefix_hash, tp_rank, layer_index, bi);

      if (store_->is_exist(key.c_str())) {
        KLLM_LOG_DEBUG << "block=" << bi << " key=" << key << " already exists, skipped";
        ++skip_exist_count;
        ++ptr_idx;
        continue;
      }

      cudaError_t err = cudaMemcpy(cpu_buffer.data(), block_ptrs[ptr_idx], block_byte_size, cudaMemcpyDeviceToHost);
      if (err != cudaSuccess) {
        KLLM_LOG_WARNING << "block=" << bi << " key=" << key << " cudaMemcpy D2H failed: " << cudaGetErrorString(err);
        ++ptr_idx;
        continue;
      }

      KLLM_LOG_DEBUG << "block=" << bi << " key=" << key << " writing, byte_size=" << block_byte_size
                     << ", ptr_idx=" << ptr_idx;
      store_->put(key.c_str(), cpu_buffer.data(), block_byte_size);
      ++put_count;
      ++ptr_idx;
    }
  }

  KLLM_LOG_DEBUG << "done: put=" << put_count << ", skip_exist=" << skip_exist_count;

  return Status();
}

Status MooncakeMockConnector::PutFirstTokens(const std::vector<int>& token_ids, const std::vector<int>& tokens) {
  if (!initialized_) {
    return Status(RetCode::RET_INVALID_ARGUMENT, "Connector not initialized.");
  }
  if (token_ids.empty() || tokens.empty()) {
    return Status();
  }

  std::string prefix_hash = HashTokenPrefix(token_ids.data(), token_ids.size());
  std::string key = MakeFirstTokenKey(prefix_hash);

  if (store_->is_exist(key.c_str())) {
    KLLM_LOG_DEBUG << "key=" << key << " already exists, skipped";
    return Status();
  }

  // 存储格式: [count(int), token0, token1, ...]
  std::vector<int> payload;
  payload.reserve(1 + tokens.size());
  payload.push_back(static_cast<int>(tokens.size()));
  payload.insert(payload.end(), tokens.begin(), tokens.end());

  store_->put(key.c_str(), reinterpret_cast<const char*>(payload.data()), payload.size() * sizeof(int));
  KLLM_LOG_DEBUG << "key=" << key << ", token_count=" << tokens.size() << ", first_token=" << tokens[0]
                 << ", input_token_count=" << token_ids.size() << ", payload_bytes=" << (payload.size() * sizeof(int));
  return Status();
}

Status MooncakeMockConnector::QueryFirstTokens(const std::vector<int>& token_ids, std::vector<int>& tokens,
                                               bool& found) {
  found = false;
  tokens.clear();
  if (!initialized_) {
    return Status();
  }
  if (token_ids.empty()) {
    return Status();
  }

  std::string prefix_hash = HashTokenPrefix(token_ids.data(), token_ids.size());
  std::string key = MakeFirstTokenKey(prefix_hash);

  // 通过 get_size 获取实际存储大小，避免 get_into 因 buffer 不足返回 -1
  int64_t data_size = store_->get_size(key.c_str());
  if (data_size <= 0) {
    KLLM_LOG_DEBUG << "key=" << key << " miss (get_size=" << data_size << ")";
    return Status();
  }

  if (data_size < static_cast<int64_t>(sizeof(int)) || data_size % sizeof(int) != 0) {
    KLLM_LOG_WARNING << "key=" << key << " invalid data_size=" << data_size;
    return Status();
  }

  // 一次性读取完整 payload: [count, token0, token1, ...]
  size_t num_ints = static_cast<size_t>(data_size) / sizeof(int);
  std::vector<int> payload(num_ints);
  int64_t fetched =
      store_->get_into(key.c_str(), reinterpret_cast<uintptr_t>(payload.data()), static_cast<size_t>(data_size));
  if (fetched != data_size) {
    KLLM_LOG_WARNING << "key=" << key << " incomplete fetch: expected=" << data_size << ", fetched=" << fetched;
    return Status();
  }

  int count = payload[0];
  if (count <= 0 || static_cast<size_t>(count) + 1 != num_ints) {
    KLLM_LOG_WARNING << "key=" << key << " header mismatch: count=" << count << ", num_ints=" << num_ints;
    return Status();
  }

  tokens.assign(payload.begin() + 1, payload.end());
  found = true;
  KLLM_LOG_DEBUG << "key=" << key << " hit, token_count=" << tokens.size()
                 << ", first_token=" << (tokens.empty() ? -1 : tokens[0]) << ", input_token_count=" << token_ids.size();
  return Status();
}

Status MooncakeMockConnector::Shutdown() {
  KLLM_LOG_INFO << "start, initialized=" << initialized_;

  if (!initialized_) {
    return Status();
  }

  if (store_) {
    store_->close();
    store_.reset();
  }

  initialized_ = false;
  KLLM_LOG_INFO << "done";
  return Status();
}

}  // namespace ksana_llm
