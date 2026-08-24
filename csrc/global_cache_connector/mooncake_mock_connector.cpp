/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "global_cache_connector/mooncake_mock_connector.h"

#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <functional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "device/device_utils.h"
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

int MooncakeMockConnector::AlignHitBoundaryBySwa(const std::vector<int>& token_ids, int tp_rank, int layer_index,
                                                 int block_size, int start_block, int fa_contig_end,
                                                 const GlobalCacheSwaContext& swa_ctx, GlobalCacheSwaResult* swa_result,
                                                 const std::vector<std::string>& block_hashes, int hash_start_block) {
  // mock 与单测一致用 tp_rank 编 key；真机 MooncakeConnector 用 device_id_。
  (void)token_ids;
  const int window = static_cast<int>(std::max<size_t>(swa_ctx.swa_window_block_num, 1));
  if (fa_contig_end <= start_block) {
    return start_block;
  }

  std::unordered_map<size_t, void*> local_swa_ptrs;
  const size_t local_swa_num = std::min(swa_ctx.local_swa_block_idxs.size(), swa_ctx.swa_block_ptrs.size());
  for (size_t i = 0; i < local_swa_num; ++i) {
    local_swa_ptrs[swa_ctx.local_swa_block_idxs[i]] = swa_ctx.swa_block_ptrs[i];
  }

  std::vector<std::string> swa_keys;
  swa_keys.reserve(static_cast<size_t>(fa_contig_end - start_block));
  for (int bi = start_block; bi < fa_contig_end; ++bi) {
    const size_t hash_idx = static_cast<size_t>(bi - hash_start_block);
    if (hash_idx >= block_hashes.size()) {
      KLLM_LOG_ERROR << "prefix hash missing for block=" << bi;
      return start_block;
    }
    swa_keys.push_back(MakeSwaBlockKey(block_hashes[hash_idx], tp_rank, layer_index, bi));
  }

  const std::vector<bool> swa_exist = StoreBatchIsExist(swa_keys);
  if (swa_exist.size() != swa_keys.size()) {
    KLLM_LOG_ERROR << "SWA BatchIsExist size mismatch, got=" << swa_exist.size() << "/" << swa_keys.size();
    return start_block;
  }

  auto swa_usable = [&](int idx) {
    if (idx < start_block) {
      return true;
    }
    const size_t off = static_cast<size_t>(idx - start_block);
    return swa_exist[off] && local_swa_ptrs.count(static_cast<size_t>(idx)) > 0;
  };

  int boundary = start_block;
  for (int b = fa_contig_end; b > start_block; --b) {
    bool ok = true;
    for (int idx = std::max(b - window, 0); idx < b; ++idx) {
      if (!swa_usable(idx)) {
        ok = false;
        break;
      }
    }
    if (ok) {
      boundary = b;
      break;
    }
  }

  if (boundary <= start_block) {
    return start_block;
  }

  // 边界窗口必须完整拉到，缺一块就整体放弃。
  std::vector<char> cpu_buffer(swa_ctx.swa_block_byte_size);
  for (int idx = std::max(boundary - window, start_block); idx < boundary; ++idx) {
    const std::string& key = swa_keys[static_cast<size_t>(idx - start_block)];
    const int64_t fetched =
        store_->get_into(key.c_str(), reinterpret_cast<uintptr_t>(cpu_buffer.data()), swa_ctx.swa_block_byte_size);
    if (fetched != static_cast<int64_t>(swa_ctx.swa_block_byte_size)) {
      KLLM_LOG_WARNING << "swa block=" << idx << " key=" << key
                       << " incomplete fetch: expected=" << swa_ctx.swa_block_byte_size << ", fetched=" << fetched;
      swa_result->filled_block_idxs.clear();
      return start_block;
    }
    try {
      DoMemcpy(local_swa_ptrs[static_cast<size_t>(idx)], cpu_buffer.data(), swa_ctx.swa_block_byte_size,
               MEMCPY_HOST_TO_DEVICE);
    } catch (const std::exception& e) {
      KLLM_LOG_WARNING << "swa block=" << idx << " key=" << key << " Memcpy H2D failed: " << e.what();
      swa_result->filled_block_idxs.clear();
      return start_block;
    }
    swa_result->filled_block_idxs.push_back(static_cast<size_t>(idx));
  }

  swa_result->swa_aligned_hit_len = static_cast<size_t>(boundary) * static_cast<size_t>(block_size);
  KLLM_LOG_DEBUG << "swa aligned: start_block=" << start_block << ", fa_contig_end=" << fa_contig_end
                 << ", boundary=" << boundary << ", swa_fetched=" << swa_result->filled_block_idxs.size();
  return boundary;
}

Status MooncakeMockConnector::QueryAndFetch(const std::vector<int>& token_ids, size_t local_cached_len, int tp_size,
                                            int tp_rank, int layer_index, int block_size, size_t block_byte_size,
                                            const std::vector<void*>& block_ptrs, std::vector<int>& matched_ranges,
                                            bool retry_until_full_hit, int timeout_ms,
                                            const GlobalCacheSwaContext* swa_ctx, GlobalCacheSwaResult* swa_result) {
  matched_ranges.clear();
  if (swa_result != nullptr) {
    *swa_result = GlobalCacheSwaResult{};
  }

  if (!initialized_ || block_size <= 0) {
    KLLM_LOG_DEBUG << "skipped: initialized=" << initialized_ << ", block_size=" << block_size;
    return Status();
  }

  if (swa_ctx != nullptr && swa_result == nullptr) {
    return Status(RetCode::RET_INVALID_ARGUMENT, "QueryAndFetch: swa_ctx given without swa_result.");
  }

  int total_tokens = static_cast<int>(token_ids.size());
  int total_blocks = static_cast<int>(block_ptrs.size());
  int start_block = static_cast<int>(local_cached_len) / block_size;
  if (total_blocks <= start_block) {
    return Status();
  }

  std::vector<std::string> block_hashes;
  HashTokenPrefixesIncremental(token_ids.data(), total_tokens, block_size, start_block, total_blocks, block_hashes);

  int exist_end = total_blocks;
  if (swa_ctx != nullptr) {
    const int cap_blocks = static_cast<int>(swa_ctx->max_hit_token_num / static_cast<size_t>(block_size));
    const int full_blocks = total_tokens / block_size;
    exist_end = std::min({total_blocks, cap_blocks, full_blocks});
    if (exist_end <= start_block) {
      return Status();
    }
  }

  // 与真机一致：先组主 KV keys，再一次 BatchIsExist。
  std::vector<std::string> fa_keys;
  fa_keys.reserve(static_cast<size_t>(exist_end - start_block));
  for (int bi = start_block; bi < exist_end; ++bi) {
    const size_t hash_idx = static_cast<size_t>(bi - start_block);
    if (hash_idx >= block_hashes.size()) {
      break;
    }
    fa_keys.push_back(MakeBlockKey(block_hashes[hash_idx], tp_rank, layer_index, bi));
  }
  if (fa_keys.empty()) {
    return Status();
  }

  const std::vector<bool> fa_exist = StoreBatchIsExist(fa_keys);
  if (fa_exist.size() != fa_keys.size()) {
    KLLM_LOG_ERROR << "BatchIsExist size mismatch, expected=" << fa_keys.size() << ", got=" << fa_exist.size();
    return Status(RetCode::RET_RUNTIME_FAILED, "BatchIsExist result size mismatch");
  }

  // 连续前缀只用于 SWA 裁边界；known_exist 覆盖全部候选，非 SWA 仍可拉洞后命中块。
  int fa_contig_end = start_block;
  for (size_t i = 0; i < fa_keys.size(); ++i) {
    if (!fa_exist[i]) {
      break;
    }
    fa_contig_end = start_block + static_cast<int>(i) + 1;
  }
  std::vector<bool> known_exist(static_cast<size_t>(total_blocks), false);
  for (size_t i = 0; i < fa_keys.size(); ++i) {
    known_exist[static_cast<size_t>(start_block) + i] = fa_exist[i];
  }

  if (swa_ctx != nullptr) {
    total_blocks = AlignHitBoundaryBySwa(token_ids, tp_rank, layer_index, block_size, start_block, fa_contig_end,
                                         *swa_ctx, swa_result, block_hashes, start_block);
    if (total_blocks <= start_block) {
      return Status();
    }
  }

  KLLM_LOG_DEBUG << "start: total_tokens=" << total_tokens << ", total_blocks=" << total_blocks
                 << ", local_cached_len=" << local_cached_len << ", start_block=" << start_block
                 << ", tp_size=" << tp_size << ", tp_rank=" << tp_rank << ", layer_index=" << layer_index
                 << ", block_size=" << block_size << ", retry=" << retry_until_full_hit
                 << ", timeout_ms=" << timeout_ms;

  std::vector<char> cpu_buffer(block_byte_size);
  std::vector<bool> block_fetched(static_cast<size_t>(total_blocks), false);

  auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  constexpr int kRetryIntervalMs = 10;
  int attempt = 0;
  bool reuse_known_exist = true;

  while (true) {
    ++attempt;
    int hit_count = 0;
    int miss_count = 0;

    std::vector<std::string> pending_keys;
    std::vector<int> pending_block_indices;
    for (int bi = start_block; bi < total_blocks; ++bi) {
      if (block_fetched[static_cast<size_t>(bi)]) {
        ++hit_count;
        continue;
      }
      const size_t hash_idx = static_cast<size_t>(bi - start_block);
      if (hash_idx >= block_hashes.size()) {
        ++miss_count;
        continue;
      }
      pending_keys.push_back(MakeBlockKey(block_hashes[hash_idx], tp_rank, layer_index, bi));
      pending_block_indices.push_back(bi);
    }

    if (pending_keys.empty()) {
      break;
    }

    std::vector<bool> pending_exist(pending_keys.size(), false);
    if (reuse_known_exist) {
      for (size_t i = 0; i < pending_block_indices.size(); ++i) {
        pending_exist[i] = known_exist[static_cast<size_t>(pending_block_indices[i])];
      }
      reuse_known_exist = false;
    } else {
      pending_exist = StoreBatchIsExist(pending_keys);
      if (pending_exist.size() != pending_keys.size()) {
        KLLM_LOG_ERROR << "BatchIsExist size mismatch, expected=" << pending_keys.size()
                       << ", got=" << pending_exist.size();
        return Status(RetCode::RET_RUNTIME_FAILED, "BatchIsExist result size mismatch");
      }
    }

    for (size_t i = 0; i < pending_keys.size(); ++i) {
      const int bi = pending_block_indices[i];
      const std::string& key = pending_keys[i];
      if (!pending_exist[i]) {
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

      try {
        DoMemcpy(block_ptrs[bi], cpu_buffer.data(), block_byte_size, MEMCPY_HOST_TO_DEVICE);
      } catch (const std::exception& e) {
        KLLM_LOG_WARNING << "block=" << bi << " key=" << key << " Memcpy H2D failed: " << e.what();
        ++miss_count;
        continue;
      }

      KLLM_LOG_DEBUG << "block=" << bi << " key=" << key << " hit, block_byte_size=" << block_byte_size;
      block_fetched[static_cast<size_t>(bi)] = true;
      ++hit_count;
    }

    KLLM_LOG_DEBUG << "attempt=" << attempt << ", hit=" << hit_count << ", miss=" << miss_count;

    if (miss_count == 0) {
      break;
    }
    if (!retry_until_full_hit) {
      break;
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      KLLM_LOG_WARNING << "retry timeout after " << attempt << " attempts, remaining miss=" << miss_count;
      break;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(kRetryIntervalMs));
  }

  int range_start = -1;
  int last_hit = -1;
  for (int bi = start_block; bi < total_blocks; ++bi) {
    if (block_fetched[static_cast<size_t>(bi)]) {
      if (range_start < 0) {
        range_start = bi;
      }
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

  if (swa_ctx != nullptr) {
    const bool full_cover = matched_ranges.size() == 2 && matched_ranges[0] == start_block * block_size &&
                            matched_ranges[1] == std::min(total_blocks * block_size, total_tokens) - 1;
    if (!full_cover) {
      KLLM_LOG_WARNING << "swa boundary abandoned: main kv not fully fetched, start_block=" << start_block
                       << ", boundary=" << total_blocks;
      matched_ranges.clear();
      *swa_result = GlobalCacheSwaResult{};
    }
  }

  KLLM_LOG_DEBUG << "done: matched_ranges_size=" << matched_ranges.size() << ", attempts=" << attempt;
  for (size_t i = 0; i + 1 < matched_ranges.size(); i += 2) {
    KLLM_LOG_DEBUG << "matched_range[" << (i / 2) << "]: [" << matched_ranges[i] << ", " << matched_ranges[i + 1]
                   << "]";
  }

  return Status();
}

Status MooncakeMockConnector::PutSwaBlocks(const std::vector<int>& token_ids, int tp_rank, int layer_index,
                                           int block_size, size_t swa_block_byte_size,
                                           const std::vector<size_t>& block_idxs, const std::vector<void*>& block_ptrs,
                                           int* put_ok_out) {
  if (put_ok_out != nullptr) {
    *put_ok_out = 0;
  }
  if (!initialized_ || block_size <= 0) {
    return Status(RetCode::RET_INVALID_ARGUMENT, "Connector not initialized or invalid block_size.");
  }
  if (swa_block_byte_size == 0) {
    return Status(RetCode::RET_INVALID_ARGUMENT, "swa_block_byte_size is 0.");
  }
  if (block_idxs.size() != block_ptrs.size()) {
    return Status(RetCode::RET_INVALID_ARGUMENT, "block_idxs and block_ptrs size mismatch.");
  }
  if (block_idxs.empty() || token_ids.empty()) {
    return Status();
  }

  const int total_tokens = static_cast<int>(token_ids.size());
  std::vector<char> cpu_buffer(swa_block_byte_size);

  // 先组齐 keys，一次 BatchIsExist（与真机 PutSwa 同形）。
  std::vector<std::string> all_keys;
  all_keys.reserve(block_idxs.size());
  for (size_t i = 0; i < block_idxs.size(); ++i) {
    const size_t prefix_len =
        std::min((block_idxs[i] + 1) * static_cast<size_t>(block_size), static_cast<size_t>(total_tokens));
    const std::string prefix_hash = HashTokenPrefix(token_ids.data(), prefix_len);
    all_keys.push_back(MakeSwaBlockKey(prefix_hash, tp_rank, layer_index, static_cast<int>(block_idxs[i])));
  }
  const std::vector<bool> exist_flags = StoreBatchIsExist(all_keys);
  if (exist_flags.size() != all_keys.size()) {
    return Status(RetCode::RET_RUNTIME_FAILED, "BatchIsExist result size mismatch");
  }

  int put_count = 0;
  int skip_exist_count = 0;
  for (size_t i = 0; i < block_idxs.size(); ++i) {
    if (exist_flags[i]) {
      ++skip_exist_count;
      continue;
    }

    try {
      DoMemcpy(cpu_buffer.data(), block_ptrs[i], swa_block_byte_size, MEMCPY_DEVICE_TO_HOST);
    } catch (const std::exception& e) {
      KLLM_LOG_WARNING << "swa block=" << block_idxs[i] << " Memcpy D2H failed: " << e.what();
      continue;
    }

    store_->put(all_keys[i].c_str(), cpu_buffer.data(), swa_block_byte_size);
    ++put_count;
  }

  KLLM_LOG_DEBUG << "PutSwaBlocks done: selected=" << block_idxs.size() << ", put=" << put_count
                 << ", skip_exist=" << skip_exist_count;
  if (put_ok_out != nullptr) {
    *put_ok_out = put_count;
  }
  return Status();
}

Status MooncakeMockConnector::PutGlobalCacheBlocks(const std::vector<int>& token_ids, int tp_size, int tp_rank,
                                                   int layer_index, int block_size, size_t block_byte_size,
                                                   const std::vector<int>& block_ranges,
                                                   const std::vector<void*>& block_ptrs, int* put_ok_out) {
  if (put_ok_out != nullptr) {
    *put_ok_out = 0;
  }
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

  // 先枚举 blocks + 一次增量 hash，再 BatchIsExist（与真机 Put 同形）。
  struct BlockInfo {
    int block_idx;
    int ptr_idx;
    std::string key;
  };
  std::vector<BlockInfo> all_blocks;
  {
    int hash_start = -1;
    int hash_end_excl = 0;
    for (size_t i = 0; i < block_ranges.size(); i += 2) {
      const int start_block = block_ranges[i] / block_size;
      const int end_block = block_ranges[i + 1] / block_size;
      if (hash_start < 0 || start_block < hash_start) {
        hash_start = start_block;
      }
      if (end_block + 1 > hash_end_excl) {
        hash_end_excl = end_block + 1;
      }
    }
    std::vector<std::string> block_hashes;
    if (hash_start >= 0 && hash_end_excl > hash_start) {
      HashTokenPrefixesIncremental(token_ids.data(), total_tokens, block_size, hash_start, hash_end_excl, block_hashes);
    }

    int ptr_idx = 0;
    for (size_t i = 0; i < block_ranges.size(); i += 2) {
      const int start_block = block_ranges[i] / block_size;
      const int end_block = block_ranges[i + 1] / block_size;
      for (int bi = start_block; bi <= end_block; ++bi) {
        if (ptr_idx >= static_cast<int>(block_ptrs.size())) {
          break;
        }
        const size_t hash_idx = static_cast<size_t>(bi - hash_start);
        if (hash_idx >= block_hashes.size()) {
          ++ptr_idx;
          continue;
        }
        all_blocks.push_back({bi, ptr_idx, MakeBlockKey(block_hashes[hash_idx], tp_rank, layer_index, bi)});
        ++ptr_idx;
      }
    }
  }

  if (all_blocks.empty()) {
    return Status();
  }

  std::vector<std::string> all_keys;
  all_keys.reserve(all_blocks.size());
  for (const auto& b : all_blocks) {
    all_keys.push_back(b.key);
  }
  const std::vector<bool> exist_flags = StoreBatchIsExist(all_keys);
  if (exist_flags.size() != all_keys.size()) {
    return Status(RetCode::RET_RUNTIME_FAILED, "BatchIsExist result size mismatch");
  }

  std::vector<char> cpu_buffer(block_byte_size);
  int put_count = 0;
  int skip_exist_count = 0;
  for (size_t i = 0; i < all_blocks.size(); ++i) {
    if (exist_flags[i]) {
      KLLM_LOG_DEBUG << "block=" << all_blocks[i].block_idx << " key=" << all_blocks[i].key
                     << " already exists, skipped";
      ++skip_exist_count;
      continue;
    }
    try {
      DoMemcpy(cpu_buffer.data(), block_ptrs[all_blocks[i].ptr_idx], block_byte_size, MEMCPY_DEVICE_TO_HOST);
    } catch (const std::exception& e) {
      KLLM_LOG_WARNING << "block=" << all_blocks[i].block_idx << " key=" << all_blocks[i].key
                       << " Memcpy D2H failed: " << e.what();
      continue;
    }
    store_->put(all_blocks[i].key.c_str(), cpu_buffer.data(), block_byte_size);
    ++put_count;
  }

  KLLM_LOG_DEBUG << "done: put=" << put_count << ", skip_exist=" << skip_exist_count;

  if (put_ok_out != nullptr) {
    *put_ok_out = put_count;
  }
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

  if (StoreIsExist(key)) {
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

void MooncakeMockConnector::DoMemcpy(void* dst, const void* src, size_t count, int kind) {
  Memcpy(dst, src, count, static_cast<MemcpyKind>(kind));
}

std::vector<bool> MooncakeMockConnector::StoreBatchIsExist(const std::vector<std::string>& keys) {
  std::vector<bool> out(keys.size(), false);
  for (size_t i = 0; i < keys.size(); ++i) {
    out[i] = store_->is_exist(keys[i].c_str());
  }
  return out;
}

bool MooncakeMockConnector::StoreIsExist(const std::string& key) {
  const std::vector<bool> r = StoreBatchIsExist({key});
  return !r.empty() && r[0];
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
