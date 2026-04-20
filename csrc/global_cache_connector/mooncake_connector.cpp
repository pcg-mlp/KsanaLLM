/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "global_cache_connector/mooncake_connector.h"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "utils/logger.h"
#include "device/device_utils.h"

// MAP_HUGE_2MB is in <linux/mman.h> but not always in <sys/mman.h>.
#ifndef MAP_HUGE_SHIFT
#  define MAP_HUGE_SHIFT 26
#endif
#ifndef MAP_HUGE_2MB
#  define MAP_HUGE_2MB (21 << MAP_HUGE_SHIFT)
#endif

namespace ksana_llm {

#ifndef NDEBUG
static int64_t ComputeInt8Checksum(const void* data, size_t byte_size) {
  int64_t sum = 0;
  const int8_t* ptr = reinterpret_cast<const int8_t*>(data);
  for (size_t i = 0; i < byte_size; ++i) {
    sum += ptr[i];
  }
  return sum;
}
#endif

MooncakeConnector::~MooncakeConnector() {
  if (initialized_) {
    Shutdown();
  }
}

// ======================== Staging Buffer Pool ========================

void MooncakeConnector::InitStagingPool(int block_size, size_t block_byte_size) {
  std::call_once(staging_pool_init_flag_, [&] {
    size_t pool_count = (max_step_tokens_ + block_size - 1) / block_size;
    pool_count = std::max(pool_count, size_t(4));

    staging_pool_ = std::make_unique<StagingBufferPool>(pool_count, block_byte_size);

    // Register the entire contiguous buffer as a single RDMA memory region.
    auto reg_result =
        client_->RegisterLocalMemory(staging_pool_->BasePtr(), staging_pool_->TotalBytes(), "cpu:0", false, true);
    if (!reg_result.has_value()) {
      KLLM_LOG_ERROR << "InitStagingPool: RegisterLocalMemory failed, error=" << reg_result.error();
      staging_pool_.reset();
      return;
    }

    KLLM_LOG_INFO << "InitStagingPool: " << staging_pool_->BlockCount() << " blocks"
                  << ", total=" << staging_pool_->TotalBytes() << " bytes"
                  << ", block_size=" << block_byte_size << ", max_step_tokens=" << max_step_tokens_;
  });
}

// ======================== Initialize ========================

Status MooncakeConnector::Initialize(const GlobalCacheConnectorConfig& config) {
  KLLM_LOG_DEBUG << "MooncakeConnector::Initialize start"
                 << ", endpoint=" << config.service_endpoint << ", protocol=" << config.protocol
                 << ", master_server_addr=" << config.master_server_addr << ", device_id=" << config.device_id
                 << ", device_count=" << config.device_count;

  if (initialized_) {
    KLLM_LOG_WARNING << "MooncakeConnector::Initialize already initialized, skipped";
    return Status(RetCode::RET_INVALID_ARGUMENT, "MooncakeConnector already initialized.");
  }

  // 1. Resolve local IP
  std::string resolved_ip;
  std::string resolved_ip_source = "auto_detect";
  {
    const char* bind_addr_env = std::getenv("MC_TCP_BIND_ADDRESS");
    if (bind_addr_env != nullptr && bind_addr_env[0] != '\0') {
      resolved_ip = bind_addr_env;
      resolved_ip_source = "env:MC_TCP_BIND_ADDRESS";
      KLLM_LOG_INFO << "MooncakeConnector::Initialize using MC_TCP_BIND_ADDRESS=" << resolved_ip;
    } else {
      std::string master_host = config.master_server_addr;
      std::string master_port = "50051";
      auto colon_pos = master_host.find(':');
      if (colon_pos != std::string::npos) {
        master_port = master_host.substr(colon_pos + 1);
        master_host = master_host.substr(0, colon_pos);
      }

      struct addrinfo hints{}, *master_res = nullptr;
      hints.ai_family = AF_INET;
      hints.ai_socktype = SOCK_DGRAM;
      int gai_ret = getaddrinfo(master_host.c_str(), master_port.c_str(), &hints, &master_res);
      if (gai_ret == 0 && master_res) {
        int udp_sock = socket(AF_INET, SOCK_DGRAM, 0);
        if (udp_sock >= 0) {
          if (connect(udp_sock, master_res->ai_addr, master_res->ai_addrlen) == 0) {
            struct sockaddr_in local_addr{};
            socklen_t local_len = sizeof(local_addr);
            if (getsockname(udp_sock, reinterpret_cast<struct sockaddr*>(&local_addr), &local_len) == 0) {
              char ip_buf[INET_ADDRSTRLEN];
              inet_ntop(AF_INET, &local_addr.sin_addr, ip_buf, sizeof(ip_buf));
              resolved_ip = ip_buf;
            }
          }
          close(udp_sock);
        }
        freeaddrinfo(master_res);
      }

      if (resolved_ip.empty()) {
        KLLM_LOG_WARNING << "MooncakeConnector::Initialize failed to detect local IP"
                         << " via master_server_addr=" << config.master_server_addr << ", falling back to 127.0.0.1";
        resolved_ip = "127.0.0.1";
        resolved_ip_source = "fallback:127.0.0.1";
      }
    }
  }

  // 2. Bind ephemeral port
  int sock = socket(AF_INET, SOCK_STREAM, 0);
  if (sock < 0) {
    KLLM_LOG_ERROR << "MooncakeConnector::Initialize socket() failed: " << strerror(errno);
    return Status(RetCode::RET_RUNTIME_FAILED, "Failed to create socket for ephemeral port");
  }
  struct sockaddr_in addr;
  memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = 0;
  if (bind(sock, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) != 0) {
    close(sock);
    KLLM_LOG_ERROR << "MooncakeConnector::Initialize bind() failed: " << strerror(errno);
    return Status(RetCode::RET_RUNTIME_FAILED, "Failed to bind ephemeral port");
  }
  socklen_t addrlen = sizeof(addr);
  if (getsockname(sock, reinterpret_cast<struct sockaddr*>(&addr), &addrlen) != 0) {
    close(sock);
    KLLM_LOG_ERROR << "MooncakeConnector::Initialize getsockname() failed: " << strerror(errno);
    return Status(RetCode::RET_RUNTIME_FAILED, "Failed to get ephemeral port");
  }
  int port = ntohs(addr.sin_port);
  close(sock);

  local_hostname_ = resolved_ip + ":" + std::to_string(port);

  // 3. Client::Create
  std::optional<std::string> device_names;
  if (!config.rdma_devices.empty()) {
    device_names = config.rdma_devices;
  }

  auto client_opt = mooncake::Client::Create(local_hostname_, config.service_endpoint, config.protocol, device_names,
                                             config.master_server_addr);
  if (!client_opt.has_value()) {
    KLLM_LOG_ERROR << "MooncakeConnector::Initialize Client::Create failed"
                   << ", local_hostname=" << local_hostname_ << " (note: ephemeral port " << port
                   << " may have been taken by another process between close() and Create())";
    return Status(RetCode::RET_RUNTIME_FAILED, "mooncake::Client::Create failed");
  }
  client_ = client_opt.value();

  // 4. Store max_step_tokens for lazy pool init
  max_step_tokens_ = config.max_step_tokens;

  // 5. Segment buffer
  global_segment_size_ = config.global_segment_size;
  if (global_segment_size_ == 0) {
    global_segment_size_ = 16 * 1024 * 1024;
  }
  // Use mmap with MAP_HUGETLB to avoid kernel soft lockup.
  // With 4KB pages a 200GB buffer has ~52M pages.  Both ibv_reg_mr (RDMA pin)
  // and ibv_dereg_mr (RDMA unpin) iterate every page under a spinlock, causing
  // "soft lockup – CPU stuck for 23s" on both startup and shutdown.
  // 2MB huge pages reduce the count to ~100K — well within watchdog limits.
  // Set MC_DISABLE_HUGETLB=1 to explicitly use regular pages.
  constexpr size_t kHugePageSize2MB = 2 * 1024 * 1024;
  const char* disable_hugetlb_env = std::getenv("MC_DISABLE_HUGETLB");
  bool use_hugetlb = !(disable_hugetlb_env && disable_hugetlb_env[0] != '\0' && disable_hugetlb_env[0] != '0');

  if (use_hugetlb) {
    segment_mmap_size_ = (global_segment_size_ + kHugePageSize2MB - 1) & ~(kHugePageSize2MB - 1);
    segment_buffer_ = mmap(nullptr, segment_mmap_size_, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_HUGE_2MB, -1, 0);
    if (segment_buffer_ != MAP_FAILED) {
      KLLM_LOG_INFO << "MooncakeConnector::Initialize segment_buffer allocated with MAP_HUGETLB 2MB"
                    << ", size=" << segment_mmap_size_ << " (" << (segment_mmap_size_ / kHugePageSize2MB) << " pages)";
    } else {
      segment_buffer_ = nullptr;
      segment_mmap_size_ = 0;
      size_t needed_pages = (global_segment_size_ + kHugePageSize2MB - 1) / kHugePageSize2MB;
      KLLM_LOG_ERROR << "MooncakeConnector::Initialize MAP_HUGETLB failed (errno=" << errno << "). "
                     << "To fix, pre-allocate huge pages: "
                     << "echo " << needed_pages
                     << " > /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages  "
                     << "Or set MC_DISABLE_HUGETLB=1 to fall back to regular pages"
                     << " (may cause soft lockup on large buffers).";
      fprintf(stderr,
              "\n[MooncakeConnector] ERROR: MAP_HUGETLB allocation failed.\n"
              "  To fix, pre-allocate huge pages:\n"
              "    echo %zu > /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages\n"
              "  Or set MC_DISABLE_HUGETLB=1 to use regular pages (may cause soft lockup on large buffers).\n\n",
              needed_pages);
      client_.reset();
      return Status(RetCode::RET_RUNTIME_FAILED, "Failed to mmap segment buffer with MAP_HUGETLB");
    }
  } else {
    segment_mmap_size_ = global_segment_size_;
    segment_buffer_ = mmap(nullptr, segment_mmap_size_, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (segment_buffer_ != MAP_FAILED) {
      madvise(segment_buffer_, segment_mmap_size_, MADV_HUGEPAGE);
      KLLM_LOG_INFO << "MooncakeConnector::Initialize segment_buffer allocated with regular mmap + MADV_HUGEPAGE"
                    << " (MC_DISABLE_HUGETLB is set), size=" << segment_mmap_size_;
    } else {
      segment_buffer_ = nullptr;
      segment_mmap_size_ = 0;
      KLLM_LOG_ERROR << "MooncakeConnector::Initialize mmap segment_buffer failed, size=" << global_segment_size_;
      client_.reset();
      return Status(RetCode::RET_RUNTIME_FAILED, "Failed to mmap segment buffer");
    }
  }

  auto mount_result = client_->MountSegment(segment_buffer_, global_segment_size_);
  if (!mount_result.has_value()) {
    KLLM_LOG_ERROR << "MooncakeConnector::Initialize MountSegment failed"
                   << ", error=" << mount_result.error();
    munmap(segment_buffer_, segment_mmap_size_);
    segment_buffer_ = nullptr;
    segment_mmap_size_ = 0;
    client_.reset();
    return Status(RetCode::RET_RUNTIME_FAILED, "MountSegment failed");
  }

  initialized_ = true;
  KLLM_LOG_INFO << "MooncakeConnector::Initialize success"
                << ", local_hostname=" << local_hostname_ << ", endpoint=" << config.service_endpoint
                << ", protocol=" << config.protocol << ", max_step_tokens=" << max_step_tokens_
                << ", global_segment_size=" << global_segment_size_;

  // Eagerly initialize staging pool if block config is available (avoids first-request latency).
  if (config.block_token_num > 0 && config.block_byte_size > 0) {
    InitStagingPool(config.block_token_num, config.block_byte_size);
  }

  return Status();
}

// ======================== QueryAndFetch (BatchGet) ========================

Status MooncakeConnector::QueryAndFetch(const std::vector<int>& token_ids, size_t local_cached_len, int tp_size,
                                        int tp_rank, int layer_index, int block_size, size_t block_byte_size,
                                        const std::vector<void*>& block_ptrs, std::vector<int>& matched_ranges,
                                        bool retry_until_full_hit, int timeout_ms) {
  matched_ranges.clear();

  if (!initialized_ || block_size <= 0) {
    return Status();
  }

  if (token_ids.empty() || block_ptrs.empty()) {
    return Status();
  }

  InitStagingPool(block_size, block_byte_size);  // fallback if not eagerly initialized

  int total_tokens = static_cast<int>(token_ids.size());
  int total_blocks = static_cast<int>(block_ptrs.size());
  int start_block = static_cast<int>(local_cached_len) / block_size;

  KLLM_LOG_DEBUG << "QueryAndFetch start: total_tokens=" << total_tokens << ", total_blocks=" << total_blocks
                 << ", start_block=" << start_block << ", retry=" << retry_until_full_hit
                 << ", timeout_ms=" << timeout_ms;

  std::vector<bool> block_fetched(total_blocks, false);
  auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  constexpr int kRetryIntervalMs = 10;
  int attempt = 0;
  int remaining = total_blocks - start_block;

  // Per-hit item: groups key, block index, and staging buffer together.
  struct HitItem {
    std::string key;
    int block_idx;
    StagingBuffer* staging;
  };

  while (remaining > 0) {
    ++attempt;

    // Phase 1: Collect keys for unfetched blocks
    std::vector<std::string> pending_keys;
    std::vector<int> pending_block_indices;
    for (int bi = start_block; bi < total_blocks; ++bi) {
      if (block_fetched[bi]) continue;
      size_t prefix_len = std::min(static_cast<size_t>(bi + 1) * block_size, static_cast<size_t>(total_tokens));
      std::string prefix_hash = HashTokenPrefix(token_ids.data(), prefix_len);
      pending_keys.push_back(MakeBlockKey(prefix_hash, tp_rank, layer_index, bi));
      pending_block_indices.push_back(bi);
    }

    if (pending_keys.empty()) break;

    // Phase 2: BatchIsExist
    auto exist_results = client_->BatchIsExist(pending_keys);
    if (exist_results.size() != pending_keys.size()) {
      KLLM_LOG_ERROR << "QueryAndFetch: BatchIsExist size mismatch, expected=" << pending_keys.size()
                     << ", got=" << exist_results.size();
      return Status(RetCode::RET_RUNTIME_FAILED, "BatchIsExist result size mismatch");
    }

    // Phase 3: Filter hits and acquire staging buffers
    std::vector<HitItem> hits;
    std::unordered_map<std::string, std::vector<mooncake::Slice> > slices_map;

    for (size_t i = 0; i < pending_keys.size(); ++i) {
      if (!exist_results[i].has_value() || !exist_results[i].value()) continue;

      auto* blk = staging_pool_->TryGet();
      if (!blk) {
        KLLM_LOG_WARNING << "QueryAndFetch: staging pool exhausted, acquired " << hits.size() << " buffers";
        break;
      }
      slices_map[pending_keys[i]] = {{blk->ptr, block_byte_size}};
      hits.push_back({pending_keys[i], pending_block_indices[i], blk});
    }

    // Phase 4: BatchGet + Memcpy H2D + release staging buffers
    int hit_count = 0;
    if (!hits.empty()) {
      std::vector<std::string> hit_keys;
      hit_keys.reserve(hits.size());
      for (const auto& h : hits) hit_keys.push_back(h.key);

      auto get_results = client_->BatchGet(hit_keys, slices_map);
      if (get_results.size() != hits.size()) {
        KLLM_LOG_ERROR << "QueryAndFetch: BatchGet size mismatch, expected=" << hits.size()
                       << ", got=" << get_results.size();
        for (auto& h : hits) staging_pool_->Put(h.staging);
        return Status(RetCode::RET_RUNTIME_FAILED, "BatchGet result size mismatch");
      }

      for (size_t i = 0; i < hits.size(); ++i) {
        if (get_results[i].has_value()) {
#ifndef NDEBUG
          int64_t checksum = ComputeInt8Checksum(hits[i].staging->ptr, block_byte_size);
          KLLM_LOG_DEBUG << "QueryAndFetch block=" << hits[i].block_idx << " checksum=" << checksum;
#endif
          try {
            Memcpy(block_ptrs[hits[i].block_idx], hits[i].staging->ptr, block_byte_size, MEMCPY_HOST_TO_DEVICE);
            block_fetched[hits[i].block_idx] = true;
            ++hit_count;
          } catch (const std::exception& e) {
            KLLM_LOG_WARNING << "QueryAndFetch: Memcpy H2D failed for block=" << hits[i].block_idx
                             << ", error=" << e.what();
          }
        } else {
          KLLM_LOG_WARNING << "QueryAndFetch: BatchGet failed for block=" << hits[i].block_idx
                           << ", key=" << hits[i].key << ", error=" << get_results[i].error();
        }
        staging_pool_->Put(hits[i].staging);
      }
    }

    remaining -= hit_count;
    KLLM_LOG_DEBUG << "QueryAndFetch attempt=" << attempt << ", hit=" << hit_count << ", remaining=" << remaining;

    if (remaining <= 0) break;
    if (!retry_until_full_hit) break;
    if (std::chrono::steady_clock::now() >= deadline) {
      KLLM_LOG_WARNING << "QueryAndFetch retry timeout after " << attempt << " attempts, remaining=" << remaining;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(kRetryIntervalMs));
  }

  // Build matched_ranges
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

  KLLM_LOG_DEBUG << "QueryAndFetch done: matched_ranges_size=" << matched_ranges.size() << ", attempts=" << attempt;
  return Status();
}

// ======================== PutGlobalCacheBlocks (BatchPut) ========================

Status MooncakeConnector::PutGlobalCacheBlocks(const std::vector<int>& token_ids, int tp_size, int tp_rank,
                                               int layer_index, int block_size, size_t block_byte_size,
                                               const std::vector<int>& block_ranges,
                                               const std::vector<void*>& block_ptrs) {
  if (!initialized_ || block_size <= 0) {
    return Status(RetCode::RET_INVALID_ARGUMENT, "Connector not initialized or invalid block_size.");
  }
  if (block_ranges.size() % 2 != 0) {
    return Status(RetCode::RET_INVALID_ARGUMENT, "block_ranges size must be a multiple of 2.");
  }
  if (block_byte_size == 0) {
    return Status(RetCode::RET_INVALID_ARGUMENT, "block_byte_size is 0.");
  }

  InitStagingPool(block_size, block_byte_size);  // fallback if not eagerly initialized

  int total_tokens = static_cast<int>(token_ids.size());
  KLLM_LOG_DEBUG << "PutGlobalCacheBlocks start: total_tokens=" << total_tokens
                 << ", block_ptrs_size=" << block_ptrs.size() << ", block_ranges_size=" << block_ranges.size()
                 << ", block_size=" << block_size << ", block_byte_size=" << block_byte_size;

  // Phase 1: Enumerate all blocks and their keys
  struct BlockInfo {
    int block_idx;
    int ptr_idx;
    std::string key;
  };
  std::vector<BlockInfo> all_blocks;
  std::vector<std::string> all_keys;
  int ptr_idx = 0;
  for (size_t i = 0; i < block_ranges.size(); i += 2) {
    int start_block = block_ranges[i] / block_size;
    int end_block = block_ranges[i + 1] / block_size;
    for (int bi = start_block; bi <= end_block; ++bi) {
      if (ptr_idx >= static_cast<int>(block_ptrs.size())) break;
      size_t prefix_len = std::min(static_cast<size_t>(bi + 1) * block_size, static_cast<size_t>(total_tokens));
      std::string prefix_hash = HashTokenPrefix(token_ids.data(), prefix_len);
      std::string key = MakeBlockKey(prefix_hash, tp_rank, layer_index, bi);
      all_blocks.push_back({bi, ptr_idx, key});
      all_keys.push_back(key);
      ++ptr_idx;
    }
  }

  if (all_keys.empty()) return Status();

  // Phase 2: BatchIsExist to filter out already stored blocks
  auto exist_results = client_->BatchIsExist(all_keys);
  if (exist_results.size() != all_keys.size()) {
    KLLM_LOG_ERROR << "PutGlobalCacheBlocks: BatchIsExist size mismatch, expected=" << all_keys.size()
                   << ", got=" << exist_results.size();
    return Status(RetCode::RET_RUNTIME_FAILED, "BatchIsExist result size mismatch");
  }

  std::vector<BlockInfo> need_put;
  int skip_exist_count = 0;
  for (size_t i = 0; i < all_blocks.size(); ++i) {
    if (exist_results[i].has_value() && exist_results[i].value()) {
      ++skip_exist_count;
    } else {
      need_put.push_back(all_blocks[i]);
    }
  }

  if (need_put.empty()) {
    KLLM_LOG_DEBUG << "PutGlobalCacheBlocks: all " << skip_exist_count << " blocks already exist";
    return Status();
  }

  // Phase 3: Acquire staging buffers and Memcpy D2H (process in batches if pool < need_put)
  int put_count = 0;
  int fail_count = 0;
  size_t offset = 0;

  while (offset < need_put.size()) {
    std::vector<mooncake::ObjectKey> batch_keys;
    std::vector<std::vector<mooncake::Slice> > batch_slices;
    std::vector<StagingBuffer*> acquired_blks;

    for (; offset < need_put.size(); ++offset) {
      auto* blk = staging_pool_->TryGet();
      if (!blk) break;

      const auto& info = need_put[offset];
      try {
        Memcpy(blk->ptr, block_ptrs[info.ptr_idx], block_byte_size, MEMCPY_DEVICE_TO_HOST);
      } catch (const std::exception& e) {
        KLLM_LOG_WARNING << "PutGlobalCacheBlocks: Memcpy D2H failed for block=" << info.block_idx
                         << ", error=" << e.what();
        staging_pool_->Put(blk);
        ++fail_count;
        continue;
      }

#ifndef NDEBUG
      int64_t checksum = ComputeInt8Checksum(blk->ptr, block_byte_size);
      KLLM_LOG_DEBUG << "PutGlobalCacheBlocks block=" << info.block_idx << " key=" << info.key
                     << " checksum=" << checksum;
#endif

      batch_keys.push_back(info.key);
      batch_slices.push_back({{blk->ptr, block_byte_size}});
      acquired_blks.push_back(blk);
    }

    // Phase 4: BatchPut
    if (!batch_keys.empty()) {
      mooncake::ReplicateConfig rep_config;
      auto put_results = client_->BatchPut(batch_keys, batch_slices, rep_config);
      if (put_results.size() != batch_keys.size()) {
        KLLM_LOG_ERROR << "PutGlobalCacheBlocks: BatchPut size mismatch, expected=" << batch_keys.size()
                       << ", got=" << put_results.size();
        for (auto* blk : acquired_blks) staging_pool_->Put(blk);
        return Status(RetCode::RET_RUNTIME_FAILED, "BatchPut result size mismatch");
      }
      for (size_t i = 0; i < put_results.size(); ++i) {
        if (put_results[i].has_value()) {
          ++put_count;
        } else {
          KLLM_LOG_WARNING << "PutGlobalCacheBlocks: BatchPut failed for key=" << batch_keys[i]
                           << ", error=" << put_results[i].error();
          ++fail_count;
        }
      }
    }

    // Phase 5: Release staging buffers
    for (auto* blk : acquired_blks) {
      staging_pool_->Put(blk);
    }
  }

  KLLM_LOG_DEBUG << "PutGlobalCacheBlocks done: put=" << put_count << ", skip_exist=" << skip_exist_count
                 << ", fail=" << fail_count;
  return Status();
}

// ======================== PutFirstTokens ========================

Status MooncakeConnector::PutFirstTokens(const std::vector<int>& token_ids, const std::vector<int>& tokens) {
  if (!initialized_) {
    return Status(RetCode::RET_INVALID_ARGUMENT, "Connector not initialized.");
  }
  if (token_ids.empty() || tokens.empty()) {
    return Status();
  }

  std::string prefix_hash = HashTokenPrefix(token_ids.data(), token_ids.size());
  std::string key = MakeFirstTokenKey(prefix_hash);

  auto exist_result = client_->IsExist(key);
  if (exist_result.has_value() && exist_result.value()) {
    return Status();
  }

  std::vector<int> payload;
  payload.reserve(1 + tokens.size());
  payload.push_back(static_cast<int>(tokens.size()));
  payload.insert(payload.end(), tokens.begin(), tokens.end());
  size_t payload_bytes = payload.size() * sizeof(int);

  auto* blk = staging_pool_ ? staging_pool_->TryGet() : nullptr;
  if (!blk) {
    KLLM_LOG_WARNING << "PutFirstTokens: staging pool unavailable, key=" << key;
    return Status(RetCode::RET_RUNTIME_FAILED, "PutFirstTokens: staging pool unavailable");
  }

  memcpy(blk->ptr, payload.data(), payload_bytes);
  std::vector<mooncake::Slice> slices = {{blk->ptr, payload_bytes}};
  mooncake::ReplicateConfig rep_config;
  auto put_result = client_->Put(key, slices, rep_config);
  staging_pool_->Put(blk);

  if (!put_result.has_value()) {
    return Status(RetCode::RET_RUNTIME_FAILED, "PutFirstTokens Put failed");
  }

  KLLM_LOG_DEBUG << "PutFirstTokens key=" << key << ", token_count=" << tokens.size();
  return Status();
}

// ======================== QueryFirstTokens ========================

Status MooncakeConnector::QueryFirstTokens(const std::vector<int>& token_ids, std::vector<int>& tokens, bool& found) {
  found = false;
  tokens.clear();

  if (!initialized_ || token_ids.empty()) {
    return Status();
  }

  std::string prefix_hash = HashTokenPrefix(token_ids.data(), token_ids.size());
  std::string key = MakeFirstTokenKey(prefix_hash);

  auto exist_result = client_->IsExist(key);
  if (!exist_result.has_value() || !exist_result.value()) {
    return Status();
  }

  auto query_result = client_->Query(key);
  if (!query_result.has_value()) {
    return Status();
  }

  size_t total_size = 0;
  bool found_replica = false;
  for (const auto& replica : query_result.value().replicas) {
    if (replica.status != mooncake::ReplicaStatus::COMPLETE) continue;
    if (replica.is_memory_replica()) {
      const auto& mem_desc = replica.get_memory_descriptor();
      total_size = mem_desc.buffer_descriptor.size_;
      found_replica = true;
      break;
    } else if (replica.is_disk_replica()) {
      total_size = replica.get_disk_descriptor().object_size;
      found_replica = true;
      break;
    }
  }

  if (!found_replica || total_size == 0 || total_size < sizeof(int) || total_size % sizeof(int) != 0) {
    return Status();
  }

  auto* blk = staging_pool_ ? staging_pool_->TryGet() : nullptr;
  if (!blk) {
    KLLM_LOG_WARNING << "QueryFirstTokens: staging pool exhausted";
    return Status();
  }

  if (total_size > blk->capacity) {
    KLLM_LOG_ERROR << "QueryFirstTokens: total_size=" << total_size
                   << " exceeds staging buffer capacity=" << blk->capacity;
    staging_pool_->Put(blk);
    return Status();
  }

  std::vector<mooncake::Slice> slices = {{blk->ptr, total_size}};
  auto get_result = client_->Get(key, query_result.value(), slices);
  if (!get_result.has_value()) {
    staging_pool_->Put(blk);
    return Status();
  }

  size_t num_ints = total_size / sizeof(int);
  const int* payload_ptr = reinterpret_cast<const int*>(blk->ptr);
  int count = payload_ptr[0];
  if (count <= 0 || static_cast<size_t>(count) + 1 != num_ints) {
    staging_pool_->Put(blk);
    return Status();
  }

  tokens.assign(payload_ptr + 1, payload_ptr + num_ints);
  found = true;
  staging_pool_->Put(blk);

  KLLM_LOG_DEBUG << "QueryFirstTokens key=" << key << " hit, token_count=" << tokens.size();
  return Status();
}

// ======================== Shutdown ========================

Status MooncakeConnector::Shutdown() {
  KLLM_LOG_INFO << "MooncakeConnector::Shutdown start, initialized=" << initialized_;

  if (!initialized_) {
    return Status();
  }

  if (client_) {
    if (staging_pool_) {
      client_->unregisterLocalMemory(staging_pool_->BasePtr());
      staging_pool_.reset();
    }

    if (segment_buffer_) {
      auto unmount_result = client_->UnmountSegment(segment_buffer_, global_segment_size_);
      if (!unmount_result.has_value()) {
        KLLM_LOG_WARNING << "MooncakeConnector::Shutdown UnmountSegment failed, error=" << unmount_result.error();
      }
    }

    client_.reset();
  }

  if (segment_buffer_) {
    munmap(segment_buffer_, segment_mmap_size_);
    segment_buffer_ = nullptr;
    segment_mmap_size_ = 0;
    global_segment_size_ = 0;
  }

  initialized_ = false;
  KLLM_LOG_INFO << "MooncakeConnector::Shutdown done";
  return Status();
}

}  // namespace ksana_llm

// ======================== Plugin Export ========================

#ifdef __cplusplus
extern "C" {
#endif

ksana_llm::GlobalCacheConnectorInterface* CreateMooncakeConnector() { return new ksana_llm::MooncakeConnector(); }

#ifdef __cplusplus
}
#endif
