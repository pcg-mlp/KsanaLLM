/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "global_cache_connector/mooncake_connector.h"

#include <arpa/inet.h>
#include <cuda_runtime.h>
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
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "device/device_utils.h"
#include "profiler/timer.h"
#include "utils/logger.h"

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

    try {
      staging_pool_ = std::make_unique<StagingBufferPool>(pool_count, block_byte_size);
    } catch (const std::exception& e) {
      KLLM_LOG_ERROR << "StagingBufferPool init failed, device=" << device_id_ << " pool_count=" << pool_count
                     << " block_byte_size=" << block_byte_size << " total_bytes=" << (pool_count * block_byte_size)
                     << " what=" << e.what();
      throw;
    }

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

void MooncakeConnector::InitCopyStreamPools(int max_concurrent_threads) {
  KLLM_CHECK_WITH_INFO(max_concurrent_threads >= 1,
                       fmt::format("max_concurrent_threads={} must be >= 1", max_concurrent_threads));

  const size_t pool_size = static_cast<size_t>(max_concurrent_threads);
  SetDevice(device_id_);
  query_h2d_streams_.clear();
  put_d2h_streams_.clear();
  query_h2d_streams_.reserve(pool_size);
  put_d2h_streams_.reserve(pool_size);
  query_h2d_next_slot_.store(0);
  put_d2h_next_slot_.store(0);
  for (size_t i = 0; i < pool_size; ++i) {
    // 默认优先级（与 compute 同级），不再用 kLow，避免传输长期被挤掉。
    query_h2d_streams_.push_back(std::make_unique<Stream>());
    put_d2h_streams_.push_back(std::make_unique<Stream>());
  }
  KLLM_LOG_INFO << "copy stream pools ready, pool_size=" << pool_size << ", device_id=" << device_id_;
}

Stream& MooncakeConnector::GetQueryH2DStream() {
  KLLM_CHECK_WITH_INFO(!query_h2d_streams_.empty(), "query_h2d_streams_ is empty");
  thread_local size_t tls_slot = static_cast<size_t>(-1);
  thread_local const MooncakeConnector* tls_owner = nullptr;
  if (tls_owner != this || tls_slot == static_cast<size_t>(-1)) {
    const size_t slot = query_h2d_next_slot_.fetch_add(1);
    KLLM_CHECK_WITH_INFO(
        slot < query_h2d_streams_.size(),
        fmt::format("query H2D stream 1:1 bind failed: thread_slot={} pool_size={}", slot, query_h2d_streams_.size()));
    tls_slot = slot;
    tls_owner = this;
  }
  return *query_h2d_streams_[tls_slot];
}

Stream& MooncakeConnector::GetPutD2HStream() {
  KLLM_CHECK_WITH_INFO(!put_d2h_streams_.empty(), "put_d2h_streams_ is empty");
  thread_local size_t tls_slot = static_cast<size_t>(-1);
  thread_local const MooncakeConnector* tls_owner = nullptr;
  if (tls_owner != this || tls_slot == static_cast<size_t>(-1)) {
    const size_t slot = put_d2h_next_slot_.fetch_add(1);
    KLLM_CHECK_WITH_INFO(
        slot < put_d2h_streams_.size(),
        fmt::format("put D2H stream 1:1 bind failed: thread_slot={} pool_size={}", slot, put_d2h_streams_.size()));
    tls_slot = slot;
    tls_owner = this;
  }
  return *put_d2h_streams_[tls_slot];
}

void MooncakeConnector::DestroyCopyStreamPools() {
  auto cleanup = [](std::vector<std::unique_ptr<Stream>>& streams, const char* name) {
    for (size_t i = 0; i < streams.size(); ++i) {
      if (!streams[i]) {
        continue;
      }
      try {
        StreamSynchronize(*streams[i]);
        streams[i]->Destroy();
      } catch (const std::exception& e) {
        KLLM_LOG_WARNING << name << "[" << i << "] cleanup failed: " << e.what();
      }
      streams[i].reset();
    }
    streams.clear();
  };
  cleanup(query_h2d_streams_, "query_h2d_stream");
  cleanup(put_d2h_streams_, "put_d2h_stream");
  query_h2d_next_slot_.store(0);
  put_d2h_next_slot_.store(0);
}

bool MooncakeConnector::AllowPutByHotkeyFrequency(const std::string& key) {
  if (put_hotkey_window_us_ <= 0) {
    return true;
  }
  const int64_t now_us = ProfileTimer::GetCurrentTimeInUs();
  std::lock_guard<std::mutex> lock(put_hotkey_mu_);
  // 表过大时清掉窗口外过期项，避免无限增长。
  constexpr size_t kMaxHotkeyEntries = 500000;
  if (put_hotkey_last_us_.size() > kMaxHotkeyEntries) {
    for (auto it = put_hotkey_last_us_.begin(); it != put_hotkey_last_us_.end();) {
      if (now_us - it->second > put_hotkey_window_us_) {
        it = put_hotkey_last_us_.erase(it);
      } else {
        ++it;
      }
    }
  }
  auto it = put_hotkey_last_us_.find(key);
  if (it == put_hotkey_last_us_.end()) {
    // 第一次出现：记时并跳过 Put。
    put_hotkey_last_us_.emplace(key, now_us);
    return false;
  }
  if (now_us - it->second > put_hotkey_window_us_) {
    // 超过窗口：重新当作第一次。
    it->second = now_us;
    return false;
  }
  // 窗口内再次出现：允许 Put，并刷新时间戳。
  it->second = now_us;
  return true;
}

void MooncakeConnector::InitSwaStagingPool(int block_size, size_t swa_block_byte_size) {
  std::call_once(swa_staging_pool_init_flag_, [&] {
    size_t pool_count = (max_step_tokens_ + block_size - 1) / block_size;
    pool_count = std::max(pool_count, size_t(4));

    swa_staging_pool_ = std::make_unique<StagingBufferPool>(pool_count, swa_block_byte_size);

    auto reg_result = client_->RegisterLocalMemory(swa_staging_pool_->BasePtr(), swa_staging_pool_->TotalBytes(),
                                                   "cpu:0", false, true);
    if (!reg_result.has_value()) {
      KLLM_LOG_ERROR << "InitSwaStagingPool: RegisterLocalMemory failed, error=" << reg_result.error();
      swa_staging_pool_.reset();
      return;
    }

    KLLM_LOG_INFO << "InitSwaStagingPool: " << swa_staging_pool_->BlockCount() << " blocks"
                  << ", total=" << swa_staging_pool_->TotalBytes() << " bytes"
                  << ", block_size=" << swa_block_byte_size << ", max_step_tokens=" << max_step_tokens_;
  });
}

// ======================== Initialize ========================

Status MooncakeConnector::Initialize(const GlobalCacheConnectorConfig& config) {
  KLLM_LOG_INFO << "Initialize start"
                << ", endpoint=" << config.service_endpoint << ", protocol=" << config.protocol
                << ", master_server_addr=" << config.master_server_addr << ", device_id=" << config.device_id
                << ", device_count=" << config.device_count;

  if (initialized_) {
    KLLM_LOG_WARNING << "Initialize already initialized, skipped";
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
      KLLM_LOG_INFO << "using MC_TCP_BIND_ADDRESS=" << resolved_ip;
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
        KLLM_LOG_WARNING << "Initialize failed to detect local IP"
                         << " via master_server_addr=" << config.master_server_addr << ", falling back to 127.0.0.1";
        resolved_ip = "127.0.0.1";
        resolved_ip_source = "fallback:127.0.0.1";
      }
    }
  }

  // 2–3. 选 ephemeral 端口并 Client::Create。
  // bind→close→Create 之间存在 TOCTOU：多卡并发初始化时端口可能被抢走，需换端口重试。
  std::optional<std::string> device_names;
  if (!config.rdma_devices.empty()) {
    device_names = config.rdma_devices;
  }

  constexpr int kClientCreateMaxAttempts = 8;
  bool client_created = false;
  for (int attempt = 1; attempt <= kClientCreateMaxAttempts; ++attempt) {
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
      KLLM_LOG_ERROR << "socket() failed: " << strerror(errno);
      return Status(RetCode::RET_RUNTIME_FAILED, "Failed to create socket for ephemeral port");
    }
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = 0;
    if (bind(sock, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) != 0) {
      close(sock);
      KLLM_LOG_ERROR << "bind() failed: " << strerror(errno);
      return Status(RetCode::RET_RUNTIME_FAILED, "Failed to bind ephemeral port");
    }
    socklen_t addrlen = sizeof(addr);
    if (getsockname(sock, reinterpret_cast<struct sockaddr*>(&addr), &addrlen) != 0) {
      close(sock);
      KLLM_LOG_ERROR << "getsockname() failed: " << strerror(errno);
      return Status(RetCode::RET_RUNTIME_FAILED, "Failed to get ephemeral port");
    }
    const int port = ntohs(addr.sin_port);
    close(sock);

    local_hostname_ = resolved_ip + ":" + std::to_string(port);
    auto client_opt = mooncake::Client::Create(local_hostname_, config.service_endpoint, config.protocol, device_names,
                                               config.master_server_addr);
    if (client_opt.has_value()) {
      client_ = client_opt.value();
      client_created = true;
      if (attempt > 1) {
        KLLM_LOG_WARNING << "Client::Create succeeded after retry, attempt=" << attempt
                         << ", local_hostname=" << local_hostname_ << ", device_id=" << config.device_id;
      }
      break;
    }
    KLLM_LOG_WARNING << "Client::Create failed, attempt=" << attempt << "/" << kClientCreateMaxAttempts
                     << ", local_hostname=" << local_hostname_ << ", device_id=" << config.device_id
                     << " (ephemeral port may have been taken between close() and Create())";
    if (attempt < kClientCreateMaxAttempts) {
      std::this_thread::sleep_for(std::chrono::milliseconds(20 * attempt));
    }
  }
  if (!client_created) {
    KLLM_LOG_ERROR << "Client::Create failed after " << kClientCreateMaxAttempts
                   << " attempts, last_local_hostname=" << local_hostname_ << ", device_id=" << config.device_id;
    return Status(RetCode::RET_RUNTIME_FAILED, "mooncake::Client::Create failed");
  }

  // 4. Store max_step_tokens for lazy pool init
  max_step_tokens_ = config.max_step_tokens;

  // 保存多卡分片信息，PutGlobalCacheBlocks 中按 device_id/device_count 平分上传任务，
  // 避免多张卡重复上传内容完全一致的 block。
  device_id_ = config.device_id;
  device_count_ = config.device_count > 0 ? config.device_count : 1;

  // 5. Segment buffer
  // global_segment_size=0：dry-run，不挂本地 store 容量；Query 仍走 BatchIsExist（通常全 miss），Put 直接跳过。
  global_segment_size_ = config.global_segment_size;
  if (global_segment_size_ == 0) {
    initialized_ = true;
    InitCopyStreamPools(config.fetch_thread_num);
    KLLM_LOG_INFO
        << "dry-run mode (global_segment_size=0): skip MountSegment, Put skipped; Query still runs (expect miss)"
        << ", local_hostname=" << local_hostname_ << ", endpoint=" << config.service_endpoint
        << ", protocol=" << config.protocol;
    return Status();
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
      KLLM_LOG_INFO << "segment_buffer allocated with MAP_HUGETLB 2MB"
                    << ", size=" << segment_mmap_size_ << " (" << (segment_mmap_size_ / kHugePageSize2MB) << " pages)";
    } else {
      segment_buffer_ = nullptr;
      segment_mmap_size_ = 0;
      size_t needed_pages = (global_segment_size_ + kHugePageSize2MB - 1) / kHugePageSize2MB;
      KLLM_LOG_ERROR << "Initialize MAP_HUGETLB failed (errno=" << errno << "). "
                     << "To fix, pre-allocate huge pages: "
                     << "echo " << needed_pages << " > /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages  "
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
      KLLM_LOG_INFO << "segment_buffer allocated with regular mmap + MADV_HUGEPAGE"
                    << " (MC_DISABLE_HUGETLB is set), size=" << segment_mmap_size_;
    } else {
      segment_buffer_ = nullptr;
      segment_mmap_size_ = 0;
      KLLM_LOG_ERROR << "Initialize mmap segment_buffer failed, size=" << global_segment_size_;
      client_.reset();
      return Status(RetCode::RET_RUNTIME_FAILED, "Failed to mmap segment buffer");
    }
  }

  auto mount_result = client_->MountSegment(segment_buffer_, global_segment_size_);
  if (!mount_result.has_value()) {
    KLLM_LOG_ERROR << "Initialize MountSegment failed"
                   << ", error=" << mount_result.error();
    munmap(segment_buffer_, segment_mmap_size_);
    segment_buffer_ = nullptr;
    segment_mmap_size_ = 0;
    client_.reset();
    return Status(RetCode::RET_RUNTIME_FAILED, "MountSegment failed");
  }

  initialized_ = true;

  // Put 热 key 窗口：0=关闭；>0 时仅窗口内第二次及以后出现才上传。
  put_hotkey_window_us_ = config.put_hotkey_window_sec > 0 ? config.put_hotkey_window_sec * 1000000 : 0;
  {
    std::lock_guard<std::mutex> lock(put_hotkey_mu_);
    put_hotkey_last_us_.clear();
  }

  // Query H2D / Put D2H 各一组拷贝流，与计算流隔离；仅在成功初始化后创建。
  InitCopyStreamPools(config.fetch_thread_num);

  KLLM_LOG_INFO << "Initialize success"
                << ", local_hostname=" << local_hostname_ << ", endpoint=" << config.service_endpoint
                << ", protocol=" << config.protocol << ", max_step_tokens=" << max_step_tokens_
                << ", global_segment_size=" << global_segment_size_
                << ", put_hotkey_window_sec=" << config.put_hotkey_window_sec;

  // Eagerly initialize staging pool if block config is available (avoids first-request latency).
  if (config.block_token_num > 0 && config.block_byte_size > 0) {
    InitStagingPool(config.block_token_num, config.block_byte_size);
  }
  if (config.block_token_num > 0 && config.swa_block_byte_size > 0) {
    InitSwaStagingPool(config.block_token_num, config.swa_block_byte_size);
  }

  return Status();
}

// ======================== SWA 命中边界对齐 ========================

int MooncakeConnector::AlignHitBoundaryBySwa(const std::vector<int>& token_ids, int tp_rank, int layer_index,
                                             int block_size, int start_block, int fa_contig_end,
                                             const GlobalCacheSwaContext& swa_ctx, GlobalCacheSwaResult* swa_result,
                                             const std::vector<std::string>& block_hashes, int hash_start_block) {
  (void)token_ids;
  (void)tp_rank;
  const int window = static_cast<int>(std::max<size_t>(swa_ctx.swa_window_block_num, 1));
  if (fa_contig_end <= start_block) {
    return start_block;
  }

  // 本地 SWA 落点：拉下来的内容只能写进这些位置，因此它也是边界合法性的判据之一。
  std::unordered_map<size_t, void*> local_swa_ptrs;
  const size_t local_swa_num = std::min(swa_ctx.local_swa_block_idxs.size(), swa_ctx.swa_block_ptrs.size());
  for (size_t i = 0; i < local_swa_num; ++i) {
    local_swa_ptrs[swa_ctx.local_swa_block_idxs[i]] = swa_ctx.swa_block_ptrs[i];
  }

  // 只需查 [start_block, fa_contig_end)：边界不会超过主 KV 连续命中上界。
  std::vector<std::string> swa_keys;
  swa_keys.reserve(static_cast<size_t>(fa_contig_end - start_block));
  for (int bi = start_block; bi < fa_contig_end; ++bi) {
    const size_t hash_idx = static_cast<size_t>(bi - hash_start_block);
    if (hash_idx >= block_hashes.size()) {
      KLLM_LOG_ERROR << "prefix hash missing for block=" << bi;
      return start_block;
    }
    // SWA 按本卡 device_id_ 分 key（与主 KV 调用方 tp_rank 可能不同）。
    swa_keys.push_back(MakeSwaBlockKey(block_hashes[hash_idx], device_id_, layer_index, bi));
  }

  auto swa_exist = client_->BatchIsExist(swa_keys);
  if (swa_exist.size() != swa_keys.size()) {
    KLLM_LOG_ERROR << "SWA BatchIsExist size mismatch, got=" << swa_exist.size() << "/" << swa_keys.size();
    return start_block;
  }

  auto exists = [](const auto& results, size_t i) { return results[i].has_value() && results[i].value(); };

  // 从最远往回找第一个满足「末 window 个位置可用」的边界。
  // 位置 < start_block 落在本地命中范围内，其 SWA 由本地保证有效，无需全局内容。
  auto swa_usable = [&](int idx) {
    if (idx < start_block) {
      return true;
    }
    return exists(swa_exist, idx - start_block) && local_swa_ptrs.count(static_cast<size_t>(idx)) > 0;
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

  // 把边界窗口内需要全局内容的位置拉下来。
  std::vector<int> need_idxs;
  for (int idx = std::max(boundary - window, start_block); idx < boundary; ++idx) {
    need_idxs.push_back(idx);
  }

  if (!need_idxs.empty()) {
    InitSwaStagingPool(block_size, swa_ctx.swa_block_byte_size);
    if (!swa_staging_pool_) {
      KLLM_LOG_WARNING << "swa staging pool unavailable, drop global extension";
      return start_block;
    }

    struct SwaHit {
      std::string key;
      int block_idx;
      StagingBuffer* staging;
    };
    std::vector<SwaHit> hits;
    std::unordered_map<std::string, std::vector<mooncake::Slice>> slices_map;
    for (int idx : need_idxs) {
      auto* blk = swa_staging_pool_->TryGet();
      if (!blk) {
        KLLM_LOG_WARNING << "swa staging pool exhausted at idx=" << idx;
        break;
      }
      const std::string& key = swa_keys[idx - start_block];
      slices_map[key] = {{blk->ptr, swa_ctx.swa_block_byte_size}};
      hits.push_back({key, idx, blk});
    }

    // 边界窗口 SWA 拉到 staging 后，走独立 query H2D stream 写入本地 SWA（与 FA Query 同模型）。
    bool all_ok = hits.size() == need_idxs.size();
    if (all_ok) {
      std::vector<std::string> hit_keys;
      hit_keys.reserve(hits.size());
      for (const auto& h : hits) {
        hit_keys.push_back(h.key);
      }
      auto get_results = client_->BatchGet(hit_keys, slices_map);
      all_ok = get_results.size() == hits.size();
      std::vector<size_t> pending_h2d;
      pending_h2d.reserve(hits.size());
      for (size_t i = 0; i < hits.size() && all_ok; ++i) {
        if (!get_results[i].has_value()) {
          KLLM_LOG_WARNING << "BatchGet failed for swa block=" << hits[i].block_idx
                           << ", error=" << get_results[i].error();
          all_ok = false;
          break;
        }
        pending_h2d.push_back(i);
      }
      if (all_ok && !pending_h2d.empty()) {
        if (query_h2d_streams_.empty()) {
          KLLM_LOG_WARNING << "query_h2d_streams empty, drop global extension";
          all_ok = false;
        } else {
          try {
            SetDevice(device_id_);
            Stream& h2d_stream = GetQueryH2DStream();
            for (size_t hi : pending_h2d) {
              void* dst = local_swa_ptrs[static_cast<size_t>(hits[hi].block_idx)];
              void* src = hits[hi].staging->ptr;
              if (dst == nullptr || src == nullptr) {
                KLLM_LOG_WARNING << "H2D bad ptr for swa block=" << hits[hi].block_idx;
                all_ok = false;
                break;
              }
              MemcpyAsync(dst, src, swa_ctx.swa_block_byte_size, MEMCPY_HOST_TO_DEVICE, h2d_stream);
            }
            if (all_ok) {
              StreamSynchronize(h2d_stream);
              for (size_t hi : pending_h2d) {
                swa_result->filled_block_idxs.push_back(static_cast<size_t>(hits[hi].block_idx));
              }
            }
          } catch (const std::exception& e) {
            KLLM_LOG_WARNING << "MemcpyAsync H2D failed, error=" << e.what();
            all_ok = false;
          }
        }
      }
    }

    for (auto& h : hits) {
      swa_staging_pool_->Put(h.staging);
    }

    if (!all_ok) {
      swa_result->filled_block_idxs.clear();
      return start_block;
    }
  }

  swa_result->swa_aligned_hit_len = static_cast<size_t>(boundary) * static_cast<size_t>(block_size);
  KLLM_LOG_DEBUG << "start_block=" << start_block << ", fa_contig_end=" << fa_contig_end << ", boundary=" << boundary
                 << ", swa_fetched=" << swa_result->filled_block_idxs.size();
  return boundary;
}

// ======================== QueryAndFetch (BatchGet) ========================

Status MooncakeConnector::QueryAndFetch(const std::vector<int>& token_ids, size_t local_cached_len, int tp_size,
                                        int tp_rank, int layer_index, int block_size, size_t block_byte_size,
                                        const std::vector<void*>& block_ptrs, std::vector<int>& matched_ranges,
                                        bool retry_until_full_hit, int timeout_ms, const GlobalCacheSwaContext* swa_ctx,
                                        GlobalCacheSwaResult* swa_result) {
  matched_ranges.clear();
  if (swa_result != nullptr) {
    *swa_result = GlobalCacheSwaResult{};
  }

  if (!initialized_ || block_size <= 0) {
    return Status();
  }

  if (token_ids.empty() || block_ptrs.empty()) {
    return Status();
  }

  if (swa_ctx != nullptr && swa_result == nullptr) {
    return Status(RetCode::RET_INVALID_ARGUMENT, "QueryAndFetch: swa_ctx given without swa_result.");
  }

  InitStagingPool(block_size, block_byte_size);  // fallback if not eagerly initialized

  int total_tokens = static_cast<int>(token_ids.size());
  int total_blocks = static_cast<int>(block_ptrs.size());
  int start_block = static_cast<int>(local_cached_len) / block_size;
  if (total_blocks <= start_block) {
    return Status();
  }

  // 入口只增量扫一次；Exist / Align / Get 共用。
  std::vector<std::string> block_hashes;
  HashTokenPrefixesIncremental(token_ids.data(), total_tokens, block_size, start_block, total_blocks, block_hashes);

  // SWA：命中上限与满块约束在 Exist 之前收紧扫描上界，避免查到不可用的尾部。
  int exist_end = total_blocks;
  if (swa_ctx != nullptr) {
    const int cap_blocks = static_cast<int>(swa_ctx->max_hit_token_num / static_cast<size_t>(block_size));
    const int full_blocks = total_tokens / block_size;
    exist_end = std::min({total_blocks, cap_blocks, full_blocks});
    if (exist_end <= start_block) {
      return Status();
    }
  }

  // 主 KV Exist 只做一次（重试轮除外）；结果供 SWA 裁剪与首轮 Get 复用。
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

  auto fa_exist = client_->BatchIsExist(fa_keys);
  if (fa_exist.size() != fa_keys.size()) {
    KLLM_LOG_ERROR << "BatchIsExist size mismatch, expected=" << fa_keys.size() << ", got=" << fa_exist.size();
    return Status(RetCode::RET_RUNTIME_FAILED, "BatchIsExist result size mismatch");
  }

  auto fa_key_exists = [&](size_t i) { return fa_exist[i].has_value() && fa_exist[i].value(); };

  // 连续前缀上界（开区间）；非 SWA 路径仍可按非整段连续命中拉后续洞后块。
  int fa_contig_end = start_block;
  for (size_t i = 0; i < fa_keys.size(); ++i) {
    if (!fa_key_exists(i)) {
      break;
    }
    fa_contig_end = start_block + static_cast<int>(i) + 1;
  }

  // 各 block 是否已知存在：首轮 Get 直接用，避免与上面 Exist 重复。
  std::vector<bool> known_exist(static_cast<size_t>(total_blocks), false);
  for (size_t i = 0; i < fa_keys.size(); ++i) {
    known_exist[static_cast<size_t>(start_block) + i] = fa_key_exists(i);
  }

  if (swa_ctx != nullptr) {
    total_blocks = AlignHitBoundaryBySwa(token_ids, tp_rank, layer_index, block_size, start_block, fa_contig_end,
                                         *swa_ctx, swa_result, block_hashes, start_block);
    if (total_blocks <= start_block) {
      return Status();
    }
  }

  KLLM_LOG_DEBUG << "start: total_tokens=" << total_tokens << ", total_blocks=" << total_blocks
                 << ", start_block=" << start_block << ", retry=" << retry_until_full_hit
                 << ", timeout_ms=" << timeout_ms;

  std::vector<bool> block_fetched(static_cast<size_t>(total_blocks), false);
  auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  constexpr int kRetryIntervalMs = 10;
  int attempt = 0;
  int remaining = total_blocks - start_block;
  // 首轮复用入口 Exist；仅 retry 时再查未拉块。
  bool reuse_known_exist = true;

  struct HitItem {
    std::string key;
    int block_idx;
    StagingBuffer* staging;
  };

  while (remaining > 0) {
    ++attempt;

    std::vector<std::string> pending_keys;
    std::vector<int> pending_block_indices;
    for (int bi = start_block; bi < total_blocks; ++bi) {
      if (block_fetched[static_cast<size_t>(bi)]) {
        continue;
      }
      const size_t hash_idx = static_cast<size_t>(bi - start_block);
      if (hash_idx >= block_hashes.size()) {
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
      decltype(client_->BatchIsExist(pending_keys)) exist_results = client_->BatchIsExist(pending_keys);
      if (exist_results.size() != pending_keys.size()) {
        KLLM_LOG_ERROR << "BatchIsExist size mismatch, expected=" << pending_keys.size()
                       << ", got=" << exist_results.size();
        return Status(RetCode::RET_RUNTIME_FAILED, "BatchIsExist result size mismatch");
      }
      for (size_t i = 0; i < pending_keys.size(); ++i) {
        pending_exist[i] = exist_results[i].has_value() && exist_results[i].value();
      }
    }

    std::vector<HitItem> hits;
    std::unordered_map<std::string, std::vector<mooncake::Slice> > slices_map;

    for (size_t i = 0; i < pending_keys.size(); ++i) {
      if (!pending_exist[i]) {
        continue;
      }

      auto* blk = staging_pool_->TryGet();
      if (!blk) {
        KLLM_LOG_WARNING << "staging pool exhausted, acquired " << hits.size() << " buffers";
        break;
      }
      slices_map[pending_keys[i]] = {{blk->ptr, block_byte_size}};
      hits.push_back({pending_keys[i], pending_block_indices[i], blk});
    }

    int hit_count = 0;
    if (!hits.empty()) {
      std::vector<std::string> hit_keys;
      hit_keys.reserve(hits.size());
      for (const auto& h : hits) {
        hit_keys.push_back(h.key);
      }

      decltype(client_->BatchGet(hit_keys, slices_map)) get_results = client_->BatchGet(hit_keys, slices_map);
      if (get_results.size() != hits.size()) {
        KLLM_LOG_ERROR << "BatchGet size mismatch, expected=" << hits.size() << ", got=" << get_results.size();
        for (auto& h : hits) {
          staging_pool_->Put(h.staging);
        }
        return Status(RetCode::RET_RUNTIME_FAILED, "BatchGet result size mismatch");
      }

      std::vector<size_t> pending_h2d;
      pending_h2d.reserve(hits.size());
      for (size_t i = 0; i < hits.size(); ++i) {
        if (get_results[i].has_value()) {
#ifndef NDEBUG
          int64_t checksum = ComputeInt8Checksum(hits[i].staging->ptr, block_byte_size);
          KLLM_LOG_DEBUG << "block=" << hits[i].block_idx << " checksum=" << checksum;
#endif
          pending_h2d.push_back(i);
        } else {
          KLLM_LOG_WARNING << "BatchGet failed for block=" << hits[i].block_idx << ", key=" << hits[i].key
                           << ", error=" << get_results[i].error();
        }
      }

      if (!pending_h2d.empty() && !query_h2d_streams_.empty()) {
        try {
          SetDevice(device_id_);
          Stream& h2d_stream = GetQueryH2DStream();
          for (size_t hi : pending_h2d) {
            const int bidx = hits[hi].block_idx;
            void* dst = (bidx >= 0 && static_cast<size_t>(bidx) < block_ptrs.size()) ? block_ptrs[bidx] : nullptr;
            void* src = hits[hi].staging != nullptr ? hits[hi].staging->ptr : nullptr;
            if (dst == nullptr || src == nullptr) {
              KLLM_LOG_ERROR << "H2D bad ptr, block_idx=" << bidx << " dst=" << dst << " src=" << src
                             << " n_ptrs=" << block_ptrs.size();
              continue;
            }
            MemcpyAsync(dst, src, block_byte_size, MEMCPY_HOST_TO_DEVICE, h2d_stream);
          }
          StreamSynchronize(h2d_stream);
          {
            const cudaError_t after = cudaPeekAtLastError();
            if (after != cudaSuccess) {
              std::ostringstream oss;
              const size_t dump_n = std::min<size_t>(pending_h2d.size(), 8);
              for (size_t di = 0; di < dump_n; ++di) {
                const int bidx = hits[pending_h2d[di]].block_idx;
                void* dst = (bidx >= 0 && static_cast<size_t>(bidx) < block_ptrs.size()) ? block_ptrs[bidx] : nullptr;
                if (di > 0) {
                  oss << ';';
                }
                oss << "b" << bidx << "=" << dst;
              }
              KLLM_LOG_ERROR << "H2D sync error, device=" << device_id_ << " err=" << cudaGetErrorString(after)
                             << " pending_h2d=" << pending_h2d.size() << " block_byte_size=" << block_byte_size
                             << " sample_dst=[" << oss.str() << "]";
            }
          }
          for (size_t hi : pending_h2d) {
            const int bidx = hits[hi].block_idx;
            if (bidx < 0 || static_cast<size_t>(bidx) >= block_ptrs.size() || block_ptrs[bidx] == nullptr ||
                hits[hi].staging == nullptr) {
              continue;
            }
            block_fetched[static_cast<size_t>(bidx)] = true;
            ++hit_count;
          }
        } catch (const std::exception& e) {
          KLLM_LOG_WARNING << "Query H2D copy/sync failed: " << e.what();
        }
      }

      for (auto& h : hits) {
        staging_pool_->Put(h.staging);
      }
    }

    remaining -= hit_count;
    KLLM_LOG_DEBUG << "attempt=" << attempt << ", hit=" << hit_count << ", remaining=" << remaining;

    if (remaining <= 0) {
      break;
    }
    if (!retry_until_full_hit) {
      break;
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      KLLM_LOG_WARNING << "retry timeout after " << attempt << " attempts, remaining=" << remaining;
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

  // SWA 对齐边界只在主 KV 完整覆盖 [start_block, boundary) 时才成立。
  if (swa_ctx != nullptr && remaining > 0) {
    KLLM_LOG_WARNING << "swa boundary abandoned, remaining=" << remaining << ", start_block=" << start_block
                     << ", total_blocks=" << total_blocks;
    matched_ranges.clear();
    if (swa_result != nullptr) {
      *swa_result = GlobalCacheSwaResult{};
    }
  }

  KLLM_LOG_SCHEDULER << "done, attempts=" << attempt << ", remaining=" << remaining
                     << ", matched_ranges=" << matched_ranges.size()
                     << ", swa_aligned_hit=" << (swa_result != nullptr ? swa_result->swa_aligned_hit_len : 0);

  return Status();
}

// ======================== PutGlobalCacheBlocks (BatchPut) ========================

Status MooncakeConnector::PutGlobalCacheBlocks(const std::vector<int>& token_ids, int tp_size, int tp_rank,
                                               int layer_index, int block_size, size_t block_byte_size,
                                               const std::vector<int>& block_ranges,
                                               const std::vector<void*>& block_ptrs, int* put_ok_out) {
  if (put_ok_out != nullptr) {
    *put_ok_out = 0;
  }
  if (!initialized_ || block_size <= 0) {
    return Status(RetCode::RET_INVALID_ARGUMENT, "Connector not initialized or invalid block_size.");
  }
  // dry-run：无本地 segment，跳过上传（Query 路径仍会走，用于观察全 miss 排队）
  if (global_segment_size_ == 0) {
    KLLM_LOG_DEBUG << "dry-run skip PutGlobalCacheBlocks, tokens=" << token_ids.size();
    return Status();
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
  // 多卡分片优化：由于所有 device 上的 block 内容和 key 都一致，
  // 通过 (global_idx % device_count_ == device_id_) 让每张卡只负责其中一部分 block，
  // 避免多卡重复上传同一份数据，从而摊薄整体上传带宽。
  // 注意：ptr_idx 仍然按原顺序递增，保证与 block_ptrs 中 GPU buffer 的对应关系不变。
  struct BlockInfo {
    int block_idx;
    int ptr_idx;
    std::string key;
  };
  std::vector<BlockInfo> all_blocks;
  std::vector<std::string> all_keys;
  int ptr_idx = 0;
  int global_idx = 0;
  int sharded_skip_count = 0;
  {
    // 先对将要覆盖的 block 范围一次增量扫完 hash，再组 key（禁止 per-block 全量重算）
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

    for (size_t i = 0; i < block_ranges.size(); i += 2) {
      int start_block = block_ranges[i] / block_size;
      int end_block = block_ranges[i + 1] / block_size;
      for (int bi = start_block; bi <= end_block; ++bi) {
        if (ptr_idx >= static_cast<int>(block_ptrs.size())) {
          break;
        }
        // 仅本卡负责的分片才参与后续 IsExist/Put 流程。
        if (device_count_ > 1 && (global_idx % device_count_) != device_id_) {
          ++ptr_idx;
          ++global_idx;
          ++sharded_skip_count;
          continue;
        }
        const size_t hash_idx = static_cast<size_t>(bi - hash_start);
        if (hash_idx >= block_hashes.size()) {
          ++ptr_idx;
          ++global_idx;
          continue;
        }
        std::string key = MakeBlockKey(block_hashes[hash_idx], tp_rank, layer_index, bi);
        all_blocks.push_back({bi, ptr_idx, key});
        all_keys.push_back(key);
        ++ptr_idx;
        ++global_idx;
      }
    }
  }

  if (all_keys.empty()) {
    KLLM_LOG_DEBUG << "nothing to put for device_id=" << device_id_ << "/device_count=" << device_count_
                   << ", sharded_skip=" << sharded_skip_count;
    return Status();
  }

  // Phase 2: BatchIsExist to filter out already stored blocks
  decltype(client_->BatchIsExist(all_keys)) exist_results = client_->BatchIsExist(all_keys);
  if (exist_results.size() != all_keys.size()) {
    KLLM_LOG_ERROR << "BatchIsExist size mismatch, expected=" << all_keys.size() << ", got=" << exist_results.size();
    return Status(RetCode::RET_RUNTIME_FAILED, "BatchIsExist result size mismatch");
  }

  std::vector<BlockInfo> need_put;
  for (size_t i = 0; i < all_blocks.size(); ++i) {
    if (exist_results[i].has_value() && exist_results[i].value()) {
      continue;
    }
    // 热 key 频率门控：窗口内首次跳过，第二次及以后才上传。
    if (!AllowPutByHotkeyFrequency(all_blocks[i].key)) {
      continue;
    }
    need_put.push_back(all_blocks[i]);
  }

  if (need_put.empty()) {
    return Status();
  }

  // Phase 3: Acquire staging buffers and Memcpy D2H (process in batches if pool < need_put)
  size_t offset = 0;
  int put_count = 0;

  while (offset < need_put.size()) {
    std::vector<mooncake::ObjectKey> batch_keys;
    std::vector<std::vector<mooncake::Slice> > batch_slices;
    std::vector<StagingBuffer*> acquired_blks;
    std::vector<size_t> pending_info_idx;  // need_put 下标，与 acquired_blks 对齐

    // 先占满本批 staging，再批量 MemcpyAsync D2H，一次 Sync 后再组 BatchPut
    for (; offset < need_put.size(); ++offset) {
      auto* blk = staging_pool_->TryGet();
      if (!blk) {
        break;
      }
      acquired_blks.push_back(blk);
      pending_info_idx.push_back(offset);
    }

    std::vector<StagingBuffer*> ready_blks;
    ready_blks.reserve(acquired_blks.size());
    if (!acquired_blks.empty() && !put_d2h_streams_.empty()) {
      // 独立 put D2H stream：MemcpyAsync + 本流同步。调用方须保证 GPU 上 KV 已写完再进入 Put。
      try {
        SetDevice(device_id_);
        Stream& d2h_stream = GetPutD2HStream();
        for (size_t j = 0; j < acquired_blks.size(); ++j) {
          const auto& info = need_put[pending_info_idx[j]];
          void* dst = acquired_blks[j] != nullptr ? acquired_blks[j]->ptr : nullptr;
          void* src = (info.ptr_idx < block_ptrs.size()) ? block_ptrs[info.ptr_idx] : nullptr;
          if (dst == nullptr || src == nullptr) {
            KLLM_LOG_ERROR << "D2H bad ptr, block_idx=" << info.block_idx << " ptr_idx=" << info.ptr_idx
                           << " dst=" << dst << " src=" << src;
            continue;
          }
          MemcpyAsync(dst, src, block_byte_size, MEMCPY_DEVICE_TO_HOST, d2h_stream);
        }
        StreamSynchronize(d2h_stream);
        {
          const cudaError_t after = cudaPeekAtLastError();
          if (after != cudaSuccess) {
            KLLM_LOG_ERROR << "D2H sync error, device=" << device_id_ << " err=" << cudaGetErrorString(after);
          }
        }
        for (size_t j = 0; j < acquired_blks.size(); ++j) {
          const auto& info = need_put[pending_info_idx[j]];
          if (acquired_blks[j] == nullptr || info.ptr_idx >= block_ptrs.size() || block_ptrs[info.ptr_idx] == nullptr) {
            continue;
          }
#ifndef NDEBUG
          int64_t checksum = ComputeInt8Checksum(acquired_blks[j]->ptr, block_byte_size);
          KLLM_LOG_DEBUG << "PutGlobalCacheBlocks block=" << info.block_idx << " key=" << info.key
                         << " checksum=" << checksum;
#endif
          batch_keys.push_back(info.key);
          batch_slices.push_back({{acquired_blks[j]->ptr, block_byte_size}});
          ready_blks.push_back(acquired_blks[j]);
          acquired_blks[j] = nullptr;
        }
      } catch (const std::exception& e) {
        KLLM_LOG_WARNING << "Put D2H copy/sync failed: " << e.what();
        for (auto* blk : acquired_blks) {
          if (blk != nullptr) {
            staging_pool_->Put(blk);
          }
        }
        acquired_blks.clear();
      }

      // 兜底：未成功进入 ready 的 staging 还回池
      for (auto* blk : acquired_blks) {
        if (blk != nullptr) {
          staging_pool_->Put(blk);
        }
      }
    } else if (!acquired_blks.empty()) {
      for (auto* blk : acquired_blks) {
        staging_pool_->Put(blk);
      }
      acquired_blks.clear();
    }
    acquired_blks = std::move(ready_blks);

    // Phase 4: BatchPut（soft_pin：降低热前缀被提前淘汰的概率）
    if (!batch_keys.empty()) {
      mooncake::ReplicateConfig rep_config;
      rep_config.with_soft_pin = true;
      decltype(client_->BatchPut(batch_keys, batch_slices, rep_config)) put_results =
          client_->BatchPut(batch_keys, batch_slices, rep_config);
      if (put_results.size() != batch_keys.size()) {
        KLLM_LOG_ERROR << "BatchPut size mismatch, expected=" << batch_keys.size() << ", got=" << put_results.size();
        for (auto* blk : acquired_blks) {
          staging_pool_->Put(blk);
        }
        if (put_ok_out != nullptr) {
          *put_ok_out = put_count;
        }
        return Status(RetCode::RET_RUNTIME_FAILED, "BatchPut result size mismatch");
      }
      for (size_t i = 0; i < put_results.size(); ++i) {
        if (!put_results[i].has_value()) {
          KLLM_LOG_WARNING << "BatchPut failed for key=" << batch_keys[i] << ", error=" << put_results[i].error();
        } else {
          ++put_count;
        }
      }
    }

    // Phase 5: Release staging buffers
    for (auto* blk : acquired_blks) {
      staging_pool_->Put(blk);
    }
  }

  if (put_ok_out != nullptr) {
    *put_ok_out = put_count;
  }
  return Status();
}

// ======================== PutSwaBlocks ========================

Status MooncakeConnector::PutSwaBlocks(const std::vector<int>& token_ids, int tp_rank, int layer_index, int block_size,
                                       size_t swa_block_byte_size, const std::vector<size_t>& block_idxs,
                                       const std::vector<void*>& block_ptrs, int* put_ok_out) {
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

  InitSwaStagingPool(block_size, swa_block_byte_size);
  if (!swa_staging_pool_) {
    return Status(RetCode::RET_RUNTIME_FAILED, "swa staging pool unavailable.");
  }

  // put worker 线程池已 SetDevice；此处再切一次，避免其它路径直接调入时读错卡。
  SetDevice(device_id_);

  const int total_tokens = static_cast<int>(token_ids.size());

  // SWA 各 TP 卡内容不同：每卡用本卡 tp_rank（= device_id）写全量选中块，不做主 KV 式分片跳过。
  struct SwaBlockInfo {
    size_t block_idx;
    size_t ptr_idx;
    std::string key;
  };
  std::vector<SwaBlockInfo> candidates;
  std::vector<std::string> all_keys;
  for (size_t i = 0; i < block_idxs.size(); ++i) {
    const size_t prefix_len =
        std::min((block_idxs[i] + 1) * static_cast<size_t>(block_size), static_cast<size_t>(total_tokens));
    const std::string prefix_hash = HashTokenPrefix(token_ids.data(), prefix_len);
    // 调用方传入的 tp_rank 应为 device_id；与 AlignHitBoundaryBySwa 的 device_id_ 对齐。
    const std::string key = MakeSwaBlockKey(prefix_hash, tp_rank, layer_index, static_cast<int>(block_idxs[i]));
    candidates.push_back({block_idxs[i], i, key});
    all_keys.push_back(key);
  }

  if (all_keys.empty()) {
    return Status();
  }

  decltype(client_->BatchIsExist(all_keys)) exist_results = client_->BatchIsExist(all_keys);
  if (exist_results.size() != all_keys.size()) {
    KLLM_LOG_ERROR << "PutSwaBlocks: BatchIsExist size mismatch, expected=" << all_keys.size()
                   << ", got=" << exist_results.size();
    return Status(RetCode::RET_RUNTIME_FAILED, "BatchIsExist result size mismatch");
  }

  std::vector<SwaBlockInfo> need_put;
  for (size_t i = 0; i < candidates.size(); ++i) {
    if (!(exist_results[i].has_value() && exist_results[i].value())) {
      need_put.push_back(candidates[i]);
    }
  }

  int put_count = 0;
  int fail_count = 0;
  size_t offset = 0;
  while (offset < need_put.size()) {
    std::vector<mooncake::ObjectKey> batch_keys;
    std::vector<std::vector<mooncake::Slice> > batch_slices;
    std::vector<StagingBuffer*> acquired_blks;
    std::vector<size_t> pending_info_idx;

    for (; offset < need_put.size(); ++offset) {
      auto* blk = swa_staging_pool_->TryGet();
      if (!blk) {
        break;
      }
      acquired_blks.push_back(blk);
      pending_info_idx.push_back(offset);
    }

    std::vector<StagingBuffer*> ready_blks;
    ready_blks.reserve(acquired_blks.size());
    if (!acquired_blks.empty() && !put_d2h_streams_.empty()) {
      try {
        SetDevice(device_id_);
        Stream& d2h_stream = GetPutD2HStream();
        for (size_t j = 0; j < acquired_blks.size(); ++j) {
          const auto& info = need_put[pending_info_idx[j]];
          void* dst = acquired_blks[j] != nullptr ? acquired_blks[j]->ptr : nullptr;
          void* src = (info.ptr_idx < block_ptrs.size()) ? block_ptrs[info.ptr_idx] : nullptr;
          if (dst == nullptr || src == nullptr) {
            KLLM_LOG_WARNING << "PutSwaBlocks: D2H bad ptr for block=" << info.block_idx;
            continue;
          }
          MemcpyAsync(dst, src, swa_block_byte_size, MEMCPY_DEVICE_TO_HOST, d2h_stream);
        }
        StreamSynchronize(d2h_stream);
        for (size_t j = 0; j < acquired_blks.size(); ++j) {
          const auto& info = need_put[pending_info_idx[j]];
          if (acquired_blks[j] == nullptr || info.ptr_idx >= block_ptrs.size() || block_ptrs[info.ptr_idx] == nullptr) {
            if (acquired_blks[j] != nullptr) {
              swa_staging_pool_->Put(acquired_blks[j]);
              acquired_blks[j] = nullptr;
            }
            ++fail_count;
            continue;
          }
          batch_keys.push_back(info.key);
          batch_slices.push_back({{acquired_blks[j]->ptr, swa_block_byte_size}});
          ready_blks.push_back(acquired_blks[j]);
          acquired_blks[j] = nullptr;
        }
      } catch (const std::exception& e) {
        KLLM_LOG_WARNING << "PutSwaBlocks: MemcpyAsync D2H failed, error=" << e.what();
      }
      for (auto* blk : acquired_blks) {
        if (blk != nullptr) {
          swa_staging_pool_->Put(blk);
          ++fail_count;
        }
      }
    } else if (!acquired_blks.empty()) {
      KLLM_LOG_WARNING << "PutSwaBlocks: put_d2h_streams empty, skip D2H batch";
      for (auto* blk : acquired_blks) {
        swa_staging_pool_->Put(blk);
        ++fail_count;
      }
      acquired_blks.clear();
    }
    acquired_blks = std::move(ready_blks);

    if (!batch_keys.empty()) {
      mooncake::ReplicateConfig rep_config;
      rep_config.with_soft_pin = true;
      auto put_results = client_->BatchPut(batch_keys, batch_slices, rep_config);
      if (put_results.size() != batch_keys.size()) {
        KLLM_LOG_ERROR << "PutSwaBlocks: BatchPut size mismatch, expected=" << batch_keys.size()
                       << ", got=" << put_results.size();
        for (auto* blk : acquired_blks) {
          swa_staging_pool_->Put(blk);
        }
        if (put_ok_out != nullptr) {
          *put_ok_out = put_count;
        }
        return Status(RetCode::RET_RUNTIME_FAILED, "BatchPut result size mismatch");
      }
      for (size_t i = 0; i < put_results.size(); ++i) {
        if (put_results[i].has_value()) {
          ++put_count;
        } else {
          const auto err = put_results[i].error();
          // 段满时逐 key WARNING 会刷屏；汇总见下方 put_ok/put_fail。
          if (err != mooncake::ErrorCode::NO_AVAILABLE_HANDLE) {
            KLLM_LOG_WARNING << "PutSwaBlocks: BatchPut failed for key=" << batch_keys[i] << ", error=" << err;
          } else {
            KLLM_LOG_DEBUG << "PutSwaBlocks: BatchPut failed for key=" << batch_keys[i] << ", error=" << err;
          }
          ++fail_count;
        }
      }
    }

    for (auto* blk : acquired_blks) {
      swa_staging_pool_->Put(blk);
    }
  }

  if (fail_count > 0) {
    KLLM_LOG_WARNING << "PutSwaBlocks: put_ok=" << put_count << ", put_fail=" << fail_count << ", device=" << device_id_
                     << " (NO_AVAILABLE_HANDLE means Mooncake segment full; reduce swa_tail_block_num or "
                     << "increase global_segment_size)";
  }
  if (put_ok_out != nullptr) {
    *put_ok_out = put_count;
  }
  return Status();
}

// ======================== PutFirstTokens ========================

Status MooncakeConnector::PutFirstTokens(const std::vector<int>& token_ids, const std::vector<int>& tokens) {
  if (!initialized_) {
    return Status(RetCode::RET_INVALID_ARGUMENT, "Connector not initialized.");
  }
  if (global_segment_size_ == 0) {
    return Status();
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
  if (!AllowPutByHotkeyFrequency(key)) {
    KLLM_LOG_DEBUG << "PutFirstTokens skip by hotkey frequency, key=" << key;
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
  rep_config.with_soft_pin = true;
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
  KLLM_LOG_INFO << "Shutdown start, initialized=" << initialized_;

  if (!initialized_) {
    return Status();
  }

  DestroyCopyStreamPools();

  if (client_) {
    if (staging_pool_) {
      client_->unregisterLocalMemory(staging_pool_->BasePtr());
      staging_pool_.reset();
    }
    if (swa_staging_pool_) {
      client_->unregisterLocalMemory(swa_staging_pool_->BasePtr());
      swa_staging_pool_.reset();
    }

    if (segment_buffer_) {
      auto unmount_result = client_->UnmountSegment(segment_buffer_, global_segment_size_);
      if (!unmount_result.has_value()) {
        KLLM_LOG_WARNING << "Shutdown UnmountSegment failed, error=" << unmount_result.error();
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
  KLLM_LOG_INFO << "Shutdown done";
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
