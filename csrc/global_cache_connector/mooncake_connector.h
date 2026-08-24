/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "client_service.h"
#include "device/device_utils.h"
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
                       std::vector<int>& matched_ranges, bool retry_until_full_hit = false, int timeout_ms = 5000,
                       const GlobalCacheSwaContext* swa_ctx = nullptr,
                       GlobalCacheSwaResult* swa_result = nullptr) override;

  Status PutGlobalCacheBlocks(const std::vector<int>& token_ids, int tp_size, int tp_rank, int layer_index,
                              int block_size, size_t block_byte_size, const std::vector<int>& block_ranges,
                              const std::vector<void*>& block_ptrs, int* put_ok_out = nullptr) override;

  Status PutSwaBlocks(const std::vector<int>& token_ids, int tp_rank, int layer_index, int block_size,
                      size_t swa_block_byte_size, const std::vector<size_t>& block_idxs,
                      const std::vector<void*>& block_ptrs, int* put_ok_out = nullptr) override;

  Status PutFirstTokens(const std::vector<int>& token_ids, const std::vector<int>& tokens) override;

  Status QueryFirstTokens(const std::vector<int>& token_ids, std::vector<int>& tokens, bool& found) override;

  Status Shutdown() override;

 private:
  std::shared_ptr<mooncake::Client> client_;

  // Staging buffer pool: eagerly initialized in Initialize() if block config is provided.
  // Falls back to lazy initialization on first Put/Get call if block_byte_size was not in config.
  // Blocks are registered with Mooncake for RDMA after allocation.
  void InitStagingPool(int block_size, size_t block_byte_size);

  // 按 max_concurrent_threads 预建 Query/Put 拷贝流；调用线程 TLS 与槽位 1:1 绑定。
  void InitCopyStreamPools(int max_concurrent_threads);
  Stream& GetQueryH2DStream();
  Stream& GetPutD2HStream();
  void DestroyCopyStreamPools();

  // Put 热 key 频率门控：窗口内第二次及以后才允许 Put，并刷新时间戳；首次或过期后重计为首次则跳过。
  // put_hotkey_window_us_==0 时恒允许。返回 true=允许上传。
  bool AllowPutByHotkeyFrequency(const std::string& key);

  std::once_flag staging_pool_init_flag_;
  std::unique_ptr<StagingBufferPool> staging_pool_;

  // SWA block 的字节数与主 KV block 不同（DSv4 实测约 7.6 倍），必须用独立的 staging pool。
  void InitSwaStagingPool(int block_size, size_t swa_block_byte_size);

  std::once_flag swa_staging_pool_init_flag_;
  std::unique_ptr<StagingBufferPool> swa_staging_pool_;

  // 在已查到的主 KV 连续前缀 fa_contig_end 上，再查 SWA、裁边界，并拉窗口 SWA。
  // 不再查主 KV（由 QueryAndFetch 入口统一 BatchIsExist）。
  // block_hashes[i] 对应全局 block (hash_start_block + i)。
  // 返回对齐后的 block 数；等于 start_block 表示没有可用的全局扩展。
  int AlignHitBoundaryBySwa(const std::vector<int>& token_ids, int tp_rank, int layer_index, int block_size,
                            int start_block, int fa_contig_end, const GlobalCacheSwaContext& swa_ctx,
                            GlobalCacheSwaResult* swa_result, const std::vector<std::string>& block_hashes,
                            int hash_start_block);

  size_t max_step_tokens_ = 0;

  // Segment buffer registered with MountSegment for storage allocation
  void* segment_buffer_ = nullptr;
  size_t global_segment_size_ = 0;
  size_t segment_mmap_size_ = 0;  // actual size passed to mmap, used by munmap

  // Local hostname:port used during Client::Create
  std::string local_hostname_;

  // 多卡分片信息：PutGlobalCacheBlocks（主 KV）假设各卡内容一致，只让
  // (global_idx % device_count_ == device_id_) 的卡实际上传。
  // SWA 相反：各卡内容不同，PutSwaBlocks / AlignHitBoundaryBySwa 用 device_id_ 作 tp 分 key，
  // 每卡各自 put/get 全量选中块。
  int device_id_ = 0;
  int device_count_ = 1;

  // Query(H2D) / Put(D2H) 各一组拷贝流；长度 = max_concurrent_threads，TLS 1:1 绑槽。
  std::vector<std::unique_ptr<Stream>> query_h2d_streams_;
  std::vector<std::unique_ptr<Stream>> put_d2h_streams_;
  std::atomic<size_t> query_h2d_next_slot_{0};
  std::atomic<size_t> put_d2h_next_slot_{0};

  int64_t put_hotkey_window_us_ = 0;
  std::mutex put_hotkey_mu_;
  std::unordered_map<std::string, int64_t> put_hotkey_last_us_;

  bool initialized_ = false;
};

}  // namespace ksana_llm
