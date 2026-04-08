/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "cache_manager/cache_manager_interface.h"
#include "runtime/infer_request.h"

namespace ksana_llm {

// Engine 端追踪每个请求的待完成 KV cache put 操作。
// 当 put 未全部完成时延迟析构，等待所有设备的 put 完成后再释放 block。
class GlobalCachePutTracker {
 public:
  GlobalCachePutTracker(int device_count, std::shared_ptr<CacheManagerInterface> cache_manager);

  // 设置多 DP cache managers（事件驱动调度器使用）。
  // 设置后 TryFinishDeferred 会使用对应 dp_idx 的 cache_manager 执行延迟析构。
  void SetDpCacheManagers(std::vector<std::shared_ptr<CacheManagerInterface>> managers);

  // 在 Schedule 发出 batch 时调用：为 batch 中每个 prefill 请求增加 device_count 个 pending put。
  void OnBatchDispatched(const std::vector<std::shared_ptr<InferRequest>>& running_reqs);

  // Executor IPC 回调：某 device 上单个请求的 put 完成。
  void OnDevicePutDone(int device_id, int64_t req_id);

  // 在 ResetRequest 中调用：若有 pending put 则延迟析构，返回 true 表示已延迟。
  // dp_idx 仅在设置了多 DP cache managers 时使用，用于延迟析构时选择正确的 cache_manager。
  bool DeferDestructionIfNeeded(int64_t req_id, size_t dp_idx = 0);

  // 是否有延迟析构中的请求
  bool HasDeferredRequests() const;

 private:
  // 检查并完成已经就绪的延迟析构请求
  void TryFinishDeferred(int64_t req_id);

  mutable std::mutex mutex_;
  int device_count_ = 1;
  std::shared_ptr<CacheManagerInterface> cache_manager_;
  std::vector<std::shared_ptr<CacheManagerInterface>> dp_cache_managers_;

  // 每个请求的 pending put 计数（所有设备完成时归零）
  std::unordered_map<int64_t, int> pending_counts_;
  // 已标记延迟析构但尚未完成 put 的请求 ID
  std::unordered_set<int64_t> deferred_req_ids_;
  // 延迟析构请求对应的 dp_idx（用于多 DP 时选择正确的 cache_manager）
  std::unordered_map<int64_t, size_t> deferred_dp_idx_;
};

}  // namespace ksana_llm
