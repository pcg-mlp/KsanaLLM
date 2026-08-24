/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "cache_manager/cache_manager_interface.h"
#include "runtime/infer_request.h"

namespace ksana_llm {

// Engine 端追踪每个请求的待完成 KV cache put 操作。
// 当 put 未全部完成时延迟析构，等待所有设备的 put 完成后再释放 block。
//
// 线程模型：
// - OnDevicePutDone 可能在 IPC 线程调用，只更新计数并把完成项入队；
// - EndSwaPutHold / DestroyFinishedRequest 必须在 scheduler 线程经 DrainPutCompletions 执行，
//   避免与 UpdateRequestTokens 等并发改 SWA LFU / prefix 状态。
class GlobalCachePutTracker {
 public:
  GlobalCachePutTracker(int device_count, std::shared_ptr<CacheManagerInterface> cache_manager);
  ~GlobalCachePutTracker() = default;

  // 设置多 DP cache managers（事件驱动调度器使用）。
  // 设置后 TryFinishDeferred 会使用对应 dp_idx 的 cache_manager 执行延迟析构。
  void SetDpCacheManagers(std::vector<std::shared_ptr<CacheManagerInterface>> managers);

  // 可选：put 全部完成入队后唤醒 scheduler（例如 event_queue_.Put），降低 Drain 延迟。
  void SetSchedulerWake(std::function<void()> wake);

  // 在 Schedule 发出 batch 时调用：仅为「最后一段」prefill 登记 pending put
  // （SetPlanningTask 后 remaining.prefill_token_num==0；中间 split-fuse chunk 跳过）。
  void OnBatchDispatched(const std::vector<std::shared_ptr<InferRequest>>& running_reqs);

  // Executor IPC 回调：某 device 上单个请求的 put 完成（可在非 scheduler 线程）。
  // fa_put_blocks / swa_put_blocks 为本卡本次 put 的 block 数。
  void OnDevicePutDone(int device_id, int64_t req_id, int fa_put_blocks = 0, int swa_put_blocks = 0);

  // 仅在 scheduler 线程调用：执行已入队的 EndSwaPutHold / 延迟 DestroyFinishedRequest。
  // 返回本次处理的完成项数量。
  size_t DrainPutCompletions();

  // 是否有待 Drain 的 put 完成项（用于 scheduler 带超时轮询，避免饿死）。
  bool HasPendingCompletions() const;

  // 在 ResetRequest 中调用：若有 pending put 则延迟析构，返回 true 表示已延迟。
  // dp_idx 仅在设置了多 DP cache managers 时使用，用于延迟析构时选择正确的 cache_manager。
  bool DeferDestructionIfNeeded(int64_t req_id, size_t dp_idx = 0);

  // 是否有延迟析构中的请求
  bool HasDeferredRequests() const;

 private:
  struct PendingCompletion {
    int64_t req_id = 0;
    std::shared_ptr<CacheManagerInterface> cache_manager;
    std::shared_ptr<Request> request;
    int fa_put_blocks = 0;   // 各卡 sum（主 KV 分片上传）
    int swa_put_blocks = 0;  // 各卡 sum
  };

  void TryFinishDeferred(int64_t req_id);
  void StampPutMetricsOnRequest(const PendingCompletion& completion);

  mutable std::mutex mutex_;
  int device_count_ = 1;
  std::shared_ptr<CacheManagerInterface> cache_manager_;
  std::vector<std::shared_ptr<CacheManagerInterface>> dp_cache_managers_;
  std::function<void()> scheduler_wake_;

  std::unordered_map<int64_t, int> pending_counts_;
  std::unordered_set<int64_t> deferred_req_ids_;
  std::unordered_map<int64_t, size_t> deferred_dp_idx_;
  std::unordered_map<int64_t, int64_t> deferred_start_us_;
  // 登记 put 时的 cache_manager，用于 put 完成后 EndSwaPutHold（多 DP 时各请求可能不同）。
  std::unordered_map<int64_t, std::shared_ptr<CacheManagerInterface>> pending_cache_managers_;
  // 登记 put 时的 Request，用于完成后写 access-log 指标并唤醒 WaitGlobalCachePutDone。
  std::unordered_map<int64_t, std::shared_ptr<Request>> pending_requests_;
  // 各卡 put 计数累加（完成前）。
  std::unordered_map<int64_t, int> pending_fa_put_blocks_;
  std::unordered_map<int64_t, int> pending_swa_put_blocks_;
  // 已上报 put-done 的 device_id，用于去重。
  std::unordered_map<int64_t, std::unordered_set<int>> pending_done_devices_;
  // IPC 线程入队，scheduler Drain。
  std::vector<PendingCompletion> pending_completions_;
};

}  // namespace ksana_llm
