/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include "runtime/infer_request.h"
#include "runtime/threadpool.h"

namespace ksana_llm {

class IpcFuncWrapper;

// 全局 cache 异步拉取的完成信息，由 executor 通过 IPC 报告给 engine。
struct GlobalCacheQueryResult {
  int64_t req_id = -1;
  int device_id = -1;
  std::vector<int> matched_ranges;
  bool success = false;
  // 全局 cache 中缓存的生成 token(s)（含 generated + draft）
  std::vector<int> cached_tokens;
  bool has_cached_tokens = false;
  // 单卡阶段耗时（来自 executor），单位 us；未上报为 -1。
  int64_t wait_us = -1;        // enqueue → Call 开始（IPC/long-task 排队）
  int64_t qf_us = -1;          // QueryAndFetch
  int64_t token_us = -1;       // QueryFirstTokens（通常仅卡 0，其它为 0）
  int64_t report_us = -1;      // ReportGlobalCacheQueryDone Invoke（消息里常为 0，真值见 executor 日志）
  int64_t call_us = -1;        // Call 入口 → 发起 report 前
  int64_t report_wall_us = 0;  // 发起 report 时的 wall clock，engine 可算回报延迟
  // SWA 对齐后的命中 token 数；0 表示该卡没有给出 SWA 对齐结果。
  size_t swa_aligned_hit_len = 0;
  std::vector<size_t> swa_filled_block_idxs;
};

// GlobalCacheFetcher 管理 PrepareRequest 之后的全局 cache 异步拉取生命周期。
//
// 流程:
//   1. PrepareRequest 完成后，调用 SubmitRequest 将请求放入传输队列。
//   2. GlobalCacheFetcher 通过 IPC 广播通知每个 executor 执行 QueryAndFetch。
//   3. 每个 executor 完成后通过 IPC 回调 OnDeviceComplete。
//   4. 当某个请求在所有卡上都完成时，将其移入已完成队列。
//   5. 调度线程通过 TakeCompletedRequests 取出已完成的请求。
class GlobalCacheFetcher {
 public:
  // role:
  //   ""               standalone (无 PD 分离)
  //   "prefill"        V1 PD Prefill 节点 (max_new_tokens=1)
  //   "decode"         V1 PD Decode 节点 (未覆盖满 seq + tokens 则 reject)
  //   "pd_v2_prefill"  pd_v2 Prefill (命中硬上限 seq-1，不 flip kDecode)
  //   "pd_v2_decode"   pd_v2 Decode (无 reject, 部分命中由 peer Prefill 兜底)
  // fetch_thread_num: 异步 dispatch 线程数，对应 yaml fetch_thread_num（至少为 1）
  // fetch_timeout_ms: 从 Submit 起超过该时间仍未收齐各卡 Done，则按失败收尾，避免 preprocessing 永久挂起
  GlobalCacheFetcher(int device_count, std::shared_ptr<IpcFuncWrapper> ipc_func_wrapper, const std::string& role = "",
                     int fetch_thread_num = 4, int fetch_timeout_ms = 5000);
  ~GlobalCacheFetcher();

  void Start();
  void Stop();

  // 提交一个已完成 PrepareRequest 的请求，开始异步全局 cache 拉取。
  // 本地 prefix cache 匹配的 token 数从 req 的 workload 中获取。
  void SubmitRequest(std::shared_ptr<InferRequest> req);

  // 由 IPC callable 调用，报告某个 executor（device）上的查询完成。
  void OnDeviceComplete(const GlobalCacheQueryResult& result);

  // 取出所有已完成全局 cache 拉取的请求（由调度线程调用）。
  std::vector<std::shared_ptr<InferRequest>> TakeCompletedRequests();

  // 是否还有正在传输中的请求。
  bool HasPendingRequests() const;

  // 是否工作在异步模式（有 IPC 后端）。
  // 异步模式下请求完成通过 CompletionCallback + UpdateAsyncResult 传递，
  // 同步模式下通过 TakeCompletedRequests 在 Process 中直接取出。
  bool IsAsync() const;

  // PD role ("" / "prefill" / "decode" / "pd_v2_prefill" / "pd_v2_decode")。
  const std::string& GetRole() const { return role_; }

  // 请求完成时的回调类型：参数为 (req_id, success)
  // success=false 表示请求被拒绝（如 V1 decode 角色未覆盖满 seq + tokens）
  using CompletionCallback = std::function<void(int64_t req_id, bool success)>;

  // 设置请求完成时的回调（由 GlobalCacheQueryHook 调用，用于驱动事件驱动调度器）
  void SetCompletionCallback(CompletionCallback callback);

  // 取出被拒绝的请求（由 hook 的 UpdateAsyncResult 使用）
  std::vector<std::shared_ptr<InferRequest>> TakeRejectedRequests();

  // Exposed for unit tests.
  // SWA 滑窗大小（token 数），0 表示未启用 SWA。用于算窗口覆盖的 block 数 W，
  // 公式与 PrefixCacheManager 判本地命中时用的完全一致，保证两条路径的截断语义相同。
  void SetSlidingWindowSize(size_t sliding_window_size) { sliding_window_size_ = sliding_window_size; }

  // W = ceil(sliding_window_size / block_token_num)，与本地 prefix 命中判据同一公式。
  size_t SwaWindowBlockNum(size_t block_token_num) const;

  // swa_aligned_hit_len > 0 时 new_cached_len 来自 SWA 对齐边界：此时内部的 cap+floor
  // 只作兜底，一旦它改变了值就说明边界与 SWA 落点不再对应，整体放弃全局命中。
  void UpdateRequestWorkload(std::shared_ptr<InferRequest>& req, size_t new_cached_len,
                             const std::vector<int>& cached_tokens = {}, bool has_cached_tokens = false,
                             size_t swa_aligned_hit_len = 0);

 private:
  // 单个请求的异步拉取跟踪信息
  struct PendingRequest {
    std::shared_ptr<InferRequest> req;
    int devices_completed = 0;
    int total_devices = 0;
    // 每个设备返回的区间（闭区间，每两个元素一组 [start, end]）
    std::vector<std::vector<int>> per_device_ranges;
    bool all_success = true;
    // 来自全局 cache 的生成 token(s)（任一设备查到即可，含 generated + draft）
    std::vector<int> cached_tokens;
    bool has_cached_tokens = false;
    // SubmitRequest / 各卡完成相对 submit 的耗时
    int64_t submit_time_us = 0;
    int64_t last_done_us = -1;
    // device_id -> 相对 submit 的完成时刻(us)；-1 表示未完成
    std::vector<int64_t> device_done_us;

    // ========== SWA 一致性跟踪 ==========
    // 本请求是否启用 SWA（dispatch 时按 req->swa_kv_cache_blocks 是否为空判定）。
    bool swa_enabled = false;
    // 第一张卡报回的 SWA 对齐命中长度；后续卡必须完全一致。
    size_t swa_aligned_hit_len = 0;
    // 各卡 swa_aligned_hit_len 是否全部一致且非 0。
    // 不一致意味着有的卡缺 SWA 内容，此时整体放弃全局命中而不是取最小值——
    // 取最小值会让命中点落在某些卡没有拉 SWA 的位置上，产生静默的错误输出。
    bool swa_consistent = true;

    // Submit 时刻的序列长度快照，供异步 dispatch 计算 max_cached_len。
    // 线程池真正跑 Dispatch 时 GetPlanningSequenceLen() 可能已变 0（warmup 等），
    // 但 token 列表与 SWA block 仍有效，需用此处快照避免 SWA 上下文被误判 incomplete。
    size_t query_seq_len = 0;
  };

  // 向所有 executor 发送全局 cache 查询 IPC
  void DispatchQueryToExecutors(const PendingRequest& pending);

  // 从 local_cached_len 出发，在所有设备的区间中求最长连续前缀。
  // 逐 token 位置推进，任一设备不覆盖即终止。
  static size_t GetAllDevicesContiguousPrefixLen(const std::vector<std::vector<int>>& per_device_ranges,
                                                 size_t local_cached_len);

  // 收齐各卡（或超时强制失败）后更新 workload、移出 pending，并在锁外触发 callback。
  // 调用方持有 mutex_；本函数可能 unlock。
  void FinalizePendingLocked(std::unordered_map<int64_t, PendingRequest>::iterator it,
                             std::unique_lock<std::mutex>& lock);

  void TimeoutLoop();
  void SweepTimeouts();

  mutable std::mutex mutex_;
  std::unordered_map<int64_t, PendingRequest> pending_requests_;
  std::vector<std::shared_ptr<InferRequest>> completed_requests_;
  std::vector<std::shared_ptr<InferRequest>> rejected_requests_;


  int device_count_ = 1;
  size_t sliding_window_size_ = 0;
  std::shared_ptr<IpcFuncWrapper> ipc_func_wrapper_;
  std::string role_;  // "" / "prefill" / "decode" / "pd_v2_prefill" / "pd_v2_decode"
  int fetch_timeout_ms_ = 5000;
  std::unique_ptr<ThreadPool> dispatch_thread_pool_;
  std::atomic<bool> terminated_{false};
  CompletionCallback completion_callback_;
  std::thread timeout_thread_;
  std::mutex timeout_wake_mutex_;
  std::condition_variable timeout_wake_cv_;
};

}  // namespace ksana_llm
