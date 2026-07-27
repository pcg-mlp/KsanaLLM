/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
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
  GlobalCacheFetcher(int device_count, std::shared_ptr<IpcFuncWrapper> ipc_func_wrapper, const std::string& role = "");
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
  void UpdateRequestWorkload(std::shared_ptr<InferRequest>& req, size_t new_cached_len,
                             const std::vector<int>& cached_tokens = {}, bool has_cached_tokens = false);

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
  };

  // 向所有 executor 发送全局 cache 查询 IPC
  void DispatchQueryToExecutors(const PendingRequest& pending);

  // 从 local_cached_len 出发，在所有设备的区间中求最长连续前缀。
  // 逐 token 位置推进，任一设备不覆盖即终止。
  static size_t GetAllDevicesContiguousPrefixLen(const std::vector<std::vector<int>>& per_device_ranges,
                                                 size_t local_cached_len);

  mutable std::mutex mutex_;
  std::unordered_map<int64_t, PendingRequest> pending_requests_;
  std::vector<std::shared_ptr<InferRequest>> completed_requests_;
  std::vector<std::shared_ptr<InferRequest>> rejected_requests_;


  int device_count_ = 1;
  std::shared_ptr<IpcFuncWrapper> ipc_func_wrapper_;
  std::string role_;  // "" / "prefill" / "decode" / "pd_v2_prefill" / "pd_v2_decode"
  std::unique_ptr<ThreadPool> dispatch_thread_pool_;
  std::atomic<bool> terminated_{false};
  CompletionCallback completion_callback_;
};

}  // namespace ksana_llm
