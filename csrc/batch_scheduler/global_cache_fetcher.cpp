/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "batch_scheduler/global_cache_fetcher.h"

#include <algorithm>
#include <climits>

#include "configure/environment.h"
#include "ipc/ipc_func_wrapper.h"
#include "profiler/reporter.h"
#include "utils/logger.h"
#include "utils/string_utils.h"

namespace ksana_llm {

GlobalCacheFetcher::GlobalCacheFetcher(int device_count, std::shared_ptr<IpcFuncWrapper> ipc_func_wrapper,
                                       const std::string& role)
    : device_count_(device_count), ipc_func_wrapper_(ipc_func_wrapper), role_(role) {
  dispatch_thread_pool_ = std::make_unique<ThreadPool>(2);
  KLLM_LOG_INFO << "GlobalCacheFetcher created: device_count=" << device_count_
                << ", role=" << (role_.empty() ? "(standalone)" : role_)
                << ", ipc_enabled=" << (ipc_func_wrapper_ != nullptr);
}

GlobalCacheFetcher::~GlobalCacheFetcher() { Stop(); }


void GlobalCacheFetcher::Start() {
  terminated_ = false;
  dispatch_thread_pool_->Start();
}

void GlobalCacheFetcher::Stop() {
  terminated_ = true;

  CompletionCallback cb;
  std::vector<int64_t> orphan_req_ids;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    cb = completion_callback_;
    for (auto& [req_id, pending] : pending_requests_) {
      orphan_req_ids.push_back(req_id);
    }
    pending_requests_.clear();
  }

  // 在锁外通知调用方这些请求已失败，避免请求永久挂起
  if (cb) {
    for (int64_t req_id : orphan_req_ids) {
      cb(req_id, false);
    }
  }

  if (dispatch_thread_pool_) {
    dispatch_thread_pool_->Stop();
  }
}

void GlobalCacheFetcher::SubmitRequest(std::shared_ptr<InferRequest> req) {
  // V1 prefill 角色: 只负责计算首 token, 调度前将 max_new_tokens 限制为 1.
  // pd_v2_prefill 不在此处限制 (pd_v2 自己控制 sampling).
  if (role_ == "prefill") {
    req->sampling_config.max_new_tokens = 1;
    KLLM_LOG_DEBUG << "GlobalCacheFetcher: prefill role, set max_new_tokens=1 for req_id=" << req->req_id;
  }

  size_t local_cached_len = req->GetPlanningWorkload().prefill_start_offset;

  // 没有 ipc_func_wrapper_ 说明未配置全局 cache 后端，仅基于本地 cache 更新 workload 后直接完成。
  if (!ipc_func_wrapper_) {
    UpdateRequestWorkload(req, local_cached_len);
    std::lock_guard<std::mutex> lock(mutex_);
    if (req->finished) {
      rejected_requests_.push_back(std::move(req));
    } else {
      completed_requests_.push_back(std::move(req));
    }
    return;
  }

  PendingRequest pending;
  pending.req = req;
  pending.devices_completed = 0;
  pending.total_devices = device_count_;

  int64_t req_id = req->req_id;

  {
    std::lock_guard<std::mutex> lock(mutex_);
    pending_requests_[req_id] = std::move(pending);
  }

  KLLM_LOG_SCHEDULER << "GlobalCacheFetcher: submit req_id=" << req_id << " to global cache fetch queue"
                     << ", local_cached_len=" << local_cached_len << ", device_count=" << device_count_;

  // 在线程池中异步发送 IPC 到所有 executor
  dispatch_thread_pool_->Submit([this, req_id]() {
    if (terminated_) return;

    PendingRequest pending_copy;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto it = pending_requests_.find(req_id);
      if (it == pending_requests_.end()) return;
      pending_copy = it->second;
    }
    DispatchQueryToExecutors(pending_copy);
  });
}

void GlobalCacheFetcher::DispatchQueryToExecutors(const PendingRequest& pending) {
  size_t local_cached_len = pending.req->GetPlanningWorkload().prefill_start_offset;
  for (int executor_id = 0; executor_id < device_count_; ++executor_id) {
    Status status = ipc_func_wrapper_->DispatchGlobalCacheQuery(
        executor_id, pending.req->req_id, pending.req->GetPrefillingTokens(), local_cached_len,
        static_cast<int>(pending.req->block_token_num), pending.req->kv_cache_blocks);

    if (!status.OK()) {
      KLLM_LOG_ERROR << "GlobalCacheFetcher: dispatch global cache query IPC failed, req_id=" << pending.req->req_id
                     << ", executor_id=" << executor_id << ", error=" << status.GetMessage();
      GlobalCacheQueryResult result;
      result.req_id = pending.req->req_id;
      result.device_id = executor_id;
      result.success = false;
      OnDeviceComplete(result);
    }
  }
}

void GlobalCacheFetcher::OnDeviceComplete(const GlobalCacheQueryResult& result) {
  std::unique_lock<std::mutex> lock(mutex_);

  auto it = pending_requests_.find(result.req_id);
  if (it == pending_requests_.end()) {
    KLLM_LOG_WARNING << "GlobalCacheFetcher: received completion for unknown req_id=" << result.req_id;
    return;
  }

  PendingRequest& pending = it->second;
  pending.devices_completed++;

  if (result.success) {
    pending.per_device_ranges.push_back(result.matched_ranges);
    // 任一设备查到 token(s) 即采用（token 与设备无关，所有设备的结果一致）
    if (result.has_cached_tokens && !pending.has_cached_tokens) {
      pending.cached_tokens = result.cached_tokens;
      pending.has_cached_tokens = true;
      KLLM_LOG_DEBUG << "GlobalCacheFetcher: req_id=" << result.req_id
                     << " cached_tokens collected from device_id=" << result.device_id
                     << ", token_count=" << result.cached_tokens.size();
    }
  } else {
    pending.all_success = false;
    KLLM_LOG_DEBUG << "GlobalCacheFetcher: req_id=" << result.req_id << " device_id=" << result.device_id
                   << " reported failure";
  }

  KLLM_LOG_SCHEDULER << "GlobalCacheFetcher: req_id=" << result.req_id << " device_id=" << result.device_id << " done, "
                     << pending.devices_completed << "/" << pending.total_devices << ", success=" << result.success;

  if (pending.devices_completed >= pending.total_devices) {
    size_t local_cached_len = pending.req->GetPlanningWorkload().prefill_start_offset;
    size_t total_cached_len = local_cached_len;

    if (pending.all_success && !pending.per_device_ranges.empty()) {
      size_t global_prefix_len = GetAllDevicesContiguousPrefixLen(pending.per_device_ranges, local_cached_len);
      total_cached_len += global_prefix_len;

      if (global_prefix_len > 0) {
        KLLM_LOG_SCHEDULER << "GlobalCacheFetcher: req_id=" << result.req_id
                           << " global cache extended prefix, local=" << local_cached_len
                           << ", global_prefix=" << global_prefix_len << ", total_cached=" << total_cached_len;
      }
    }

    UpdateRequestWorkload(pending.req, total_cached_len, pending.cached_tokens, pending.has_cached_tokens);

    bool success = !pending.req->finished;
    if (success) {
      KLLM_LOG_SCHEDULER << "GlobalCacheFetcher: req_id=" << result.req_id
                         << " all devices done, moved to prepared queue";
      completed_requests_.push_back(std::move(pending.req));
    } else {
      KLLM_LOG_SCHEDULER << "GlobalCacheFetcher: req_id=" << result.req_id << " all devices done, request rejected";
      rejected_requests_.push_back(std::move(pending.req));
    }

    CompletionCallback cb = completion_callback_;
    pending_requests_.erase(it);

    if (cb) {
      // 回调在锁外执行，避免死锁（回调可能重新获取锁或操作调度器事件队列）
      lock.unlock();
      cb(result.req_id, success);
      return;
    }
  }
}

size_t GlobalCacheFetcher::GetAllDevicesContiguousPrefixLen(const std::vector<std::vector<int>>& per_device_ranges,
                                                            size_t local_cached_len) {
  if (per_device_ranges.empty()) return 0;

  // 每个设备维护一个区间索引，从 local_cached_len 开始逐步推进。
  // 对每个设备单独求从当前位置能连续覆盖到哪里，取所有设备的最小值作为实际推进位置。
  // 任一设备无法覆盖当前位置则立即终止。
  size_t num_devices = per_device_ranges.size();
  std::vector<size_t> cursors(num_devices, 0);  // 每个设备当前扫描到的区间索引

  for (size_t d = 0; d < num_devices; ++d) {
    KLLM_CHECK_WITH_INFO(per_device_ranges[d].size() % 2 == 0,
                         FormatStr("device %zu matched_ranges size %zu is not even", d, per_device_ranges[d].size()));
  }

  int pos = static_cast<int>(local_cached_len);

  while (true) {
    int min_end = INT_MAX;
    bool all_cover = true;

    for (size_t d = 0; d < num_devices; ++d) {
      const auto& ranges = per_device_ranges[d];

      // 跳过已经在 pos 左边的区间
      while (cursors[d] + 1 < ranges.size() && ranges[cursors[d] + 1] < pos) {
        cursors[d] += 2;
      }

      if (cursors[d] + 1 >= ranges.size() || ranges[cursors[d]] > pos) {
        // 该设备没有覆盖 pos 的区间
        all_cover = false;
        break;
      }

      // ranges[cursors[d]] <= pos <= ranges[cursors[d]+1]
      int end = ranges[cursors[d] + 1];
      if (end < min_end) {
        min_end = end;
      }
    }

    if (!all_cover) break;

    // 所有设备都覆盖 [pos, min_end]，推进到 min_end + 1
    pos = min_end + 1;
  }

  return static_cast<size_t>(pos) - local_cached_len;
}

void GlobalCacheFetcher::UpdateRequestWorkload(std::shared_ptr<InferRequest>& req, size_t new_cached_len,
                                               const std::vector<int>& cached_tokens, bool has_cached_tokens) {
  // V1 decode 角色无论命中与否都要走完判定逻辑 (未全覆盖+tokens 需要 reject); 其它角色无命中直接返回.
  if (new_cached_len == 0 && role_ != "decode") {
    return;
  }

  size_t seq_token_num = req->GetPlanningSequenceLen();
  // pd_v2_decode 像 standalone：无 reject，partial 由 peer Prefill 兜底.
  bool is_decode_role = (role_ == "decode");
  bool is_pd_v2_prefill = (role_ == "pd_v2_prefill");
  // 远端已覆盖完整 prompt（用于 V1 decode 准入 / tokens_ready 短路）；本地算 Q 的命中仍硬上限 seq-1。
  bool remote_covers_full_seq = (new_cached_len >= seq_token_num);
  bool tokens_ready = has_cached_tokens && !cached_tokens.empty();
  const size_t max_cached_len = (seq_token_num > 0) ? (seq_token_num - 1) : 0;
  size_t usable_cached_len = std::min(new_cached_len, max_cached_len);
  // SWA：prefill_start_offset 须按 block 下取整（与旧 HandleFullPrefixCacheHit 一致），否则窗口错位。
  if (!req->swa_kv_cache_blocks.empty() && req->block_token_num > 0) {
    usable_cached_len = (usable_cached_len / req->block_token_num) * req->block_token_num;
  }

  KLLM_LOG_DEBUG << "UpdateRequestWorkload: req_id=" << req->req_id
                 << ", role=" << (role_.empty() ? "standalone" : role_) << ", new_cached_len=" << new_cached_len
                 << ", usable_cached_len=" << usable_cached_len << ", seq_token_num=" << seq_token_num
                 << ", remote_covers_full_seq=" << remote_covers_full_seq << ", tokens_ready=" << tokens_ready
                 << ", cached_token_count=" << cached_tokens.size()
                 << ", infer_stage=" << static_cast<int>(req->infer_stage);

  // V1 decode 角色: 必须 KV 覆盖满 seq + token(s) 就绪, 否则拒绝 (V1 Decode 节点无法 prefill).
  // pd_v2_decode 用独立 role 字符串走过这条守卫 (peer Prefill 通过 RDMA 兜底 partial hit).
  if (is_decode_role && !(remote_covers_full_seq && tokens_ready)) {
    KLLM_LOG_WARNING << "decode role reject: remote_covers_full_seq=" << remote_covers_full_seq
                     << ", tokens_ready=" << tokens_ready << ", cached=" << new_cached_len << ", seq=" << seq_token_num
                     << ", req_id=" << req->req_id;
    REPORT_COUNTER("decode_role_reject_req_num", static_cast<size_t>(1));

    req->finish_status =
        Status(RET_RUNTIME_FAILED, "Decode node failed to fetch complete KV cache and tokens from global cache.");
    req->finished = true;
    req->Notify();
    return;
  }

  // 远端已算完 prompt 并带出 generated tokens：本地不再算 Q，直接 decode 短路。
  // pd_v2 Prefill 不能 flip kDecode，仍走下方 seq-1 partial。
  if (remote_covers_full_seq && tokens_ready && !is_pd_v2_prefill) {
    KLLM_LOG_SCHEDULER << "req_id=" << req->req_id
                       << " remote full cover with cached_tokens(count=" << cached_tokens.size()
                       << "), switch to decode";
    REPORT_COUNTER("full_prompt_matched_req_num", static_cast<size_t>(1));
    REPORT_COUNTER("first_token_cache_hit_req_num", static_cast<size_t>(1));

    req->infer_stage = InferStage::kDecode;
    req->AppendCachedTokens(cached_tokens);
    req->NotifyStep();

    ScheduleTaskWorkload planning_workload = req->GetPlanningWorkload();
    planning_workload.prefill_start_offset = seq_token_num;
    planning_workload.prefill_token_num = static_cast<int>(cached_tokens.size());
    req->SetPlanningWorkload(planning_workload);
    req->SetRemainingWorkload(planning_workload);
    return;
  }
  if (remote_covers_full_seq && tokens_ready && is_pd_v2_prefill) {
    // pd_v2_prefill 自行算 Q/写 KV，不消费远端 generated tokens；丢弃是刻意行为。
    KLLM_LOG_DEBUG << "req_id=" << req->req_id << " pd_v2_prefill: discarding " << cached_tokens.size()
                   << " remote tokens, will prefill locally from offset " << usable_cached_len;
  }

  // 命中硬上限 seq-1：无 tokens 的「假全命中」与 partial 统一走此路径，不再 FullHit 回退。
  if (usable_cached_len == 0) {
    return;
  }
  if (remote_covers_full_seq) {
    REPORT_COUNTER("full_prompt_matched_req_num", static_cast<size_t>(1));
  }

  ScheduleTaskWorkload planning_workload = req->GetPlanningWorkload();
  size_t old_total = planning_workload.prefill_token_num + planning_workload.prefill_start_offset;
  planning_workload.prefill_start_offset = usable_cached_len;
  planning_workload.prefill_token_num = old_total - usable_cached_len;
  req->SetPlanningWorkload(planning_workload);

  ScheduleTaskWorkload remaining_workload = req->GetRemainingWorkload();
  size_t remaining_total = remaining_workload.prefill_token_num + remaining_workload.prefill_start_offset;
  remaining_workload.prefill_start_offset = usable_cached_len;
  remaining_workload.prefill_token_num = remaining_total - usable_cached_len;
  req->SetRemainingWorkload(remaining_workload);

  KLLM_LOG_DEBUG << "workload set (capped hit): req_id=" << req->req_id
                 << ", prefill_start_offset=" << planning_workload.prefill_start_offset
                 << ", prefill_token_num=" << planning_workload.prefill_token_num;
}

std::vector<std::shared_ptr<InferRequest>> GlobalCacheFetcher::TakeCompletedRequests() {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<std::shared_ptr<InferRequest>> result;
  result.swap(completed_requests_);
  return result;
}

bool GlobalCacheFetcher::HasPendingRequests() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return !pending_requests_.empty() || !completed_requests_.empty();
}

bool GlobalCacheFetcher::IsAsync() const { return ipc_func_wrapper_ != nullptr; }

// NOTE: SetCompletionCallback must only be called during initialization (before Start()),
// not at runtime. OnDeviceComplete copies the callback under lock then invokes it
// outside the lock to avoid deadlocks.
void GlobalCacheFetcher::SetCompletionCallback(CompletionCallback callback) {
  std::lock_guard<std::mutex> lock(mutex_);
  completion_callback_ = std::move(callback);
}

std::vector<std::shared_ptr<InferRequest>> GlobalCacheFetcher::TakeRejectedRequests() {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<std::shared_ptr<InferRequest>> result;
  result.swap(rejected_requests_);
  return result;
}

}  // namespace ksana_llm
