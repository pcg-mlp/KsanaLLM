/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "batch_scheduler/global_cache_fetcher.h"

#include <climits>

#include "batch_scheduler/full_cache_hit_handler.h"
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
  // prefill 角色只负责计算首 token，调度前将 max_new_tokens 限制为 1
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

    // 统一处理 workload 更新和全命中切换 decode（decode 角色不满足条件时会拒绝请求）
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
  if (new_cached_len == 0 && role_ != "decode") return;

  size_t seq_token_num = req->GetPlanningSequenceLen();
  bool is_decode_role = (role_ == "decode");
  bool full_hit = (new_cached_len >= seq_token_num);
  bool tokens_ready = has_cached_tokens && !cached_tokens.empty();

  KLLM_LOG_DEBUG << "GlobalCacheFetcher::UpdateRequestWorkload: req_id=" << req->req_id
                 << ", role=" << (role_.empty() ? "standalone" : role_) << ", new_cached_len=" << new_cached_len
                 << ", seq_token_num=" << seq_token_num << ", full_hit=" << full_hit
                 << ", tokens_ready=" << tokens_ready << ", cached_token_count=" << cached_tokens.size()
                 << ", infer_stage=" << static_cast<int>(req->infer_stage);

  // decode 角色：必须 KV 全命中且 token(s) 就绪，否则拒绝请求（decode 节点无法 prefill）
  if (is_decode_role && !(full_hit && tokens_ready)) {
    KLLM_LOG_WARNING << "GlobalCacheFetcher: req_id=" << req->req_id << " decode role reject: full_hit=" << full_hit
                     << ", tokens_ready=" << tokens_ready << ", cached=" << new_cached_len << ", seq=" << seq_token_num;
    REPORT_COUNTER("decode_role_reject_req_num", static_cast<size_t>(1));

    req->finish_status =
        Status(RET_RUNTIME_FAILED, "Decode node failed to fetch complete KV cache and tokens from global cache.");
    req->finished = true;
    req->Notify();
    return;
  }

  if (new_cached_len == 0) return;

  if (full_hit) {
    // 本地 + 全局 cache 合计覆盖所有 token，切换 decode
    KLLM_LOG_SCHEDULER << "GlobalCacheFetcher: req_id=" << req->req_id << " full cache hit (cached=" << new_cached_len
                       << ", seq=" << seq_token_num << ", has_cached_tokens=" << has_cached_tokens
                       << ", cached_token_count=" << cached_tokens.size() << "), switch to decode";
    REPORT_COUNTER("full_prompt_matched_req_num", static_cast<size_t>(1));

    req->infer_stage = InferStage::kDecode;

    if (tokens_ready) {
      // 缓存的 token(s) 命中：追加到序列中，直接以 decode 方式处理其 KV 计算。
      REPORT_COUNTER("first_token_cache_hit_req_num", static_cast<size_t>(1));

      req->AppendCachedTokens(cached_tokens);
      req->NotifyStep();

      KLLM_LOG_SCHEDULER << "GlobalCacheFetcher: req_id=" << req->req_id
                         << " cached_tokens(count=" << cached_tokens.size() << ") appended, notify step";

      ScheduleTaskWorkload planning_workload = req->GetPlanningWorkload();
      planning_workload.prefill_start_offset = seq_token_num;
      planning_workload.prefill_token_num = static_cast<int>(cached_tokens.size());
      req->SetPlanningWorkload(planning_workload);
      req->SetRemainingWorkload(planning_workload);

      KLLM_LOG_DEBUG << "GlobalCacheFetcher: req_id=" << req->req_id
                     << " workload set (tokens_ready): prefill_start_offset=" << planning_workload.prefill_start_offset
                     << ", prefill_token_num=" << planning_workload.prefill_token_num;
    } else {
      // standalone/prefill 模式下未查到 token(s)：prefill 最后 1 个 input token 来采样
      ScheduleTaskWorkload planning_workload = req->GetPlanningWorkload();
      HandleFullPrefixCacheHit(req, planning_workload, seq_token_num);

      KLLM_LOG_DEBUG << "GlobalCacheFetcher: req_id=" << req->req_id
                     << " workload set (no tokens): prefill_start_offset=" << planning_workload.prefill_start_offset
                     << ", prefill_token_num=" << planning_workload.prefill_token_num;
    }
  } else {
    // 部分命中（仅 standalone/prefill 角色会到这里，decode 角色已在前面被拒绝）
    ScheduleTaskWorkload planning_workload = req->GetPlanningWorkload();
    size_t old_total = planning_workload.prefill_token_num + planning_workload.prefill_start_offset;
    planning_workload.prefill_start_offset = new_cached_len;
    planning_workload.prefill_token_num = old_total - new_cached_len;
    req->SetPlanningWorkload(planning_workload);

    ScheduleTaskWorkload remaining_workload = req->GetRemainingWorkload();
    size_t remaining_total = remaining_workload.prefill_token_num + remaining_workload.prefill_start_offset;
    remaining_workload.prefill_start_offset = new_cached_len;
    remaining_workload.prefill_token_num = remaining_total - new_cached_len;
    req->SetRemainingWorkload(remaining_workload);

    KLLM_LOG_DEBUG << "GlobalCacheFetcher: req_id=" << req->req_id
                   << " workload set (partial hit): prefill_start_offset=" << planning_workload.prefill_start_offset
                   << ", prefill_token_num=" << planning_workload.prefill_token_num;
  }
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
