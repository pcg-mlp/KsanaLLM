/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "batch_scheduler/global_cache_fetcher.h"

#include <algorithm>
#include <chrono>
#include <climits>
#include <future>
#include <string>

#include "configure/environment.h"
#include "ipc/ipc_func_wrapper.h"
#include "profiler/reporter.h"
#include "profiler/timer.h"
#include "utils/logger.h"
#include "utils/string_utils.h"

namespace ksana_llm {

GlobalCacheFetcher::GlobalCacheFetcher(int device_count, std::shared_ptr<IpcFuncWrapper> ipc_func_wrapper,
                                       const std::string& role, int fetch_thread_num, int fetch_timeout_ms)
    : device_count_(device_count),
      ipc_func_wrapper_(ipc_func_wrapper),
      role_(role),
      fetch_timeout_ms_(std::max(1, fetch_timeout_ms)) {
  const size_t thread_num = static_cast<size_t>(std::max(1, fetch_thread_num));
  dispatch_thread_pool_ = std::make_unique<ThreadPool>(thread_num);
  KLLM_LOG_INFO << "created: device_count=" << device_count_ << ", fetch_thread_num=" << thread_num
                << ", fetch_timeout_ms=" << fetch_timeout_ms_ << ", role=" << (role_.empty() ? "(standalone)" : role_)
                << ", ipc_enabled=" << (ipc_func_wrapper_ != nullptr);
}

GlobalCacheFetcher::~GlobalCacheFetcher() { Stop(); }


void GlobalCacheFetcher::Start() {
  terminated_ = false;
  dispatch_thread_pool_->Start();
  // 仅异步 IPC 路径需要超时扫尾；无后端时 Submit 同步完成
  if (ipc_func_wrapper_ != nullptr) {
    timeout_thread_ = std::thread([this] { TimeoutLoop(); });
  }
}

void GlobalCacheFetcher::Stop() {
  terminated_ = true;
  timeout_wake_cv_.notify_all();
  if (timeout_thread_.joinable()) {
    timeout_thread_.join();
  }

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
    size_t input_token_num = req->GetPrefillingTokens().size();
    size_t seq_len = req->GetPlanningSequenceLen();
    KLLM_LOG_DEBUG << "GlobalCacheFetcher: req_id=" << req->req_id << ", input_tokens=" << input_token_num
                   << ", seq_len=" << seq_len << ", local_cache=" << local_cached_len
                   << ", global_cache=0, total_cached=" << local_cached_len
                   << ", remaining_prefill=" << (seq_len > local_cached_len ? seq_len - local_cached_len : 0)
                   << ", full_hit=" << (local_cached_len >= seq_len);
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
  pending.submit_time_us = ProfileTimer::GetCurrentTimeInUs();
  const size_t device_slots = static_cast<size_t>(std::max(device_count_, 0));
  pending.device_done_us.assign(device_slots, -1);
  pending.swa_enabled = !req->swa_kv_cache_blocks.empty() &&
                        req->swa_kv_cache_blocks.size() == req->swa_kv_cache_block_idx_offsets.size();
  {
    const size_t planning_seq_len = req->GetPlanningSequenceLen();
    const size_t token_seq_len = req->GetPrefillingTokens().size();
    pending.query_seq_len = std::max(planning_seq_len, token_seq_len);
  }

  int64_t req_id = req->req_id;

  {
    std::lock_guard<std::mutex> lock(mutex_);
    pending_requests_[req_id] = std::move(pending);
  }

  KLLM_LOG_SCHEDULER << "GlobalCacheFetcher: submit req_id=" << req_id << " to global cache fetch queue"
                     << ", local_cached_len=" << local_cached_len << ", device_count=" << device_count_;

  // 在线程池中异步发送 IPC 到所有 executor
  dispatch_thread_pool_->Submit([this, req_id]() {
    if (terminated_) {
      return;
    }

    PendingRequest pending_copy;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto it = pending_requests_.find(req_id);
      if (it == pending_requests_.end()) {
        return;
      }
      pending_copy = it->second;
    }
    DispatchQueryToExecutors(pending_copy);
  });
}

size_t GlobalCacheFetcher::SwaWindowBlockNum(size_t block_token_num) const {
  if (sliding_window_size_ == 0 || block_token_num == 0) {
    return 1;
  }
  return (sliding_window_size_ + block_token_num - 1) / block_token_num;
}

void GlobalCacheFetcher::DispatchQueryToExecutors(const PendingRequest& pending) {
  // 先拷贝只读字段，再并行 Invoke：同一请求尽量同时发到各卡，避免串行发送拉开起点
  const int64_t req_id = pending.req->req_id;
  const std::vector<int> token_ids = pending.req->GetPrefillingTokens();
  const size_t local_cached_len = pending.req->GetPlanningWorkload().prefill_start_offset;
  const int block_size = static_cast<int>(pending.req->block_token_num);
  const std::vector<int> kv_cache_block_ids = pending.req->kv_cache_blocks;
  const int n = device_count_;

  // SWA：命中边界必须落在「末 W 个位置有 SWA」处，因此把本地 SWA 落点与命中上限一起发给
  // executor，让求边界与拉取在同一处完成。上限用 seq_len - 1，engine 侧的 cap 只作兜底。
  const std::vector<int> swa_kv_cache_block_ids =
      pending.swa_enabled ? pending.req->swa_kv_cache_blocks : std::vector<int>{};
  const std::vector<size_t> swa_block_idx_offsets =
      pending.swa_enabled ? pending.req->swa_kv_cache_block_idx_offsets : std::vector<size_t>{};
  const size_t swa_window_block_num = pending.swa_enabled ? SwaWindowBlockNum(pending.req->block_token_num) : 1;
  const size_t seq_len = pending.query_seq_len > 0 ? pending.query_seq_len : pending.req->GetPlanningSequenceLen();
  const size_t max_cached_len = pending.swa_enabled && seq_len > 0 ? seq_len - 1 : 0;

  std::vector<std::future<void>> futs;
  futs.reserve(static_cast<size_t>(std::max(n, 0)));

  for (int executor_id = 0; executor_id < n; ++executor_id) {
    futs.push_back(std::async(std::launch::async, [this, executor_id, req_id, &token_ids, local_cached_len, block_size,
                                                   &kv_cache_block_ids, &swa_kv_cache_block_ids, &swa_block_idx_offsets,
                                                   swa_window_block_num, max_cached_len]() {
      Status status = ipc_func_wrapper_->DispatchGlobalCacheQuery(
          executor_id, req_id, token_ids, local_cached_len, block_size, kv_cache_block_ids, swa_kv_cache_block_ids,
          swa_block_idx_offsets, swa_window_block_num, max_cached_len);
      if (!status.OK()) {
        KLLM_LOG_ERROR << "dispatch global cache query IPC failed, req_id=" << req_id << ", executor_id=" << executor_id
                       << ", error=" << status.GetMessage();
        GlobalCacheQueryResult result;
        result.req_id = req_id;
        result.device_id = executor_id;
        result.success = false;
        OnDeviceComplete(result);
      }
    }));
  }
  for (auto& fut : futs) {
    fut.wait();
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
  const int64_t now_us = ProfileTimer::GetCurrentTimeInUs();
  const int64_t done_offset = now_us - pending.submit_time_us;
  if (result.device_id >= 0 && static_cast<size_t>(result.device_id) < pending.device_done_us.size()) {
    pending.device_done_us[static_cast<size_t>(result.device_id)] = done_offset;
  }
  pending.last_done_us = done_offset;

  if (result.success) {
    pending.per_device_ranges.push_back(result.matched_ranges);

    // SWA：各卡必须报回完全相同的对齐边界。不一致说明有的卡缺 SWA 内容，
    // 取最小值会让命中点落在那些卡没有拉 SWA 的位置上，只能整体放弃全局命中。
    if (pending.swa_enabled) {
      if (pending.swa_aligned_hit_len == 0) {
        pending.swa_aligned_hit_len = result.swa_aligned_hit_len;
      } else if (pending.swa_aligned_hit_len != result.swa_aligned_hit_len) {
        pending.swa_consistent = false;
      }
      if (result.swa_aligned_hit_len == 0) {
        pending.swa_consistent = false;
      }
    }

    // 任一设备查到 token(s) 即采用（token 与设备无关，所有设备的结果一致）
    if (result.has_cached_tokens && !pending.has_cached_tokens) {
      pending.cached_tokens = result.cached_tokens;
      pending.has_cached_tokens = true;
      KLLM_LOG_DEBUG << "GlobalCacheFetcher: req_id=" << result.req_id
                     << " cached_tokens collected from device_id=" << result.device_id
                     << ", token_count=" << pending.cached_tokens.size();
    }
  } else {
    pending.all_success = false;
    KLLM_LOG_DEBUG << "GlobalCacheFetcher: req_id=" << result.req_id << " device_id=" << result.device_id
                   << " reported failure";
  }

  KLLM_LOG_SCHEDULER << "GlobalCacheFetcher: req_id=" << result.req_id << " device_id=" << result.device_id << " done, "
                     << pending.devices_completed << "/" << pending.total_devices << ", success=" << result.success;

  if (pending.devices_completed >= pending.total_devices) {
    FinalizePendingLocked(it, lock);
  }
}

void GlobalCacheFetcher::FinalizePendingLocked(std::unordered_map<int64_t, PendingRequest>::iterator it,
                                               std::unique_lock<std::mutex>& lock) {
  PendingRequest& pending = it->second;
  const int64_t req_id = pending.req->req_id;

  size_t local_cached_len = pending.req->GetPlanningWorkload().prefill_start_offset;
  size_t total_cached_len = local_cached_len;

  // SWA 命中的边界要被 hook 用来算 DropUnbackedSwaBlocks 的保留起点，所以要跟着 pending 传出去。
  size_t swa_aligned_hit_len = 0;

  if (pending.all_success && !pending.per_device_ranges.empty()) {
    size_t global_prefix_len = GetAllDevicesContiguousPrefixLen(pending.per_device_ranges, local_cached_len);
    total_cached_len += global_prefix_len;

    if (pending.swa_enabled) {
      // executor 侧保证：主 KV 没有完整覆盖 [local, L) 时会把 matched_ranges 一起清空，
      // 所以这里要求两个信号严格对齐；对不上就说明 L 与实际可用前缀不一致，放弃全局命中。
      if (!pending.swa_consistent || total_cached_len != pending.swa_aligned_hit_len) {
        if (global_prefix_len > 0) {
          KLLM_LOG_WARNING << "GlobalCacheFetcher: req_id=" << req_id
                           << " drop global hit, swa_consistent=" << pending.swa_consistent
                           << ", swa_aligned_hit_len=" << pending.swa_aligned_hit_len
                           << ", contiguous_total=" << total_cached_len << ", local=" << local_cached_len;
        } else {
          KLLM_LOG_DEBUG << "GlobalCacheFetcher: req_id=" << req_id
                         << " swa align unavailable, no global extension, swa_consistent=" << pending.swa_consistent
                         << ", swa_aligned_hit_len=" << pending.swa_aligned_hit_len << ", local=" << local_cached_len;
        }
        total_cached_len = local_cached_len;
        global_prefix_len = 0;
      } else {
        swa_aligned_hit_len = pending.swa_aligned_hit_len;
      }
    }

    if (global_prefix_len > 0) {
      KLLM_LOG_DEBUG << "GlobalCacheFetcher: req_id=" << req_id
                     << " global cache extended prefix, local=" << local_cached_len
                     << ", global_prefix=" << global_prefix_len << ", total_cached=" << total_cached_len
                     << ", swa_aligned=" << swa_aligned_hit_len;
    }
  }

  // get 耗时：相对 submit 的最晚一卡完成时刻（us）。无 IPC 的本地旁路路径保持 0。
  if (pending.last_done_us >= 0) {
    pending.req->global_cache_get_us.store(pending.last_done_us, std::memory_order_relaxed);
    REPORT_METRIC("global_cache_get_us", pending.last_done_us);
  }

  // 统一处理 workload 更新和全命中切换 decode（decode 角色不满足条件时会拒绝请求）
  UpdateRequestWorkload(pending.req, total_cached_len, pending.cached_tokens, pending.has_cached_tokens,
                        swa_aligned_hit_len);

  // 按实际采纳的 workload 记扩展长度（被放弃/截断后为 0）。
  {
    const size_t applied = pending.req->GetPlanningWorkload().prefill_start_offset;
    const int ext = (applied > local_cached_len) ? static_cast<int>(applied - local_cached_len) : 0;
    pending.req->global_cache_ext_tokens.store(ext, std::memory_order_relaxed);
    // 与 prefix_cache_hit_token_num（仅本地）对称：这里只记全局相对本地多出来的 token。
    if (ext > 0) {
      REPORT_METRIC("global_cache_hit_req_num", 1);
      REPORT_METRIC("global_cache_ext_token_num", ext);
    }
  }

  // 交给 hook 释放命中范围内未写入的 SWA block。UpdateRequestWorkload 可能因兜底截断放弃
  // 全局命中，此时 prefill_start_offset 不等于 L，不能再让 hook 按 L 去 drop。
  pending.req->global_swa_aligned_hit_len =
      (swa_aligned_hit_len > 0 && pending.req->GetPlanningWorkload().prefill_start_offset == swa_aligned_hit_len)
          ? swa_aligned_hit_len
          : 0;

  bool success = !pending.req->finished;
  if (success) {
    KLLM_LOG_SCHEDULER << "GlobalCacheFetcher: req_id=" << req_id << " all devices done, moved to prepared queue";
    completed_requests_.push_back(std::move(pending.req));
  } else {
    KLLM_LOG_SCHEDULER << "GlobalCacheFetcher: req_id=" << req_id << " all devices done, request rejected";
    rejected_requests_.push_back(std::move(pending.req));
  }

  CompletionCallback cb = completion_callback_;
  pending_requests_.erase(it);

  if (cb) {
    // 回调在锁外执行，避免死锁（回调可能重新获取锁或操作调度器事件队列）
    lock.unlock();
    cb(req_id, success);
  }
}

void GlobalCacheFetcher::TimeoutLoop() {
  while (!terminated_) {
    {
      std::unique_lock<std::mutex> wake_lock(timeout_wake_mutex_);
      timeout_wake_cv_.wait_for(wake_lock, std::chrono::milliseconds(100), [this] { return terminated_.load(); });
    }
    if (terminated_) {
      break;
    }
    SweepTimeouts();
  }
}

void GlobalCacheFetcher::SweepTimeouts() {
  const int64_t now_us = ProfileTimer::GetCurrentTimeInUs();
  const int64_t timeout_us = static_cast<int64_t>(fetch_timeout_ms_) * 1000;
  std::vector<int64_t> timed_out_ids;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    timed_out_ids.reserve(pending_requests_.size());
    for (const auto& [req_id, pending] : pending_requests_) {
      if (now_us - pending.submit_time_us >= timeout_us) {
        timed_out_ids.push_back(req_id);
      }
    }
  }

  for (int64_t req_id : timed_out_ids) {
    std::unique_lock<std::mutex> lock(mutex_);
    auto it = pending_requests_.find(req_id);
    if (it == pending_requests_.end()) {
      continue;
    }

    PendingRequest& pending = it->second;
    // 再次确认仍超时（期间可能已正常完成）
    if (now_us - pending.submit_time_us < timeout_us) {
      continue;
    }

    std::vector<int> missing_devs;
    for (int d = 0; d < pending.total_devices; ++d) {
      if (static_cast<size_t>(d) >= pending.device_done_us.size() ||
          pending.device_done_us[static_cast<size_t>(d)] < 0) {
        missing_devs.push_back(d);
      }
    }

    KLLM_LOG_WARNING << "fetch timeout, req_id=" << req_id << " age_us=" << (now_us - pending.submit_time_us)
                     << " timeout_ms=" << fetch_timeout_ms_ << " done=" << pending.devices_completed << "/"
                     << pending.total_devices << " missing_dev=" << Vector2Str(missing_devs)
                     << " dev_us=" << Vector2Str(pending.device_done_us);

    // 强制按失败收尾：保留已收到的部分命中区间，但 all_success=false 不再扩展 global prefix
    pending.all_success = false;
    pending.devices_completed = pending.total_devices;
    if (pending.last_done_us < 0) {
      pending.last_done_us = now_us - pending.submit_time_us;
    }
    FinalizePendingLocked(it, lock);
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
                                               const std::vector<int>& cached_tokens, bool has_cached_tokens,
                                               size_t swa_aligned_hit_len) {
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

  // 命中长度来自 SWA 对齐边界时，executor 已经把 seq-1 上限和 block 对齐算进边界里了，
  // 上面这两步只该是恒等变换。真被改动说明边界与「有 SWA 的位置」不再对应：
  // 命中点会退到没拉 SWA 的 block 上，恢复计算读到垃圾，因此整体放弃全局命中。
  if (swa_aligned_hit_len > 0 && usable_cached_len != new_cached_len) {
    KLLM_LOG_WARNING << "swa-aligned hit truncated by cap/floor, abandon global hit: req_id=" << req->req_id
                     << ", new_cached_len=" << new_cached_len << ", usable_cached_len=" << usable_cached_len
                     << ", seq_token_num=" << seq_token_num << ", block_token_num=" << req->block_token_num;
    return;
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
