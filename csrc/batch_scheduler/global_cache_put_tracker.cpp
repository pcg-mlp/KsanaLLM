/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "batch_scheduler/global_cache_put_tracker.h"

#include "profiler/reporter.h"
#include "profiler/timer.h"
#include "utils/logger.h"
#include "utils/request.h"

namespace ksana_llm {

GlobalCachePutTracker::GlobalCachePutTracker(int device_count, std::shared_ptr<CacheManagerInterface> cache_manager)
    : device_count_(device_count), cache_manager_(cache_manager) {}

void GlobalCachePutTracker::SetDpCacheManagers(std::vector<std::shared_ptr<CacheManagerInterface>> managers) {
  std::lock_guard<std::mutex> lock(mutex_);
  dp_cache_managers_ = std::move(managers);
}

void GlobalCachePutTracker::SetSchedulerWake(std::function<void()> wake) {
  std::lock_guard<std::mutex> lock(mutex_);
  scheduler_wake_ = std::move(wake);
}

void GlobalCachePutTracker::OnBatchDispatched(const std::vector<std::shared_ptr<InferRequest>>& running_reqs) {
  std::lock_guard<std::mutex> lock(mutex_);
  for (const auto& req : running_reqs) {
    if (req == nullptr || req->infer_stage != InferStage::kContext) {
      continue;
    }
    // 仅在最后一段 prefill 登记：SetPlanningTask 之后 remaining.prefill>0 说明 input 尚未算完。
    // 中间 split-fuse chunk 若也 += pending，会与按 device 去重的 PutDone 错位，导致 EndSwaPutHold 永不触发。
    if (req->GetRemainingWorkload().prefill_token_num > 0) {
      KLLM_LOG_SCHEDULER << "skip put register, incomplete prefill, req_id=" << req->req_id
                         << ", remaining_prefill=" << req->GetRemainingWorkload().prefill_token_num;
      continue;
    }
    pending_counts_[req->req_id] += device_count_;
    pending_fa_put_blocks_[req->req_id] = 0;
    pending_swa_put_blocks_[req->req_id] = 0;
    pending_done_devices_[req->req_id].clear();
    if (req->request_ptr != nullptr) {
      pending_requests_[req->req_id] = req->request_ptr;
      req->global_cache_put_pending.store(true, std::memory_order_release);
      req->global_cache_put_done_time_us.store(0, std::memory_order_relaxed);
      req->global_cache_put_fa_blocks.store(0, std::memory_order_relaxed);
      req->global_cache_put_swa_blocks.store(0, std::memory_order_relaxed);
    }
    if (req->cache_manager != nullptr) {
      pending_cache_managers_[req->req_id] = req->cache_manager;
      // 在本步 UpdateRequestTokens 滑窗之前扣住 SWA；executor 异步 put 仍要读这些块。
      // BeginSwaPutHold 仅在 scheduler 线程调用（OnBatchDispatched），与 Drain 同线程。
      if (!req->swa_kv_cache_blocks.empty()) {
        req->cache_manager->BeginSwaPutHold(req->req_id);
      }
    }
    KLLM_LOG_SCHEDULER << "register pending put, req_id=" << req->req_id
                       << ", pending=" << pending_counts_[req->req_id];
  }
}

void GlobalCachePutTracker::OnDevicePutDone(int device_id, int64_t req_id, int fa_put_blocks, int swa_put_blocks) {
  // 锁内拷贝 wake / completion，锁外 stamp+wake，避免与 Wait/SetSchedulerWake 交叉持锁。
  std::function<void()> wake;
  PendingCompletion completion_for_metrics;
  bool stamp_metrics = false;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = pending_counts_.find(req_id);
    if (it == pending_counts_.end()) {
      KLLM_LOG_DEBUG << "put done for unknown/skipped req_id=" << req_id << ", device_id=" << device_id;
      return;
    }
    auto& done_devices = pending_done_devices_[req_id];
    if (!done_devices.insert(device_id).second) {
      KLLM_LOG_WARNING << "duplicate put done ignored, req_id=" << req_id << ", device_id=" << device_id
                       << ", remaining=" << it->second;
      return;
    }
    it->second -= 1;
    // FA / SWA 均按本卡实际上传数累加：主 KV 多卡按 block 分片上传，各卡子集不同。
    pending_fa_put_blocks_[req_id] += fa_put_blocks;
    pending_swa_put_blocks_[req_id] += swa_put_blocks;
    KLLM_LOG_SCHEDULER << "device_id=" << device_id << " put done, req_id=" << req_id << ", remaining=" << it->second
                       << ", fa_blocks=" << fa_put_blocks << ", swa_blocks=" << swa_put_blocks;

    if (it->second > 0) {
      return;
    }

    pending_counts_.erase(it);
    pending_done_devices_.erase(req_id);
    PendingCompletion completion;
    completion.req_id = req_id;
    completion.fa_put_blocks = pending_fa_put_blocks_[req_id];
    completion.swa_put_blocks = pending_swa_put_blocks_[req_id];
    pending_fa_put_blocks_.erase(req_id);
    pending_swa_put_blocks_.erase(req_id);

    auto mgr_it = pending_cache_managers_.find(req_id);
    if (mgr_it != pending_cache_managers_.end()) {
      completion.cache_manager = mgr_it->second;
      pending_cache_managers_.erase(mgr_it);
    }
    auto req_it = pending_requests_.find(req_id);
    if (req_it != pending_requests_.end()) {
      completion.request = req_it->second;
      pending_requests_.erase(req_it);
    }
    completion_for_metrics = completion;
    stamp_metrics = true;
    pending_completions_.push_back(std::move(completion));
    wake = scheduler_wake_;
  }

  if (stamp_metrics) {
    StampPutMetricsOnRequest(completion_for_metrics);
  }
  if (wake) {
    wake();
  }
}

void GlobalCachePutTracker::StampPutMetricsOnRequest(const PendingCompletion& completion) {
  if (completion.request == nullptr) {
    return;
  }
  completion.request->global_cache_put_fa_blocks.store(completion.fa_put_blocks, std::memory_order_relaxed);
  completion.request->global_cache_put_swa_blocks.store(completion.swa_put_blocks, std::memory_order_relaxed);
  const uint64_t put_done_us = ProfileTimer::GetCurrentTimeInUs();
  completion.request->global_cache_put_done_time_us.store(put_done_us, std::memory_order_relaxed);
  completion.request->NotifyGlobalCachePutDone();

  // 与 ACCESS 的 put_fa / put_swa 对齐；有实际写入再计 req。
  if (completion.fa_put_blocks > 0) {
    REPORT_METRIC("global_cache_put_fa_block_num", completion.fa_put_blocks);
  }
  if (completion.swa_put_blocks > 0) {
    REPORT_METRIC("global_cache_put_swa_block_num", completion.swa_put_blocks);
  }
  if (completion.fa_put_blocks > 0 || completion.swa_put_blocks > 0) {
    REPORT_METRIC("global_cache_put_req_num", 1);
  }
  // put 可能晚于 finish；finish 已打点时上报 ACCESS 同名字段。
  const uint64_t finish_us = completion.request->finish_time_us.load(std::memory_order_relaxed);
  if (finish_us > 0 && put_done_us > finish_us) {
    REPORT_METRIC("global_cache_put_after_finish_ms", static_cast<double>(put_done_us - finish_us) / 1000.0);
  }
}

size_t GlobalCachePutTracker::DrainPutCompletions() {
  std::vector<PendingCompletion> local;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (pending_completions_.empty()) {
      return 0;
    }
    local.swap(pending_completions_);
  }

  for (const auto& completion : local) {
    if (completion.cache_manager != nullptr) {
      completion.cache_manager->EndSwaPutHold(completion.req_id);
    }
    TryFinishDeferred(completion.req_id);
  }
  return local.size();
}

bool GlobalCachePutTracker::HasPendingCompletions() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return !pending_completions_.empty();
}

bool GlobalCachePutTracker::DeferDestructionIfNeeded(int64_t req_id, size_t dp_idx) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = pending_counts_.find(req_id);
  if (it != pending_counts_.end() && it->second > 0) {
    deferred_req_ids_.insert(req_id);
    deferred_dp_idx_[req_id] = dp_idx;
    deferred_start_us_[req_id] = ProfileTimer::GetCurrentTimeInUs();
    KLLM_LOG_SCHEDULER << "deferred destruction, req_id=" << req_id << ", dp_idx=" << dp_idx
                       << ", pending=" << it->second;
    return true;
  }
  return false;
}

bool GlobalCachePutTracker::HasDeferredRequests() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return !deferred_req_ids_.empty();
}

void GlobalCachePutTracker::TryFinishDeferred(int64_t req_id) {
  // 调用方：DrainPutCompletions（scheduler 线程），不持有 mutex_。
  size_t dp_idx = 0;
  int64_t start_us = -1;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = deferred_req_ids_.find(req_id);
    if (it == deferred_req_ids_.end()) {
      return;
    }
    deferred_req_ids_.erase(it);

    auto dp_it = deferred_dp_idx_.find(req_id);
    if (dp_it != deferred_dp_idx_.end()) {
      dp_idx = dp_it->second;
      deferred_dp_idx_.erase(dp_it);
    }

    auto start_it = deferred_start_us_.find(req_id);
    if (start_it != deferred_start_us_.end()) {
      start_us = start_it->second;
      deferred_start_us_.erase(start_it);
    }
  }

  if (start_us >= 0) {
    const int64_t retention_us = ProfileTimer::GetCurrentTimeInUs() - start_us;
    KLLM_LOG_INFO << "deferred release now, req_id=" << req_id << ", dp_idx=" << dp_idx
                  << ", put_retention_us=" << retention_us;
  } else {
    KLLM_LOG_INFO << "deferred release now, req_id=" << req_id << ", dp_idx=" << dp_idx;
  }

  std::shared_ptr<CacheManagerInterface> mgr = nullptr;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!dp_cache_managers_.empty() && dp_idx < dp_cache_managers_.size()) {
      mgr = dp_cache_managers_[dp_idx];
    } else {
      mgr = cache_manager_;
    }
  }

  if (mgr) {
    mgr->DestroyFinishedRequest(req_id);
  }
}

}  // namespace ksana_llm
