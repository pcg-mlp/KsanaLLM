/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "batch_scheduler/global_cache_put_tracker.h"

#include "utils/logger.h"

namespace ksana_llm {

GlobalCachePutTracker::GlobalCachePutTracker(int device_count, std::shared_ptr<CacheManagerInterface> cache_manager)
    : device_count_(device_count), cache_manager_(cache_manager) {}

void GlobalCachePutTracker::SetDpCacheManagers(std::vector<std::shared_ptr<CacheManagerInterface>> managers) {
  std::lock_guard<std::mutex> lock(mutex_);
  dp_cache_managers_ = std::move(managers);
}

void GlobalCachePutTracker::OnBatchDispatched(const std::vector<std::shared_ptr<InferRequest>>& running_reqs) {
  std::lock_guard<std::mutex> lock(mutex_);
  for (const auto& req : running_reqs) {
    // 仅对 prefill 请求（kContext）产生 pending 计数，decode 请求不发起 put
    if (req->infer_stage == InferStage::kContext) {
      pending_counts_[req->req_id] += device_count_;
      KLLM_LOG_SCHEDULER << "GlobalCachePutTracker: register pending put, req_id=" << req->req_id
                         << ", pending=" << pending_counts_[req->req_id];
    }
  }
}

void GlobalCachePutTracker::OnDevicePutDone(int device_id, int64_t req_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = pending_counts_.find(req_id);
  if (it == pending_counts_.end()) {
    KLLM_LOG_WARNING << "GlobalCachePutTracker: received put done for unknown req_id=" << req_id;
    return;
  }
  it->second -= 1;
  KLLM_LOG_SCHEDULER << "GlobalCachePutTracker: device_id=" << device_id << " put done, req_id=" << req_id
                     << ", remaining=" << it->second;

  if (it->second <= 0) {
    pending_counts_.erase(it);
    TryFinishDeferred(req_id);
  }
}

bool GlobalCachePutTracker::DeferDestructionIfNeeded(int64_t req_id, size_t dp_idx) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = pending_counts_.find(req_id);
  if (it != pending_counts_.end() && it->second > 0) {
    deferred_req_ids_.insert(req_id);
    deferred_dp_idx_[req_id] = dp_idx;
    KLLM_LOG_SCHEDULER << "GlobalCachePutTracker: deferred destruction, req_id=" << req_id << ", dp_idx=" << dp_idx
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
  // 调用方已持有 mutex_
  auto it = deferred_req_ids_.find(req_id);
  if (it != deferred_req_ids_.end()) {
    deferred_req_ids_.erase(it);

    // 优先使用多 DP cache manager（事件驱动调度器路径），回退到单 cache_manager（BatchScheduler 路径）
    size_t dp_idx = 0;
    auto dp_it = deferred_dp_idx_.find(req_id);
    if (dp_it != deferred_dp_idx_.end()) {
      dp_idx = dp_it->second;
      deferred_dp_idx_.erase(dp_it);
    }

    std::shared_ptr<CacheManagerInterface> mgr = nullptr;
    if (!dp_cache_managers_.empty() && dp_idx < dp_cache_managers_.size()) {
      mgr = dp_cache_managers_[dp_idx];
    } else {
      mgr = cache_manager_;
    }

    KLLM_LOG_SCHEDULER << "GlobalCachePutTracker: deferred destruction completed, releasing blocks, req_id=" << req_id
                       << ", dp_idx=" << dp_idx;
    if (mgr) {
      mgr->DestroyFinishedRequest(req_id);
    }
  }
}

}  // namespace ksana_llm
