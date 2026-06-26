/* Copyright 2026 Tencent Inc.  All rights reserved.
 *
 * 事件驱动调度器：Hook 与调度器共享的上下文。
 * ==============================================================================*/
#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "batch_scheduler/event_driven_scheduler/event_driven_state.h"
#include "batch_scheduler/event_driven_scheduler/event_driven_types.h"
#include "cache_manager/cache_manager_interface.h"

namespace ksana_llm {

// Hook 异步任务完成时调用的回调类型（通常由调度器设为 UpdateAfterAsyncTaskFinished）
using AsyncTaskFinishedCallback = std::function<void(const EventAsyncTaskFinishedInfo&)>;

// 责任链 / Hook 与调度器共享的上下文，可扩展（如挂异步线程池等）
class ScheduleContext {
 public:
  void SetScheduleState(ScheduleState* schedule_state) { schedule_state_ = schedule_state; }

  // 本轮调度开始时间（毫秒时间戳），用于 PreSchedule 的 timeout 判定基准。
  void SetScheduleStartTimeInMs(int64_t schedule_start_time_in_ms) {
    schedule_start_time_in_ms_ = schedule_start_time_in_ms;
  }
  int64_t GetScheduleStartTimeInMs() const { return schedule_start_time_in_ms_; }

  // 一个 DP 对应一个 cache manager，索引即 dp_idx。
  void SetDpCacheManagers(const std::vector<std::shared_ptr<CacheManagerInterface>>* dp_cache_managers) {
    dp_cache_managers_ = dp_cache_managers;
  }

  // ---- multi_batch_id context ----

  // Set/get the PP batch index currently being scheduled (used inside RunScheduleForBatches loop).
  void SetCurrentMultiBatchId(size_t multi_batch_id) { current_multi_batch_id_ = multi_batch_id; }
  size_t GetCurrentMultiBatchId() const { return current_multi_batch_id_; }

  // ---- DpScheduleState access ----

  // 输入 multi_batch_id + dp_idx 获取 const DpScheduleState 引用。
  // 旧的 1D 签名 GetDpScheduleState(size_t dp_idx) 已删除，强制所有调用点显式传递 multi_batch_id。
  const DpScheduleState& GetDpScheduleState(size_t multi_batch_id, size_t dp_idx) const {
    assert(schedule_state_ != nullptr);
    return schedule_state_->schedule_states.at(multi_batch_id).at(dp_idx);
  }

  // Return the number of PP batches (max_pp_batch_num).
  size_t GetNumPpBatches() const {
    assert(schedule_state_ != nullptr);
    return schedule_state_->schedule_states.size();
  }

  // 输入 req_id 获取对应 InferRequestPtr。
  std::shared_ptr<InferRequest> GetReqById(int64_t req_id) const {
    if (schedule_state_ == nullptr) {
      return nullptr;
    }
    auto it = schedule_state_->req_id_to_req_ptr.find(req_id);
    if (it == schedule_state_->req_id_to_req_ptr.end()) {
      return nullptr;
    }
    return it->second;
  }

  // 输入 dp_idx 获取对应的 std::shared_ptr<CacheManagerInterface> 引用。
  const std::shared_ptr<CacheManagerInterface>& GetDpCacheManager(size_t dp_idx) const {
    KLLM_CHECK_WITH_INFO(dp_cache_managers_ != nullptr, "dp_cache_managers_ is nullptr");
    return dp_cache_managers_->at(dp_idx);
  }

  // 输入 multi_batch_id + dp_idx 获取对应的可修改 DpScheduleState.in_plan_schedule_ops 引用。
  // 旧的 1D 签名 GetMutableInPlanScheduleOps(size_t dp_idx) 已删除。
  DpScheduleOpsInPlan& GetMutableInPlanScheduleOps(size_t multi_batch_id, size_t dp_idx) const {
    return schedule_state_->schedule_states[multi_batch_id][dp_idx].in_plan_schedule_ops;
  }

  // ---- req_to_batch_idx helpers ----

  // Look up which PP batch a request belongs to. Returns multi_batch_id if found.
  // Callers MUST check existence first; missing entry means state machine bug → KLLM_THROW.
  size_t GetReqBatchIdx(int64_t req_id) const {
    assert(schedule_state_ != nullptr);
    auto it = schedule_state_->req_to_batch_idx.find(req_id);
    if (it == schedule_state_->req_to_batch_idx.end()) {
      KLLM_THROW(FormatStr("req_id=%ld not found in req_to_batch_idx_, state machine bug", req_id));
    }
    return it->second;
  }

  void SetReqBatchIdx(int64_t req_id, size_t multi_batch_id) {
    schedule_state_->req_to_batch_idx[req_id] = multi_batch_id;
  }

  void RemoveReqBatchIdx(int64_t req_id) { schedule_state_->req_to_batch_idx.erase(req_id); }

 private:
  ScheduleState* schedule_state_ = nullptr;
  int64_t schedule_start_time_in_ms_ = 0;
  const std::vector<std::shared_ptr<CacheManagerInterface>>* dp_cache_managers_ = nullptr;
  size_t current_multi_batch_id_ = 0;
};

}  // namespace ksana_llm
