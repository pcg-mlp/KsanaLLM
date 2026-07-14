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
  void SetDpCacheManagers(
      const std::vector<std::shared_ptr<CacheManagerInterface>>* dp_cache_managers) {
    dp_cache_managers_ = dp_cache_managers;
  }

  // 1) 输入 dp_idx 获取 const DpScheduleState 引用。
  const DpScheduleState& GetDpScheduleState(size_t dp_idx) const {
    assert(schedule_state_ != nullptr);
    return schedule_state_->schedule_states.at(dp_idx);
  }

  // 2) 输入 req_id 获取对应 InferRequestPtr。
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

  // 3) 输入 dp_idx 获取对应的 std::shared_ptr<CacheManagerInterface> 引用。
  const std::shared_ptr<CacheManagerInterface>& GetDpCacheManager(size_t dp_idx) const {
    KLLM_CHECK_WITH_INFO(dp_cache_managers_ != nullptr, "dp_cache_managers_ is nullptr");
    return dp_cache_managers_->at(dp_idx);
  }

  // 4) 输入 dp_idx 获取对应的可修改 DpScheduleState.in_plan_schedule_ops 引用。
  DpScheduleOpsInPlan& GetMutableInPlanScheduleOps(size_t dp_idx) const {
    return schedule_state_->schedule_states[dp_idx].in_plan_schedule_ops;
  }

 private:
  ScheduleState* schedule_state_ = nullptr;
  int64_t schedule_start_time_in_ms_ = 0;
  const std::vector<std::shared_ptr<CacheManagerInterface>>* dp_cache_managers_ = nullptr;
};

}  // namespace ksana_llm
