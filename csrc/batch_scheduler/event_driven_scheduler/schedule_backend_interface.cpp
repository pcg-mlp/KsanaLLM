/* Copyright 2026 Tencent Inc.  All rights reserved.
 *
 * 事件驱动调度器：调度后端抽象接口默认实现。
 * ==============================================================================*/

#include "batch_scheduler/event_driven_scheduler/schedule_backend_interface.h"

#include <unordered_set>

#include "profiler/timer.h"

namespace ksana_llm {

void ScheduleBackendInterface::PreSchedule(const ScheduleContext& context,
                                           const std::vector<std::shared_ptr<PlanningBatch>>& planning_batches,
                                           std::vector<ScheduleBackendEvent>& events) {
  events.clear();
  // TOOD(robertyuan): support multi-batch
  if (planning_batches.empty() || planning_batches[0] == nullptr) {
    return;
  }

  const int64_t schedule_start_time_in_ms = (context.GetScheduleStartTimeInMs() > 0)
                                                ? context.GetScheduleStartTimeInMs()
                                                : ProfileTimer::GetCurrentTimeInMs();
  auto is_timeout = [this, schedule_start_time_in_ms](const std::shared_ptr<InferRequest>& req) -> bool {
    if (req == nullptr || req->IsStopped()) {
      return false;
    }
    const int64_t deadline_ms = static_cast<int64_t>(req->timestamp_in_us / 1000) +
                                static_cast<int64_t>(batch_scheduler_config_.waiting_timeout_in_ms);
    return schedule_start_time_in_ms > deadline_ms;
  };

  const PlanningBatch* base_batch = planning_batches[0].get();  // TODO(robertyuan): support multi-batch
  for (size_t dp_idx = 0; dp_idx < base_batch->dp_batches.size(); ++dp_idx) {
    const DpScheduleState& dp_state = context.GetDpScheduleState(dp_idx);
    DpScheduleOpsInPlan& in_plan_ops = context.GetMutableInPlanScheduleOps(dp_idx);
    std::unordered_set<int64_t> timeout_req_ids_seen;
    auto collect_timeout_req = [&events, dp_idx, &timeout_req_ids_seen, &is_timeout,
                                &in_plan_ops](const std::shared_ptr<InferRequest>& req) {
      if (!is_timeout(req)) {
        return;
      }
      if (timeout_req_ids_seen.insert(req->req_id).second) {
        ScheduleBackendEvent timeout_event;
        timeout_event.type = ScheduleBackendEventType::kStopRequest;
        timeout_event.dp_idx = dp_idx;
        timeout_event.req_ptr = req;
        timeout_event.status = Status(RET_REQUEST_TIMEOUT, "timeout in running.");
        events.push_back(std::move(timeout_event));

        // Mark stop op in plan so later planning stages can observe it.
        in_plan_ops.stop_req_ids.insert(req->req_id);
      }
    };

    for (const auto& req_kv : dp_state.preprocessing_reqs) {
      collect_timeout_req(req_kv.second);
    }
    for (const auto& req : dp_state.prefilling_reqs) {
      collect_timeout_req(req);
    }
    for (const auto& req : dp_state.decoding_reqs) {
      collect_timeout_req(req);
    }

    for (const auto& running_kv : base_batch->dp_batches[dp_idx].running_reqs) {
      collect_timeout_req(running_kv.second);
    }
  }
}

}  // namespace ksana_llm
