/* Copyright 2026 Tencent Inc.  All rights reserved.
 *
 * 事件驱动调度器：调度后端抽象接口，Schedule / UpdateWithGenerationResult 均以 ScheduleContext 为上下文。
 * 与 BatchSchedulerInterface 解耦，后续删除 BatchSchedulerInterface 时仅需提供本接口的其它实现。
 * ==============================================================================*/
#pragma once

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "batch_scheduler/event_driven_scheduler/event_driven_state.h"
#include "batch_scheduler/event_driven_scheduler/schedule_context.h"
#include "batch_scheduler/event_driven_scheduler/workload_balance/workload_balancer_interface.h"
#include "configure/schedule_config_parser.h"
#include "runtime/infer_request.h"
#include "runtime/schedule_output.h"
#include "utils/status.h"

namespace ksana_llm {

enum class ScheduleBackendEventType {
  kNone,
  kStopRequest,
  kRecomputeRequest,
  // Notify processor to remove request from DpScheduleState and add to decoding queue.
  kMoveToDecode,
};

inline std::string ScheduleBackendEventType2Str(ScheduleBackendEventType e) {
  switch (e) {
    case ScheduleBackendEventType::kNone:
      return "kNone";
    case ScheduleBackendEventType::kStopRequest:
      return "kStopRequest";
    case ScheduleBackendEventType::kRecomputeRequest:
      return "kRecomputeRequest";
    case ScheduleBackendEventType::kMoveToDecode:
      return "kMoveToDecode";
    default:
      return "Wrong type";
  }
}

struct ScheduleBackendEvent {
  ScheduleBackendEventType type = ScheduleBackendEventType::kNone;
  size_t dp_idx = 0;
  // 该事件关联的请求指针；用于延迟执行 Stop/Recompute 的 InferRequest 级操作。
  std::shared_ptr<InferRequest> req_ptr = nullptr;
  // Status carried by this event (e.g. timeout, alloc failed, cache update failed).
  Status status = Status(RET_SUCCESS);
};

// 调度后端：只读访问 ScheduleState，输出本轮可执行 PlanningBatch 以及需处理的事件列表。
class ScheduleBackendInterface {
 public:
  explicit ScheduleBackendInterface(const BatchSchedulerConfig& batch_scheduler_config)
      : batch_scheduler_config_(batch_scheduler_config) {}
  virtual ~ScheduleBackendInterface() = default;

  // 通用预调度阶段：默认实现 timeout 检查并生成 stop 事件。
  virtual void PreSchedule(const ScheduleContext& context,
                           const std::vector<std::shared_ptr<PlanningBatch>>& planning_batches,
                           std::vector<ScheduleBackendEvent>& events);

  virtual void Schedule(const ScheduleContext& context, std::vector<std::shared_ptr<PlanningBatch>>& planning_batches,
                        std::vector<ScheduleBackendEvent>& events) = 0;

  virtual void UpdateWithGenerationResult(size_t multi_batch_id, const GenerationOutputGroup& generation_output,
                                          const ScheduleContext& context,
                                          std::vector<ScheduleBackendEvent>& events) = 0;

  virtual void SetWorkloadBalancer(std::shared_ptr<WorkloadBalancerInterface> workload_balancer) {
    workload_balancer_ = std::move(workload_balancer);
  }

  const BatchSchedulerConfig& GetBatchSchedulerConfig() const { return batch_scheduler_config_; }

  // Keep admission policy behind the backend boundary so the processor does
  // not depend on individual BatchSchedulerConfig fields.
  Status CheckRequestAdmission(const std::shared_ptr<InferRequest>& req) const;

  virtual void Stop() = 0;

 protected:
  BatchSchedulerConfig batch_scheduler_config_;
  std::shared_ptr<WorkloadBalancerInterface> workload_balancer_ = nullptr;
};

}  // namespace ksana_llm
