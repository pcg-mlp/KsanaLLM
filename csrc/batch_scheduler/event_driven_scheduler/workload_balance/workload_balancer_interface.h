/* Copyright 2026 Tencent Inc.  All rights reserved.
 *
 * Event-driven scheduler workload balance interface.
 * ==============================================================================*/
#pragma once

#include <memory>
#include <vector>

#include "../event_driven_state.h"
#include "../schedule_context.h"

namespace ksana_llm {

struct WorkloadQuota {
  size_t dp_idx = 0;
  int step_token_limit = 0;
};

class WorkloadBalancerInterface {
 public:
  using InferRequestPtr = std::shared_ptr<InferRequest>;

  virtual ~WorkloadBalancerInterface() = default;

  // Assign attn_dp_group_id for incoming requests.
  virtual void AddInferRequests(std::vector<InferRequestPtr>& requests, const ScheduleContext& context) = 0;

  // Update each dp batch workload quota and return dp scheduling order for the current stage.
  virtual std::vector<WorkloadQuota> GetDpSchedulingOrderAndQuota(const PlanningBatch& batch,
                                                                  const ScheduleContext& context, InferStage stage) = 0;

  // Record launched inflight dp workload after result_queue_.Put(result).
  virtual void RecordInflightBatch(const PlanningBatch& batch, const ScheduleContext& context) = 0;

  // Check whether request can be accepted into the target dp batch.
  virtual bool IsRequestAcceptable(const InferRequestPtr& req, const PlanningBatch& batch,
                                   const WorkloadQuota& workload_quota, size_t seq_token_num) const = 0;
};

}  // namespace ksana_llm
