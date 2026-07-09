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
  // seq_token_num: tokens counted against the GLOBAL max_step_token_num (kernel/显存 单步上限).
  //   prefill = 整条 seq (含命中 prefix, MLA prefix_kv_buffer); decode = per-step query.
  // per_dp_step_token_num: tokens counted against the per-dp quota (调度公平). prefill = req_step
  //   (本步新算 token); decode = per-step query. per-dp quota 仅在 dp 本步已有请求时生效。
  virtual bool IsRequestAcceptable(const InferRequestPtr& req, const PlanningBatch& batch,
                                   const WorkloadQuota& workload_quota, size_t seq_token_num,
                                   size_t per_dp_step_token_num) const = 0;
};

}  // namespace ksana_llm
