/* Copyright 2026 Tencent Inc.  All rights reserved.
 * ==============================================================================*/

#include "batch_scheduler/event_driven_scheduler/model_perf/model_perf_workload_balancer.h"

#include <climits>

#include "utils/logger.h"

namespace ksana_llm {

void ModelPerfWorkloadBalancer::AddInferRequests(std::vector<InferRequestPtr>& requests,
                                                 const ScheduleContext& context) {
  // 仅做日志记录，不修改任何 attn_dp_group_id。
  for (const auto& req : requests) {
    if (req != nullptr) {
      KLLM_LOG_DEBUG << "ModelPerfWorkloadBalancer: req_id=" << req->req_id
                     << " attn_dp_group_id=" << req->attn_dp_group_id;
    }
  }
}

std::vector<WorkloadQuota> ModelPerfWorkloadBalancer::GetDpSchedulingOrderAndQuota(const PlanningBatch& batch,
                                                                                    const ScheduleContext& context,
                                                                                    InferStage stage) {
  std::vector<WorkloadQuota> quotas;
  quotas.reserve(dp_num_);
  for (size_t dp_idx = 0; dp_idx < dp_num_; ++dp_idx) {
    WorkloadQuota q;
    q.dp_idx = dp_idx;
    q.step_token_limit = INT_MAX;
    quotas.push_back(q);
  }
  return quotas;
}

}  // namespace ksana_llm
