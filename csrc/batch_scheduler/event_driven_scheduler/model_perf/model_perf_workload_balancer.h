/* Copyright 2026 Tencent Inc.  All rights reserved.
 *
 * ModelPerfWorkloadBalancer：ModelPerfBackend 专用 workload balancer。
 * 在 SubmitBatch 时 InferRequest.attn_dp_group_id 已被 ModelPerfBackend 提前写好，
 * 本 balancer 直接透传，不做轮询覆写。
 * ==============================================================================*/
#pragma once

#include <climits>
#include <vector>

#include "batch_scheduler/event_driven_scheduler/workload_balance/workload_balancer_interface.h"

namespace ksana_llm {

class ModelPerfWorkloadBalancer : public WorkloadBalancerInterface {
 public:
  explicit ModelPerfWorkloadBalancer(size_t dp_num) : dp_num_(dp_num) {}
  ~ModelPerfWorkloadBalancer() override = default;

  // 透传：不修改 req->attn_dp_group_id，直接按已有值路由到对应 DP。
  void AddInferRequests(std::vector<InferRequestPtr>& requests, const ScheduleContext& context) override;

  // 不会被 ModelPerfBackend 调用；返回全 DP 最大 quota，满足潜在的健壮性检查。
  std::vector<WorkloadQuota> GetDpSchedulingOrderAndQuota(const PlanningBatch& batch, const ScheduleContext& context,
                                                          InferStage stage) override;

  // no-op
  void RecordInflightBatch(const PlanningBatch& batch, const ScheduleContext& context) override {}

  // 始终接受（ModelPerfBackend 自行控制 running_reqs，此处不做过滤）。
  bool IsRequestAcceptable(const InferRequestPtr& req, const PlanningBatch& batch,
                            const WorkloadQuota& workload_quota, size_t seq_token_num,
                            size_t per_dp_step_token_num) const override {
    return true;
  }

 private:
  size_t dp_num_;
};

}  // namespace ksana_llm
