/* Copyright 2026 Tencent Inc.  All rights reserved.
 *
 * Event-driven scheduler workload balance interface.
 * ==============================================================================*/
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include "batch_scheduler/event_driven_scheduler/workload_balance/workload_balancer_interface.h"

namespace ksana_llm {

class BasicWorkloadBalancer : public WorkloadBalancerInterface {
 public:
  BasicWorkloadBalancer(size_t dp_num, size_t max_batch_size, size_t max_decode_tokens_per_req,
                        size_t max_step_token_num);

  // Assign attn_dp_group_id for incoming requests.
  virtual void AddInferRequests(std::vector<InferRequestPtr>& requests, const ScheduleContext& context) override;

  // Update each dp batch workload quota and return dp scheduling order for the current stage.
  virtual std::vector<WorkloadQuota> GetDpSchedulingOrderAndQuota(const PlanningBatch& batch,
                                                                  const ScheduleContext& context,
                                                                  InferStage stage) override;

  // Record launched inflight dp workload after result_queue_.Put(result).
  virtual void RecordInflightBatch(const PlanningBatch& batch, const ScheduleContext& context) override;

  // Check whether request can be accepted into the target dp batch.
  virtual bool IsRequestAcceptable(const InferRequestPtr& req, const PlanningBatch& batch,
                                   const WorkloadQuota& workload_quota, size_t seq_token_num,
                                   size_t per_dp_step_token_num) const override;

 private:
  size_t dp_num_ = 0;
  size_t max_batch_size_ = 0;
  size_t dp_max_logits_num_ = 0;
  size_t max_step_token_num_ = 0;
  size_t next_dp_idx_ = 0;
  std::vector<size_t> inflight_dp_loads_;
  mutable std::mutex mu_;
};

}  // namespace ksana_llm
