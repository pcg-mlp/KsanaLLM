/* Copyright 2026 Tencent Inc.  All rights reserved.
 *
 * Event-driven scheduler: PP batch workload distributor.
 * Distributes incoming requests across multiple PP batches using a greedy
 * minimum-workload strategy. Analogous to PPMultibatchWorkloadBalancer in the
 * traditional scheduler but adapted for the event-driven scheduler's 2D
 * schedule_states_[multi_batch_id][dp_idx] data layout.
 * ==============================================================================*/
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include "configure/schedule_config_parser.h"
#include "runtime/infer_request.h"

namespace ksana_llm {

// Forward declaration for the 2D schedule states.
struct DpScheduleState;

// Distributes incoming requests across PP batches.
// Phase 2: static greedy min-workload assignment only.
// Phase 5+: dynamic rebalancing (OffloadBatchWorkload) to be added.
class PpBatchDistributor {
 public:
  explicit PpBatchDistributor(PPMultibatchWBStrategy strategy);

  // Assign multi_batch_id to each request and place it into the corresponding
  // schedule_states_[multi_batch_id][dp_idx].prefilling_reqs.
  // Also populates req_to_batch_idx[req_id] = multi_batch_id.
  //
  // dp_idx is already set on each request via attn_dp_group_id (assigned by
  // the DP-level workload balancer before this call).
  void DistributeRequests(std::vector<std::shared_ptr<InferRequest>>& requests, size_t max_pp_batch_num, size_t dp_num,
                          std::vector<std::vector<DpScheduleState>>& schedule_states,
                          std::unordered_map<int64_t, size_t>& req_to_batch_idx);

  // TODO(yancyliu): add OffloadBatchWorkload for dynamic rebalancing in Phase 5+.

 private:
  // Calculate workload for a single request based on the configured strategy.
  int CalculateWorkload(const std::shared_ptr<InferRequest>& req) const;

  // Calculate total workload for all queues in a given (multi_batch_id, dp_idx) slot.
  int CalculateSlotWorkload(const std::vector<std::vector<DpScheduleState>>& schedule_states, size_t multi_batch_id,
                            size_t dp_idx) const;

  PPMultibatchWBStrategy strategy_;
};

}  // namespace ksana_llm