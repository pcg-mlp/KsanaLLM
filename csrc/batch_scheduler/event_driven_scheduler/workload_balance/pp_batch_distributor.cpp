/* Copyright 2026 Tencent Inc.  All rights reserved.
 *
 * Event-driven scheduler: PP batch workload distributor implementation.
 * ==============================================================================*/

#include "batch_scheduler/event_driven_scheduler/workload_balance/pp_batch_distributor.h"

#include <algorithm>
#include <sstream>

#include "batch_scheduler/event_driven_scheduler/event_driven_state.h"
#include "utils/logger.h"

namespace ksana_llm {

PpBatchDistributor::PpBatchDistributor(PPMultibatchWBStrategy strategy) : strategy_(strategy) {
  KLLM_LOG_INFO << "PpBatchDistributor created with strategy=" << static_cast<int>(strategy_);
}

void PpBatchDistributor::DistributeRequests(std::vector<std::shared_ptr<InferRequest>>& requests,
                                            size_t max_pp_batch_num, size_t dp_num,
                                            std::vector<std::vector<DpScheduleState>>& schedule_states,
                                            std::unordered_map<int64_t, size_t>& req_to_batch_idx) {
  if (requests.empty() || max_pp_batch_num == 0) {
    return;
  }

  // Compute current workload for each PP batch.
  // Workload = sum of all request workloads across all DP slots for this batch.
  std::vector<int> batch_workloads(max_pp_batch_num, 0);
  for (size_t bid = 0; bid < max_pp_batch_num; ++bid) {
    int workload = 0;
    for (size_t dp = 0; dp < dp_num; ++dp) {
      workload += CalculateSlotWorkload(schedule_states, bid, dp);
    }
    batch_workloads[bid] = workload;
  }

  std::stringstream ss;
  std::vector<std::vector<int64_t>> assigned_req_ids(max_pp_batch_num);

  // Greedy min-workload assignment: each request goes to the least-loaded batch.
  for (auto& req : requests) {
    if (req == nullptr) {
      continue;
    }

    auto min_it = std::min_element(batch_workloads.begin(), batch_workloads.end());
    size_t min_batch_id = static_cast<size_t>(std::distance(batch_workloads.begin(), min_it));
    size_t dp_idx = static_cast<size_t>(req->attn_dp_group_id);

    // Place request into the chosen batch's prefilling queue.
    schedule_states[min_batch_id][dp_idx].prefilling_reqs.push_back(req);

    // Record batch membership.
    req_to_batch_idx[req->req_id] = min_batch_id;

    // Update workload counter.
    int req_workload = CalculateWorkload(req);
    batch_workloads[min_batch_id] += req_workload;

    assigned_req_ids[min_batch_id].push_back(req->req_id);
  }

  for (size_t bid = 0; bid < max_pp_batch_num; ++bid) {
    const auto& req_ids = assigned_req_ids[bid];
    if (!req_ids.empty()) {
      ss << "[batch " << bid << " add " << req_ids.size() << " reqs, workload=" << batch_workloads[bid]
         << ", req_ids=" << Vector2Str(req_ids) << "] ";
    }
  }
  KLLM_LOG_SCHEDULER << "Distributed " << requests.size() << " reqs across " << max_pp_batch_num
                     << " PP batches: " << ss.str();
}

int PpBatchDistributor::CalculateWorkload(const std::shared_ptr<InferRequest>& req) const {
  if (!req) {
    return 0;
  }
  switch (strategy_) {
    case PPMultibatchWBStrategy::NO_WB:
      return 1;
    case PPMultibatchWBStrategy::NO_DYNAMIC_WB:
      return 1;
    case PPMultibatchWBStrategy::WB_BATCH_REQ:
      return 1;
    case PPMultibatchWBStrategy::WB_BATCH_TOKEN:
      // Coarse token-based: each 200-token chunk counts as 1 workload unit.
      return static_cast<int>(req->forwarding_tokens.size() / 200) + 1;
    case PPMultibatchWBStrategy::WB_REQ_TOKEN:
      // Fine token-based: actual uncached tokens.
      return std::max(1, static_cast<int>(req->output_tokens.size()) - static_cast<int>(req->kv_cached_token_num));
    default:
      return 1;
  }
}

int PpBatchDistributor::CalculateSlotWorkload(const std::vector<std::vector<DpScheduleState>>& schedule_states,
                                              size_t multi_batch_id, size_t dp_idx) const {
  const DpScheduleState& dp_state = schedule_states[multi_batch_id][dp_idx];
  int workload = 0;

  // Sum workload from all queues: preprocessing, prefilling, decoding.
  for (const auto& kv : dp_state.preprocessing_reqs) {
    workload += CalculateWorkload(kv.second);
  }
  for (const auto& req : dp_state.prefilling_reqs) {
    workload += CalculateWorkload(req);
  }
  for (const auto& req : dp_state.decoding_reqs) {
    workload += CalculateWorkload(req);
  }
  return workload;
}

}  // namespace ksana_llm