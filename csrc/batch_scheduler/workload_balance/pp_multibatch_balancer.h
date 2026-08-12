/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <utility>

#include "batch_scheduler/state/batch_state.h"

namespace ksana_llm {

class PPMultibatchWorkloadBalancer {
 public:
  explicit PPMultibatchWorkloadBalancer(const BatchSchedulerConfig& batch_scheduler_config);

  // Whether the balancer is currently holding waiting reqs in its private queue
  bool HasHeldWaitingReqs() const { return !held_waiting_reqs_.empty(); }

  // Balance requests between batch states, considering waiting_reqs
  void BalancePPMultiBatchReqs(size_t multi_batch_id, std::vector<std::shared_ptr<InferRequest>>& waiting_reqs,
                               std::vector<std::shared_ptr<BatchState>>& batch_states);

  // Shift chunked-prefill requests (prefill_start_offset > 0) to the next multi_batch.
  void ShiftChunkedReqs(size_t multi_batch_id, std::vector<std::shared_ptr<BatchState>>& batch_states);

 private:
  // Distribute waiting requests to multiple batch states based on capacity
  void DistributeWaitingReqs(size_t multi_batch_id, std::vector<std::shared_ptr<InferRequest>>& waiting_reqs,
                             std::vector<std::shared_ptr<BatchState>>& batch_states);

  // Tokens consumed in the *next two* iters for req (capped by split_fuse_token_num_ if enabled).
  std::pair<size_t, size_t> StepLoadOf(const InferRequest* req);

 private:
  PPMultibatchWBStrategy pp_multibatch_wb_strategy_;
  size_t step_token_num_;
  size_t split_fuse_token_num_;
  size_t step_token_num_bound_;  // saturation threshold for prefill
  size_t max_draft_token_num_;

  std::vector<size_t> workloads_;       // per-microbatch step token counts (current iter)
  std::vector<size_t> last_workloads_;  // per-microbatch step token counts (prev iter, snapshot)

  std::vector<std::shared_ptr<InferRequest>> held_waiting_reqs_;  // reqs being held by the balancer between iters
  int hold_iter_ = 0;  // number of consecutive iters the balancer has held waiting reqs before firing
};

}  // namespace ksana_llm
