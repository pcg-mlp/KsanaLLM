/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>

#include "runtime/infer_request.h"

namespace ksana_llm {

// Estimate planning workload for the inflight task and set planning workload from remaining workload.
// max_draft_token_num: the maximum number of draft tokens, derived from SpecSchedulingInfo::workload_multiplier.
inline void EstimateRequestPlanningWorkload(const std::shared_ptr<InferRequest>& req, size_t max_draft_token_num) {
  if (req->HasInflightTask()) {
    if (req->IsSplitPrefillStep()) {
      req->SetInflightTaskGenResultEstimation(0, 0);
    } else {
      req->SetInflightTaskGenResultEstimation(kStepGenerateTokenNum, max_draft_token_num);
    }
  }
  req->SetPlanningWorkload(req->GetRemainingWorkload());
}

}  // namespace ksana_llm
