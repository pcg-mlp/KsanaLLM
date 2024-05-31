/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/runtime/sampling_request.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/tensor.h"

namespace ksana_llm {

class BeamSearchSampling {
 public:
  BeamSearchSampling(std::shared_ptr<Context> context);
  void Update(std::vector<std::shared_ptr<InferRequest>>& req_group, int dst_idx, int src_idx, int token_idx,
              float cumulative_score);
  Status Sampling(SamplingRequest& sampling_req);

 private:
  std::shared_ptr<Context> context_;
};

}  // namespace ksana_llm
