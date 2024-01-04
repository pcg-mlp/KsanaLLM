/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/samplers/sampler.h"
#include "numerous_llm/utils/logger.h"

namespace numerous_llm {

Status Sampler::Sampling(std::vector<SamplingRequest>& sampling_reqs) {
  if (rank_ == 0) {
    // NOTE(karlluo): fake generate result for break the inference loop
    // start_id = 0
    // end_id = 1
    for (auto& sampling_req : sampling_reqs) {
      // NOTE(karlluo): copy sampling_reqs to local or device
      SamplingRequest local_sampling_req = sampling_req;

      if ((local_sampling_req.output_tokens->back()) == local_sampling_req.model_config->start_id) {
        NLLM_LOG_INFO << "Last token is " << local_sampling_req.output_tokens->back() << " push end_id "
                      << local_sampling_req.model_config->end_id << " at back";
        local_sampling_req.output_tokens->push_back(local_sampling_req.model_config->end_id);
      } else {
        NLLM_LOG_INFO << "Last token is " << local_sampling_req.output_tokens->back() << " push start_id "
                      << local_sampling_req.model_config->start_id << " at back";
        local_sampling_req.output_tokens->push_back(local_sampling_req.model_config->start_id);
      }

      sampling_req = local_sampling_req;
    }
  }
  return Status();
}

}  // namespace numerous_llm
