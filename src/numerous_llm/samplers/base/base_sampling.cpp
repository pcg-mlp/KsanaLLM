/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/samplers/base/base_sampling.h"

namespace numerous_llm {

Status BaseSampling::Forward(const float* logits, const uint32_t* offsets, uint32_t* output_token,
                             const SamplingConfig* sampling_config, const ModelConfig* model_config,
                             cudaStream_t& stream) {
  // the same for all type sampling

  // specialized sampling for different type
  STATUS_CHECK_RETURN(RunSampling(logits, offsets, output_token, sampling_config, model_config, stream));
  return Status();
}

}  // namespace numerous_llm
