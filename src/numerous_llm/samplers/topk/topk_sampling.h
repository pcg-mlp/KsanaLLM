/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "numerous_llm/samplers/base/base_sampling.h"

namespace numerous_llm {

class TopkSampling : public BaseSampling {
 public:
  TopkSampling() {}
  ~TopkSampling() {}

 private:
  Status RunSampling(const float* logits, const uint32_t* offsets, uint32_t* output_token,
                     const SamplingConfig* sampling_config, const ModelConfig* model_config,
                     cudaStream_t& stream) override;
};

}  // namespace numerous_llm
