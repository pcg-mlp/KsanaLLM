/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "numerous_llm/utils/request.h"
#include "numerous_llm/utils/status.h"

namespace numerous_llm {

class BaseSampling {
 public:
  BaseSampling() {}
  Status Forward(const float* logits, const uint32_t* offsets, uint32_t* output_token,
                 const SamplingConfig* sampling_config, const ModelConfig* model_config, cudaStream_t& stream);
  virtual ~BaseSampling() {}

 protected:
  virtual Status RunSampling(const float* logits, const uint32_t* offsets, uint32_t* output_token,
                             const SamplingConfig* sampling_config, const ModelConfig* model_config,
                             cudaStream_t& stream) = 0;
};

}  // namespace numerous_llm
