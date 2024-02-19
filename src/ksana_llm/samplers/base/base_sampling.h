/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/utils/request.h"
#include "ksana_llm/utils/status.h"

namespace ksana_llm {

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

}  // namespace ksana_llm
