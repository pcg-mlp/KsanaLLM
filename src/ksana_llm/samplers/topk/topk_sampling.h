/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/samplers/base/base_sampling.h"

namespace ksana_llm {

class TopkSampling : public BaseSampling {
 public:
  TopkSampling(size_t max_batch_size, size_t max_vocab_size, curandState_t* device_curandstates);
  ~TopkSampling();

 private:
  Status RunSampling(float* logits, const uint32_t* offsets, uint32_t* output_token,
                     const SamplingConfig* sampling_config, SamplingDevideParameter sampling_devide_parameter,
                     const ModelConfig* model_config, cudaStream_t& stream) override;

  int workspace_block_id_;
  void* workspace_ = nullptr;
  size_t workspace_size_ = 0;
};

}  // namespace ksana_llm
