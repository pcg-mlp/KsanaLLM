/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/samplers/topk/topk_sampling.h"
#include "ksana_llm/utils/logger.h"

#include "3rdparty/LLM_kernels/csrc/kernels/nvidia/samplers/greedy.h"

#include <cstdint>

namespace ksana_llm {

Status TopkSampling::RunSampling(const float* logits, const uint32_t* offsets, uint32_t* output_token,
                                 const SamplingConfig* sampling_config, const ModelConfig* model_config,
                                 cudaStream_t& stream) {
  if (sampling_config->temperature == 0) {
    // greedy
    llm_kernels::nvidia::InvokeArgMaxReduce(logits, offsets, 1, model_config->vocab_size, output_token, stream);
  } else {
    return Status(RET_INVALID_ARGUMENT, "topk_sampling only support greedy.");
  }
  return Status();
}

}  // namespace ksana_llm
