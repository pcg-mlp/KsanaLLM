/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/utils/request.h"
#include "ksana_llm/utils/status.h"

struct curandStateXORWOW;
typedef struct curandStateXORWOW curandState_t;

namespace ksana_llm {
struct SamplingDevideParameter {
  int* device_topKs = nullptr;
  float* device_topPs = nullptr;
  float* device_temperatures = nullptr;
  int** device_output_tokens_ptrs = nullptr;
#ifdef ENABLE_CUDA
  curandState_t* device_curandstates = nullptr;
#endif
  int vocab_size_padded = 0;
  int max_topK = 0;
  int bs = 0;
};

class BaseSampling {
 public:
  BaseSampling(size_t max_batch_size, size_t max_vocab_size)
      : max_batch_size_(max_batch_size), max_vocab_size_(max_vocab_size) {}
#ifdef ENABLE_CUDA
  Status Forward(float* logits, const uint32_t* offsets, uint32_t* output_token, const SamplingConfig* sampling_config,
                 SamplingDevideParameter sampling_devide_parameter, const ModelConfig* model_config,
                 cudaStream_t& stream);
#endif
#ifdef ENABLE_ACL
  Status Forward(float* logits, const uint32_t* offsets, uint32_t* output_token, const SamplingConfig* sampling_config,
                 SamplingDevideParameter sampling_devide_parameter, const ModelConfig* model_config,
                 aclrtStream& stream);
#endif
  virtual ~BaseSampling() {}

 protected:
#ifdef ENABLE_CUDA
  virtual Status RunSampling(float* logits, const uint32_t* offsets, uint32_t* output_token,
                             const SamplingConfig* sampling_config, SamplingDevideParameter sampling_devide_parameter,
                             const ModelConfig* model_config, cudaStream_t& stream) = 0;
#endif

#ifdef ENABLE_ACL
  virtual Status RunSampling(float* logits, const uint32_t* offsets, uint32_t* output_token,
                             const SamplingConfig* sampling_config, SamplingDevideParameter sampling_devide_parameter,
                             const ModelConfig* model_config, aclrtStream& stream) = 0;
#endif
  // The max batch size.
  size_t max_batch_size_ = 8;

  // The max vocab size.
  size_t max_vocab_size_ = 0;
};

}  // namespace ksana_llm
