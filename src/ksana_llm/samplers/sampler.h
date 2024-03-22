/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/runtime/sampling_request.h"
#include "ksana_llm/samplers/base/base_sampling.h"
#include "ksana_llm/samplers/topk/topk_sampling.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/tensor.h"

namespace ksana_llm {

class Sampler {
 public:
  Sampler(const BatchSchedulerConfig& batch_scheduler_config, int rank);
  ~Sampler();
#ifdef ENABLE_CUDA
  Status Sampling(std::vector<SamplingRequest>& sampling_reqs, cudaStream_t& stream);
#endif
#ifdef ENABLE_ACL
  Status Sampling(std::vector<SamplingRequest>& sampling_reqs, aclrtStream& stream);
#endif

 private:
  BatchSchedulerConfig batch_schedule_config_;
  int rank_;
  TopkSampling* topk_sampling_{nullptr};
  int device_buffer_block_id_{-1};
  void* device_buffer_;
  uint32_t* device_output_tokens_;
  uint32_t* device_offset_;
  int* device_topKs_;
  float* device_topPs_;
  float* device_temperatures_;
  int** device_output_tokens_ptrs_;
#ifdef ENABLE_CUDA
  curandState_t* device_curandstates_{nullptr};
#endif

  std::vector<int> host_output_tokens_;
  std::vector<uint32_t> host_offset_;
  std::vector<int> host_topKs_;
  std::vector<float> host_topPs_;
  std::vector<float> host_temperatures_;
  std::vector<const float*> host_logits_;
};

}  // namespace ksana_llm
