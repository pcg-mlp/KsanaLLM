/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/runtime/sampling_request.h"
#include "ksana_llm/samplers/base/base_sampling.h"
#include "ksana_llm/samplers/beam_search/beam_search_sampling.h"
#include "ksana_llm/samplers/topk/topk_sampling.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/tensor.h"

namespace ksana_llm {

class Sampler {
 public:
  Sampler(const BatchSchedulerConfig& batch_scheduler_config, int rank, std::shared_ptr<Context> context);
  ~Sampler();
  Status Sampling(std::vector<SamplingRequest>& sampling_reqs, Stream& stream);
  Status SamplingAndCalcLogprobs(std::vector<SamplingRequest>& sampling_reqs, float* device_logits,
                                 SamplingDevideParameter& sampling_devide_parameter, Stream& stream);
  void SamplingParameterToDevide(bool use_top_p, bool use_temperature, bool logits_softmax,
                                 SamplingDevideParameter& sampling_devide_parameter, Stream& stream);
  void CopyProbsOutput(std::vector<SamplingRequest>& sampling_reqs, Stream& stream,
                             std::vector<std::vector<float>>& probs_output);
  void ApplyRepetitionPenalty(float* logits, std::vector<int>* input_tokens, std::vector<int>* output_tokens,
                              const int vocab_size, const float repetition_penalty, Stream& stream);

 private:
  BatchSchedulerConfig batch_schedule_config_;
  int rank_;
  TopkSampling* topk_sampling_{nullptr};
  BeamSearchSampling beam_search_sampling_;
  int device_buffer_block_id_{-1};
  void* device_buffer_;
  uint32_t* device_output_tokens_;
  uint32_t* device_offset_;
  int* device_topKs_;
  float* device_topPs_;
  float* device_temperatures_;
  int** device_output_tokens_ptrs_;
  float* device_inv_repetition_penalties_;
  RandState* device_curandstates_{nullptr};

  std::vector<int> host_output_tokens_;
  std::vector<uint32_t> host_offset_;
  std::vector<int> host_topKs_;
  std::vector<float> host_topPs_;
  std::vector<float> host_temperatures_;
  std::vector<const float*> host_logits_;

  // The context
  std::shared_ptr<Context> context_;
  std::vector<float> inv_repetition_penalties_;
};

}  // namespace ksana_llm
