/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#ifdef ENABLE_CUDA
#  include <curand_kernel.h>
#  include "ksana_llm/kernels/nvidia/kernel_wrapper.h"
#endif

#include "ksana_llm/samplers/sampler.h"
#include "ksana_llm/utils/common_device.h"
#include "ksana_llm/utils/device_utils.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/memory_utils.h"

namespace ksana_llm {

Sampler::Sampler(const BatchSchedulerConfig& batch_scheduler_config, int rank, std::shared_ptr<Context> context) {
#ifdef ENABLE_CUDA
  batch_schedule_config_ = batch_scheduler_config;
  context_ = context;
  rank_ = rank;
  auto max_batch_size = batch_scheduler_config.max_batch_size;
  // need to allocate device buffer for sampling
  GetBlockManager()->SetDeviceId(rank_);

  GetBlockManager()->AllocateContiguous(
      (sizeof(uint32_t) * 2 + sizeof(int) + sizeof(float) * 2 + sizeof(curandState_t) + sizeof(int*)) * max_batch_size +
          sizeof(float) * batch_schedule_config_.max_vocab_size,
      device_buffer_block_id_);
  GetBlockManager()->GetContiguousPtr(device_buffer_block_id_, device_buffer_);
  NLLM_LOG_DEBUG << "AllocateContiguous device_buffer_ " << device_buffer_ << " size "
                 << (sizeof(uint32_t) * 2 + sizeof(int) + sizeof(curandState_t) + sizeof(int*)) * max_batch_size +
                        sizeof(float) * batch_schedule_config_.max_vocab_size;
  device_output_tokens_ = static_cast<uint32_t*>(device_buffer_);
  device_offset_ = device_output_tokens_ + max_batch_size;
  device_topKs_ = reinterpret_cast<int*>(device_offset_ + max_batch_size);
  device_topPs_ = reinterpret_cast<float*>(device_topKs_ + max_batch_size);
  device_temperatures_ = reinterpret_cast<float*>(device_topPs_ + max_batch_size);
  device_curandstates_ = reinterpret_cast<curandState_t*>(device_temperatures_ + max_batch_size);
  device_output_tokens_ptrs_ = reinterpret_cast<int**>(device_curandstates_ + max_batch_size);
  device_inv_repetition_penalties_ = reinterpret_cast<float*>(device_output_tokens_ptrs_ + max_batch_size);

  inv_repetition_penalties_.resize(batch_schedule_config_.max_vocab_size);

  if (sizeof(uint32_t) != sizeof(int)) {
    NLLM_LOG_ERROR << fmt::format("sizeof(uint32_t)({}) != sizeof(int)({})", sizeof(uint32_t), sizeof(int));
    abort();
    exit(RetCode::RET_SEGMENT_FAULT);
  }
  std::vector<uint32_t*> host_device_output_tokens_ptrs(max_batch_size);
  for (int i = 0; i < max_batch_size; i++) {
    host_device_output_tokens_ptrs[i] = device_output_tokens_ + i;
  }

  MemcpyAsync(device_output_tokens_ptrs_, host_device_output_tokens_ptrs.data(), sizeof(uint32_t*) * max_batch_size,
              MEMCPY_HOST_TO_DEVICE, context_->GetH2DStreams()[0]);

  host_offset_.resize(max_batch_size);
  host_topKs_.resize(max_batch_size);
  host_topPs_.resize(max_batch_size);
  host_temperatures_.resize(max_batch_size);
  host_output_tokens_.resize(max_batch_size);

  topk_sampling_ = new TopkSampling(max_batch_size, batch_scheduler_config.max_vocab_size, device_curandstates_);
#endif
}

Sampler::~Sampler() {
  // free device buffer of output tokens
  GetBlockManager()->SetDeviceId(rank_);
  if (topk_sampling_ != nullptr) {
    delete topk_sampling_;
  }
  if (device_buffer_block_id_ != -1) {
    GetBlockManager()->FreeContiguous(device_buffer_block_id_);
  }
}

void Sampler::ApplyRepetitionPenalty(float* logits, std::vector<int>* input_tokens, std::vector<int>* output_tokens,
                                     const int vocab_size, const float repetition_penalty, Stream& stream) {
#ifdef ENABLE_CUDA
  // inv_repetition_penalties_ is filled with 1.0f
  std::fill(inv_repetition_penalties_.begin(), inv_repetition_penalties_.end(), 1.0f);
  // If a token has appeared before, repetition_penalties is inv_repetition_penalty.
  float inv_repetition_penalty = 1.0f / repetition_penalty;
  for (int i = 0; i < input_tokens->size(); ++i) {
    inv_repetition_penalties_[input_tokens->at(i)] = inv_repetition_penalty;
  }
  for (int i = 0; i < output_tokens->size(); ++i) {
    inv_repetition_penalties_[output_tokens->at(i)] = inv_repetition_penalty;
  }
  // copy inv_repetition_penalties_ to device
  MemcpyAsync(device_inv_repetition_penalties_, inv_repetition_penalties_.data(), sizeof(float) * vocab_size,
              MEMCPY_HOST_TO_DEVICE, stream);
  // logits = mul(logits, device_inv_repetition_penalties_)
  Mul(logits, device_inv_repetition_penalties_, logits, vocab_size, rank_);
#endif
}

Status Sampler::Sampling(std::vector<SamplingRequest>& sampling_reqs, Stream& stream) {

  if (rank_ == 0) {
    bool use_arg_max = true;
    bool use_top_p = false;
    bool use_temperature = false;
    int req_index = 0;
    float* device_logits = nullptr;
    SamplingDevideParameter sampling_devide_parameter;
    sampling_devide_parameter.bs = sampling_reqs.size();

    for (auto& sampling_req : sampling_reqs) {
      const ModelConfig* model_config = sampling_req.model_config;
      const SamplingConfig* sampling_config = sampling_req.sampling_config;
      float* logits = sampling_req.logits_buf[rank_];
      if (device_logits == logits || device_logits == nullptr) {
        device_logits = logits;
        sampling_devide_parameter.vocab_size_padded = batch_schedule_config_.max_vocab_size;
      } else {
        return Status(RET_SEGMENT_FAULT, "sampling for different logits not implemented");
      }
      int offset = sampling_req.logits_offset;
      if (offset >= sampling_devide_parameter.bs) {
        return Status(RET_SEGMENT_FAULT, "sampling check sampling_req.logits_offset >= sampling_devide_parameter.bs");
      }
      host_offset_[req_index] = offset;
      if (sampling_config->beam_width == 1) {
        if (sampling_config->topk > 1024) {
          return Status(RET_INVALID_ARGUMENT, "topk > 1024.");
        }
        host_topKs_[offset] = sampling_config->topk;
        host_topPs_[offset] = sampling_config->topp == 0.0f ? 1.0f : sampling_config->topp;
        host_temperatures_[offset] = sampling_config->temperature == 0.0f ? 1.0f : sampling_config->temperature;
        sampling_devide_parameter.max_topK = sampling_devide_parameter.max_topK > sampling_config->topk
                                                 ? sampling_devide_parameter.max_topK
                                                 : sampling_config->topk;
        use_arg_max = use_arg_max && sampling_config->topk == 1;
        use_top_p = use_top_p || !(host_topPs_[offset] == 1.0f);
        use_temperature = use_temperature || !(host_temperatures_[offset] == 1.0f);

      } else {
        return Status(RET_INVALID_ARGUMENT, "sampling for beam_width > 1 not implemented");
      }
      if (sampling_config->repetition_penalty != 1.0f) {
        int vocab_size = batch_schedule_config_.max_vocab_size;
        ApplyRepetitionPenalty(logits + req_index * vocab_size, sampling_req.input_tokens, sampling_req.output_tokens,
                               vocab_size, sampling_config->repetition_penalty, stream);
      }
      req_index++;
    }

    if (!use_arg_max) {
      MemcpyAsync(device_topKs_, host_topKs_.data(), sizeof(int) * sampling_devide_parameter.bs, MEMCPY_HOST_TO_DEVICE,
                  stream);
      sampling_devide_parameter.device_topKs = device_topKs_;
      sampling_devide_parameter.device_output_tokens_ptrs = device_output_tokens_ptrs_;
#ifdef ENABLE_CUDA
      sampling_devide_parameter.device_curandstates = device_curandstates_;
#endif
      if (use_top_p) {
        MemcpyAsync(device_topPs_, host_topPs_.data(), sizeof(float) * sampling_devide_parameter.bs,
                    MEMCPY_HOST_TO_DEVICE, stream);
        sampling_devide_parameter.device_topPs = device_topPs_;
      }
      if (use_temperature) {
        MemcpyAsync(device_temperatures_, host_temperatures_.data(), sizeof(float) * sampling_devide_parameter.bs,
                    MEMCPY_HOST_TO_DEVICE, stream);
        sampling_devide_parameter.device_temperatures = device_temperatures_;
      }
    }
#ifdef ENABLE_CUDA
    STATUS_CHECK_RETURN(topk_sampling_->Forward(device_logits, nullptr, device_output_tokens_, nullptr,
                                                sampling_devide_parameter, nullptr, stream));
#endif
    MemcpyAsync(host_output_tokens_.data(), device_output_tokens_, sizeof(uint32_t) * sampling_devide_parameter.bs,
                MEMCPY_DEVICE_TO_HOST, stream);
    StreamSynchronize(stream);
    for (int i = 0; i < sampling_devide_parameter.bs; i++) {
      sampling_reqs[i].output_tokens->push_back(host_output_tokens_[host_offset_[i]]);
    }
  }

  return Status();
}

}  // namespace ksana_llm
