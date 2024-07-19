/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#ifdef ENABLE_CUDA
#  include <curand_kernel.h>
#  include "3rdparty/LLM_kernels/csrc/kernels/nvidia/samplers/copy_elements.cuh"
#  include "ksana_llm/kernels/nvidia/kernel_wrapper.h"
#endif

#include "ksana_llm/samplers/sampler.h"
#include "ksana_llm/utils/common_device.h"
#include "ksana_llm/utils/device_utils.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/memory_utils.h"

namespace ksana_llm {

static size_t kCudaMemAlignmentSize = alignof(std::max_align_t);

Sampler::Sampler(const BatchSchedulerConfig& batch_scheduler_config, int rank, std::shared_ptr<Context> context)
    : beam_search_sampling_(context) {
  batch_schedule_config_ = batch_scheduler_config;
  context_ = context;
  rank_ = rank;
  auto max_batch_size = batch_scheduler_config.max_batch_size;
  // need to allocate device buffer for sampling
  GetBlockManager()->SetDeviceId(rank_);
  auto allocator = [this](size_t size) -> void* {
    GetBlockManager()->SetDeviceId(rank_);
    GetBlockManager()->AllocateContiguous(size, device_buffer_block_id_);
    GetBlockManager()->GetContiguousPtr(device_buffer_block_id_, device_buffer_);
    return device_buffer_;
  };
  AlignedMemoryQueue aligned_memory_queue(kCudaMemAlignmentSize, allocator);
  aligned_memory_queue.Add(device_output_tokens_, max_batch_size);
  aligned_memory_queue.Add(device_offset_, max_batch_size);
  aligned_memory_queue.Add(device_topKs_, max_batch_size);
  aligned_memory_queue.Add(device_topPs_, max_batch_size);
  aligned_memory_queue.Add(device_temperatures_, max_batch_size);
  aligned_memory_queue.Add(device_curandstates_, max_batch_size);
  aligned_memory_queue.Add(device_output_tokens_ptrs_, max_batch_size);
  aligned_memory_queue.Add(device_inv_repetition_penalties_, batch_schedule_config_.max_vocab_size);
  aligned_memory_queue.Add(device_prob_, max_batch_size);
  aligned_memory_queue.Add(device_prob_ptrs_, max_batch_size);
  aligned_memory_queue.AllocateAndAlign();

  inv_repetition_penalties_.resize(batch_schedule_config_.max_vocab_size);

  if (sizeof(uint32_t) != sizeof(int)) {
    KLLM_LOG_ERROR << fmt::format("sizeof(uint32_t)({}) != sizeof(int)({})", sizeof(uint32_t), sizeof(int));
    abort();
    exit(RetCode::RET_SEGMENT_FAULT);
  }
  std::vector<uint32_t*> host_device_output_tokens_ptrs(max_batch_size);
  for (size_t i = 0; i < max_batch_size; i++) {
    host_device_output_tokens_ptrs[i] = device_output_tokens_ + i;
  }

  MemcpyAsync(device_output_tokens_ptrs_, host_device_output_tokens_ptrs.data(), sizeof(uint32_t*) * max_batch_size,
              MEMCPY_HOST_TO_DEVICE, context_->GetH2DStreams()[rank_]);

  host_offset_.resize(max_batch_size);
  host_topKs_.resize(max_batch_size);
  host_topPs_.resize(max_batch_size);
  host_temperatures_.resize(max_batch_size);
  host_output_tokens_.resize(max_batch_size);

  topk_sampling_ = new TopkSampling(max_batch_size, batch_scheduler_config.max_vocab_size, device_curandstates_);
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
  // inv_repetition_penalties_ is filled with 1.0f
  std::fill(inv_repetition_penalties_.begin(), inv_repetition_penalties_.end(), 1.0f);
  // If a token has appeared before, repetition_penalties is inv_repetition_penalty.
  float inv_repetition_penalty = 1.0f / repetition_penalty;
  for (size_t i = 0; i < input_tokens->size(); ++i) {
    inv_repetition_penalties_[input_tokens->at(i)] = inv_repetition_penalty;
  }
  for (size_t i = 0; i < output_tokens->size(); ++i) {
    inv_repetition_penalties_[output_tokens->at(i)] = inv_repetition_penalty;
  }
  // copy inv_repetition_penalties_ to device
  MemcpyAsync(device_inv_repetition_penalties_, inv_repetition_penalties_.data(), sizeof(float) * vocab_size,
              MEMCPY_HOST_TO_DEVICE, stream);
  // logits = mul(logits, device_inv_repetition_penalties_)
#ifdef ENABLE_CUDA
  Mul(logits, device_inv_repetition_penalties_, logits, vocab_size, rank_);
#endif
}

Status Sampler::SamplingAndCalcLogprobs(std::vector<SamplingRequest>& sampling_reqs, float* device_logits,
                                        SamplingDevideParameter& sampling_devide_parameter, Stream& stream) {
  for (auto& sampling_req : sampling_reqs) {
    auto& logprobs_num = sampling_req.sampling_config->logprobs_num;
    auto& offset = sampling_req.logits_offset;
    auto& vocab_size = sampling_devide_parameter.vocab_size_padded;

    if (logprobs_num == 0) {
      std::unique_lock<std::mutex> lock(*sampling_req.output_mutex);
      sampling_req.logprobs->push_back({});
      continue;
    }
    std::vector<float> logprobs(logprobs_num);
    std::vector<int64_t> token_ids(logprobs_num);
#ifdef ENABLE_CUDA
    CalcLogprobs(device_logits + (offset * vocab_size), sampling_devide_parameter.device_temperatures + offset,
                 vocab_size, 1, sampling_req.sampling_config->logprobs_num, logprobs.data(), token_ids.data());
#endif
    std::vector<std::pair<int, float>> logprobs_output;
    for (int logprobs_index = 0; logprobs_index < sampling_req.sampling_config->logprobs_num; logprobs_index++) {
      logprobs_output.push_back({token_ids[logprobs_index], logprobs[logprobs_index]});
    }
    std::unique_lock<std::mutex> lock(*sampling_req.output_mutex);
    sampling_req.logprobs->emplace_back(logprobs_output);
  }
  return Status();
}

// Copies the probabilities from the logits buffer to the output vector for each sampling request.
std::function<void()> Sampler::CopyProbsOutput(std::vector<SamplingRequest>& sampling_reqs, Stream& stream,
                                               std::vector<std::vector<float>>& probs_output) {
  // Vectors to hold source and destination pointers for copying.
  std::vector<float*> src_ptr_vector;
  std::vector<float*> dst_ptr_vector;
  for (size_t i = 0; i < sampling_reqs.size(); i++) {
    if (sampling_reqs[i].logits_custom_length > 0) {
      probs_output[i].resize(sampling_reqs[i].logits_custom_length);
      auto& input_tokens = *sampling_reqs[i].input_tokens;
      auto& vocab_size = batch_schedule_config_.max_vocab_size;
      size_t probs_index = 0;
      for (auto [l, r] : sampling_reqs[i].request_target->at("logits").slice_pos) {
        for (auto index = l; index <= r; index++) {
          size_t req_logits_offset = (sampling_reqs[i].logits_offset + probs_index) * vocab_size;
          // Add destination and source pointers for copying.
          dst_ptr_vector.push_back(probs_output[i].data() + probs_index);
          src_ptr_vector.push_back(sampling_reqs[i].logits_buf[rank_] + req_logits_offset + input_tokens[index + 1]);
          probs_index++;
        }
      }
    }
  }
  std::vector<float> dst_vector(src_ptr_vector.size());
#ifdef ENABLE_CUDA
  // Copy source pointers to device memory asynchronously.
  MemcpyAsync(device_prob_ptrs_, src_ptr_vector.data(), sizeof(float*) * src_ptr_vector.size(), MEMCPY_HOST_TO_DEVICE,
              stream);
  // Invoke kernel to copy elements from source to a temporary device buffer.
  llm_kernels::nvidia::InvokeCopyElements(device_prob_ptrs_, device_prob_, src_ptr_vector.size(), stream.Get());
  // Copy the temporary device buffer to host memory asynchronously.
  MemcpyAsync(dst_vector.data(), device_prob_, sizeof(float) * src_ptr_vector.size(), MEMCPY_DEVICE_TO_HOST, stream);
#endif
  return [dst_vector = std::move(dst_vector), dst_ptr_vector = std::move(dst_ptr_vector)]() mutable {
    for (size_t i = 0; i < dst_ptr_vector.size(); i++) {
      *dst_ptr_vector[i] = dst_vector[i];
    }
  };
}

// Transfer sampling parameters to the device
void Sampler::SamplingParameterToDevide(bool use_top_p, bool use_temperature, bool logits_softmax,
                                        SamplingDevideParameter& sampling_devide_parameter, Stream& stream) {
  MemcpyAsync(device_topKs_, host_topKs_.data(), sizeof(int) * sampling_devide_parameter.bs, MEMCPY_HOST_TO_DEVICE,
              stream);
  sampling_devide_parameter.device_topKs = device_topKs_;
  sampling_devide_parameter.device_output_tokens_ptrs = device_output_tokens_ptrs_;
  sampling_devide_parameter.device_curandstates = device_curandstates_;
  if (use_top_p) {
    MemcpyAsync(device_topPs_, host_topPs_.data(), sizeof(float) * sampling_devide_parameter.bs, MEMCPY_HOST_TO_DEVICE,
                stream);
    sampling_devide_parameter.device_topPs = device_topPs_;
  }
  if (use_temperature || logits_softmax) {
    MemcpyAsync(device_temperatures_, host_temperatures_.data(), sizeof(float) * sampling_devide_parameter.bs,
                MEMCPY_HOST_TO_DEVICE, stream);
    sampling_devide_parameter.device_temperatures = device_temperatures_;
  }
}

Status Sampler::Sampling(std::vector<SamplingRequest>& sampling_reqs, Stream& stream) {
  if (rank_ == 0) {
    bool use_arg_max = true;
    bool use_top_p = false;
    bool use_temperature = false;
    int req_index = 0;
    float* device_logits = nullptr;
    SamplingDevideParameter sampling_devide_parameter;
    sampling_devide_parameter.bs = 0;
    // If true, softmax must be done.
    bool logits_softmax = false;
    for (auto& sampling_req : sampling_reqs) {
      const SamplingConfig* sampling_config = sampling_req.sampling_config;
      logits_softmax =
          logits_softmax || sampling_req.logits_custom_length > 0 || sampling_req.sampling_config->logprobs_num > 0;
      // If logits_custom_length is used, there are logits_custom_length logits that need to be calculated.
      int sampling_bs = (sampling_req.logits_custom_length > 0 ? sampling_req.logits_custom_length : 1);
      sampling_devide_parameter.bs += sampling_bs;
      float* logits = sampling_req.logits_buf[rank_];
      if (device_logits == logits || device_logits == nullptr) {
        device_logits = logits;
        sampling_devide_parameter.vocab_size_padded = batch_schedule_config_.max_vocab_size;
      } else {
        return Status(RET_SEGMENT_FAULT, "sampling for different logits not implemented");
      }
      size_t offset = sampling_req.logits_offset;
      if (offset >= batch_schedule_config_.max_batch_size) {
        return Status(RET_SEGMENT_FAULT, "sampling check sampling_req.logits_offset >= max_batch_size");
      }
      host_offset_[req_index] = offset;
      if (sampling_config->topk > 1024) {
        return Status(RET_INVALID_ARGUMENT, "topk > 1024.");
      }
      for (int sampling_index = 0; sampling_index < sampling_bs; sampling_index++) {
        host_topKs_[offset + sampling_index] = sampling_config->topk;
        host_topPs_[offset + sampling_index] = sampling_config->topp == 0.0f ? 1.0f : sampling_config->topp;
        host_temperatures_[offset + sampling_index] =
            sampling_config->temperature == 0.0f ? 1.0f : sampling_config->temperature;
      }
      if (sampling_devide_parameter.max_topK < sampling_config->topk) {
        sampling_devide_parameter.max_topK = sampling_config->topk;
      }
      use_arg_max = use_arg_max && sampling_config->topk == 1;
      use_top_p = use_top_p || !(host_topPs_[offset] == 1.0f);
      use_temperature = use_temperature || !(host_temperatures_[offset] == 1.0f);

      if (sampling_config->repetition_penalty != 1.0f) {
        int vocab_size = batch_schedule_config_.max_vocab_size;
        ApplyRepetitionPenalty(logits + req_index * vocab_size, sampling_req.input_tokens, sampling_req.output_tokens,
                               vocab_size, sampling_config->repetition_penalty, stream);
      }
      req_index++;
    }
    if (!use_arg_max || logits_softmax) {
      SamplingParameterToDevide(use_top_p, use_temperature, logits_softmax, sampling_devide_parameter, stream);
    }
    SamplingAndCalcLogprobs(sampling_reqs, device_logits, sampling_devide_parameter, stream);
    STATUS_CHECK_RETURN(topk_sampling_->Forward(device_logits, nullptr, device_output_tokens_, nullptr,
                                                sampling_devide_parameter, nullptr, stream));
    MemcpyAsync(host_output_tokens_.data(), device_output_tokens_, sizeof(uint32_t) * sampling_devide_parameter.bs,
                MEMCPY_DEVICE_TO_HOST, stream);
    std::vector<std::vector<float>> probs_output(sampling_reqs.size());
    auto copy_probs_after_synchronize = CopyProbsOutput(sampling_reqs, stream, probs_output);
    StreamSynchronize(stream);
    copy_probs_after_synchronize();
    for (size_t i = 0; i < sampling_reqs.size(); i++) {
      std::unique_lock<std::mutex> lock(*sampling_reqs[i].output_mutex);
      sampling_reqs[i].output_tokens->push_back(host_output_tokens_[host_offset_[i]]);
      beam_search_sampling_.Sampling(sampling_reqs[i]);
      if (!probs_output[i].empty()) {
        PythonTensor& ret_tensor = (*sampling_reqs[i].response)["logits"];
        ret_tensor.shape = {probs_output[i].size()};
        ret_tensor.dtype = GetTypeString(TYPE_FP32);
        ret_tensor.data.resize(probs_output[i].size() * sizeof(float));
        memcpy(ret_tensor.data.data(), probs_output[i].data(), ret_tensor.data.size());
      }
    }
  }
  return Status();
}

}  // namespace ksana_llm
