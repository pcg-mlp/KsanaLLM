/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#ifdef ENABLE_CUDA
#  include <curand_kernel.h>
#  include <nccl.h>
#  include "3rdparty/LLM_kernels/csrc/kernels/nvidia/samplers/copy_elements.cuh"
#  include "3rdparty/LLM_kernels/csrc/kernels/nvidia/samplers/decoding_common.h"
#  include "3rdparty/LLM_kernels/csrc/kernels/nvidia/samplers/repetition_penalty.h"
#  include "device/nvidia/nccl_utils.h"
#  include "kernels/nvidia/kernel_wrapper.h"
#  include "layers/nvidia/replace_last_token.h"
#endif

#include "device/common_device.h"
#include "device/device_utils.h"
#include "profiler/profile_event.h"
#include "runtime/fast_path_controller.h"
#include "samplers/sampler.h"
#include "utils/logger.h"
#include "utils/memory_utils.h"

namespace ksana_llm {

static size_t kCudaMemAlignmentSize = alignof(std::max_align_t);
static constexpr float kBitsPerInt = 32.0f;  // Number of bits in an integer for bitmask calculations

Sampler::Sampler(const BatchSchedulerConfig& batch_scheduler_config, const int rank, std::shared_ptr<Context> context)
    : batch_schedule_config_(batch_scheduler_config), rank_(rank), context_(context) {
  KLLM_CHECK_WITH_INFO(sizeof(uint32_t) == sizeof(int),
                       fmt::format("sizeof(uint32_t)({}) != sizeof(int)({})", sizeof(uint32_t), sizeof(int)));

  // need to allocate device buffer for sampling
  const size_t max_logits_num =
      batch_scheduler_config.max_batch_size * batch_schedule_config_.max_decode_tokens_per_req;
  AlignedMemoryQueue aligned_memory_queue(kCudaMemAlignmentSize, [this](const size_t size) {
    Malloc(&device_buffer_, size);
    return device_buffer_;
  });
  aligned_memory_queue.Add(device_output_tokens_, max_logits_num);
  aligned_memory_queue.Add(device_topKs_, max_logits_num);
  aligned_memory_queue.Add(device_topPs_, max_logits_num);
  aligned_memory_queue.Add(device_temperatures_, max_logits_num);
  aligned_memory_queue.Add(device_curandstates_, max_logits_num);
  aligned_memory_queue.Add(device_output_tokens_ptrs_, max_logits_num);
  aligned_memory_queue.Add(device_repetition_processor_, batch_schedule_config_.max_vocab_size);
  aligned_memory_queue.Add(device_prob_, max_logits_num);
  aligned_memory_queue.Add(device_prob_ptrs_, max_logits_num);

  // vocab mask buffer
  // TODO(ethanyczeng): if(!grammar_backend_)
  if (batch_schedule_config_.enable_xgrammar) {
    const int bitmask_elements = static_cast<int>(std::ceil(batch_schedule_config_.max_vocab_size / kBitsPerInt));
    const size_t vocab_mask_elements = batch_schedule_config_.max_batch_size * bitmask_elements;
    aligned_memory_queue.Add(device_vocab_mask_, vocab_mask_elements);
    host_vocab_mask_.resize(vocab_mask_elements);
    KLLM_LOG_INFO << "Grammar vocab mask buffers initialized successfully with " << vocab_mask_elements
                  << " elements (batch_size=" << batch_schedule_config_.max_batch_size
                  << ", bitmask_elements=" << bitmask_elements << ")";
  }

  aligned_memory_queue.AllocateAndAlign();

  repetition_penalties_.resize(batch_schedule_config_.max_vocab_size);
  norepeat_ngrams_.resize(batch_schedule_config_.max_vocab_size);

  std::vector<uint32_t*> output_tokens_ptrs_host(max_logits_num);
  iota(output_tokens_ptrs_host.begin(), output_tokens_ptrs_host.end(), device_output_tokens_);
  MemcpyAsync(device_output_tokens_ptrs_, output_tokens_ptrs_host.data(),
              sizeof(decltype(output_tokens_ptrs_host)::value_type) * output_tokens_ptrs_host.size(),
              MEMCPY_HOST_TO_DEVICE, context_->device->GetH2DStream());

  host_topKs_.resize(max_logits_num);
  host_topPs_.resize(max_logits_num);
  host_temperatures_.resize(max_logits_num);
  host_output_tokens_.resize(max_logits_num);

  topk_sampling_ = new TopkSampling(max_logits_num, batch_scheduler_config.max_vocab_size, device_curandstates_);

  // build-ahead slot 1 buffer：仅 FastPathController 启动期 gate 通过时预分配。
  if (FastPathController::GetInstance().IsEnabledAtStartup()) {
    AlignedMemoryQueue slot1_queue(kCudaMemAlignmentSize, [this](const size_t size) {
      Malloc(&device_buffer_slot1_, size);
      return device_buffer_slot1_;
    });
    slot1_queue.Add(device_output_tokens_slot1_, max_logits_num);
    slot1_queue.Add(device_output_tokens_ptrs_slot1_, max_logits_num);
    slot1_queue.AllocateAndAlign();

    // Initialize slot1 device_output_tokens_ptrs (each ptr -> &device_output_tokens_slot1_[i]).
    std::vector<uint32_t*> slot1_ptrs_host(max_logits_num);
    iota(slot1_ptrs_host.begin(), slot1_ptrs_host.end(), device_output_tokens_slot1_);
    MemcpyAsync(device_output_tokens_ptrs_slot1_, slot1_ptrs_host.data(),
                sizeof(decltype(slot1_ptrs_host)::value_type) * slot1_ptrs_host.size(), MEMCPY_HOST_TO_DEVICE,
                context_->device->GetH2DStream());

    host_output_tokens_slot1_.resize(max_logits_num);
    KLLM_LOG_INFO << "Sampler rank=" << rank_ << " allocated slot1 buffer (fast path enabled).";

    // 分配双 slot 配对 ring buffer（int64）：[count, (req_id, token) * max_batch_size]。
    // 初始化 count=0；首步即使误触 ReplaceKernel 也循环零次，placeholder 不变。
    device_ring_capacity_ = 1 + 2 * batch_schedule_config_.max_batch_size;
    const size_t ring_bytes = device_ring_capacity_ * sizeof(int64_t);
    Malloc(reinterpret_cast<void**>(&device_ring_slot0_), ring_bytes);
    Malloc(reinterpret_cast<void**>(&device_ring_slot1_), ring_bytes);
    MemsetAsync(device_ring_slot0_, 0, ring_bytes, context_->device->GetH2DStream());
    MemsetAsync(device_ring_slot1_, 0, ring_bytes, context_->device->GetH2DStream());
    KLLM_LOG_INFO << "Sampler rank=" << rank_ << " allocated fast-path pair ring (cap=" << device_ring_capacity_
                  << " int64 each slot).";
  }
}

Sampler::~Sampler() {
  // free device buffer of output tokens
  SetDevice(rank_);
  if (topk_sampling_ != nullptr) {
    delete topk_sampling_;
  }
  if (device_buffer_ != nullptr) {
    Free(device_buffer_);
    device_buffer_ = nullptr;
  }
  // 释放 build-ahead slot 1 buffer（仅总开关打开时分配）。
  if (device_buffer_slot1_ != nullptr) {
    Free(device_buffer_slot1_);
    device_buffer_slot1_ = nullptr;
  }
  // 释放 build-ahead 配对 ring buffer。
  if (device_ring_slot0_ != nullptr) {
    Free(device_ring_slot0_);
    device_ring_slot0_ = nullptr;
  }
  if (device_ring_slot1_ != nullptr) {
    Free(device_ring_slot1_);
    device_ring_slot1_ = nullptr;
  }
}

void Sampler::ApplyRepetitionPenalty(float* logits, std::vector<int>* input_tokens, std::vector<int>* forwarding_tokens,
                                     const int vocab_size, const float repetition_penalty, Stream& stream) {
  // repetition_penalties_ is filled with 1.0f
  std::fill(repetition_penalties_.begin(), repetition_penalties_.end(), 1.0f);
  // If a token has appeared before, repetition_penalty is applied.
  const int input_tokens_size = input_tokens->size();
  repetition_penalties_[input_tokens->back()] = repetition_penalty;

  if (forwarding_tokens->size() > input_tokens_size) {
    for (size_t i = input_tokens_size; i < forwarding_tokens->size(); ++i) {
      repetition_penalties_[forwarding_tokens->at(i)] = repetition_penalty;
    }
  }

  // copy repetition_penalties_ to device
  MemcpyAsync(device_repetition_processor_, repetition_penalties_.data(), sizeof(float) * vocab_size,
              MEMCPY_HOST_TO_DEVICE, stream);
#ifdef ENABLE_CUDA
  llm_kernels::nvidia::InvokeRepetitionPenalty<float>(logits, device_repetition_processor_, logits, vocab_size,
                                                      stream.Get());
#endif
}

void Sampler::GetNgrams(const int ngram_size, const int cur_output_size, const std::vector<int>* output_tokens,
                        NgramDict* ngram_dict) {
  if (!ngram_dict->empty()) {
    return;
  }

  std::vector<std::vector<int>> ngrams;
  // for tokens recompute
  for (int i = 0; i <= cur_output_size - ngram_size; ++i) {
    std::vector<int> sub_ngram(output_tokens->begin() + i, output_tokens->begin() + i + ngram_size);
    ngrams.push_back(sub_ngram);
  }

  for (const auto& ngram : ngrams) {
    std::vector<int> ngram_excluding_last(ngram.begin(), ngram.end() - 1);
    int last_elem = ngram.back();
    (*ngram_dict)[ngram_excluding_last].push_back(last_elem);
  }
}

void Sampler::BanRepeatTokens(float* logits, const int ngram_size, const int cur_output_size,
                              const std::vector<int>* output_tokens, NgramDict* ngram_dict, const int vocab_size,
                              Stream& stream) {
  std::vector<int> repeat_ids;
  int start_idx = cur_output_size - ngram_size + 1;
  std::vector<int> ngram_idx(output_tokens->begin() + start_idx, output_tokens->begin() + cur_output_size);
  if (ngram_dict->find(ngram_idx) != ngram_dict->end()) {
    repeat_ids = (*ngram_dict)[ngram_idx];
  } else {
    repeat_ids = {};
  }

  if (repeat_ids.size() > 0) {
    std::fill(norepeat_ngrams_.begin(), norepeat_ngrams_.end(), 0.0f);
    for (size_t i = 0; i < repeat_ids.size(); ++i) {
      norepeat_ngrams_[repeat_ids[i]] = -std::numeric_limits<float>::infinity();
    }
    MemcpyAsync(device_repetition_processor_, norepeat_ngrams_.data(), sizeof(float) * vocab_size,
                MEMCPY_HOST_TO_DEVICE, stream);
#ifdef ENABLE_CUDA
    InvokeAddBiasResidual<float>(logits, device_repetition_processor_, nullptr, 1, vocab_size, logits, stream.Get());
#endif
  }
}

void Sampler::NoRepeatNgramProcessor(float* logits, const int ngram_size, const int input_tokens_size,
                                     const std::vector<int>* output_tokens, NgramDict* ngram_dict, const int vocab_size,
                                     size_t last_step_token_num, Stream& stream) {
  int cur_output_size = output_tokens->size();
  if (ngram_size > cur_output_size) {
    KLLM_LOG_WARNING << fmt::format(
        "The no_repeat_ngram_size must be less than the number of tokens output by the Forward. {} < {}", ngram_size,
        cur_output_size);
    return;
  }
  if (input_tokens_size == cur_output_size) {
    KLLM_LOG_DEBUG << "for input and output tokens no repeat ngram sample";
    // TODO(winminkong): consider the computational approach for ngrams with re-computation.
    GetNgrams(ngram_size, cur_output_size, output_tokens, ngram_dict);
  } else if (input_tokens_size < cur_output_size) {
    for (size_t i = 0; i < last_step_token_num; ++i) {  // For MTP
      std::vector<int> sub_ngram(output_tokens->end() - ngram_size - i, output_tokens->end() - i);
      std::vector<int> ngram_excluding_last(sub_ngram.begin(), sub_ngram.end() - 1);
      int last_elem = sub_ngram.back();
      (*ngram_dict)[ngram_excluding_last].push_back(last_elem);
    }
  }
  BanRepeatTokens(logits, ngram_size, cur_output_size, output_tokens, ngram_dict, vocab_size, stream);
}

void Sampler::EncoderNoRepeatNgramProcessor(float* logits, const int ngram_size, const int input_tokens_size,
                                            const std::vector<int>* output_tokens, NgramDict* ngram_dict,
                                            const int vocab_size, Stream& stream) {
  int cur_output_size = output_tokens->size();
  if (ngram_size > cur_output_size) {
    KLLM_LOG_WARNING << fmt::format(
        "The encoder_no_repeat_ngram_size must be less than the number of tokens output by the Forward. {} < {}",
        ngram_size, cur_output_size);
    return;
  }
  if (input_tokens_size == cur_output_size) {
    KLLM_LOG_DEBUG << "for input tokens no repeat ngram sample";
    GetNgrams(ngram_size, cur_output_size, output_tokens, ngram_dict);
  }
  BanRepeatTokens(logits, ngram_size, cur_output_size, output_tokens, ngram_dict, vocab_size, stream);
}

void Sampler::DecoderNoRepeatNgramProcessor(float* logits, const int ngram_size, const int input_tokens_size,
                                            const std::vector<int>* output_tokens, NgramDict* ngram_dict,
                                            const int vocab_size, size_t last_step_token_num, Stream& stream) {
  int cur_output_size = output_tokens->size();
  if (ngram_size > cur_output_size - input_tokens_size) {
    KLLM_LOG_WARNING << fmt::format(
        "The decoder_no_repeat_ngram_size must be less than the number of tokens output by the Forward. {} < {}",
        ngram_size, cur_output_size - input_tokens_size);
    return;
  } else {
    for (size_t i = 0; i < last_step_token_num; ++i) {  // For MTP
      std::vector<int> sub_ngram(output_tokens->end() - ngram_size - i, output_tokens->end() - i);
      std::vector<int> ngram_excluding_last(sub_ngram.begin(), sub_ngram.end() - 1);
      int last_elem = sub_ngram.back();
      (*ngram_dict)[ngram_excluding_last].push_back(last_elem);
    }
  }
  BanRepeatTokens(logits, ngram_size, cur_output_size, output_tokens, ngram_dict, vocab_size, stream);
}

void Sampler::WriteRing(size_t slot, const int64_t* cur_pairs, int batch_size, Stream& stream) {
  // 调用方 (ModelRunner::WriteBuildAheadRingAfterSampling) 仅在 build-ahead 启用时进入此函数。
  // ring slot / device_output_tokens_slot1_ 必须已在构造函数按相同条件分配——任一缺失即配置不一致, hard fail.
  // batch_size==0 是合法情况 (kernel 内 count=0, 仅写头部置零), 不视作错误.
  int64_t* ring = GetRingSlot(slot);
  KLLM_CHECK_WITH_INFO(
      ring != nullptr,
      fmt::format("Sampler::WriteRing slot {} ring not allocated (build_ahead config mismatch).", slot));
  KLLM_CHECK_WITH_INFO(batch_size >= 0, fmt::format("Sampler::WriteRing got negative batch_size={}.", batch_size));
  KLLM_CHECK_WITH_INFO(static_cast<size_t>(1 + 2 * batch_size) <= device_ring_capacity_,
                       fmt::format("Sampler::WriteRing batch_size={} exceeds ring capacity={} (max_batch_size).",
                                   batch_size, device_ring_capacity_));
#ifdef ENABLE_CUDA
  // sampling 输出在 device_output_tokens_<slot>; ring 与 sampling 用相同 slot, 写入即固化本步 (req_id, token).
  uint32_t* sampled_tokens = GetDeviceOutputTokens(slot);
  KLLM_CHECK_WITH_INFO(sampled_tokens != nullptr,
                       fmt::format("Sampler::WriteRing slot {} output buffer is null.", slot));
  LaunchWriteRingKernel(ring, cur_pairs, sampled_tokens, batch_size, stream.Get());
#endif
}

void Sampler::BroadcastRingSlot(size_t slot, Stream& stream) {
  if (device_ring_capacity_ == 0) {
    return;
  }
  const size_t tp_size = context_->global->GetTensorParallelSize();
  if (tp_size <= 1) {
    return;
  }
#ifdef ENABLE_CUDA
  int64_t* ring = GetRingSlot(slot);
  KLLM_CHECK_WITH_INFO(
      ring != nullptr,
      fmt::format("Sampler::BroadcastRingSlot slot {} ring not allocated (build_ahead config mismatch).", slot));
  // 固定广播整槽容量，避免按 B_cur 变长；rank 0 为 root（与 SelectTopkTokens / NCCL comm 一致）。
  NCCL_CHECK(ncclBroadcast(ring, ring, device_ring_capacity_, ncclInt64, /*root=*/0,
                           context_->device->ext->GetNCCLParam().nccl_comm, stream.Get()));
#endif
}

void Sampler::CopyProbsOutputToRequests(std::vector<SamplingRequest>& sampling_reqs,
                                        std::vector<std::vector<float>>& probs_output, Stream& stream, size_t slot) {
  PROFILE_EVENT_SCOPE(sampler_copy_probs_total, "Exec.Sampler.CopyProbsOutputToRequests.Total");
  auto copy_probs_after_synchronize = CopyProbsOutput(sampling_reqs, stream, probs_output);
  {
    PROFILE_EVENT_SCOPE(sampler_stream_sync, "Exec.Sampler.CopyProbsOutputToRequests.StreamSync");
    StreamSynchronize(stream);
  }
  copy_probs_after_synchronize();
  // 根据 slot 选用对应的 host_output_tokens (slot 1 未分配时回落 slot 0).
  std::vector<int>& host_out_tokens = GetHostOutputTokens(slot);
  for (size_t i = 0; i < sampling_reqs.size(); i++) {
    auto& req = sampling_reqs[i];
    req.sampling_result_tokens->insert(req.sampling_result_tokens->end(), host_out_tokens.begin() + req.logits_offset,
                                       host_out_tokens.begin() + req.logits_offset + req.sampling_token_num);
    if (req.request_target != nullptr) {
      auto it = req.request_target->find("logits");
      if (it != req.request_target->end()) {
        if (it->second.token_reduce_mode == TokenReduceMode::GATHER_ALL) {
          continue;
        }
      }
    }
    if (!probs_output[i].empty()) {
      PythonTensor& ret_tensor = (*req.response)["logits"];
      ret_tensor.shape = {probs_output[i].size()};
      ret_tensor.dtype = GetTypeString(TYPE_FP32);
      ret_tensor.data.resize(probs_output[i].size() * sizeof(float));
      memcpy(ret_tensor.data.data(), probs_output[i].data(), ret_tensor.data.size());
    }
  }
}

Status Sampler::SamplingAndCalcLogprobs(std::vector<SamplingRequest>& sampling_reqs, float* device_logits,
                                        SamplingDeviceParameter& sampling_device_parameter, Stream& stream,
                                        size_t /*slot*/) {
  // 当前实现内部并未直接读写 device_output_tokens, 故 slot 仅用于接口对齐 (PrepareDeviceLogitsAndParameter
  // 已通过 sampling_device_parameter.device_output_tokens_ptrs 间接选定 slot 的 buffer).
  if (rank_ != 0) {
    return Status();
  }
  for (auto& sampling_req : sampling_reqs) {
    if (sampling_req.enable_mtp_sampler) {
      // Do not calculate output token logprobs when sample req is mtp req.
      continue;
    }

    auto& logprobs_num = sampling_req.sampling_config->logprobs_num;
    if (logprobs_num == 0) {
      sampling_req.logprobs->emplace_back();
      continue;
    }

    int universal_logprobs_num = logprobs_num > kMinLogprobsNum ? logprobs_num : kMinLogprobsNum;
    std::vector<float> logprobs(universal_logprobs_num);
    std::vector<int64_t> token_ids(universal_logprobs_num);
#ifdef ENABLE_CUDA
    auto& offset = sampling_req.logits_offset;
    auto& vocab_size = sampling_device_parameter.vocab_size_padded;
    float* device_temperatures_ptr = sampling_device_parameter.device_temperatures == nullptr
                                         ? nullptr
                                         : sampling_device_parameter.device_temperatures + offset;
    for (size_t sampling_index = 0; sampling_index < sampling_req.sampling_token_num; sampling_index++) {
      CalcLogprobs(device_logits + (offset + sampling_index) * vocab_size, device_temperatures_ptr, vocab_size, 1,
                   universal_logprobs_num, logprobs.data(), token_ids.data());
#endif
      std::vector<std::pair<int, float>> logprobs_output;
      for (int logprobs_index = 0; logprobs_index < universal_logprobs_num; logprobs_index++) {
        logprobs_output.push_back({token_ids[logprobs_index], logprobs[logprobs_index]});
      }
      sampling_req.logprobs->emplace_back(logprobs_output);
#ifdef ENABLE_CUDA
    }
#endif
  }
  return Status();
}

// Co#endifpies the probabilities from the logits buffer to the output vector for each sampling request.
std::function<void()> Sampler::CopyProbsOutput(std::vector<SamplingRequest>& sampling_reqs, Stream& stream,
                                               std::vector<std::vector<float>>& probs_output) {
  // Vectors to hold source and destination pointers for copying.
  std::vector<float*> src_ptr_vector;
  std::vector<float*> dst_ptr_vector;
  for (size_t i = 0; i < sampling_reqs.size(); i++) {
    if (sampling_reqs[i].logits_custom_length > 0) {
      if (sampling_reqs[i].request_target != nullptr) {
        auto it = sampling_reqs[i].request_target->find("logits");
        if (it != sampling_reqs[i].request_target->end()) {
          if (it->second.token_reduce_mode == TokenReduceMode::GATHER_ALL) {
            continue;
          }
        }
      }
      probs_output[i].resize(sampling_reqs[i].logits_custom_length);
      auto& input_tokens = *sampling_reqs[i].input_tokens;
      auto& vocab_size = batch_schedule_config_.max_vocab_size;
      size_t probs_index = 0;
      for (auto [l, r] : sampling_reqs[i].request_target->at("logits").slice_pos) {
        for (auto index = l; index <= r; index++) {
          size_t req_logits_offset = (sampling_reqs[i].logits_offset + probs_index) * vocab_size;
          // Add destination and source pointers for copying.
          dst_ptr_vector.push_back(probs_output[i].data() + probs_index);
          // For any part that exceeds the input token size, directly take the value of the zeroth position.
          size_t token_idx_offset = (index + 1) < input_tokens.size() ? input_tokens[index + 1] : 0;
          src_ptr_vector.push_back(sampling_reqs[i].logits_buf + req_logits_offset + token_idx_offset);
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
  CUDA_CHECK_LAST_ERROR(
      llm_kernels::nvidia::InvokeCopyElements(device_prob_ptrs_, device_prob_, src_ptr_vector.size(), stream.Get()));
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
void Sampler::SamplingParameterToDevice(bool use_top_k, bool use_top_p, bool use_temperature,
                                        SamplingDeviceParameter& sampling_device_parameter, Stream& stream,
                                        size_t slot) {
  if (use_top_k) {
    MemcpyAsync(device_topKs_, host_topKs_.data(), sizeof(int) * sampling_device_parameter.bs, MEMCPY_HOST_TO_DEVICE,
                stream);
    sampling_device_parameter.device_topKs = device_topKs_;
    // 根据 slot 选用 device_output_tokens_ptrs (slot 1 未分配时回落 slot 0).
    sampling_device_parameter.device_output_tokens_ptrs = GetDeviceOutputTokensPtrs(slot);
    sampling_device_parameter.device_curandstates = device_curandstates_;
  }
  if (use_top_p) {
    MemcpyAsync(device_topPs_, host_topPs_.data(), sizeof(float) * sampling_device_parameter.bs, MEMCPY_HOST_TO_DEVICE,
                stream);
    sampling_device_parameter.device_topPs = device_topPs_;
  }
  if (use_temperature) {
    MemcpyAsync(device_temperatures_, host_temperatures_.data(), sizeof(float) * sampling_device_parameter.bs,
                MEMCPY_HOST_TO_DEVICE, stream);
    sampling_device_parameter.device_temperatures = device_temperatures_;
  }
}

Status Sampler::PrepareDeviceLogitsAndParameter(std::vector<SamplingRequest>& sampling_reqs,
                                                SamplingDeviceParameter& sampling_device_parameter,
                                                float*& device_logits, Stream& stream, size_t slot) {
  PROFILE_EVENT_SCOPE(PrepareDeviceLogitsAndParameter, "PrepareDeviceLogitsAndParameter", rank_);
  bool use_top_k = false;
  bool use_top_p = false;
  bool use_temperature = false;
  sampling_device_parameter.logits_softmax = false;
  sampling_device_parameter.do_sampling = false;
  const size_t max_logits_num =
      batch_schedule_config_.max_batch_size * batch_schedule_config_.max_decode_tokens_per_req;

  for (auto& sampling_req : sampling_reqs) {
    SamplingConfig* const sampling_config = sampling_req.sampling_config;
    STATUS_CHECK_RETURN(sampling_config->VerifyArgs());
    sampling_device_parameter.logits_softmax |= sampling_req.logits_custom_length > 0;
    sampling_device_parameter.do_sampling |= sampling_req.logits_custom_length == 0;
    // In cases of logits_custom_length and speculative decoding, a single request may correspond to multiple logits
    sampling_device_parameter.bs += sampling_req.sampling_token_num;
    float* const logits = sampling_req.logits_buf;
    if (device_logits != logits && device_logits != nullptr) {
      return Status(RET_SEGMENT_FAULT, "sampling for different logits not implemented");
    }
    device_logits = logits;
    sampling_device_parameter.vocab_size_padded = batch_schedule_config_.max_vocab_size;
    const size_t offset = sampling_req.logits_offset;
    if (offset >= max_logits_num) {
      return Status(RET_SEGMENT_FAULT, "sampling check sampling_req.logits_offset >= max_logits_num");
    }
    for (size_t sampling_index = 0; sampling_index < sampling_req.sampling_token_num; sampling_index++) {
      host_topKs_[offset + sampling_index] = sampling_config->topk;
      host_topPs_[offset + sampling_index] = sampling_config->topp;
      host_temperatures_[offset + sampling_index] = sampling_config->temperature;
    }
    if (sampling_device_parameter.max_topK < sampling_config->topk) {
      sampling_device_parameter.max_topK = sampling_config->topk;
    }
    use_top_k |= sampling_config->topk > 1;
    use_top_p |= sampling_config->topp != 1.0f;
    use_temperature |= sampling_config->temperature != 1.0f;

    auto it = sampling_req.request_target->find("logits");
    if (it != sampling_req.request_target->end()) {
      int input_top_logprobs_num = it->second.input_top_logprobs_num;
      sampling_req.input_top_logprobs_num = input_top_logprobs_num;
      sampling_device_parameter.max_input_top_logprobs_num =
          std::max(sampling_device_parameter.max_input_top_logprobs_num, input_top_logprobs_num);
    }

    const int vocab_size = batch_schedule_config_.max_vocab_size;
    if (sampling_config->repetition_penalty != 1.0f) {
      for (size_t sampling_index = 0; sampling_index < sampling_req.sampling_token_num; sampling_index++) {
        ApplyRepetitionPenalty(logits + (offset + sampling_index) * vocab_size, sampling_req.input_tokens.get(),
                               sampling_req.forwarding_tokens, vocab_size, sampling_config->repetition_penalty,
                               stream);
      }
    }

    const int input_tokens_size = sampling_req.input_tokens->size();
    // NOTE(winminkong): Do not apply NoRepeatNgram sampling when sample req is mtp req.
    if (sampling_req.enable_mtp_sampler) {
      continue;
    }
    // NOTE(winminkong): When mtp_step_num > 0, the NoRepeatNgram sampling is applied only to the first token generated.
    if (sampling_config->no_repeat_ngram_size > 0) {
      NoRepeatNgramProcessor(logits + offset * vocab_size, sampling_config->no_repeat_ngram_size, input_tokens_size,
                             sampling_req.forwarding_tokens, sampling_req.ngram_dict, vocab_size,
                             sampling_req.last_step_token_num, stream);
    } else if (sampling_config->encoder_no_repeat_ngram_size > 0) {
      EncoderNoRepeatNgramProcessor(logits + offset * vocab_size, sampling_config->encoder_no_repeat_ngram_size,
                                    input_tokens_size, sampling_req.forwarding_tokens, sampling_req.ngram_dict,
                                    vocab_size, stream);
    } else if (sampling_config->decoder_no_repeat_ngram_size > 0) {
      DecoderNoRepeatNgramProcessor(logits + offset * vocab_size, sampling_config->decoder_no_repeat_ngram_size,
                                    input_tokens_size, sampling_req.forwarding_tokens, sampling_req.ngram_dict,
                                    vocab_size, sampling_req.last_step_token_num, stream);
    }
  }

  // top_p and temperature are applyed on the logits after softmax.
  sampling_device_parameter.logits_softmax |= use_top_p | use_temperature;
  sampling_device_parameter.logits_softmax &= (sampling_device_parameter.max_input_top_logprobs_num == 0);
  SamplingParameterToDevice(use_top_k, use_top_p, use_temperature, sampling_device_parameter, stream, slot);
  return Status();
}
void Sampler::SelectTopkTokens(const SamplingDeviceParameter& sampling_device_parameter, float* const device_logits,
                               Stream& stream, size_t slot) {
  if (!sampling_device_parameter.do_sampling) {
    return;
  }

  const auto run_select = [&]() {
    // 根据 slot 选用对应的 device/host buffer (slot 1 未分配时回落 slot 0).
    uint32_t* dev_out_tokens = GetDeviceOutputTokens(slot);
    std::vector<int>& host_out_tokens = GetHostOutputTokens(slot);

    if (rank_ == 0) {
      STATUS_CHECK_FAILURE(
          topk_sampling_->Forward(device_logits, dev_out_tokens, nullptr, sampling_device_parameter, nullptr, stream));
    }

#ifdef ENABLE_CUDA
    if (context_->global->GetTensorParallelSize() > 1 && context_->global->GetMtpStepNum() > 0) {
      NCCL_CHECK(ncclBroadcast(dev_out_tokens, dev_out_tokens, sampling_device_parameter.bs, ncclUint32, 0,
                               context_->device->ext->GetNCCLParam().nccl_comm, stream.Get()));
    }
#endif

    // D2H 在 PrimaryComputeStream 上异步完成; host 可读由 CopyProbsOutputToRequests 内 StreamSynchronize 保证.
    MemcpyAsync(host_out_tokens.data(), dev_out_tokens, sizeof(uint32_t) * sampling_device_parameter.bs,
                MEMCPY_DEVICE_TO_HOST, stream);
  };

  if (FastPathController::GetInstance().IsEnabledAtStartup()) {
    PROFILE_EVENT_SCOPE(SelectTopkTokens_, fmt::format("SelectTopkTokens_slot{}", slot), rank_);
    run_select();
  } else {
    run_select();
  }
}

Status Sampler::Sampling(std::vector<SamplingRequest>& sampling_reqs, Stream& stream, size_t slot) {
  PROFILE_EVENT_SCOPE(Sampling_, fmt::format("Sampling_{}", rank_), rank_);
  float* device_logits = nullptr;
  SamplingDeviceParameter sampling_device_parameter;
  STATUS_CHECK_RETURN(
      PrepareDeviceLogitsAndParameter(sampling_reqs, sampling_device_parameter, device_logits, stream, slot));

  SamplingAndCalcLogprobs(sampling_reqs, device_logits, sampling_device_parameter, stream, slot);

  // Apply grammar mask after logits processing
  ApplyGrammarMask(sampling_reqs, device_logits, sampling_device_parameter, stream);

  // Apply softmax on logits.
  if (sampling_device_parameter.logits_softmax) {
#ifdef ENABLE_CUDA
    CUDA_CHECK_LAST_ERROR(tensorrt_llm::kernels::InvokeAddBiasSoftMax<float>(
        device_logits, nullptr, sampling_device_parameter.device_temperatures, nullptr, nullptr, nullptr, nullptr,
        sampling_device_parameter.bs, 0, 1, sampling_device_parameter.vocab_size_padded,
        sampling_device_parameter.vocab_size_padded, false, true, stream.Get()));
#else
    KLLM_THROW("Softmax is not supported on NPU.");
#endif
  }

  if (sampling_device_parameter.max_input_top_logprobs_num > 0 && rank_ == 0) {
    int max_top_num = sampling_device_parameter.max_input_top_logprobs_num;
    std::vector<std::vector<std::pair<int, float>>> input_top_logprobs_res(
        sampling_device_parameter.bs, std::vector<std::pair<int, float>>(max_top_num));
#ifdef ENABLE_CUDA
    CalcInputLogprobs(device_logits, sampling_device_parameter.device_temperatures,
                      sampling_device_parameter.vocab_size_padded, sampling_device_parameter.bs, input_top_logprobs_res,
                      max_top_num);
#else
    KLLM_THROW("Input logprobs calculation is not supported on NPU.");
#endif
    int pruned_len = 0;
    for (size_t req_index = 0; req_index < sampling_reqs.size(); ++req_index) {
      auto& sampling_req = sampling_reqs[req_index];
      if (sampling_req.enable_mtp_sampler) {
        KLLM_LOG_WARNING << "MTP sampler not support input_top_logprobs, please set mtp_step_num = 0";
        continue;
      }
      sampling_req.logprobs->clear();
      if (sampling_req.logits_custom_length <= 0) {
        sampling_req.logprobs->emplace_back();
      } else {
        for (int i = 0; i < sampling_req.logits_custom_length; ++i) {
          sampling_req.logprobs->emplace_back(
              input_top_logprobs_res[pruned_len + i].begin(),
              input_top_logprobs_res[pruned_len + i].begin() + sampling_req.input_top_logprobs_num);
        }
        pruned_len += sampling_req.logits_custom_length;
      }
    }
  }

  SelectTopkTokens(sampling_device_parameter, device_logits, stream, slot);

  std::vector<std::vector<float>> probs_output(sampling_reqs.size());
  CopyProbsOutputToRequests(sampling_reqs, probs_output, stream, slot);

  // Update grammar state after sampling
  UpdateGrammarState(sampling_reqs);

  return Status();
}

void Sampler::ApplyGrammarMask(std::vector<SamplingRequest>& sampling_reqs, float* device_logits,
                               const SamplingDeviceParameter& sampling_device_parameter, Stream& stream) {
  if (!batch_schedule_config_.enable_xgrammar) {
    return;
  }

  // allocate vocab mask
  const int bitmask_elements = static_cast<int>(std::ceil(batch_schedule_config_.max_vocab_size / kBitsPerInt));
  const int32_t full_mask = -1;  // All bits set to 1

  // Track which requests have grammar constraints
  std::vector<size_t> grammar_req_indices;
  std::vector<size_t> grammar_req_offsets;

  // Process each sampling request to identify grammar-enabled requests
  for (size_t req_idx = 0; req_idx < sampling_reqs.size(); ++req_idx) {
    auto& req = sampling_reqs[req_idx];

    if (!req.grammar_matcher || !req.apply_grammar_constraint) {
      continue;
    }

    // Record the request index and its logits offset (only first token position for MTP)
    grammar_req_indices.push_back(req_idx);
    grammar_req_offsets.push_back(req.logits_offset);
  }

  if (grammar_req_indices.empty()) {
    return;  // No requests with grammar constraints
  }

  // Allocate bitmask only for grammar-enabled requests
  const size_t grammar_req_num = grammar_req_indices.size();
  std::fill(host_vocab_mask_.begin(), host_vocab_mask_.begin() + grammar_req_num * bitmask_elements, full_mask);

  // Fill bitmasks for grammar-enabled requests
  bool has_active_grammar = false;
  for (size_t i = 0; i < grammar_req_num; ++i) {
    size_t req_idx = grammar_req_indices[i];
    auto& req = sampling_reqs[req_idx];

    // Fill the bitmask at position i (not req_idx)
    int32_t* batch_bitmask = host_vocab_mask_.data() + i * bitmask_elements;
    bool needs_mask = req.grammar_matcher->FillNextTokenBitmask(batch_bitmask, 0);

    if (needs_mask) {
      has_active_grammar = true;
      KLLM_LOG_DEBUG << fmt::format("Grammar applied: req={} idx={}/{}, sampling_token_num={}", req.req_id, req_idx, i,
                                    req.sampling_token_num);
    }
  }

  if (has_active_grammar) {
    KLLM_LOG_DEBUG << "Applying grammar mask: " << grammar_req_num << "/" << sampling_reqs.size() << " requests";

    // Copy bitmask to device (only for grammar requests)
    const size_t vocab_mask_size = grammar_req_num * bitmask_elements * sizeof(int32_t);
    MemcpyAsync(device_vocab_mask_, host_vocab_mask_.data(), vocab_mask_size, MEMCPY_HOST_TO_DEVICE, stream);

    // Apply bitmask only to grammar-enabled requests
    ApplyTokenBitmaskSelective(device_logits, device_vocab_mask_, sampling_device_parameter.vocab_size_padded,
                               grammar_req_offsets, stream);
  }
}

void Sampler::UpdateGrammarState(std::vector<SamplingRequest>& sampling_reqs) {
  for (auto& req : sampling_reqs) {
    if (!req.grammar_matcher || !req.apply_grammar_constraint || !req.sampling_result_tokens ||
        req.sampling_result_tokens->empty()) {
      continue;
    }

    if (req.grammar_matcher->IsTerminated()) {
      // Note: The request termination should be handled by the caller
      KLLM_LOG_DEBUG << "Grammar completed for request " << req.req_id;
    }

    // In MTP mode, only update grammar state for the first token (verify_token)
    // The second token (new_token) will be handled in DraftTokenFilter if needed
    int token_id = req.sampling_result_tokens->front();
    bool accepted = req.grammar_matcher->AcceptToken(token_id);

    if (!accepted) {
      // In production, this should rarely happen if the mask was applied correctly
      KLLM_LOG_WARNING << "Grammar rejected token " << token_id << " for request " << req.req_id;
    }
  }
}

void Sampler::ApplyTokenBitmaskSelective(float* logits, void* bitmask_data, int vocab_size,
                                         const std::vector<size_t>& logits_offsets, Stream& stream) {
  const int bitmask_stride = (vocab_size + kBitsPerInt - 1) / kBitsPerInt;
  KLLM_LOG_DEBUG << "BitmaskSelective parameters: vocab_size=" << vocab_size
                 << ", num_requests=" << logits_offsets.size()
                 << ", bitmask_stride=" << bitmask_stride;

#ifdef ENABLE_CUDA

  // Apply bitmask to each grammar-enabled request individually
  for (size_t i = 0; i < logits_offsets.size(); ++i) {
    float* request_logits = logits + logits_offsets[i] * vocab_size;
    int32_t* request_bitmask = static_cast<int32_t*>(bitmask_data) + i * bitmask_stride;

    // Apply bitmask for single request
    InvokeApplyTokenBitmaskInplace<float>(request_logits, request_bitmask, nullptr, vocab_size, vocab_size,
                                          bitmask_stride, 1, stream.Get());
  }
#else
  // NPU implementation: empty implementation for now
  KLLM_THROW("ApplyTokenBitmask is not supported on NPU.");
#endif
}
}  // namespace ksana_llm
