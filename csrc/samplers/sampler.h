/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "configure/environment.h"
#include "runtime/sampling_request.h"
#include "samplers/base/base_sampling.h"
#include "samplers/topk/topk_sampling.h"
#include "utils/grammar_matcher.h"
#include "utils/status.h"
#include "utils/tensor.h"

namespace ksana_llm {

// TODO(winminkong): To ensure that the sampling_result_tokens align with the indices of the top log probabilities of
// the output tokens, we will temporarily standardize by computing the top 5 log probabilities. It will be deleted
// later.
constexpr int kMinLogprobsNum = 5;

class Sampler {
 public:
  Sampler(const BatchSchedulerConfig& batch_scheduler_config, const int rank, std::shared_ptr<Context> context);
  ~Sampler();
  // slot: 与 ForwardingContext / NewModelInput 同名槽位编号。慢路径恒为 0。
  // build-ahead 在 0/1 间 ping-pong，使连续两步采样结果落到独立 device/host buffer，
  // 避免 D2H 竞争与 host 数据被下一步覆盖。slot 1 仅在 build-ahead 启用时分配。
  Status Sampling(std::vector<SamplingRequest>& sampling_reqs, Stream& stream, size_t slot = 0);
  Status SamplingAndCalcLogprobs(std::vector<SamplingRequest>& sampling_reqs, float* device_logits,
                                 SamplingDeviceParameter& sampling_device_parameter, Stream& stream, size_t slot = 0);

  void SamplingParameterToDevice(bool use_top_k, bool use_top_p, bool use_temperature,
                                 SamplingDeviceParameter& sampling_device_parameter, Stream& stream, size_t slot = 0);

  Status PrepareDeviceLogitsAndParameter(std::vector<SamplingRequest>& sampling_reqs,
                                         SamplingDeviceParameter& sampling_device_parameter, float*& device_logits,
                                         Stream& stream, size_t slot = 0);

  // 双 sampler buffer 是否已分配 slot 1（build-ahead 启用条件之一）。
  bool HasDoubleSlot() const { return device_output_tokens_slot1_ != nullptr; }

  // (req_id, token) 配对 ring：返回 slot 对应 device ring 起始指针。
  // 布局 [count, req_id_0, token_0, req_id_1, token_1, ...]；ReplaceLastTokenKernel 以 prev_slot=cur_slot^1 读取。
  // 仅 build-ahead 启用时分配；未启用返回 nullptr，调用方应跳过 launch。
  int64_t* GetRingSlot(size_t slot) {
    if (slot == 1) return device_ring_slot1_;
    return device_ring_slot0_;
  }

  // 把本步 sampling 输出与 cur_pairs 中的 req_id 配对写入 ring slot（含 count 头部）。
  // SelectTopkTokens 后立即调用，slot 与本步 sampling 一致。ring 未分配时静默 no-op。
  // TP 下仅 rank 0 写入有效采样结果，其余 rank 通过 BroadcastRingSlot 同步。
  void WriteRing(size_t slot, const int64_t* cur_pairs, int batch_size, Stream& stream);

  // TP>1 时把 rank 0 本步 WriteRing 的 slot 广播到组内所有卡，供下一步 ReplaceLastTokenKernel 读取。
  // 所有 TP rank 必须同一步 collective 调用；tp_size==1 或 ring 未分配时 no-op。
  void BroadcastRingSlot(size_t slot, Stream& stream);

  // 暴露 slot 对应 device_output_tokens 指针，供 WriteRingKernel 使用。
  // 慢路径不应调用 slot=1；slot 1 未分配时内部回落 slot 0。
  uint32_t* GetDeviceOutputTokensPtr(size_t slot) { return GetDeviceOutputTokens(slot); }

  // Copies the probabilities from the logits buffer to the output vector for each sampling request.
  std::function<void()> CopyProbsOutput(std::vector<SamplingRequest>& sampling_reqs, Stream& stream,
                                        std::vector<std::vector<float>>& probs_output);

  void ApplyRepetitionPenalty(float* logits, std::vector<int>* input_tokens, std::vector<int>* forwarding_tokens,
                              const int vocab_size, const float repetition_penalty, Stream& stream);

  void CopyProbsOutputToRequests(std::vector<SamplingRequest>& sampling_reqs,
                                 std::vector<std::vector<float>>& probs_output, Stream& stream, size_t slot = 0);

  void GetNgrams(const int ngram_size, const int cur_output_size, const std::vector<int>* output_tokens,
                 NgramDict* ngram_dict);

  void BanRepeatTokens(float* logits, const int ngram_size, const int cur_output_size,
                       const std::vector<int>* output_tokens, NgramDict* ngram_dict, const int vocab_size,
                       Stream& stream);

  void NoRepeatNgramProcessor(float* logits, const int ngram_size, const int input_tokens_size,
                              const std::vector<int>* output_tokens, NgramDict* ngram_dict, const int vocab_size,
                              size_t last_step_token_num, Stream& stream);

  void EncoderNoRepeatNgramProcessor(float* logits, const int ngram_size, const int input_tokens_size,
                                     const std::vector<int>* output_tokens, NgramDict* ngram_dict, const int vocab_size,
                                     Stream& stream);

  void DecoderNoRepeatNgramProcessor(float* logits, const int ngram_size, const int input_tokens_size,
                                     const std::vector<int>* output_tokens, NgramDict* ngram_dict, const int vocab_size,
                                     size_t last_step_token_num, Stream& stream);

  // Apply grammar constraints to logits for requests with grammar matchers
  void ApplyGrammarMask(std::vector<SamplingRequest>& sampling_reqs, float* device_logits,
                        const SamplingDeviceParameter& sampling_device_parameter, Stream& stream);

  // Update grammar state after token selection for grammar-enabled requests
  void UpdateGrammarState(std::vector<SamplingRequest>& sampling_reqs);

  // Apply token bitmask selectively to grammar-enabled requests only
  void ApplyTokenBitmaskSelective(float* logits, void* bitmask_data, int vocab_size,
                                  const std::vector<size_t>& logits_offsets, Stream& stream);

  // Get the next tokens based on logits and the sampling parameters.
  void SelectTopkTokens(const SamplingDeviceParameter& sampling_device_parameter, float* const device_logits,
                        Stream& stream, size_t slot = 0);

 private:
  // 选取 slot 对应的 device output tokens 起始指针 (slot 1 未分配时回落 slot 0).
  uint32_t* GetDeviceOutputTokens(size_t slot) {
    return (slot == 1 && device_output_tokens_slot1_ != nullptr) ? device_output_tokens_slot1_ : device_output_tokens_;
  }

  // 选取 slot 对应的 device ptr 数组 (slot 1 未分配时回落 slot 0).
  int** GetDeviceOutputTokensPtrs(size_t slot) {
    return (slot == 1 && device_output_tokens_ptrs_slot1_ != nullptr) ? device_output_tokens_ptrs_slot1_
                                                                      : device_output_tokens_ptrs_;
  }

  // 选取 slot 对应的 host output tokens (slot 1 未分配时回落 slot 0).
  std::vector<int>& GetHostOutputTokens(size_t slot) {
    return (slot == 1 && !host_output_tokens_slot1_.empty()) ? host_output_tokens_slot1_ : host_output_tokens_;
  }

  const BatchSchedulerConfig batch_schedule_config_;
  const int rank_;
  TopkSampling* topk_sampling_{nullptr};
  void* device_buffer_{nullptr};
  uint32_t* device_output_tokens_;
  uint32_t* device_offset_;
  int* device_topKs_;
  float* device_topPs_;
  float* device_temperatures_;
  int** device_output_tokens_ptrs_;
  float* device_repetition_processor_;
  float* device_prob_;
  float** device_prob_ptrs_;
  RandState* device_curandstates_{nullptr};
  std::vector<int> host_output_tokens_;

  // build-ahead 双实例：slot 1 独立 buffer；慢路径不分配，对原路径零影响。
  // 物理隔离 device/host 数据，避免相邻 step sampling 互相覆盖。
  void* device_buffer_slot1_{nullptr};
  uint32_t* device_output_tokens_slot1_{nullptr};
  int** device_output_tokens_ptrs_slot1_{nullptr};
  std::vector<int> host_output_tokens_slot1_;

  // (req_id, token) 配对 ring buffer，各 slot 独立。布局 int64[1 + 2 * max_batch_size]。
  // 仅 build-ahead 启用时分配；否则 nullptr。
  int64_t* device_ring_slot0_{nullptr};
  int64_t* device_ring_slot1_{nullptr};
  size_t device_ring_capacity_{0};  // 已分配槽长度 (int64 元素数), 即 1 + 2*max_batch_size
  std::vector<int> host_topKs_;
  std::vector<float> host_topPs_;
  std::vector<float> host_temperatures_;
  std::vector<const float*> host_logits_;

  // The context
  std::shared_ptr<Context> context_;
  std::vector<float> repetition_penalties_;
  std::vector<float> norepeat_ngrams_;

  // Grammar-related buffers
  int32_t* device_vocab_mask_{nullptr};
  std::vector<int32_t> host_vocab_mask_;
};

}  // namespace ksana_llm
