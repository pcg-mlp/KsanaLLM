/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <atomic>
#include <future>
#include <memory>
#include <string>
#include <vector>

#include "ksana_llm/runtime/infer_stage.h"
#include "ksana_llm/runtime/model_instance.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/request.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/tensor.h"
#include "ksana_llm/utils/waiter.h"

namespace ksana_llm {

// The infer request, it is the unit of batch manager's scheduler.
class InferRequest {
 public:
  InferRequest(std::shared_ptr<Request> &request, int index);
  ~InferRequest();

  void SetReqGroup(const std::vector<std::shared_ptr<InferRequest>> &beam_search_infer_group) {
    req_group = beam_search_infer_group;
  }

  void ClearReqGroup() { req_group.clear(); }

  // Notify after request finished.
  void Notify();

  // Notify after step finished.
  void NotifyStep();

  // Get logits ptr on every device, that is, output of forward and input of sampling.
  std::vector<float *> GetLogitsPtr();

  // Get addr ptr of blocks.
  std::vector<std::vector<void *>> GetBlockPtrs();

  // Adjust the infer stage.
  void AdjustInferStage();

  // Get the next token number for next step.
  // For all waiting queue's reqs(context decoding stage), it is 1 + input token number.
  // For all otheqr queue's reqs(decoding stage), it is always 1.
  size_t GetStepTokenNumber();

  // Get the total token number.
  // that is, the current tokens and the next token.
  size_t GetTotalTokenNumber();

  // Get the next wanted block number for next step.
  // It is determited by next token number.
  size_t GetStepBlockNumber();

  // Get the total block number for current request.
  // that is, the current tokens and the next token.
  size_t GetTotalBlockNumber();

  // Get the current block number
  // Include all the generated tokens, except the next token.
  size_t GetCurrentBlockNumber();

  // Swap in/out this request asynchronous.
  Status SwapInAsync();
  Status SwapOutAsync(const int host_block_num_to_add);

  // Drop this swapped request.
  Status DropSwappedAsync();

  // Free blocks this request hold.
  Status FreeBlocks();

  // Check whether the model instance enable lora.
  bool CheckLoraEnable();

  // Get the block number for lora weights.
  size_t GetLoraBlockNumber();

  // Swap in/out request's lora weights.
  Status SwapInLoraAsync();
  Status SwapOutLoraAsync();

  // Allocate blocks for next step.
  Status AllocateStepBlocks();

 public:
  // The req id of the user's request.
  int64_t &req_id;

  // The name of model instance.
  std::string &model_name;

  // The offsets of the tokens for the prompt_probs that need to be returned.
  size_t prompt_probs_offset = 0;

  // Probs of specific tokens at certain positions in the prompt.
  std::vector<float> &prompt_probs;

  // The input tokens.
  std::vector<int> &input_tokens;

  // The subinput_pos indicates the start position of the embedding to be replaced.
  std::vector<int> &subinput_pos;

  // The subinput_embedding is the embedding value to be used for the replacement, from the request.
  std::vector<std::vector<float>> &subinput_embedding;

  // The subinput_url is the multimodal resources url
  std::vector<std::string> &subinput_url;

  // The output tokens, always contain input tokens on the left.
  std::vector<int> &output_tokens;

  // Store token and their corresponding float probability values.
  std::vector<std::vector<std::pair<int, float>>> &logprobs;

  float cumulative_score;

  // The sampling config of this request.
  SamplingConfig &sampling_config;

  // The waiter used to notify when request finished.
  std::shared_ptr<Waiter> &waiter;

  // The waiter used to notify when step finished.
  std::shared_ptr<Waiter> &step_waiter;

  // Whether the request is finished.
  bool &finished;

  // The final status of this request.
  Status &finish_status;

  // Protect parallel access for output token.
  std::mutex &output_mutex;

  // The padded token num.
  int &padded_size;

  std::vector<std::shared_ptr<InferRequest>> req_group;

  // The intermediate result of beam_search
  std::vector<OutputTuple> &beam_search_group;

  // The model instance pointer.
  std::shared_ptr<ModelInstance> model_instance;

  // Padding token id of the model.
  int pad_id;

  // The arrive time.
  unsigned long timestamp_in_ms;

  // context decode or decode stage.
  InferStage infer_stage;

  // The decode step, 1 for context decode, and then 2, 3, 4...
  int step;

  // The kv cache blocks this request used, the index is used as device_id.
  // The key and value are stored in same blocks.
  std::vector<std::vector<int>> kv_cache_blocks;

  // The block size for every kv cache block.
  size_t block_size;

  // The offset for model forward's logits output.
  size_t logits_offset = 0;

  // Whether the current req is in pending status of swappiness.
  bool swap_pending = false;

  // The swappiness future.
  std::future<void> swap_future;

  // The flag for tagging request prefix cache usage
  bool is_use_prefix_cache = false;

  // The prefix cache tokens number
  int prefix_cache_len = 0;

  // The prefix cache blocks number
  int prefix_cache_blocks_number = 0;
};

}  // namespace ksana_llm
