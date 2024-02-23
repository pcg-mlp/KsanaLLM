/* Copyright 2023 Tencent Inc.  All rights reserved.

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
  InferRequest(std::shared_ptr<Request> &request);
  ~InferRequest();

  // Notify after request finished.
  void Notify();

  // Notify after step finished.
  void NotifyStep();

  // Get logits ptr on every device, that is, output of forward and input of sampling.
  std::vector<float *> GetLogitsPtr();

  // Get addr ptr of blocks.
  std::vector<std::vector<void *>> GetBlockPtrs();

  // Get the block size.
  size_t GetBlockSize() const;

  // Reset the infer stage.
  void ResetInferStage();

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
  Status SwapOutAsync();

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

  // The input tokens.
  std::vector<int> &input_tokens;

  // The output tokens, always contain input tokens on the left.
  std::vector<int> &output_tokens;

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

  // The model instance pointer.
  std::shared_ptr<ModelInstance> model_instance;

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
};

}  // namespace ksana_llm
