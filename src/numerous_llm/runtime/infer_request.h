/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <atomic>
#include <string>
#include <vector>

#include "numerous_llm/runtime/infer_stage.h"
#include "numerous_llm/runtime/model_instance.h"
#include "numerous_llm/utils/environment.h"
#include "numerous_llm/utils/request.h"
#include "numerous_llm/utils/status.h"
#include "numerous_llm/utils/tensor.h"
#include "numerous_llm/utils/waiter.h"

namespace numerous_llm {

// The infer request, it is the unit of batch manager's scheduler.
class InferRequest {
 public:
  InferRequest();
  ~InferRequest();

  // Get logits ptr on every device, that is, output of forward and input of sampling.
  std::vector<float*> GetLogitsPtr();

  // Get addr ptr of blocks.
  std::vector<std::vector<void*>> GetBlockPtrs();

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
  size_t GetTotalBlockNumber();

  // Swap in/out this request asynchronous.
  Status SwapInAsync();
  Status SwapOutAsync();

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
  int64_t req_id;

  // The unique id of this request.
  int infer_id;

  // The name of model instance.
  std::string model_name;

  // The model instance pointer.
  std::shared_ptr<ModelInstance> model_instance;

  // The arrive time.
  unsigned long timestamp_in_ms;

  // context decode or decode stage.
  InferStage infer_stage;

  // The decode step, 1 for context decode, and then 2, 3, 4...
  int step;

  // The final status of this request.
  Status finish_status;

  // The input tensor map.
  TensorMap input_tensor_map;

  // The sampling config of this request.
  SamplingConfig sampling_config;

  // The output tensor map.
  TensorMap output_tensor_map;

  // The waiter used to nofity caller client.
  std::shared_ptr<Waiter> waiter;

  // The kv cache blocks this request used, the index is used as device_id.
  // The key and value are stored in same blocks.
  std::vector<std::vector<int>> kv_cache_blocks;

  // The block size for every kv cache block.
  size_t block_size;

  // The input tokens.
  std::vector<int> input_tokens;

  // The output tokens, always contain input tokens on the left.
  std::vector<int> output_tokens;

  // The offset for model forward's logits output.
  size_t logits_offset;
};

}  // namespace numerous_llm
