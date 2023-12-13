/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <atomic>
#include <string>

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

  // Get the next token number for next step.
  // For all waiting queue's reqs(context decoding stage), it is input token number.
  // For all otheqr queue's reqs(decoding stage), it is always 1.
  size_t GetStepTokenNumber();

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

 public:
  // The req id of the user's request.
  int req_id;

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
};

}  // namespace numerous_llm
