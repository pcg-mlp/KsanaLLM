/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/runtime/infer_request.h"

#include <atomic>
#include <limits>
#include "numerous_llm/utils/status.h"

static std::atomic index_counter = 0;

namespace numerous_llm {

InferRequest::InferRequest() {
  constexpr int max = std::numeric_limits<int>::max();

  ++index_counter;
  if (index_counter == max) {
    index_counter = 1;
  }

  infer_id = index_counter;
}

size_t InferRequest::GetStepTokenNumber() { return 0; }

size_t InferRequest::GetStepBlockNumber() { return 0; }

size_t InferRequest::GetTotalBlockNumber() { return 0; }

Status InferRequest::SwapInAsync() { return Status(); }

Status InferRequest::SwapOutAsync() { return Status(); }

bool InferRequest::CheckLoraEnable() { return false; }

size_t InferRequest::GetLoraBlockNumber() { return 0; }

Status InferRequest::SwapInLoraAsync() { return Status(); }

Status InferRequest::SwapOutLoraAsync() { return Status(); }

}  // namespace numerous_llm
