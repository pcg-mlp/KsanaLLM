/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/runtime/infer_request.h"

#include <atomic>
#include <limits>

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

} // namespace numerous_llm
