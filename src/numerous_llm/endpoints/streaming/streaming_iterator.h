/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include <vector>
#include "numerous_llm/utils/request.h"
#include "numerous_llm/utils/status.h"
#include "numerous_llm/utils/waiter.h"

namespace numerous_llm {

// The iterator used to return output token in streaming mode.
class StreamingIterator {
 public:
  StreamingIterator() {}
  StreamingIterator(const std::shared_ptr<Request> request) : request_(request) {}
  ~StreamingIterator() {}

  // Get the next token id, blocked if no token
  Status GetNext(int& token_id);

 private:
  // The user request.
  std::shared_ptr<Request> request_;
};

}  // namespace numerous_llm
