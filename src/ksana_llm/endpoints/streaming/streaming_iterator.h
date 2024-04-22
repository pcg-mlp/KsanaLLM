/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include <vector>
#include "ksana_llm/utils/request.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/waiter.h"

namespace ksana_llm {

// The iterator used to return output token in streaming mode.
class StreamingIterator {
  public:
    StreamingIterator() {}
    StreamingIterator(const std::shared_ptr<Request> request) : request_(request) {
      cur_index_ = request->input_tokens.size();
    }
    ~StreamingIterator() {}

    // Get the next token id, blocked if no token
    Status GetNext(int& token_id);

  private:
    // The user request.
    std::shared_ptr<Request> request_;

    // The current index, useful when notify event lost.
    size_t cur_index_;
};

}  // namespace ksana_llm
