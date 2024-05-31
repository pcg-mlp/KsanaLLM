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
  StreamingIterator(const std::shared_ptr<Request> request, bool return_logprobs)
      : request_(request), return_logprobs_(return_logprobs), all_finished(false), total_token_nums_(0) {}
  ~StreamingIterator() {}

  // Get the next token id, blocked if no token
  Status GetNext(ksana_llm::KsanaPythonOutput &ksana_python_output);
  bool AddOutput(ksana_llm::KsanaPythonOutput &ksana_python_output);

 private:
  // The user request.
  std::shared_ptr<Request> request_;
  bool return_logprobs_;
  bool all_finished;

  // The total_token_nums, useful when notify event lost.
  size_t total_token_nums_;
};

}  // namespace ksana_llm
