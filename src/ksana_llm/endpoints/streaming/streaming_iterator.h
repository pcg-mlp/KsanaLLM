/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include <vector>
#include "ksana_llm/utils/request.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/waiter.h"

namespace ksana_llm {

// The dying request that to be reaped.
struct DyingRequest {
  std::shared_ptr<Request> request;
  std::shared_ptr<KsanaPythonInput> ksana_python_input;
};

// The iterator used to return output token in streaming mode.
class StreamingIterator {
 public:
  StreamingIterator() {}
  StreamingIterator(const std::shared_ptr<Request> request, const std::shared_ptr<KsanaPythonInput> &ksana_python_input)
      : request_(request), all_finished(false), ksana_python_input_(ksana_python_input), total_token_nums_(0) {
    return_logprobs_ = ksana_python_input_->sampling_config.logprobs_num > 0;
  }
  ~StreamingIterator();

  // Get the next token id, blocked if no token
  Status GetNext(ksana_llm::KsanaPythonOutput &ksana_python_output);
  bool AddOutput(ksana_llm::KsanaPythonOutput &ksana_python_output);

 private:
  // The user request.
  std::shared_ptr<Request> request_;
  bool return_logprobs_;
  bool all_finished;

  std::shared_ptr<KsanaPythonInput> ksana_python_input_;

  // The total_token_nums, useful when notify event lost.
  size_t total_token_nums_;
};

}  // namespace ksana_llm
