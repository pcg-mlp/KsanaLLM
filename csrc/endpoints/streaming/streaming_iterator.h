/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include <vector>
#include "utils/request.h"
#include "utils/status.h"
#include "utils/waiter.h"

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
      : request_(request),
        all_finished(false),
        ksana_python_input_(ksana_python_input),
        total_token_nums_(0),
        prefix_cache_hit_tokens_read_(false),
        prefix_cache_hit_tokens_(0) {
    return_logprobs_ = ksana_python_input_->sampling_config.logprobs_num > 0;
  }
  ~StreamingIterator();

  // Get the next token id, blocked if no token
  Status GetNext(ksana_llm::KsanaPythonOutput &ksana_python_output);
  bool AddOutput(ksana_llm::KsanaPythonOutput &ksana_python_output);

  // Mark request aborted and unblock any in-flight GetNext.
  // Why: ~StreamingIterator is the only abort hook today, but it cannot fire
  // while GetNext is blocked in step_waiter->Wait() inside a thread-pool worker
  // (the worker holds a reference to the shared_ptr). Calling Abort from the
  // event loop forces GetNext to return so the destructor can run.
  void Abort();

  // Snapshot the current access-log timestamps (microseconds, 0 if not yet stamped).
  // Used by Python to retrieve timing fields after StopIteration.
  void SnapshotTimestamps(ksana_llm::KsanaPythonOutput &ksana_python_output, bool wait_put_done = true) const;

 private:
  // The user request.
  std::shared_ptr<Request> request_;
  bool return_logprobs_;
  bool all_finished;

  std::shared_ptr<KsanaPythonInput> ksana_python_input_;

  // The total_token_nums, useful when notify event lost.
  size_t total_token_nums_;

  // Cache for prefix_cache_hit_tokens, read only once
  bool prefix_cache_hit_tokens_read_;
  int prefix_cache_hit_tokens_;
};

}  // namespace ksana_llm
