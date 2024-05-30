/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/endpoints/streaming/streaming_iterator.h"

#include "ksana_llm/utils/ret_code.h"
#include "ksana_llm/utils/status.h"

namespace ksana_llm {

bool StreamingIterator::AddOutput(ksana_llm::KsanaPythonOutput& ksana_python_output) {
  size_t total_token_nums = 0;
  for (int i = 0; i < request_->output_group.size(); i++) {
    OutputTuple& output = request_->output_group[i];
    total_token_nums += std::get<0>(output).size();
  }
  if (total_token_nums == total_token_nums_) return false;
  total_token_nums_ = total_token_nums;

  for (int i = 0; i < request_->output_group.size(); i++) {
    OutputTuple& output = request_->output_group[i];
    ksana_python_output.output_tokens.push_back(std::get<0>(output));
    if (return_logprobs_) ksana_python_output.logprobs.push_back(std::get<1>(output));
  }
  ksana_python_output.prompt_probs = std::move(request_->prompt_probs);
  return true;
}

Status StreamingIterator::GetNext(ksana_llm::KsanaPythonOutput& ksana_python_output) {
  while (true) {
    if (all_finished) {
      // failure, no token generated.
      if (!request_->finish_status.OK()) {
        return request_->finish_status;
      }
      return Status(RET_STOP_ITERATION);
    }
    // Waiting util next token generated.
    request_->step_waiter->Wait();
    request_->step_waiter->Reset(1);
    all_finished = true;
    for (bool req_finished : request_->finisheds) {
      all_finished = all_finished && req_finished;
    }
    {
      // Fetch next token util the last token is fetched.
      std::unique_lock<std::mutex> lock(request_->output_mutex);
      if (!AddOutput(ksana_python_output)) continue;
    }
    return Status();
  }
  return Status();
}

}  // namespace ksana_llm
