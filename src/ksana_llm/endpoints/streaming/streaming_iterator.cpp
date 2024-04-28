/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/endpoints/streaming/streaming_iterator.h"

#include "ksana_llm/utils/ret_code.h"
#include "ksana_llm/utils/status.h"

namespace ksana_llm {

Status StreamingIterator::GetNext(int& token_id, std::vector<std::pair<int, float>>& logprobs) {
  while (true) {
    if (request_->finished) {
      // failure, no token generated.
      if (!request_->finish_status.OK()) {
        return request_->finish_status;
      }

      // Fetch next token util the last token is fetched.
      std::unique_lock<std::mutex> lock(request_->output_mutex);
      if (cur_index_ < request_->output_tokens.size()) {
        logprobs = request_->logprobs[cur_index_ - request_->input_tokens.size()];
        token_id = request_->output_tokens[cur_index_++];
        return Status();
      }

      return Status(RET_STOP_ITERATION);
    }

    // Have more token that not fetched.
    {
      std::unique_lock<std::mutex> lock(request_->output_mutex);
      if (cur_index_ < request_->output_tokens.size()) {
        logprobs = request_->logprobs[cur_index_ - request_->input_tokens.size()];
        token_id = request_->output_tokens[cur_index_++];
        return Status();
      }
    }

    // Waiting util next token generated.
    request_->step_waiter->Wait();
    request_->step_waiter->Reset(1);
  }

  return Status();
}

}  // namespace ksana_llm
