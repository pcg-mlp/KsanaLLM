/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/endpoints/streaming/streaming_iterator.h"

#include "numerous_llm/utils/ret_code.h"
#include "numerous_llm/utils/status.h"

namespace numerous_llm {

Status StreamingIterator::GetNext(int& token_id) {
  while (true) {
    if (request_->finished) {
      // failure, no token generated.
      if (!request_->finish_status.OK()) {
        return request_->finish_status;
      }

      // Make sure the last token is fetched.
      if (!last_token_fetched_) {
        last_token_fetched_ = true;
        std::unique_lock<std::mutex> lock(request_->output_mutex);
        if (cur_index_ < request_->output_tokens.size()) {
          token_id = request_->output_tokens[cur_index_++];
          return Status();
        }
      }

      return Status(RET_STOP_ITERATION);
    }

    // Have more token that not fetched.
    {
      std::unique_lock<std::mutex> lock(request_->output_mutex);
      if (cur_index_ < request_->output_tokens.size()) {
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

}  // namespace numerous_llm
