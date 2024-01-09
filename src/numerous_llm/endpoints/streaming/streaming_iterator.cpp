/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/endpoints/streaming/streaming_iterator.h"

#include "numerous_llm/utils/ret_code.h"
#include "numerous_llm/utils/status.h"

namespace numerous_llm {

Status StreamingIterator::GetNext(int& token_id) {
  if (request_->finished) {
    // failure, no token generated.
    if (!request_->finish_status.OK()) {
      return request_->finish_status;
    }

    return Status(RET_STOP_ITERATION);
  }

  request_->step_waiter->Wait();
  token_id = request_->output_tokens.back();
  request_->step_waiter->Reset(1);

  return Status();
}

}  // namespace numerous_llm
