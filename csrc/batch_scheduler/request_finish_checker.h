/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>

#include "configure/schedule_config_parser.h"
#include "runtime/infer_request.h"

namespace ksana_llm {

// Returns true when the request still has remaining prefill workload (e.g. split-fuse chunks).
// output_tokens is initialized from input_tokens, so length-based finish checks must be skipped
// until prefill is fully done.
inline bool HasRemainingPrefillWorkload(const std::shared_ptr<InferRequest>& req) {
  return req->GetRemainingWorkload().prefill_token_num > 0;
}

// True if request is finished (EOS / max_new_tokens / max_token_len).
// Returns false while split-fuse prefill is still in progress.
inline bool IsRequestFinished(const std::shared_ptr<InferRequest>& req, const BatchSchedulerConfig& config) {
  if (HasRemainingPrefillWorkload(req)) {
    return false;
  }
  return req->IsEosGenerated() ||
         (req->sampling_config.max_new_tokens > 0 &&
          req->output_tokens.size() >= req->input_tokens.size() + req->sampling_config.max_new_tokens) ||
         // max_token_len limits the total sequence length (prompt + generated tokens), not generated-only length.
         req->output_tokens.size() >= config.max_token_len;
}

}  // namespace ksana_llm
