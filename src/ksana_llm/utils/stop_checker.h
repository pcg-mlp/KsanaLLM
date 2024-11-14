/* Copyright 2024 Tencent Inc.  All rights reserved.
==============================================================================*/
#pragma once

#include "ksana_llm/runtime/infer_request.h"
#include "ksana_llm/utils/tokenizer.h"

namespace ksana_llm {

// Stop Checker to check stop strings
class StopChecker {
 public:
  // Do increment stop strings check during generation phase to do early stop
  bool CheckIncrementalStopStrings(const std::shared_ptr<InferRequest> req,
    const std::shared_ptr<Tokenizer> tokenizer);

    void CheckCompleteStopStrings(const std::shared_ptr<InferRequest> req, const std::shared_ptr<Tokenizer> tokenizer);
};

}  // namespace ksana_llm
