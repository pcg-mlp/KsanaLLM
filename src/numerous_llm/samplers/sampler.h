/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/runtime/infer_request.h"
#include "numerous_llm/utils/status.h"

namespace numerous_llm {

class Sampler {
public:
  Status Sampling(std::vector<InferRequest> &reqs);
};

} // namespace numerous_llm
