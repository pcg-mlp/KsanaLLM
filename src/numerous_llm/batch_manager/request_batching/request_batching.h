/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "numerous_llm/utils/environment.h"

namespace numerous_llm {

class RequestBatching {
public:
  RequestBatching(const RequestBatchingConfig &request_batching_config);
};

} // namespace numerous_llm
