/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/utils/environment.h"

namespace ksana_llm {

class RequestBatching {
 public:
  RequestBatching(const RequestBatchingConfig &request_batching_config);
};

}  // namespace ksana_llm
