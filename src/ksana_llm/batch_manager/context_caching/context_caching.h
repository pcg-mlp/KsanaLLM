/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/utils/environment.h"

namespace ksana_llm {

class ContextCaching {
 public:
  ContextCaching(const ContextCachingConfig &context_caching_config);
};

}  // namespace ksana_llm
