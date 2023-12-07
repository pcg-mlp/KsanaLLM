/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "numerous_llm/utils/environment.h"

namespace numerous_llm {

class ContextCaching {
 public:
  ContextCaching(const ContextCachingConfig &context_caching_config);
};

}  // namespace numerous_llm
