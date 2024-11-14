/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/batch_scheduler/strategy/base_strategy.h"

namespace ksana_llm {

void BaseScheduleStrategy::SetCacheManager(std::shared_ptr<CacheManagerInterface> cache_manager) {
  cache_manager_ = cache_manager;
}

void BaseScheduleStrategy::SetTokenizer(std::shared_ptr<Tokenizer> tokenizer) {
  tokenizer_ = tokenizer;
}

}  // namespace ksana_llm
