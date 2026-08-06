/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "runtime/speculative/common/speculative_resource_profile.h"

namespace ksana_llm {

SpeculativeRegistry& SpeculativeRegistry::Get() {
  static SpeculativeRegistry instance;
  return instance;
}

void SpeculativeRegistry::Register(SpeculativeResourceEstimator estimator) {
  estimators_.push_back(std::move(estimator));
}

SpeculativeResourceProfile SpeculativeRegistry::Aggregate(const BatchSchedulerConfig& cfg) const {
  SpeculativeResourceProfile total;
  for (const auto& estimator : estimators_) {
    auto profile = estimator(cfg);
    total.extra_decode_tokens_per_req += profile.extra_decode_tokens_per_req;
    total.workload_multiplier += profile.workload_multiplier;
    total.extra_token_capacity += profile.extra_token_capacity;
    // Append buffer decls directly (no move_iterator needed for clarity).
    total.buffer_decls.insert(total.buffer_decls.end(), profile.buffer_decls.begin(), profile.buffer_decls.end());
  }
  return total;
}

void SpeculativeRegistry::RegisterDecoder(const std::string& method, DecoderFactory factory) {
  decoder_factories_[method] = std::move(factory);
}

DecoderFactoryResult SpeculativeRegistry::CreateDecoder(const std::string& method,
                                                        const BatchSchedulerConfig& cfg) const {
  auto it = decoder_factories_.find(method);
  if (it != decoder_factories_.end()) {
    return it->second(cfg);
  }
  return {};
}

bool SpeculativeRegistry::HasDecoderMethod(const std::string& method) const {
  return decoder_factories_.find(method) != decoder_factories_.end();
}

void SpeculativeRegistry::RegisterPostInitSetup(PostInitSetup setup) { post_init_setups_.push_back(std::move(setup)); }

void SpeculativeRegistry::RunPostInitSetups(const SpeculativeSetupContext& ctx) const {
  for (const auto& setup : post_init_setups_) {
    setup(ctx);
  }
}

}  // namespace ksana_llm
