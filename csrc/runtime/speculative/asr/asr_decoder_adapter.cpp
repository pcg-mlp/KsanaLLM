/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "runtime/speculative/asr/asr_decoder_adapter.h"

#include "runtime/executor_infer_request.h"
#include "runtime/speculative/common/speculative_resource_profile.h"

namespace ksana_llm {

AsrDecoderAdapter::AsrDecoderAdapter(std::shared_ptr<AsrCustomGenerator> generator)
    : generator_(std::move(generator)) {}

void AsrDecoderAdapter::BeforeForward(SpecContext& ctx) {
  if (!ctx.reqs) return;
  for (auto& req : *ctx.reqs) {
    generator_->GenerateDraft(req);
  }
}

void AsrDecoderAdapter::Verify(SpecContext& ctx) {
  if (!ctx.reqs) return;
  for (auto& req : *ctx.reqs) {
    generator_->FilterDraftTokens(req);
  }
}

bool AsrDecoderAdapter::IsCombinableWith(const SpeculativeDecoderInterface* other) const {
  if (!other) return true;
  const char* other_name = other->GetName();
  return (strcmp(other_name, "MTP") != 0);
}

}  // namespace ksana_llm

// Reverse static registration: maps "asr_custom" speculative_method to AsrDecoderAdapter.
namespace {
const bool kAsrDecoderRegistered = []() {
  ksana_llm::SpeculativeRegistry::Get().RegisterDecoder("asr_custom", [](const ksana_llm::BatchSchedulerConfig&) {
    ksana_llm::DecoderFactoryResult result;
    auto gen = std::make_shared<ksana_llm::AsrCustomGenerator>();
    result.generator = gen;
    result.adapter = std::make_unique<ksana_llm::AsrDecoderAdapter>(gen);
    return result;
  });
  return true;
}();
}  // namespace