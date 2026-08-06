/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "runtime/speculative/trie/trie_decoder_adapter.h"

#include "runtime/executor_infer_request.h"
#include "runtime/speculative/common/speculative_resource_profile.h"

namespace ksana_llm {

TrieDecoderAdapter::TrieDecoderAdapter(std::shared_ptr<TrieGenerator> generator) : generator_(std::move(generator)) {}

void TrieDecoderAdapter::BeforeForward(SpecContext& ctx) {
  if (!ctx.reqs) return;
  for (auto& req : *ctx.reqs) {
    generator_->GenerateDraft(req);
  }
}

void TrieDecoderAdapter::Verify(SpecContext& ctx) {
  if (!ctx.reqs) return;
  for (auto& req : *ctx.reqs) {
    generator_->FilterDraftTokens(req);
  }
}

bool TrieDecoderAdapter::IsCombinableWith(const SpeculativeDecoderInterface* other) const {
  if (!other) return true;
  const char* other_name = other->GetName();
  return (strcmp(other_name, "MTP") != 0);
}

}  // namespace ksana_llm

// Reverse static registration: maps "trie" speculative_method to TrieDecoderAdapter.
namespace {
const bool kTrieDecoderRegistered = []() {
  ksana_llm::SpeculativeRegistry::Get().RegisterDecoder("trie", [](const ksana_llm::BatchSchedulerConfig&) {
    ksana_llm::DecoderFactoryResult result;
    auto gen = std::make_shared<ksana_llm::TrieGenerator>();
    result.generator = gen;
    result.adapter = std::make_unique<ksana_llm::TrieDecoderAdapter>(gen);
    return result;
  });
  return true;
}();
}  // namespace