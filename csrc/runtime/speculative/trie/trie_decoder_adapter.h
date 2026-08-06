/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>

#include "runtime/speculative/trie/trie_generator.h"
#include "runtime/speculative/common/speculative_decoder_interface.h"

namespace ksana_llm {

// Adapter that wraps TrieGenerator as a SpeculativeDecoderInterface.
// TrieGenerator is a no-draft-model decoder: ExecuteForward is a no-op,
// draft tokens are generated in BeforeForward (trie tree update + prediction).
class TrieDecoderAdapter : public SpeculativeDecoderInterface {
 public:
  explicit TrieDecoderAdapter(std::shared_ptr<TrieGenerator> generator);

  const char* GetName() const override { return "Trie"; }
  SpecSchedulingInfo GetSchedulingInfo() const override { return {}; }
  void BeforeForward(SpecContext& ctx) override;
  void ExecuteForward(SpecContext& ctx) override {}
  void AfterForward(SpecContext& ctx) override {}
  void Verify(SpecContext& ctx) override;
  bool IsCombinableWith(const SpeculativeDecoderInterface* other) const override;

 private:
  std::shared_ptr<TrieGenerator> generator_;
};

}  // namespace ksana_llm