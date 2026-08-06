/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>

#include "runtime/speculative/asr/asr_custom_generator.h"
#include "runtime/speculative/common/speculative_decoder_interface.h"

namespace ksana_llm {

class AsrDecoderAdapter : public SpeculativeDecoderInterface {
 public:
  explicit AsrDecoderAdapter(std::shared_ptr<AsrCustomGenerator> generator);

  const char* GetName() const override { return "ASR"; }
  SpecSchedulingInfo GetSchedulingInfo() const override { return {}; }
  void BeforeForward(SpecContext& ctx) override;
  void ExecuteForward(SpecContext& ctx) override {}
  void AfterForward(SpecContext& ctx) override {}
  void Verify(SpecContext& ctx) override;
  bool IsCombinableWith(const SpeculativeDecoderInterface* other) const override;

 private:
  std::shared_ptr<AsrCustomGenerator> generator_;
};

}  // namespace ksana_llm