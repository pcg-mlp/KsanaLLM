/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>

#include "runtime/speculative/ptp/ptp_generator.h"
#include "runtime/speculative/common/speculative_decoder_interface.h"

namespace ksana_llm {

// Adapter that wraps PtpGenerator as a SpeculativeDecoderInterface.
// PtpGenerator is a no-draft-model decoder: ExecuteForward is a no-op,
// draft tokens are generated in BeforeForward (placeholder tokens).
class PtpDecoderAdapter : public SpeculativeDecoderInterface {
 public:
  explicit PtpDecoderAdapter(std::shared_ptr<PtpGenerator> generator);

  const char* GetName() const override { return "PTP"; }
  SpecSchedulingInfo GetSchedulingInfo() const override;
  void BeforeForward(SpecContext& ctx) override;
  void ExecuteForward(SpecContext& ctx) override {}
  void AfterForward(SpecContext& ctx) override {}
  void Verify(SpecContext& ctx) override;
  bool IsCombinableWith(const SpeculativeDecoderInterface* other) const override;

 private:
  std::shared_ptr<PtpGenerator> generator_;
};

}  // namespace ksana_llm