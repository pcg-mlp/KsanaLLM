/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>

#include "runtime/speculative/common/speculative_decoder_interface.h"

namespace ksana_llm {

// Forward declarations
class ModelInstance;
class MtpDecoder;
class ModelExecutor;
class ModelSampler;
class ModelBuilder;

// Adapter that wraps MtpDecoder as a SpeculativeDecoderInterface plugin.
// Registered with CompositeSpeculativeDecoder alongside PTP/Trie adapters
// to unify the speculative decoding architecture.
class MtpDecoderAdapter : public SpeculativeDecoderInterface {
 public:
  explicit MtpDecoderAdapter(std::shared_ptr<MtpDecoder> mtp_decoder, ModelExecutor* model_executor,
                             ModelSampler* model_sampler, ModelBuilder* model_builder);
  ~MtpDecoderAdapter() override = default;

  // SpeculativeDecoderInterface implementation
  const char* GetName() const override { return "MTP"; }
  SpecSchedulingInfo GetSchedulingInfo() const override;
  void BeforeForward(SpecContext& ctx) override;
  void ExecuteForward(SpecContext& ctx) override;
  void AfterForward(SpecContext& ctx) override;
  void Verify(SpecContext& ctx) override;
  bool IsCombinableWith(const SpeculativeDecoderInterface* other) const override;

  // Called before main forward to handle MTP serialization (Wait/Reset)
  // and hidden buffer injection.
  void PreMainForward(SpecContext& ctx) override;

  // Wake the prev_mtp_done_ Waiter to unblock the main forward thread during stop.
  void Shutdown() override;

 private:
  std::shared_ptr<MtpDecoder> mtp_decoder_;
  ModelExecutor* model_executor_;  // Non-owning, stable across iterations
  ModelSampler* model_sampler_;    // Non-owning, stable across iterations
  ModelBuilder* model_builder_;    // Non-owning, stable across iterations
};

}  // namespace ksana_llm
