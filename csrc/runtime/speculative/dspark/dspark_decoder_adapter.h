/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>

#include "runtime/speculative/common/speculative_decoder_interface.h"

namespace ksana_llm {

// Forward declarations
class DSparkDecoder;
class ModelExecutor;
class ModelSampler;
class ModelBuilder;
class ModelInstance;

// Adapter that wraps DSparkDecoder as a SpeculativeDecoderInterface plugin.
// Registered with CompositeSpeculativeDecoder alongside MTP/PTP/Trie adapters
// to unify the speculative decoding architecture.
class DSparkDecoderAdapter : public SpeculativeDecoderInterface {
 public:
  DSparkDecoderAdapter(std::shared_ptr<DSparkDecoder> dspark_decoder, ModelExecutor* model_executor,
                       ModelSampler* model_sampler, ModelBuilder* model_builder, ModelInstance* model_instance);
  ~DSparkDecoderAdapter() override = default;

  // SpeculativeDecoderInterface implementation
  const char* GetName() const override { return "DSpark"; }
  SpecSchedulingInfo GetSchedulingInfo() const override;
  void PreMainForward(SpecContext& ctx) override;
  void BeforeForward(SpecContext&) override {}
  void ExecuteForward(SpecContext& ctx) override;
  void AfterForward(SpecContext&) override {}
  void Verify(SpecContext& ctx) override;
  bool IsCombinableWith(const SpeculativeDecoderInterface* other) const override;
  void Shutdown() override;

 private:
  std::shared_ptr<DSparkDecoder> dspark_decoder_;
  ModelExecutor* model_executor_;  // Non-owning
  ModelSampler* model_sampler_;    // Non-owning
  ModelBuilder* model_builder_;    // Non-owning
  ModelInstance* model_instance_;  // Non-owning
};

}  // namespace ksana_llm
