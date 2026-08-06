/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include <unordered_map>

#include "models/base/draft_model_interface.h"

namespace ksana_llm {

// Forward declarations
class DeepSeekV3MtpLayer;
class Layernorm;
class Mhc;

// DeepSeekV3-specific draft model that encapsulates the MTP (Multi-Token Prediction)
// forward pipeline. Holds the nextn decoder layers, alternate norm, and alternate MHC
// that were previously owned by DeepSeekV3Model.
//
// This is the reference implementation of DraftModelInterface for DeepSeekV3 architecture.
// DeepSeekV4 and GLM4-MoE will have their own MtpDraftModel implementations.
//
// Uses non-owning references to layer/norm/MHC objects owned by the parent model,
// enabling parallel use by both the model's kNextN path and MtpDecoder.
class DeepSeekV3MtpDraftModel : public DraftModelInterface {
 public:
  DeepSeekV3MtpDraftModel(std::unordered_map<int, std::shared_ptr<DeepSeekV3MtpLayer>>* nextn_layers,
                          Layernorm* nextn_shared_head_norm, Mhc* nextn_final_mhc, int16_t lower_nextn_layer_idx,
                          int16_t upper_nextn_layer_idx);

  ~DeepSeekV3MtpDraftModel() override;

  // Execute the MTP forward pipeline for one step.
  // The MTP layer index (nextn_layer_idx_) is advanced internally and wraps around
  // when it reaches upper_nextn_layer_idx_.
  Status ForwardLayer(std::vector<Tensor>& residual_buffer, ForwardingContext& forwarding_context) override;
  Status ForwardMhcNormLmHead(std::vector<Tensor>& residual_buffer, std::shared_ptr<BaseWeight>& base_weight,
                              ForwardingContext& forwarding_context) override;

  // Number of MTP layers available.
  size_t GetLayerCount() const { return nextn_layers_->size(); }

 private:
  std::unordered_map<int, std::shared_ptr<DeepSeekV3MtpLayer>>* nextn_layers_;  // non-owning reference
  Layernorm* nextn_shared_head_norm_;                                           // non-owning reference
  Mhc* nextn_final_mhc_;                                                        // non-owning reference
  int16_t nextn_layer_idx_;  // Current MTP layer index, advances each Forward() call
  int16_t lower_nextn_layer_idx_;
  int16_t upper_nextn_layer_idx_;
};

}  // namespace ksana_llm