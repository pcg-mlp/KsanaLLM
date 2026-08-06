/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include <unordered_map>

#include "models/base/draft_model_interface.h"

namespace ksana_llm {

// Forward declarations
class DeepSeekV4MtpLayer;
class Layernorm;
class Mhc;

// DeepSeekV4-specific draft model encapsulating MTP forward pipeline.
// Uses non-owning references to layer/norm/MHC objects owned by the parent model,
// enabling parallel use by both the model's kNextN path and MtpDecoder.
class DeepSeekV4MtpDraftModel : public DraftModelInterface {
 public:
  DeepSeekV4MtpDraftModel(std::unordered_map<int, std::shared_ptr<DeepSeekV4MtpLayer>>* nextn_layers,
                          Layernorm* nextn_shared_head_norm, Mhc* nextn_final_mhc, int16_t lower_nextn_layer_idx,
                          int16_t upper_nextn_layer_idx);

  ~DeepSeekV4MtpDraftModel() override;

  Status ForwardLayer(std::vector<Tensor>& residual_buffer, ForwardingContext& forwarding_context) override;
  Status ForwardMhcNormLmHead(std::vector<Tensor>& residual_buffer, std::shared_ptr<BaseWeight>& base_weight,
                              ForwardingContext& forwarding_context) override;

  size_t GetLayerCount() const { return nextn_layers_->size(); }

 private:
  std::unordered_map<int, std::shared_ptr<DeepSeekV4MtpLayer>>* nextn_layers_;  // non-owning reference
  Layernorm* nextn_shared_head_norm_;                                           // non-owning reference
  Mhc* nextn_final_mhc_;                                                        // non-owning reference
  int16_t nextn_layer_idx_;
  int16_t lower_nextn_layer_idx_;
  int16_t upper_nextn_layer_idx_;
};

}  // namespace ksana_llm