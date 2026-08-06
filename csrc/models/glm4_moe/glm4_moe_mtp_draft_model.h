/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include <unordered_map>

#include "models/base/draft_model_interface.h"

namespace ksana_llm {

// Forward declarations
class Glm4MoeMtpLayer;

// GLM4-MoE-specific draft model encapsulating MTP forward pipeline.
// GLM4-MoE uses the alternate norm and MHC from CommonModel (nextn_shared_head_norm_,
// nextn_final_mhc_) rather than owning its own copies.
//
// Uses a non-owning reference to nextn_layers_ owned by the parent model,
// enabling parallel use by both the model's kNextN path and MtpDecoder.
class Glm4MoeMtpDraftModel : public DraftModelInterface {
 public:
  Glm4MoeMtpDraftModel(std::unordered_map<int, std::shared_ptr<Glm4MoeMtpLayer>>* nextn_layers,
                       int16_t lower_nextn_layer_idx, int16_t upper_nextn_layer_idx);

  ~Glm4MoeMtpDraftModel() override;

  Status ForwardLayer(std::vector<Tensor>& residual_buffer, ForwardingContext& forwarding_context) override;
  Status ForwardMhcNormLmHead(std::vector<Tensor>& residual_buffer, std::shared_ptr<BaseWeight>& base_weight,
                              ForwardingContext& forwarding_context) override;

  size_t GetLayerCount() const { return nextn_layers_->size(); }

 private:
  std::unordered_map<int, std::shared_ptr<Glm4MoeMtpLayer>>* nextn_layers_;  // non-owning reference
  int16_t nextn_layer_idx_;
  int16_t lower_nextn_layer_idx_;
  int16_t upper_nextn_layer_idx_;
};

}  // namespace ksana_llm