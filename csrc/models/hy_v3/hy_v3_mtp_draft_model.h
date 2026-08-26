/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include <unordered_map>

#include "models/base/draft_model_interface.h"

namespace ksana_llm {

// Forward declarations
class HyV3MtpLayer;
class Layernorm;
class Mhc;

// hy_v3-specific draft model encapsulating the MTP forward pipeline.
//
// Unlike GLM4-MoE (which reuses the main model.norm), hy_v3 applies a per-MTP-layer
// final_layernorm before the shared head, so the norm is injected here the same way
// DeepSeek-V3 does (see HyV3Model::CreateLayers for where it comes from).
//
// Holds a non-owning reference to nextn_layers_ owned by the parent model.
class HyV3MtpDraftModel : public DraftModelInterface {
 public:
  HyV3MtpDraftModel(std::unordered_map<int, std::shared_ptr<HyV3MtpLayer>>* nextn_layers,
                    Layernorm* nextn_shared_head_norm, Mhc* nextn_final_mhc, int16_t lower_nextn_layer_idx,
                    int16_t upper_nextn_layer_idx);

  ~HyV3MtpDraftModel() override;

  Status ForwardLayer(std::vector<Tensor>& residual_buffer, ForwardingContext& forwarding_context) override;
  Status ForwardMhcNormLmHead(std::vector<Tensor>& residual_buffer, std::shared_ptr<BaseWeight>& base_weight,
                              ForwardingContext& forwarding_context) override;

 private:
  std::unordered_map<int, std::shared_ptr<HyV3MtpLayer>>* nextn_layers_;  // non-owning reference
  Layernorm* nextn_shared_head_norm_;                                     // non-owning, may be nullptr
  Mhc* nextn_final_mhc_;                                                  // non-owning, may be nullptr
  int16_t nextn_layer_idx_;
  int16_t lower_nextn_layer_idx_;
  int16_t upper_nextn_layer_idx_;
};

}  // namespace ksana_llm
