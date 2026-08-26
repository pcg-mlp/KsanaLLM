/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "models/hy_v3/hy_v3_mtp_draft_model.h"

#include "models/hy_v3/hy_v3_model.h"
#include "modules/basic/layernorm.h"
#include "modules/basic/mhc.h"
#include "utils/logger.h"

namespace ksana_llm {

HyV3MtpDraftModel::HyV3MtpDraftModel(std::unordered_map<int, std::shared_ptr<HyV3MtpLayer>>* nextn_layers,
                                     Layernorm* nextn_shared_head_norm, Mhc* nextn_final_mhc,
                                     int16_t lower_nextn_layer_idx, int16_t upper_nextn_layer_idx)
    : nextn_layers_(nextn_layers),
      nextn_shared_head_norm_(nextn_shared_head_norm),
      nextn_final_mhc_(nextn_final_mhc),
      nextn_layer_idx_(lower_nextn_layer_idx),
      lower_nextn_layer_idx_(lower_nextn_layer_idx),
      upper_nextn_layer_idx_(upper_nextn_layer_idx) {
  KLLM_LOG_INFO << "HyV3MtpDraftModel created with " << nextn_layers_->size() << " MTP layers, layer range ["
                << lower_nextn_layer_idx_ << ", " << upper_nextn_layer_idx_ << "]";
}

HyV3MtpDraftModel::~HyV3MtpDraftModel() = default;

Status HyV3MtpDraftModel::ForwardLayer(std::vector<Tensor>& residual_buffer, ForwardingContext& forwarding_context) {
  if (nextn_layers_->empty()) {
    return Status(RET_INVALID_ARGUMENT, "No MTP layers available");
  }

  // Each MTP step consumes exactly one decoder layer.
  auto it = nextn_layers_->find(nextn_layer_idx_);
  if (it == nextn_layers_->end()) {
    return Status(RET_INVALID_ARGUMENT, fmt::format("MTP layer index {} not found", nextn_layer_idx_));
  }

  STATUS_CHECK_RETURN(it->second->Forward(residual_buffer, forwarding_context));

  nextn_layer_idx_++;
  if (nextn_layer_idx_ > upper_nextn_layer_idx_) {
    nextn_layer_idx_ = lower_nextn_layer_idx_;
  }
  return Status();
}

Status HyV3MtpDraftModel::ForwardMhcNormLmHead(std::vector<Tensor>& residual_buffer,
                                               std::shared_ptr<BaseWeight>& base_weight,
                                               ForwardingContext& forwarding_context) {
  if (nextn_final_mhc_) {
    nextn_final_mhc_->FinalForward(residual_buffer, residual_buffer);
  }
  if (nextn_shared_head_norm_) {
    nextn_shared_head_norm_->Forward(residual_buffer[0], residual_buffer[0]);
  }
  return Status();
}

}  // namespace ksana_llm
