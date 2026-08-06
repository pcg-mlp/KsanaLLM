/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "models/glm4_moe/glm4_moe_mtp_draft_model.h"

#include "models/glm4_moe/glm4_moe_model.h"
#include "utils/logger.h"

namespace ksana_llm {

Glm4MoeMtpDraftModel::Glm4MoeMtpDraftModel(std::unordered_map<int, std::shared_ptr<Glm4MoeMtpLayer>>* nextn_layers,
                                           int16_t lower_nextn_layer_idx, int16_t upper_nextn_layer_idx)
    : nextn_layers_(nextn_layers),
      nextn_layer_idx_(lower_nextn_layer_idx),
      lower_nextn_layer_idx_(lower_nextn_layer_idx),
      upper_nextn_layer_idx_(upper_nextn_layer_idx) {
  KLLM_LOG_INFO << "Glm4MoeMtpDraftModel created with " << nextn_layers_->size() << " MTP layers"
                << ", layer range [" << lower_nextn_layer_idx_ << ", " << upper_nextn_layer_idx_ << "]";
}

Glm4MoeMtpDraftModel::~Glm4MoeMtpDraftModel() = default;

Status Glm4MoeMtpDraftModel::ForwardLayer(std::vector<Tensor>& residual_buffer, ForwardingContext& forwarding_context) {
  if (nextn_layers_->empty()) {
    return Status(RET_INVALID_ARGUMENT, "No MTP layers available");
  }

  auto it = nextn_layers_->find(nextn_layer_idx_);
  if (it == nextn_layers_->end()) {
    return Status(RET_INVALID_ARGUMENT, "MTP layer not found");
  }

  STATUS_CHECK_RETURN(it->second->Forward(residual_buffer, forwarding_context));

  nextn_layer_idx_++;
  if (nextn_layer_idx_ > upper_nextn_layer_idx_) {
    nextn_layer_idx_ = lower_nextn_layer_idx_;
  }

  return Status();
}

Status Glm4MoeMtpDraftModel::ForwardMhcNormLmHead(std::vector<Tensor>& residual_buffer,
                                                  std::shared_ptr<BaseWeight>& base_weight,
                                                  ForwardingContext& forwarding_context) {
  return Status();
}

}  // namespace ksana_llm