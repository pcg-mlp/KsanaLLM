/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "models/deepseek_v4/deepseek_v4_mtp_draft_model.h"

#include "models/deepseek_v4/deepseek_v4_model.h"
#include "utils/logger.h"

namespace ksana_llm {

DeepSeekV4MtpDraftModel::DeepSeekV4MtpDraftModel(
    std::unordered_map<int, std::shared_ptr<DeepSeekV4MtpLayer>>* nextn_layers, Layernorm* nextn_shared_head_norm,
    Mhc* nextn_final_mhc, int16_t lower_nextn_layer_idx, int16_t upper_nextn_layer_idx)
    : nextn_layers_(nextn_layers),
      nextn_shared_head_norm_(nextn_shared_head_norm),
      nextn_final_mhc_(nextn_final_mhc),
      nextn_layer_idx_(lower_nextn_layer_idx),
      lower_nextn_layer_idx_(lower_nextn_layer_idx),
      upper_nextn_layer_idx_(upper_nextn_layer_idx) {
  KLLM_LOG_INFO << "DeepSeekV4MtpDraftModel created with " << nextn_layers_->size() << " MTP layers"
                << ", layer range [" << lower_nextn_layer_idx_ << ", " << upper_nextn_layer_idx_ << "]";
}

DeepSeekV4MtpDraftModel::~DeepSeekV4MtpDraftModel() = default;

Status DeepSeekV4MtpDraftModel::ForwardLayer(std::vector<Tensor>& residual_buffer,
                                             ForwardingContext& forwarding_context) {
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

Status DeepSeekV4MtpDraftModel::ForwardMhcNormLmHead(std::vector<Tensor>& residual_buffer,
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