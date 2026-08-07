/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "models/base/base_weight.h"

#include <future>
#include <regex>

#include "configure/environment.h"
#include "runtime/threadpool.h"
#include "utils/singleton.h"

namespace ksana_llm {

BaseWeight::BaseWeight(const ModelConfig& model_config, const RuntimeConfig& runtime_config, int rank,
                       std::shared_ptr<Context> context)
    : rank_(rank), context_(context), model_config_(model_config), runtime_config_(runtime_config) {
  Singleton<Environment>::GetInstance()->GetPipelineConfig(pipeline_config_);

  // extract required layer idx
  for (auto idx = pipeline_config_.lower_layer_idx; idx <= pipeline_config_.upper_layer_idx; ++idx) {
    required_layer_idx_.all.emplace(idx);
  }
  KLLM_LOG_INFO << "CommonWeight is_standalone::" << context_->global->IsStandalone()
                << ", is_distributed_master:" << context_->global->IsDistributedMaster() << ", layer:["
                << pipeline_config_.lower_layer_idx << ", " << pipeline_config_.upper_layer_idx << "].";
  // add nextn predict layers
  if (static_cast<int>(pipeline_config_.lower_nextn_layer_idx) >= static_cast<int>(model_config_.num_layer)) {
    for (auto idx = pipeline_config_.lower_nextn_layer_idx; idx <= pipeline_config_.upper_nextn_layer_idx; ++idx) {
      required_layer_idx_.all.emplace(idx);
    }
    KLLM_LOG_INFO << "CommonWeight is_standalone::" << context_->global->IsStandalone()
                  << ", is_distributed_master:" << context_->global->IsDistributedMaster() << ", nextn layer:["
                  << pipeline_config_.lower_nextn_layer_idx << ", " << pipeline_config_.upper_nextn_layer_idx << "].";
  }
  // extract dense and moe layer idx
  const std::vector<size_t>& moe_layers = model_config_.moe_config.moe_layers;
  for (const auto idx : required_layer_idx_.all) {
    if (model_config_.is_moe && idx >= static_cast<int>(model_config_.moe_config.first_k_dense_replace) &&
        (moe_layers.empty() || std::find(moe_layers.begin(), moe_layers.end(), idx) != moe_layers.end())) {
      required_layer_idx_.moe.emplace(idx);
    } else {
      required_layer_idx_.dense.emplace(idx);
    }
  }
}

}  // namespace ksana_llm
