/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "models/qwen2_moe/new_qwen2_moe_config_parser.h"

#include "configure/json_config_utils.h"

namespace ksana_llm {

Status NewQwen2MoeConfigParser::ParseModelConfig(const nlohmann::json &config_json,
                                                 const ParallelismBasicConfig &parallel_basic_config,
                                                 const std::string &model_dir,
                                                 std::shared_ptr<BaseModelConfig> &model_config) {
  std::shared_ptr<NewQwen2MoeConfig> new_qwen2_moe_model_config = std::make_shared<NewQwen2MoeConfig>();
  model_config = new_qwen2_moe_model_config;

  new_qwen2_moe_model_config->weight_data_type = GetModelDataType(config_json);

  // Reference: https://huggingface.co/Qwen/Qwen2-57B-A14B-Instruct/blob/main/config.json
  new_qwen2_moe_model_config->head_num = config_json.value("num_attention_heads", 28);
  new_qwen2_moe_model_config->num_key_value_heads = config_json.value("num_key_value_heads", 4);
  new_qwen2_moe_model_config->inter_size = config_json.value("intermediate_size", 18944);
  new_qwen2_moe_model_config->vocab_size = config_json.value("vocab_size", 151936);
  new_qwen2_moe_model_config->num_layer = config_json.value("num_hidden_layers", 28);
  new_qwen2_moe_model_config->hidden_units = config_json.value("hidden_size", 3584);
  new_qwen2_moe_model_config->rope_theta = config_json.value("rope_theta", 1000000.0f);
  new_qwen2_moe_model_config->layernorm_eps = config_json.value("rms_norm_eps", 1e-6);
  new_qwen2_moe_model_config->start_id = config_json.value("bos_token_id", 151643);
  new_qwen2_moe_model_config->end_id = config_json.value("eos_token_id", 151643);
  new_qwen2_moe_model_config->pad_id = config_json.value("pad_token_id", 0);
  new_qwen2_moe_model_config->max_position_embeddings = config_json.value("max_position_embeddings", 32768);

  if (!config_json.contains("tie_word_embeddings")) {
    new_qwen2_moe_model_config->exist_tie_embeddings_param = false;
  }
  new_qwen2_moe_model_config->tie_word_embeddings = config_json.value("tie_word_embeddings", false);
  new_qwen2_moe_model_config->is_visual = config_json.contains("visual");

  if (config_json.contains("head_dim")) {
    new_qwen2_moe_model_config->size_per_head = config_json.value("head_dim", 128);
  } else {  // for config.json which doesn't contain head_dim, init size_per_head during weight processing
    new_qwen2_moe_model_config->size_per_head = 0;
  }

  // Parse MoE config
  ParseMoeConfig(config_json, new_qwen2_moe_model_config);

  return Status();
}

Status NewQwen2MoeConfigParser::ParseMoeConfig(const nlohmann::json &config_json,
                                               std::shared_ptr<NewQwen2MoeConfig> new_qwen2_moe_config) {
  new_qwen2_moe_config->is_moe = true;

  // Reference: https://huggingface.co/Qwen/Qwen2-57B-A14B-Instruct/blob/main/config.json
  new_qwen2_moe_config->moe_config.num_experts = config_json.value("num_experts", 64);
  new_qwen2_moe_config->moe_config.experts_topk = config_json.value("num_experts_per_tok", 8);
  new_qwen2_moe_config->moe_config.moe_inter_size = config_json.value("moe_intermediate_size", 2560);

  // Check if shared experts exist
  new_qwen2_moe_config->has_shared_experts = config_json.contains("shared_expert_intermediate_size");
  if (new_qwen2_moe_config->has_shared_experts) {
    new_qwen2_moe_config->shared_expert_intermediate_size = config_json.value("shared_expert_intermediate_size", 20480);
  }

  // Parse routing strategy
  new_qwen2_moe_config->moe_config.router_aux_loss_coef = config_json.value("router_aux_loss_coef", 0.001f);
  new_qwen2_moe_config->moe_config.norm_topk_prob = config_json.value("norm_topk_prob", false);

  KLLM_LOG_INFO << fmt::format(
      "MoE Config: num_experts={}, num_experts_per_tok={}, moe_intermediate_size={}, "
      "has_shared_experts={}, norm_topk_prob={}",
      new_qwen2_moe_config->moe_config.num_experts, new_qwen2_moe_config->moe_config.experts_topk,
      new_qwen2_moe_config->moe_config.moe_inter_size, new_qwen2_moe_config->has_shared_experts,
      new_qwen2_moe_config->moe_config.norm_topk_prob);

  return Status();
}

}  // namespace ksana_llm