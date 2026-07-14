/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "models/mixtral/mixtral_config_parser.h"

#include "configure/json_config_utils.h"

namespace ksana_llm {

MixtralConfigParser::MixtralConfigParser() {}

MixtralConfigParser::~MixtralConfigParser() {}

Status MixtralConfigParser::ParseModelConfig(const nlohmann::json &config_json,
                                             const ParallelismBasicConfig &parallel_basic_config,
                                             const std::string &model_dir,
                                             std::shared_ptr<BaseModelConfig> &model_config) {
  std::shared_ptr<MixtralConfig> mixtral_config = std::make_shared<MixtralConfig>();
  model_config = mixtral_config;

  mixtral_config->weight_data_type = GetModelDataType(config_json);

  mixtral_config->head_num = config_json.value("num_attention_heads", 32);
  mixtral_config->num_key_value_heads = config_json.value("num_key_value_heads", 8);
  mixtral_config->inter_size = config_json.value("intermediate_size", 14336);
  mixtral_config->vocab_size = config_json.value("vocab_size", 32000);
  mixtral_config->num_layer = config_json.value("num_hidden_layers", 32);
  mixtral_config->hidden_units = config_json.value("hidden_size", 4096);
  mixtral_config->rope_theta = config_json.value("rope_theta", 1000000.0f);
  mixtral_config->layernorm_eps = config_json.value("rms_norm_eps", 1e-5);
  mixtral_config->start_id = config_json.value("bos_token_id", 1);
  mixtral_config->end_id = config_json.value("eos_token_id", 2);
  mixtral_config->pad_id = config_json.value("pad_token_id", 0);
  mixtral_config->max_position_embeddings = config_json.value("max_position_embeddings", 32768);
  mixtral_config->activation_function = config_json.value("hidden_act", "silu");
  mixtral_config->initializer_range = config_json.value("initializer_range", 0.02f);

  if (!config_json.contains("tie_word_embeddings")) {
    mixtral_config->exist_tie_embeddings_param = false;
  }
  mixtral_config->tie_word_embeddings = config_json.value("tie_word_embeddings", false);

  size_t size_per_head = config_json.value("head_dim", mixtral_config->hidden_units / mixtral_config->head_num);
  mixtral_config->size_per_head = size_per_head;

  // Parse MoE config
  ParseMoeConfig(config_json, mixtral_config);

  return Status();
}

Status MixtralConfigParser::ParseMoeConfig(const nlohmann::json &config_json,
                                           std::shared_ptr<MixtralConfig> mixtral_config) {
  mixtral_config->moe_config.num_experts = config_json.value("num_local_experts", 8);
  if (mixtral_config->moe_config.num_experts > 1) {
    mixtral_config->is_moe = true;
    mixtral_config->has_shared_experts = false;
    mixtral_config->moe_config.moe_inter_size = mixtral_config->inter_size;  // for mixtral model
    mixtral_config->moe_config.experts_topk = config_json.value("num_experts_per_tok", 2);
    mixtral_config->moe_config.output_router_logits = config_json.value("output_router_logits", false);
    mixtral_config->moe_config.router_aux_loss_coef = config_json.value("router_aux_loss_coef", 0.02);
    KLLM_LOG_INFO << fmt::format("Using moe model, num_experts: {}, experts_topk: {}",
                                 mixtral_config->moe_config.num_experts, mixtral_config->moe_config.experts_topk);
  }

  mixtral_config->moe_config.apply_weight = false;
  return Status();
}

}  // namespace ksana_llm
