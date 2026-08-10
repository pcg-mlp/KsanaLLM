/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "models/gpt_moe/gpt_moe_config_parser.h"

#include "configure/json_config_utils.h"
#include "models/gpt_moe/gpt_moe_config.h"

namespace ksana_llm {

Status GptMoeConfigParser::ParseModelConfig(const nlohmann::json &config_json,
                                            const ParallelismBasicConfig &parallel_basic_config,
                                            const std::string &model_dir,
                                            std::shared_ptr<BaseModelConfig> &model_config) {
  auto gpt_moe_model_config = std::make_shared<GptMoeConfig>();
  model_config = gpt_moe_model_config;

  gpt_moe_model_config->weight_data_type = GetModelDataType(config_json);

  // Reference: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/configuration_gpt2.py
  gpt_moe_model_config->head_num = config_json.value("n_heads", 8);
  gpt_moe_model_config->num_key_value_heads = gpt_moe_model_config->head_num;
  gpt_moe_model_config->vocab_size = config_json.value("vocab_size", 6999);
  gpt_moe_model_config->num_layer = config_json.value("n_layer", 6);
  gpt_moe_model_config->hidden_units = config_json.value("n_embd", 512);
  gpt_moe_model_config->inter_size = config_json.value("n_inner", 4 * gpt_moe_model_config->hidden_units);
  gpt_moe_model_config->layernorm_eps = config_json.value("layer_norm_epsilon", 1e-5);  // torch.nn uses 1e-5 by default
  gpt_moe_model_config->start_id = config_json.value("bos_token_id", 0);
  gpt_moe_model_config->end_ids = std::vector<uint32_t>{static_cast<uint32_t>(config_json.value("eos_token_id", 2))};
  gpt_moe_model_config->pad_id = config_json.value("pad_token_id", 1);
  gpt_moe_model_config->max_position_embeddings = config_json.value("n_positions", 128);
  gpt_moe_model_config->activation_function = config_json.value("afn", "gelu");

  // Tie the weights of the token_embedding and the lm_head.
  gpt_moe_model_config->tie_word_embeddings = true;

  const uint32_t size_per_head = gpt_moe_model_config->hidden_units / gpt_moe_model_config->head_num;
  gpt_moe_model_config->size_per_head = size_per_head;
  gpt_moe_model_config->rotary_embedding = size_per_head;

  gpt_moe_model_config->is_moe = true;
  gpt_moe_model_config->moe_config.num_experts = config_json.value("num_experts", 8);
  gpt_moe_model_config->moe_config.moe_inter_size =
      config_json.value("moe_intermediate_size", 2 * gpt_moe_model_config->hidden_units);
  gpt_moe_model_config->moe_config.experts_topk = config_json.value("num_experts_per_tok", 2);
  KLLM_LOG_INFO << fmt::format("using moe model, num_experts: {}, moe_inter_size: {}, experts_topk: {}",
                               gpt_moe_model_config->moe_config.num_experts,
                               gpt_moe_model_config->moe_config.moe_inter_size,
                               gpt_moe_model_config->moe_config.experts_topk);

  return Status();
}

}  // namespace ksana_llm
