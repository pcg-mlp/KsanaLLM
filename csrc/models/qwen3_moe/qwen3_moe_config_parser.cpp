/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "models/qwen3_moe/qwen3_moe_config_parser.h"

#include "configure/environment.h"
#include "configure/json_config_utils.h"

#include "utils/singleton.h"
namespace ksana_llm {

Status Qwen3MoeConfigParser::ParseModelConfig(const nlohmann::json &config_json,
                                              const ParallelismBasicConfig &parallel_basic_config,
                                              const std::string &model_dir,
                                              std::shared_ptr<BaseModelConfig> &model_config) {
  std::shared_ptr<Qwen3MoeConfig> qwen3_moe_model_config = std::make_shared<Qwen3MoeConfig>();
  model_config = qwen3_moe_model_config;

  qwen3_moe_model_config->weight_data_type = GetModelDataType(config_json);

  // Reference: https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507/blob/main/config.json
  qwen3_moe_model_config->head_num = config_json.value("num_attention_heads", 32);
  qwen3_moe_model_config->num_key_value_heads = config_json.value("num_key_value_heads", 4);
  qwen3_moe_model_config->inter_size = config_json.value("intermediate_size", 6144);
  qwen3_moe_model_config->vocab_size = config_json.value("vocab_size", 151936);
  qwen3_moe_model_config->num_layer = config_json.value("num_hidden_layers", 48);
  qwen3_moe_model_config->hidden_units = config_json.value("hidden_size", 2048);
  qwen3_moe_model_config->rope_theta = config_json.value("rope_theta", 10000000.0f);
  qwen3_moe_model_config->layernorm_eps = config_json.value("rms_norm_eps", 1e-6);
  qwen3_moe_model_config->start_id = config_json.value("bos_token_id", 151643);
  qwen3_moe_model_config->end_id = config_json.value("eos_token_id", 151645);
  qwen3_moe_model_config->pad_id = config_json.value("pad_token_id", 0);
  qwen3_moe_model_config->max_position_embeddings = config_json.value("max_position_embeddings", 262144);

  if (!config_json.contains("tie_word_embeddings")) {
    qwen3_moe_model_config->exist_tie_embeddings_param = false;
  }
  qwen3_moe_model_config->tie_word_embeddings = config_json.value("tie_word_embeddings", false);
  qwen3_moe_model_config->is_visual = config_json.contains("visual");

  if (config_json.contains("head_dim")) {
    qwen3_moe_model_config->size_per_head = config_json.value("head_dim", 128);
  } else {  // for config.json which doesn't contain head_dim, init size_per_head during weight processing
    qwen3_moe_model_config->size_per_head = 0;
  }

  // Parse MoE config
  ParseMoeConfig(config_json, qwen3_moe_model_config);

  // Parse quantization config
  auto env = Singleton<Environment>::GetInstance();
  ParseQuantConfig(config_json, qwen3_moe_model_config, env->GetYamlWeightQuantMethod(), env->GetYamlGptqBackend());

  return Status();
}

Status Qwen3MoeConfigParser::ParseMoeConfig(const nlohmann::json &config_json,
                                            std::shared_ptr<Qwen3MoeConfig> qwen3_moe_config) {
  qwen3_moe_config->is_moe = true;

  // Reference: https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507/blob/main/config.json
  qwen3_moe_config->moe_config.num_experts = config_json.value("num_experts", 128);
  qwen3_moe_config->moe_config.experts_topk = config_json.value("num_experts_per_tok", 8);
  qwen3_moe_config->moe_config.moe_inter_size = config_json.value("moe_intermediate_size", 768);

  // Check if shared experts exist
  qwen3_moe_config->has_shared_experts = config_json.value("num_shared_experts", 0) > 0;
  if (qwen3_moe_config->has_shared_experts) {
    qwen3_moe_config->moe_config.num_shared_experts = config_json.value("num_shared_experts", 0);
  }

  // Parse routing strategy
  qwen3_moe_config->moe_config.router_aux_loss_coef = config_json.value("router_aux_loss_coef", 0.001f);
  qwen3_moe_config->moe_config.norm_topk_prob = config_json.value("norm_topk_prob", true);

  KLLM_LOG_INFO << fmt::format(
      "MoE Config: num_experts={}, num_experts_per_tok={}, moe_intermediate_size={}, "
      "has_shared_experts={}, norm_topk_prob={}",
      qwen3_moe_config->moe_config.num_experts, qwen3_moe_config->moe_config.experts_topk,
      qwen3_moe_config->moe_config.moe_inter_size, qwen3_moe_config->has_shared_experts,
      qwen3_moe_config->moe_config.norm_topk_prob);

  return Status();
}

Status Qwen3MoeConfigParser::ParseQuantConfig(const nlohmann::json &config_json,
                                              std::shared_ptr<Qwen3MoeConfig> qwen3_moe_config,
                                              const std::string &yaml_weight_quant_method,
                                              const std::string &yaml_gptq_backend) {
  // Handle quantization config from config.json
  qwen3_moe_config->is_quant = config_json.contains("quantization_config");
  if (qwen3_moe_config->is_quant) {
    std::string quant_method = config_json["quantization_config"].at("quant_method");
    if (quant_method == "fp8") {
      // Handle static FP8 quantization from checkpoint
      qwen3_moe_config->quant_config.method = QUANT_FP8_E4M3;
      qwen3_moe_config->quant_config.is_checkpoint_fp8_serialized = true;
      qwen3_moe_config->quant_config.is_activation_scheme_static =
          config_json["quantization_config"].value("activation_scheme", "") == "static";
      if (!qwen3_moe_config->quant_config.is_activation_scheme_static) {
        KLLM_LOG_ERROR << "dynamic activation quant scheme is not supported for qwen3-moe FP8 quantization.";
        return Status(RET_MODEL_LOAD_FAILED,
                      "dynamic activation quant scheme is not supported for qwen3-moe FP8 quantization.");
      }
      KLLM_LOG_INFO << fmt::format(
          "using static fp8 quant model, is_checkpoint_fp8_serialized: {}, is_activation_scheme_static: {}",
          qwen3_moe_config->quant_config.is_checkpoint_fp8_serialized,
          qwen3_moe_config->quant_config.is_activation_scheme_static);
    } else {
      KLLM_LOG_ERROR << fmt::format("unsupported quant method: {}", quant_method);
      return Status(RET_MODEL_LOAD_FAILED, fmt::format("unsupported quant method: {}", quant_method));
    }
  } else if (yaml_weight_quant_method != "auto" && !yaml_weight_quant_method.empty()) {
    KLLM_LOG_ERROR << fmt::format("unsupported quant method: {}", yaml_weight_quant_method);
    return Status(RET_MODEL_LOAD_FAILED, fmt::format("unsupported quant method: {}", yaml_weight_quant_method));
  }
  return Status();
}
}  // namespace ksana_llm