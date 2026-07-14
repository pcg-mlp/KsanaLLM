/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "models/bge_reranker_minicpm/bge_reranker_minicpm_config_parser.h"

#include "configure/json_config_utils.h"
#include "utils/singleton.h"

namespace ksana_llm {

Status BgeRerankerMinicpmConfigParser::ParseModelConfig(const nlohmann::json &config_json,
                                                        const ParallelismBasicConfig &parallel_basic_config,
                                                        const std::string &model_dir,
                                                        std::shared_ptr<BaseModelConfig> &model_config) {
  auto bge_reranker_minicpm_model_config = std::make_shared<BgeRerankerMinicpmConfig>();
  model_config = bge_reranker_minicpm_model_config;

  bge_reranker_minicpm_model_config->weight_data_type = GetModelDataType(config_json);

  // Reference: https://huggingface.co/BAAI/bge-reranker-v2-minicpm-layerwise/blob/main/config.json
  bge_reranker_minicpm_model_config->head_num = config_json.value("num_attention_heads", 36);
  bge_reranker_minicpm_model_config->num_key_value_heads = config_json.value("num_key_value_heads", 36);
  bge_reranker_minicpm_model_config->inter_size = config_json.value("intermediate_size", 5760);
  bge_reranker_minicpm_model_config->vocab_size = config_json.value("vocab_size", 122753);
  bge_reranker_minicpm_model_config->num_layer = config_json.value("num_hidden_layers", 40);
  bge_reranker_minicpm_model_config->hidden_units = config_json.value("hidden_size", 2304);
  bge_reranker_minicpm_model_config->rope_theta = config_json.value("rope_theta", 10000.0f);
  bge_reranker_minicpm_model_config->layernorm_eps = config_json.value("rms_norm_eps", 1e-5);
  bge_reranker_minicpm_model_config->start_id = config_json.value("bos_token_id", 1);
  bge_reranker_minicpm_model_config->end_id = config_json.value("eos_token_id", 2);
  bge_reranker_minicpm_model_config->max_position_embeddings = config_json.value("max_position_embeddings", 2048);

  bge_reranker_minicpm_model_config->emb_scale = config_json.value("scale_emb", 1.0f);
  bge_reranker_minicpm_model_config->scale_depth = config_json.value("scale_depth", 1.0f);
  bge_reranker_minicpm_model_config->start_layer = config_json.value("start_layer", 1);

  const auto env = Singleton<Environment>::GetInstance();
  ParseQuantConfig(config_json, bge_reranker_minicpm_model_config, env->GetYamlWeightQuantMethod());
  return Status();
}

Status BgeRerankerMinicpmConfigParser::ParseQuantConfig(
    const nlohmann::json &quant_config_json,
    std::shared_ptr<BgeRerankerMinicpmConfig> bge_reranker_minicpm_model_config,
    const std::string &yaml_weight_quant_method) {
  // Handle quantization config from config.json
  bge_reranker_minicpm_model_config->is_quant = quant_config_json.contains("quantization_config");
  if (bge_reranker_minicpm_model_config->is_quant) {
    const std::string &quant_method = quant_config_json["quantization_config"].at("quant_method");
    if (quant_method == "fp8") {
      // Handle static FP8 quantization from checkpoint
      bge_reranker_minicpm_model_config->quant_config.method = QUANT_FP8_E4M3;
      bge_reranker_minicpm_model_config->quant_config.is_checkpoint_fp8_serialized = true;
      bge_reranker_minicpm_model_config->quant_config.is_activation_scheme_static =
          quant_config_json["quantization_config"].value("activation_scheme", "") == "static";
      KLLM_LOG_INFO << fmt::format(
          "using static fp8 quant model, is_checkpoint_fp8_serialized: {}, is_activation_scheme_static: {}",
          bge_reranker_minicpm_model_config->quant_config.is_checkpoint_fp8_serialized,
          bge_reranker_minicpm_model_config->quant_config.is_activation_scheme_static);
    } else {
      // TODO(jinxcwu) 支持所有量化类型后，需要修改回 KLLM_THROW
      KLLM_LOG_ERROR << fmt::format("Not support quant method: {}", quant_method);
    }
  } else if (yaml_weight_quant_method != "auto" && !yaml_weight_quant_method.empty()) {
    // Handle dynamic FP8 quantization from YAML config
    if (yaml_weight_quant_method == "fp8_e4m3") {
      bge_reranker_minicpm_model_config->is_quant = true;
      bge_reranker_minicpm_model_config->quant_config.method = QUANT_FP8_E4M3;
      bge_reranker_minicpm_model_config->quant_config.is_checkpoint_fp8_serialized = false;
      bge_reranker_minicpm_model_config->quant_config.is_activation_scheme_static = false;
      KLLM_LOG_INFO << fmt::format(
          "using dynamic fp8 quant model, quant method: {}, is_checkpoint_fp8_serialized: {}, "
          "is_activation_scheme_static: {}",
          yaml_weight_quant_method, bge_reranker_minicpm_model_config->quant_config.is_checkpoint_fp8_serialized,
          bge_reranker_minicpm_model_config->quant_config.is_activation_scheme_static);
    } else {
      KLLM_LOG_ERROR << fmt::format("Not support yaml quant_method: {}", yaml_weight_quant_method);
    }
  }
  return Status();
}

}  // namespace ksana_llm
