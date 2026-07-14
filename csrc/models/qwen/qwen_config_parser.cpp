/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "models/qwen/qwen_config_parser.h"

#include <memory>

#include "configure/json_config_utils.h"
#include "configure/quant_config_parser.h"
#include "models/qwen/qwen_config.h"
#include "utils/singleton.h"

namespace ksana_llm {

QwenConfigParser::QwenConfigParser() {}

QwenConfigParser::~QwenConfigParser() {}

Status QwenConfigParser::ParseModelConfig(const nlohmann::json &config_json,
                                             const ParallelismBasicConfig &parallel_basic_config,
                                             const std::string &model_dir,
                                             std::shared_ptr<BaseModelConfig> &model_config) {
  std::shared_ptr<QwenConfig> qwen_model_config = std::make_shared<QwenConfig>();
  model_config = qwen_model_config;

  qwen_model_config->weight_data_type = GetModelDataType(config_json);

  // Use Qwen3-32B config as default values
  qwen_model_config->head_num = config_json.value("num_attention_heads", 64);
  qwen_model_config->num_key_value_heads = config_json.value("num_key_value_heads", 8);
  qwen_model_config->inter_size = config_json.value("intermediate_size", 25600);
  qwen_model_config->vocab_size = config_json.value("vocab_size", 151936);
  qwen_model_config->num_layer = config_json.value("num_hidden_layers", 64);
  qwen_model_config->hidden_units = config_json.value("hidden_size", 5120);
  qwen_model_config->rope_theta = config_json.value("rope_theta", 1000000.0f);
  qwen_model_config->layernorm_eps = config_json.value("rms_norm_eps", 1e-6);
  qwen_model_config->start_id = config_json.value("bos_token_id", 151643);
  qwen_model_config->end_id = config_json.value("eos_token_id", 151645);
  qwen_model_config->pad_id = config_json.value("pad_token_id", 0);
  qwen_model_config->max_position_embeddings = config_json.value("max_position_embeddings", 40960);
  if (!config_json.contains("tie_word_embeddings")) {
    qwen_model_config->exist_tie_embeddings_param = false;
  }
  qwen_model_config->tie_word_embeddings = config_json.value("tie_word_embeddings", false);
  qwen_model_config->is_visual = config_json.contains("visual");

  if (config_json.contains("head_dim")) {
    qwen_model_config->size_per_head = config_json.value("head_dim", 128);
  } else {  // for config.json which doesn't contain head_dim, init size_per_head during weight processing
    qwen_model_config->size_per_head = 0;
  }

  // parse quantization config
  auto env = Singleton<Environment>::GetInstance();
  ParseQuantConfig(config_json, qwen_model_config, env->GetYamlWeightQuantMethod(), env->GetYamlGptqBackend());
  return Status();
}

void ParseModelOptQuantConfig(const nlohmann::json &config_json, std::shared_ptr<QwenConfig> qwen_config,
                              QuantConfig &quant_config) {
  std::string algo = config_json.value("quant_algo", "");
  if (algo == "W4A8_AWQ") {
    quant_config.method = QUANT_W4A8_AWQ;
    KLLM_LOG_INFO << "using quant model, quant method W4A8_AWQ";
  } else {
    // TODO(jinxcwu) 支持所有量化类型后，需要修改回 KLLM_THROW
    KLLM_LOG_ERROR << "only support W4A8_AWQ algo in modelopt";
  }
}

Status QwenConfigParser::ParseQuantConfig(const nlohmann::json &config_json,
                                             std::shared_ptr<QwenConfig> qwen_config,
                                             const std::string &yaml_weight_quant_method,
                                             const std::string &yaml_gptq_backend) {
  // Handle quantization config from config.json
  qwen_config->is_quant = config_json.contains("quantization_config");
  if (qwen_config->is_quant) {
    std::string quant_method = config_json["quantization_config"].at("quant_method");
    if (quant_method == "modelopt") {
      ParseModelOptQuantConfig(config_json["quantization_config"], qwen_config, qwen_config->quant_config);
    } else if (quant_method == "gptq") {
      ParseGPTQQuantConfig(config_json["quantization_config"], qwen_config->is_moe, qwen_config->quant_config);
    } else if (quant_method == "awq") {
      // NOTE: Qwen dense model, so is_moe = false.
      ParseAWQQuantConfig(config_json["quantization_config"], /*is_moe=*/false, qwen_config->quant_config);
    } else if (quant_method == "fp8") {
      // Handle static FP8 quantization from checkpoint
      qwen_config->quant_config.method = QUANT_FP8_E4M3;
      qwen_config->quant_config.is_checkpoint_fp8_serialized = true;
      qwen_config->quant_config.is_activation_scheme_static =
          config_json["quantization_config"].value("activation_scheme", "") == "static";
      KLLM_LOG_INFO << fmt::format(
          "using static fp8 quant model, is_checkpoint_fp8_serialized: {}, is_activation_scheme_static: {}",
          qwen_config->quant_config.is_checkpoint_fp8_serialized,
          qwen_config->quant_config.is_activation_scheme_static);
    } else {
      // TODO(jinxcwu) 支持所有量化类型后，需要修改回 KLLM_THROW
      KLLM_LOG_ERROR << fmt::format("Not support quant method: {}", quant_method);
    }
    // Choice quant backends
    if (qwen_config->quant_config.method == QUANT_GPTQ && qwen_config->quant_config.desc_act == true) {
      qwen_config->quant_config.backend = MARLIN_LINEAR_BACKEND;
      KLLM_LOG_INFO << "Using MARLIN Quant Backend, only support MARLIN backend in desc_act mode";
    } else if (qwen_config->quant_config.method == QUANT_GPTQ || qwen_config->quant_config.method == QUANT_AWQ) {
      if (yaml_gptq_backend == "cutlass") {
        qwen_config->quant_config.backend = CUTLASS_LINEAR_BACKEND;
        KLLM_LOG_INFO << "Using CUTLASS Quant Backend";
      } else if (yaml_gptq_backend == "marlin") {
        qwen_config->quant_config.backend = MARLIN_LINEAR_BACKEND;
        KLLM_LOG_INFO << "Using MARLIN Quant Backend";
      } else {
        KLLM_THROW(fmt::format("Not support quant backend {}.", yaml_gptq_backend));
      }
    }
  } else if (yaml_weight_quant_method != "auto" && !yaml_weight_quant_method.empty()) {
    // Handle dynamic FP8 quantization from YAML config
    if (yaml_weight_quant_method == "fp8_e4m3") {
      qwen_config->is_quant = true;
      qwen_config->quant_config.method = QUANT_FP8_E4M3;
      qwen_config->quant_config.is_checkpoint_fp8_serialized = false;
      qwen_config->quant_config.is_activation_scheme_static = false;
      KLLM_LOG_INFO << fmt::format(
          "using dynamic fp8 quant model, quant method: {}, is_checkpoint_fp8_serialized: {}, "
          "is_activation_scheme_static: {}",
          yaml_weight_quant_method, qwen_config->quant_config.is_checkpoint_fp8_serialized,
          qwen_config->quant_config.is_activation_scheme_static);
    } else {
      KLLM_LOG_ERROR << fmt::format("Not support yaml quant_method: {}", yaml_weight_quant_method);
    }
  }
  return Status();
}
}  // namespace ksana_llm
