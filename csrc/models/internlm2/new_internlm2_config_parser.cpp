/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "models/internlm2/new_internlm2_config_parser.h"

#include <memory>

#include "configure/json_config_utils.h"
#include "models/internlm2/internlm2_config.h"

namespace ksana_llm {

Internlm2ConfigParser::Internlm2ConfigParser() {}

Internlm2ConfigParser::~Internlm2ConfigParser() {}

Status Internlm2ConfigParser::ParseModelConfig(const nlohmann::json &config_json,
                                                  const ParallelismBasicConfig &parallel_basic_config,
                                                  const std::string &model_dir,
                                                  std::shared_ptr<BaseModelConfig> &model_config) {
  std::shared_ptr<Internlm2Config> new_internlm2_config = std::make_shared<Internlm2Config>();
  model_config = new_internlm2_config;

  new_internlm2_config->weight_data_type = GetModelDataType(config_json);

  new_internlm2_config->head_num = config_json.value("num_attention_heads", 32);
  new_internlm2_config->num_key_value_heads = config_json.value("num_key_value_heads", 8);
  new_internlm2_config->inter_size = config_json.value("intermediate_size", 8960);
  new_internlm2_config->vocab_size = config_json.value("vocab_size", 92544);
  new_internlm2_config->num_layer = config_json.value("num_hidden_layers", 32);
  new_internlm2_config->hidden_units = config_json.value("hidden_size", 4096);
  new_internlm2_config->rope_theta = config_json.value("rope_theta", 1000000.0f);
  new_internlm2_config->layernorm_eps = config_json.value("rms_norm_eps", 1e-5f);
  new_internlm2_config->start_id = config_json.value("bos_token_id", 1);
  new_internlm2_config->end_id = config_json.value("eos_token_id", 2);
  new_internlm2_config->pad_id = config_json.value("pad_token_id", 2);
  new_internlm2_config->max_position_embeddings = config_json.value("max_position_embeddings", 32768);

  if (!config_json.contains("tie_word_embeddings")) {
    new_internlm2_config->exist_tie_embeddings_param = false;
  }
  new_internlm2_config->tie_word_embeddings = config_json.value("tie_word_embeddings", false);

  if (config_json.contains("head_dim")) {
    new_internlm2_config->size_per_head = config_json.value("head_dim", 128);
  } else {
    new_internlm2_config->size_per_head = new_internlm2_config->hidden_units / new_internlm2_config->head_num;
  }

  ParseQuantConfig(config_json, new_internlm2_config);

  return Status();
}

Status Internlm2ConfigParser::ParseQuantConfig(const nlohmann::json &config_json,
                                                  std::shared_ptr<Internlm2Config> config) {
  config->is_quant = config_json.contains("quantization_config");
  if (config->is_quant) {
    std::string quant_method = config_json["quantization_config"].value("quant_method", "");
    if (quant_method == "modelopt") {
      std::string algo = config_json["quantization_config"].value("quant_algo", "");
      if (algo == "W4A8_AWQ") {
        config->quant_config.method = QUANT_W4A8_AWQ;
        KLLM_LOG_INFO << "using quant model, quant method W4A8_AWQ";
      } else {
        KLLM_LOG_ERROR << fmt::format("only support W4A8_AWQ algo in modelopt, got: {}", algo);
      }
    } else {
      KLLM_LOG_ERROR << fmt::format("Not support quant method: {}", quant_method);
    }
  }
  return Status();
}

}  // namespace ksana_llm

