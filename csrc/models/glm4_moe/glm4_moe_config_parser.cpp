/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "models/glm4_moe/glm4_moe_config_parser.h"

#include "configure/environment.h"
#include "configure/json_config_utils.h"
#include "configure/quant_config_parser.h"
#include "utils/singleton.h"

namespace ksana_llm {

Status Glm4MoeConfigParser::ParseModelConfig(const nlohmann::json &config_json,
                                             const ParallelismBasicConfig &parallel_basic_config,
                                             const std::string &model_dir,
                                             std::shared_ptr<BaseModelConfig> &model_config) {
  std::shared_ptr<Glm4MoeConfig> glm4_moe_config = std::make_shared<Glm4MoeConfig>();
  model_config = glm4_moe_config;

  glm4_moe_config->model_dir = model_dir;
  glm4_moe_config->weight_data_type = GetModelDataType(config_json);

  // Reference: GLM4-MoE config.json
  glm4_moe_config->head_num = config_json.value("num_attention_heads", 96);
  glm4_moe_config->num_key_value_heads = config_json.value("num_key_value_heads", 8);
  glm4_moe_config->inter_size = config_json.value("intermediate_size", 12288);
  glm4_moe_config->vocab_size = config_json.value("vocab_size", 151552);
  glm4_moe_config->num_layer = config_json.value("num_hidden_layers", 92);
  glm4_moe_config->num_nextn_predict_layers = config_json.value("num_nextn_predict_layers", 0);
  glm4_moe_config->hidden_units = config_json.value("hidden_size", 5120);
  glm4_moe_config->rope_theta = config_json.value("rope_theta", 1000000.0f);
  glm4_moe_config->layernorm_eps = config_json.value("rms_norm_eps", 1e-5f);
  glm4_moe_config->layernorm_eps = config_json.value("layer_norm_epsilon", glm4_moe_config->layernorm_eps);
  glm4_moe_config->start_id = config_json.value("bos_token_id", 1);

  // Handle eos_token_id - could be array or single value
  if (config_json.contains("eos_token_id")) {
    if (config_json["eos_token_id"].is_array()) {
      glm4_moe_config->end_ids = config_json["eos_token_id"].get<std::vector<uint32_t>>();
      if (!glm4_moe_config->end_ids.empty()) {
        glm4_moe_config->end_id = glm4_moe_config->end_ids[0];
      }
    } else {
      glm4_moe_config->end_id = config_json.value("eos_token_id", 2);
      glm4_moe_config->end_ids = std::vector<uint32_t>{glm4_moe_config->end_id};
    }
  } else {
    glm4_moe_config->end_id = 2;
    glm4_moe_config->end_ids = std::vector<uint32_t>{2};
  }

  glm4_moe_config->pad_id = config_json.value("pad_token_id", 151329);
  glm4_moe_config->max_position_embeddings = config_json.value("max_position_embeddings", 202752);

  if (!config_json.contains("tie_word_embeddings")) {
    glm4_moe_config->exist_tie_embeddings_param = false;
  }
  glm4_moe_config->tie_word_embeddings = config_json.value("tie_word_embeddings", false);
  glm4_moe_config->is_visual = config_json.contains("visual");

  // GLM4-MoE specific: use head_dim from config if available
  if (config_json.contains("head_dim")) {
    glm4_moe_config->size_per_head = config_json.value("head_dim", 128);
  } else {
    glm4_moe_config->size_per_head = glm4_moe_config->hidden_units / glm4_moe_config->head_num;
  }

  // Parse partial rotary factor (GLM4-MoE uses partial rotary embedding)
  glm4_moe_config->partial_rotary_factor = config_json.value("partial_rotary_factor", 0.5f);
  glm4_moe_config->rotary_embedding =
      static_cast<uint32_t>(glm4_moe_config->size_per_head * glm4_moe_config->partial_rotary_factor);

  // Parse MoE config
  ParseMoeConfig(config_json, glm4_moe_config);

  // GLM4-MoE specific features
  glm4_moe_config->use_qk_norm = config_json.value("use_qk_norm", true);
  glm4_moe_config->enable_add_qkv_bias = config_json.value("attention_bias", true);
  glm4_moe_config->load_bias = true;  // GLM4-MoE has bias on QKV projections

  // Parse quantization config
  auto env = Singleton<Environment>::GetInstance();
  ParseQuantConfig(config_json, glm4_moe_config, env->GetYamlWeightQuantMethod(), env->GetYamlGptqBackend());

  return Status();
}

Status Glm4MoeConfigParser::ParseMoeConfig(const nlohmann::json &config_json,
                                           std::shared_ptr<Glm4MoeConfig> glm4_moe_config) {
  glm4_moe_config->is_moe = true;

  // Reference: GLM4-MoE config.json
  glm4_moe_config->moe_config.num_experts = config_json.value("n_routed_experts", 160);
  glm4_moe_config->moe_config.experts_topk = config_json.value("num_experts_per_tok", 8);
  glm4_moe_config->moe_config.moe_inter_size = config_json.value("moe_intermediate_size", glm4_moe_config->inter_size);
  glm4_moe_config->moe_config.first_k_dense_replace = config_json.value("first_k_dense_replace", 3);

  // For moe group topk config
  glm4_moe_config->moe_config.num_expert_group = config_json.value("n_group", 1);
  glm4_moe_config->moe_config.expert_groups_topk = config_json.value("topk_group", 1);

  // GLM4-MoE uses Sigmoid activation for router
  // GLM4-MoE uses noaux_tc topk_method (same as DeepSeek V3)
  // When use_vllm_moe=true, the topk is computed by InvokeGroupedTopk
  glm4_moe_config->moe_config.scoring_func = "sigmoid";
  glm4_moe_config->moe_config.topk_method = config_json.value("topk_method", "fast");
  glm4_moe_config->moe_config.norm_topk_prob = config_json.value("norm_topk_prob", true);
  glm4_moe_config->moe_config.routed_scaling_factor = config_json.value("routed_scaling_factor", 2.5f);

  // GLM4-MoE always uses e_score_correction_bias
  glm4_moe_config->moe_config.use_e_score_correction_bias = true;
  glm4_moe_config->moe_config.use_vllm_moe = true;

  // Shared experts config
  // Note: The quantized model (GLM-4.7-W4AFP8) uses moe_intermediate_size for shared_expert
  // This is different from the original model config where intermediate_size should be used
  glm4_moe_config->moe_config.num_shared_experts = config_json.value("n_shared_experts", 1);
  if (glm4_moe_config->moe_config.num_shared_experts > 0) {
    glm4_moe_config->has_shared_experts = true;
    // Use moe_inter_size to match the quantized model weights (1536)
    glm4_moe_config->moe_config.shared_expert_inter_size =
        glm4_moe_config->moe_config.num_shared_experts * glm4_moe_config->moe_config.moe_inter_size;
    glm4_moe_config->shared_expert_intermediate_size = glm4_moe_config->moe_config.shared_expert_inter_size;
  }

  return Status();
}

Status Glm4MoeConfigParser::ParseQuantConfig(const nlohmann::json &config_json,
                                             std::shared_ptr<Glm4MoeConfig> glm4_moe_config,
                                             const std::string &yaml_weight_quant_method,
                                             const std::string &yaml_gptq_backend) {
  // Parse quantize_config.json and hf_quant_config.json
  nlohmann::json quantize_config = ReadJsonFromFile(glm4_moe_config->model_dir + "/quantize_config.json");
  nlohmann::json hf_quant_config = ReadJsonFromFile(glm4_moe_config->model_dir + "/hf_quant_config.json");

  // Parse three different cases
  nlohmann::json quantization_config;
  if (config_json.contains("quantization_config")) {
    // config.json has quantization_config field (default case)
    glm4_moe_config->is_quant = true;
    quantization_config = config_json["quantization_config"];
  } else if (!quantize_config.empty()) {
    // config.json doesn't have quantization_config, but quantize_config.json exists
    glm4_moe_config->is_quant = true;
    quantization_config = quantize_config;
  } else if (!hf_quant_config.empty()) {
    // config.json doesn't have quantization_config, no quantize_config.json, but hf_quant_config.json exists
    glm4_moe_config->is_quant = true;
    quantization_config = ParseAndConvertQuantConfig(hf_quant_config);
  }

  if (glm4_moe_config->is_quant) {
    std::string quant_method = quantization_config.at("quant_method");

    if (quant_method == "gptq" || quant_method == "rtn") {
      ParseGPTQQuantConfig(quantization_config, glm4_moe_config->is_moe, glm4_moe_config->quant_config);
      // For W4A8 quantization (RTN with input_scale=true), use QUANT_W4A8_AWQ instead of QUANT_GPTQ
      if (glm4_moe_config->quant_config.input_scale) {
        glm4_moe_config->quant_config.method = QUANT_W4A8_AWQ;
      }
    } else if (quant_method == "fp8") {
      ParseFP8QuantConfig(quantization_config, glm4_moe_config->is_moe, glm4_moe_config->quant_config);
    } else if (quant_method == "mixed") {
      auto configs = quantization_config["configs"];
      for (auto it = configs.begin(); it != configs.end(); ++it) {
        QuantConfig quant_config;
        quant_method = quantization_config["configs"][it.key()]["method"];
        if (quant_method == "gptq" || quant_method == "rtn") {
          ParseGPTQQuantConfig(quantization_config["configs"][it.key()], glm4_moe_config->is_moe, quant_config);
          // For W4A8 quantization (RTN with input_scale=true), use QUANT_W4A8_AWQ instead of QUANT_GPTQ
          if (quant_config.input_scale) {
            quant_config.method = QUANT_W4A8_AWQ;
          }
        } else if (quant_method == "fp8") {
          ParseFP8QuantConfig(quantization_config["configs"][it.key()], glm4_moe_config->is_moe, quant_config);
        } else {
          KLLM_THROW(fmt::format("Not support quant_method {}.", quant_method));
        }
        if (!quantization_config.contains("layer_mapping") ||
            !quantization_config["layer_mapping"].contains(it.key())) {
          KLLM_THROW(
              fmt::format("quantization_config.json missing 'layer_mapping' entry for config key '{}'", it.key()));
        }
        auto layer_mapping = quantization_config["layer_mapping"][it.key()];
        quant_config.pattern_layers = layer_mapping["pattern_layers"].get<std::vector<std::string>>();
        quant_config.ignored_layers = layer_mapping["ignored_layers"].get<std::vector<std::string>>();
        if (layer_mapping["default_config"]) {
          glm4_moe_config->quant_config = quant_config;
        } else {
          glm4_moe_config->sub_quant_configs.push_back(quant_config);
        }
      }
      // Enable moe_int4 flag when we have mixed quantization with W4A8_AWQ for expert layers
      // This is needed for the MoE layer factory to correctly select the int4 backend
      for (const auto &sub_config : glm4_moe_config->sub_quant_configs) {
        if (Glm4MoeConfig::IsW4A8Method(sub_config.method)) {
          for (const auto &pattern : sub_config.pattern_layers) {
            if (pattern.find(".mlp.experts.") != std::string::npos) {
              glm4_moe_config->quant_config.enable_moe_int4 = true;
              break;
            }
          }
          if (glm4_moe_config->quant_config.enable_moe_int4) break;
        }
      }
    } else {
      KLLM_THROW(fmt::format("Not support quant method: {}", quant_method));
    }
  } else if (yaml_weight_quant_method != "auto" && !yaml_gptq_backend.empty()) {
    if (glm4_moe_config->is_moe) {
      KLLM_THROW(fmt::format("Not support quant_method {} for moe model.", yaml_weight_quant_method));
    }
    if (yaml_weight_quant_method == "fp8_e4m3") {
      glm4_moe_config->is_quant = true;
      glm4_moe_config->quant_config.method = QUANT_FP8_E4M3;
      glm4_moe_config->quant_config.is_checkpoint_fp8_serialized = false;
      glm4_moe_config->quant_config.is_activation_scheme_static = false;
    } else {
      KLLM_THROW(fmt::format("Not support quant_method {}.", yaml_weight_quant_method));
    }
  }

  // GLM4-MoE supports multiple backends for W4A8/GPTQ quantization
  if (glm4_moe_config->is_quant && glm4_moe_config->ContainW4A8Weights()) {
    if (yaml_gptq_backend == "cutlass") {
      glm4_moe_config->quant_config.backend = CUTLASS_LINEAR_BACKEND;
    } else if (yaml_gptq_backend == "marlin") {
      glm4_moe_config->quant_config.backend = MARLIN_LINEAR_BACKEND;
    } else if (yaml_gptq_backend == "machete") {
      glm4_moe_config->quant_config.backend = MACHETE_LINEAR_BACKEND;
    } else {
      // Default to CUTLASS if not specified or unknown backend
      glm4_moe_config->quant_config.backend = CUTLASS_LINEAR_BACKEND;
      KLLM_LOG_WARNING << "Unknown gptq_backend '" << yaml_gptq_backend
                       << "', using default CUTLASS Quant Backend for GLM4-MoE";
    }
  }

  return Status();
}

}  // namespace ksana_llm
