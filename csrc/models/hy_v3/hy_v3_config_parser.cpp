/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "models/hy_v3/hy_v3_config_parser.h"

#include "configure/environment.h"
#include "configure/json_config_utils.h"
#include "configure/quant_config_parser.h"
#ifdef ENABLE_CUDA
#include "device/nvidia/cuda_utils.h"
#endif
#include "utils/singleton.h"

namespace ksana_llm {

Status HyV3ConfigParser::ParseModelConfig(const nlohmann::json &config_json,
                                          const ParallelismBasicConfig &parallel_basic_config,
                                          const std::string &model_dir,
                                          std::shared_ptr<BaseModelConfig> &model_config) {
  std::shared_ptr<HyV3Config> hy_v3_config = std::make_shared<HyV3Config>();
  model_config = hy_v3_config;

  hy_v3_config->model_dir = model_dir;
  // HunYuan V3 checkpoints are bfloat16 but config.json omits torch_dtype (GetModelDataType would
  // otherwise default to float16 and mismatch the BF16 weights). Honor an explicit dtype if present.
  hy_v3_config->weight_data_type = (config_json.contains("torch_dtype") || config_json.contains("dtype"))
                                       ? GetModelDataType(config_json)
                                       : DataType::TYPE_BF16;

  // Reference: /data1/models/Hy3-29183/config.json (HYV3ForCausalLM)
  hy_v3_config->head_num = config_json.value("num_attention_heads", 64);
  hy_v3_config->num_key_value_heads = config_json.value("num_key_value_heads", 8);
  hy_v3_config->inter_size = config_json.value("intermediate_size", 13312);
  hy_v3_config->vocab_size = config_json.value("vocab_size", 120832);
  hy_v3_config->num_layer = config_json.value("num_hidden_layers", 80);
  hy_v3_config->hidden_units = config_json.value("hidden_size", 4096);
  hy_v3_config->layernorm_eps = config_json.value("rms_norm_eps", 1e-5f);
  hy_v3_config->start_id = config_json.value("bos_token_id", 120000);
  hy_v3_config->end_id = config_json.value("eos_token_id", 120025);
  hy_v3_config->end_ids = std::vector<uint32_t>{hy_v3_config->end_id};
  hy_v3_config->pad_id = config_json.value("pad_token_id", 120002);
  hy_v3_config->max_position_embeddings = config_json.value("max_position_embeddings", 262144);

  // rope_theta lives inside a nested "rope_parameters" object for hy_v3
  // (there is no top-level "rope_theta"). Fall back to a top-level key for safety.
  if (config_json.contains("rope_parameters") && config_json["rope_parameters"].contains("rope_theta")) {
    hy_v3_config->rope_theta = config_json["rope_parameters"].value("rope_theta", 11158840.0f);
  } else {
    hy_v3_config->rope_theta = config_json.value("rope_theta", 11158840.0f);
  }

  if (!config_json.contains("tie_word_embeddings")) {
    hy_v3_config->exist_tie_embeddings_param = false;
  }
  hy_v3_config->tie_word_embeddings = config_json.value("tie_word_embeddings", false);

  // hy_v3 declares head_dim explicitly (128) and applies RoPE over the full head_dim
  // (no partial rotary factor), so rotary_embedding == size_per_head.
  hy_v3_config->size_per_head = config_json.value("head_dim", hy_v3_config->hidden_units / hy_v3_config->head_num);
  hy_v3_config->rotary_embedding = hy_v3_config->size_per_head;

  // hy_v3 applies pre-RoPE QK-Norm and has no bias on QKV projections.
  hy_v3_config->use_qk_norm = config_json.value("qk_norm", true);
  hy_v3_config->enable_add_qkv_bias = config_json.value("attention_bias", false);

  // MTP (Multi-Token Prediction): honor num_nextn_predict_layers from config.json.
  // 0 = disabled; typical hy_v3 checkpoints ship a single MTP head (num_nextn_predict_layers=1)
  // at layer index num_layer.
  hy_v3_config->num_nextn_predict_layers = config_json.value("num_nextn_predict_layers", 0);

  // Parse MoE config
  ParseMoeConfig(config_json, hy_v3_config);

  // Parse quantization config
  auto env = Singleton<Environment>::GetInstance();
  STATUS_CHECK_RETURN(
      ParseQuantConfig(config_json, hy_v3_config, env->GetYamlWeightQuantMethod(), env->GetYamlGptqBackend()));

  return Status();
}

Status HyV3ConfigParser::ParseMoeConfig(const nlohmann::json &config_json, std::shared_ptr<HyV3Config> hy_v3_config) {
  hy_v3_config->is_moe = true;

  hy_v3_config->moe_config.num_experts = config_json.value("num_experts", 192);
  hy_v3_config->moe_config.experts_topk = config_json.value("num_experts_per_tok", 8);
  // hy_v3 exposes both "expert_hidden_dim" and "moe_intermediate_size" (equal); prefer the latter.
  hy_v3_config->moe_config.moe_inter_size =
      config_json.value("moe_intermediate_size", config_json.value("expert_hidden_dim", 1536));
  hy_v3_config->moe_config.first_k_dense_replace = config_json.value("first_k_dense_replace", 1);

  // hy_v3 uses grouped-topk with a single group (equivalent to plain top-k).
  hy_v3_config->moe_config.num_expert_group = config_json.value("n_group", 1);
  hy_v3_config->moe_config.expert_groups_topk = config_json.value("topk_group", 1);

  // hy_v3 routes with sigmoid scoring + aux-loss-free expert bias correction.
  hy_v3_config->moe_config.scoring_func = config_json.value("moe_router_use_sigmoid", true) ? "sigmoid" : "softmax";
  hy_v3_config->moe_config.norm_topk_prob = config_json.value("route_norm", true);
  hy_v3_config->moe_config.routed_scaling_factor = config_json.value("router_scaling_factor", 1.0f);
  hy_v3_config->moe_config.use_e_score_correction_bias = config_json.value("moe_router_enable_expert_bias", true);
  hy_v3_config->moe_config.use_vllm_moe = true;
  // Keep in sync with PrepareHyV3Attributes (models/common_moe/moe_config.h): Fp8MoeLayer's
  // InvokeMoeCutlassGemm only wires (greedy, softmax) and (fast, sigmoid), and hy_v3 is sigmoid,
  // so "fast" is the only valid pairing. MoeConfig's default is "greedy".
  hy_v3_config->moe_config.topk_method = "fast";

  // Single merged shared expert.
  hy_v3_config->moe_config.num_shared_experts = config_json.value("num_shared_experts", 1);
  hy_v3_config->has_shared_experts = hy_v3_config->moe_config.num_shared_experts > 0;
  if (hy_v3_config->has_shared_experts) {
    hy_v3_config->moe_config.shared_expert_inter_size =
        hy_v3_config->moe_config.num_shared_experts * hy_v3_config->moe_config.moe_inter_size;
    hy_v3_config->shared_expert_intermediate_size = hy_v3_config->moe_config.shared_expert_inter_size;
  }

  KLLM_LOG_INFO << fmt::format(
      "hy_v3 MoE config: num_experts={}, experts_topk={}, moe_inter_size={}, first_k_dense_replace={}, "
      "scoring_func={}, routed_scaling_factor={}, use_e_score_correction_bias={}, num_shared_experts={}",
      hy_v3_config->moe_config.num_experts, hy_v3_config->moe_config.experts_topk,
      hy_v3_config->moe_config.moe_inter_size, hy_v3_config->moe_config.first_k_dense_replace,
      hy_v3_config->moe_config.scoring_func, hy_v3_config->moe_config.routed_scaling_factor,
      hy_v3_config->moe_config.use_e_score_correction_bias, hy_v3_config->moe_config.num_shared_experts);

  return Status();
}

Status HyV3ConfigParser::ParseQuantConfig(const nlohmann::json &config_json, std::shared_ptr<HyV3Config> hy_v3_config,
                                          const std::string &yaml_weight_quant_method,
                                          const std::string &yaml_gptq_backend) {
  // hy_v3 supports BF16 (no quantization) and FP8-E4M3 static per-tensor quantization.
  hy_v3_config->is_quant = config_json.contains("quantization_config");
  if (hy_v3_config->is_quant) {
    std::string quant_method = config_json["quantization_config"].at("quant_method");
    if (quant_method == "fp8") {
      // Shared with qwen3_moe / glm4_moe / deepseek: detects per-tensor (static) vs
      // blockwise (weight_block_size present, dynamic activation).
      ParseFP8QuantConfig(config_json["quantization_config"], hy_v3_config->is_moe, hy_v3_config->quant_config);
    } else {
      KLLM_LOG_ERROR << fmt::format("hy_v3 does not support quant method: {}", quant_method);
      return Status(RET_MODEL_LOAD_FAILED, fmt::format("hy_v3 does not support quant method: {}", quant_method));
    }
  }
  if (yaml_weight_quant_method != "auto" && !yaml_weight_quant_method.empty()) {
    KLLM_LOG_ERROR << fmt::format("hy_v3 does not support yaml quant method: {}", yaml_weight_quant_method);
    return Status(RET_MODEL_LOAD_FAILED,
                  fmt::format("hy_v3 does not support yaml quant method: {}", yaml_weight_quant_method));
  }
  return Status();
}

}  // namespace ksana_llm
