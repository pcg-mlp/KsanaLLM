/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "models/kimi_k3/kimi_k3_config_parser.h"

#include "configure/json_config_utils.h"
#include "configure/quant_config_parser.h"
#include "utils/singleton.h"

namespace ksana_llm {

Status KimiK3ConfigParser::ParseModelConfig(const nlohmann::json &config_json,
                                            const ParallelismBasicConfig &parallel_basic_config,
                                            const std::string &model_dir,
                                            std::shared_ptr<BaseModelConfig> &model_config) {
  KLLM_LOG_INFO << "Parse config using kimi_k3 config parser";
  std::shared_ptr<KimiK3Config> kimi_k3_config = std::make_shared<KimiK3Config>();
  model_config = kimi_k3_config;

  kimi_k3_config->model_dir = model_dir;

  auto env = Singleton<Environment>::GetInstance();
  env->GetExpertParallelConfig(kimi_k3_config->expert_parallel_config);
  kimi_k3_config->tensor_para_size = parallel_basic_config.tensor_parallel_size;
  kimi_k3_config->attn_data_para_size = parallel_basic_config.attn_data_parallel_size;
  size_t ep = parallel_basic_config.expert_parallel_size;
  kimi_k3_config->expert_para_size = ep == 0 ? 1 : ep;
  kimi_k3_config->moe_tensor_para_size = kimi_k3_config->tensor_para_size / kimi_k3_config->expert_para_size;

  kimi_k3_config->weight_data_type = GetModelDataType(config_json);
  // 1. Common attributes (aligned with PrepareCommonModelAttributes).
  kimi_k3_config->head_num = config_json.value("num_attention_heads", 96);
  kimi_k3_config->num_key_value_heads = config_json.value("num_key_value_heads", kimi_k3_config->head_num);
  kimi_k3_config->inter_size = config_json.value("intermediate_size", 33792);
  kimi_k3_config->vocab_size = config_json.value("vocab_size", 163840);
  kimi_k3_config->num_layer = config_json.value("num_hidden_layers", 93);
  kimi_k3_config->num_nextn_predict_layers = config_json.value("num_nextn_predict_layers", 0);
  kimi_k3_config->hidden_units = config_json.value("hidden_size", 7168);
  kimi_k3_config->rope_theta = config_json.value("rope_theta", 10000.0f);
  kimi_k3_config->layernorm_eps = config_json.value("rms_norm_eps", 1e-5);
  kimi_k3_config->start_id = config_json.value("bos_token_id", 163584);

  // eos_token_id can be an array or a single value.
  if (config_json.contains("eos_token_id") && config_json["eos_token_id"].is_array()) {
    kimi_k3_config->end_ids = config_json["eos_token_id"].get<std::vector<uint32_t>>();
  } else {
    kimi_k3_config->end_ids =
        std::vector<uint32_t>{static_cast<unsigned int>(config_json.value("eos_token_id", 163586))};
  }
  if (config_json.contains("pad_token_id") && config_json.at("pad_token_id").is_null()) {
    kimi_k3_config->pad_id = 0;
  } else {
    kimi_k3_config->pad_id = config_json.value("pad_token_id", 163839);
  }

  kimi_k3_config->max_position_embeddings = config_json.value("max_position_embeddings", 1048576);
  if (!config_json.contains("tie_word_embeddings")) {
    kimi_k3_config->exist_tie_embeddings_param = false;
  }
  kimi_k3_config->tie_word_embeddings = config_json.value("tie_word_embeddings", false);

  // 2. Activation ("situ").
  kimi_k3_config->activation_function = config_json.value("hidden_act", "situ");
  kimi_k3_config->activation_situ_beta = config_json.value("activation_situ_beta", 4.0f);
  kimi_k3_config->activation_situ_linear_beta = config_json.value("activation_situ_linear_beta", 25.0f);

  // 3. MLA config.
  kimi_k3_config->use_mla = true;
  if (config_json.contains("q_lora_rank") && config_json["q_lora_rank"].is_number()) {
    kimi_k3_config->mla_config.q_lora_rank = config_json.value("q_lora_rank", 1536);
  } else {
    kimi_k3_config->mla_config.q_lora_rank = 0;
  }
  kimi_k3_config->mla_config.kv_lora_rank = config_json.value("kv_lora_rank", 512);
  kimi_k3_config->mla_config.qk_nope_head_dim = config_json.value("qk_nope_head_dim", 128);
  kimi_k3_config->mla_config.qk_rope_head_dim = config_json.value("qk_rope_head_dim", 64);
  kimi_k3_config->mla_config.v_head_dim = config_json.value("v_head_dim", 128);
  kimi_k3_config->mla_config.mla_use_nope = config_json.value("mla_use_nope", true);
  kimi_k3_config->mla_config.mla_use_output_gate = config_json.value("mla_use_output_gate", false);
  kimi_k3_config->size_per_head =
      kimi_k3_config->mla_config.qk_nope_head_dim + kimi_k3_config->mla_config.qk_rope_head_dim;
  kimi_k3_config->rotary_embedding = kimi_k3_config->mla_config.qk_rope_head_dim;

  // 4. MoE config (kimi_k3 key names differ from deepseek).
  kimi_k3_config->moe_config.num_experts = config_json.value("num_experts", 896);
  if (kimi_k3_config->moe_config.num_experts > 1) {
    kimi_k3_config->moe_config.use_vllm_moe = true;
    kimi_k3_config->is_moe = true;
    kimi_k3_config->moe_config.moe_inter_size = config_json.value("moe_intermediate_size", 3072);
    kimi_k3_config->moe_config.experts_topk = config_json.value("num_experts_per_token", 16);
    kimi_k3_config->moe_config.first_k_dense_replace = config_json.value("first_k_dense_replace", 1);
    kimi_k3_config->moe_config.num_expert_group = config_json.value("num_expert_group", 1);
    kimi_k3_config->moe_config.expert_groups_topk = config_json.value("topk_group", 1);
    kimi_k3_config->moe_config.scoring_func = config_json.value("moe_router_activation_func", "sigmoid");
    kimi_k3_config->moe_config.topk_method = config_json.value("topk_method", "noaux_tc");
    kimi_k3_config->moe_config.norm_topk_prob = config_json.value("moe_renormalize", true);
    kimi_k3_config->moe_config.routed_scaling_factor = config_json.value("routed_scaling_factor", 1.0f);
    kimi_k3_config->moe_config.use_e_score_correction_bias = (kimi_k3_config->moe_config.topk_method == "noaux_tc");
  }
  kimi_k3_config->moe_config.num_shared_experts = config_json.value("num_shared_experts", 2);
  if (kimi_k3_config->moe_config.num_shared_experts > 0) {
    kimi_k3_config->has_shared_experts = true;
    kimi_k3_config->moe_config.shared_expert_inter_size =
        kimi_k3_config->moe_config.num_shared_experts * kimi_k3_config->moe_config.moe_inter_size;
  }
  kimi_k3_config->moe_config.routed_expert_hidden_size = config_json.value("routed_expert_hidden_size", 3584);
  kimi_k3_config->moe_config.latent_moe_use_norm = config_json.value("latent_moe_use_norm", true);
  KLLM_LOG_INFO << fmt::format(
      "Using kimi_k3 moe model, num_experts: {}, num_shared_experts: {}, experts_topk: {}, moe_inter_size: {}, "
      "scoring_func: {}, topk_method: {}",
      kimi_k3_config->moe_config.num_experts, kimi_k3_config->moe_config.num_shared_experts,
      kimi_k3_config->moe_config.experts_topk, kimi_k3_config->moe_config.moe_inter_size,
      kimi_k3_config->moe_config.scoring_func, kimi_k3_config->moe_config.topk_method);

  // 5. Kimi Linear (KDA) hybrid attention config.
  if (config_json.contains("linear_attn_config")) {
    const auto &linear_attn_json = config_json["linear_attn_config"];
    auto &kimi_linear_config = kimi_k3_config->kimi_linear_config;
    kimi_linear_config.head_dim = linear_attn_json.value("head_dim", kimi_k3_config->mla_config.v_head_dim);
    kimi_linear_config.short_conv_kernel_size = linear_attn_json.value("short_conv_kernel_size", 4u);
    kimi_linear_config.use_full_rank_gate = linear_attn_json.value("use_full_rank_gate", true);
    if (linear_attn_json.contains("gate_lower_bound") && !linear_attn_json["gate_lower_bound"].is_null()) {
      kimi_linear_config.gate_lower_bound = linear_attn_json["gate_lower_bound"].get<float>();
    }

    // Layers using KDA; the remaining ones use full (MLA) attention.
    if (linear_attn_json.contains("kda_layers") && linear_attn_json["kda_layers"].is_array()) {
      for (const auto &layer_idx : linear_attn_json["kda_layers"]) {
        kimi_linear_config.kda_layers.insert(layer_idx.get<uint32_t>());
      }
    }

    KLLM_CHECK_WITH_INFO(kimi_linear_config.head_dim == kimi_k3_config->mla_config.v_head_dim,
                         "linear_attn_config.head_dim must equal v_head_dim");
    KLLM_CHECK_WITH_INFO(linear_attn_json.value("num_heads", kimi_k3_config->head_num) == kimi_k3_config->head_num,
                         "linear_attn_config.num_heads must equal num_attention_heads");

    KLLM_LOG_INFO << fmt::format("Using kimi_linear hybrid attention, kda_layers: {}, num_heads: {}, head_dim: {}",
                                 kimi_k3_config->kimi_linear_config.kda_layers.size(), kimi_k3_config->head_num,
                                 kimi_linear_config.head_dim);
  }

  // 6. Parse attn_res config
  kimi_k3_config->attn_res_config.attn_res_block_size = config_json.value("attn_res_block_size", 0u);
  KLLM_LOG_INFO << fmt::format("AttnRes config: attn_res_block_size: {}",
                               kimi_k3_config->attn_res_config.attn_res_block_size);

  // 7. Quantization config.
  ParseQuantConfig(config_json, kimi_k3_config, env->GetYamlWeightQuantMethod(), env->GetYamlGptqBackend());
  return Status();
}

Status KimiK3ConfigParser::ParseQuantConfig(const nlohmann::json &config_json,
                                            std::shared_ptr<KimiK3Config> kimi_k3_config,
                                            const std::string &yaml_weight_quant_method,
                                            const std::string &yaml_gptq_backend) {
  // Parse quantize_config.json and hf_quant_config.json fallbacks.
  nlohmann::json quantize_config = ReadJsonFromFile(kimi_k3_config->model_dir + "/quantize_config.json");
  nlohmann::json hf_quant_config = ReadJsonFromFile(kimi_k3_config->model_dir + "/hf_quant_config.json");

  nlohmann::json quantization_config;
  if (config_json.contains("quantization_config")) {
    kimi_k3_config->is_quant = true;
    quantization_config = config_json["quantization_config"];
  } else if (!quantize_config.empty()) {
    KLLM_LOG_INFO << "No quantization_config in config.json, but find quantize_config.json";
    kimi_k3_config->is_quant = true;
    quantization_config = quantize_config;
  } else if (!hf_quant_config.empty()) {
    KLLM_LOG_INFO << "No quantization_config in config.json, no quantize_config.json, but find hf_quant_config.json";
    kimi_k3_config->is_quant = true;
    quantization_config = ParseAndConvertQuantConfig(hf_quant_config);
  }

  if (kimi_k3_config->is_quant) {
    std::string quant_method = quantization_config.at("quant_method");
    if (quant_method == "gptq" || quant_method == "rtn") {
      ParseGPTQQuantConfig(quantization_config, kimi_k3_config->is_moe, kimi_k3_config->quant_config);
    } else if (quant_method == "awq") {
      ParseAWQQuantConfig(quantization_config, kimi_k3_config->is_moe, kimi_k3_config->quant_config);
    } else if (quant_method == "fp8") {
      ParseFP8QuantConfig(quantization_config, kimi_k3_config->is_moe, kimi_k3_config->quant_config);
    } else if (quant_method == "compressed-tensors") {
      // Kimi-K3 ships mxfp4-pack-quantized weights described as compressed-tensors.
      ParseCompressedTensorsQuantConfig(quantization_config, kimi_k3_config->is_moe, kimi_k3_config->quant_config,
                                        kimi_k3_config->sub_quant_configs);
      kimi_k3_config->is_compressed_tensors = true;
    } else {
      KLLM_THROW(fmt::format("Not support quant method: {}", quant_method));
    }
  }
  kimi_k3_config->quant_config.expert_dtype = config_json.value("expert_dtype", "none");

  if (kimi_k3_config->is_quant && kimi_k3_config->ContainGptqWeights()) {
    kimi_k3_config->quant_config.backend = MACHETE_LINEAR_BACKEND;
  } else {
    KLLM_LOG_INFO << "Not using any Quant Backend";
  }

  return Status();
}

}  // namespace ksana_llm
