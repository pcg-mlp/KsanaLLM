/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "models/deepseek_v4/deepseek_v4_config_parser.h"

#include "configure/json_config_utils.h"
#include "configure/quant_config_parser.h"
#include "utils/singleton.h"

namespace ksana_llm {

Status DeepSeekV4ConfigParser::ParseModelConfig(const nlohmann::json &config_json,
                                                const ParallelismBasicConfig &parallel_basic_config,
                                                const std::string &model_dir,
                                                std::shared_ptr<BaseModelConfig> &model_config) {
  KLLM_LOG_INFO << "Parse config using deepseek v4 config parser";
  std::shared_ptr<DeepSeekV4Config> deepseek_v4_config = std::make_shared<DeepSeekV4Config>();
  model_config = deepseek_v4_config;

  deepseek_v4_config->model_dir = model_dir;

  auto env = Singleton<Environment>::GetInstance();
  RuntimeConfig runtime_config;
  env->GetRuntimeConfig(runtime_config);
  env->GetExpertParallelConfig(deepseek_v4_config->expert_parallel_config);
  deepseek_v4_config->tensor_para_size = parallel_basic_config.tensor_parallel_size;
  deepseek_v4_config->attn_data_para_size = parallel_basic_config.attn_data_parallel_size;
  size_t ep = parallel_basic_config.expert_parallel_size;
  deepseek_v4_config->expert_para_size = ep == 0 ? 1 : ep;
  deepseek_v4_config->moe_tensor_para_size =
      deepseek_v4_config->tensor_para_size / deepseek_v4_config->expert_para_size;

  deepseek_v4_config->weight_data_type = GetModelDataType(config_json);

  // 1. Parse common config (aligned with PrepareCommonModelAttributes, defaults from huggingface_config.json)
  deepseek_v4_config->type = config_json.value("model_type", "deepseek_v4");
  deepseek_v4_config->head_num = config_json.value("num_attention_heads", 64);
  deepseek_v4_config->num_key_value_heads = config_json.value("num_key_value_heads", 1);
  // deepseek_v4_config->inter_size = config_json.value("intermediate_size", 11008);
  deepseek_v4_config->vocab_size = config_json.value("vocab_size", 129280);
  deepseek_v4_config->num_layer = config_json.value("num_hidden_layers", 43);
  deepseek_v4_config->num_nextn_predict_layers =
      ReadDsparkNumLayers(model_dir, config_json.value("num_nextn_predict_layers", 1));
  deepseek_v4_config->hidden_units = config_json.value("hidden_size", 4096);
  deepseek_v4_config->rope_theta = config_json.value("rope_theta", 10000.0f);
  deepseek_v4_config->layernorm_eps = config_json.value("rms_norm_eps", 1e-6);
  deepseek_v4_config->start_id = config_json.value("bos_token_id", 0);

  // Handle eos_token_id which can be an array or a single value
  if (config_json.contains("eos_token_id") && config_json["eos_token_id"].is_array()) {
    deepseek_v4_config->end_ids = config_json["eos_token_id"].get<std::vector<uint32_t>>();
  } else {
    deepseek_v4_config->end_ids =
        std::vector<uint32_t>{static_cast<unsigned int>(config_json.value("eos_token_id", 1))};
  }

  // // Handle pad_token_id which can be null
  // if (config_json.contains("pad_token_id") && config_json.at("pad_token_id").is_null()) {
  //   deepseek_v4_config->pad_id = 0;
  // } else {
  //   deepseek_v4_config->pad_id = config_json.value("pad_token_id", 0);
  // }

  deepseek_v4_config->max_position_embeddings = config_json.value("max_position_embeddings", 1048576);
  if (!config_json.contains("tie_word_embeddings")) {
    deepseek_v4_config->exist_tie_embeddings_param = false;
  }
  deepseek_v4_config->tie_word_embeddings = config_json.value("tie_word_embeddings", false);
  deepseek_v4_config->is_visual = config_json.contains("visual");

  // size_per_head and rotary_embedding from head_dim (aligned with PrepareCommonModelAttributes)
  deepseek_v4_config->size_per_head = config_json.value("head_dim", 512);
  deepseek_v4_config->rotary_embedding = deepseek_v4_config->dsv4_attn_config.qk_rope_head_dim;

  // Parse rope_scaling config (yarn for deepseek_v4)
  if (config_json.contains("rope_scaling")) {
    auto rope_scaling = config_json["rope_scaling"];
    deepseek_v4_config->rope_scaling_factor_config.type = rope_scaling.value("type", "yarn");
    deepseek_v4_config->rope_scaling_factor_config.factor = rope_scaling.value("factor", 16.0f);
    deepseek_v4_config->rope_scaling_factor_config.original_max_position_embeddings =
        rope_scaling.value("original_max_position_embeddings", 65536);
    deepseek_v4_config->rope_scaling_factor_config.beta_fast = rope_scaling.value("beta_fast", 32.0f);
    deepseek_v4_config->rope_scaling_factor_config.beta_slow = rope_scaling.value("beta_slow", 1.0f);
    // deepseek_v4_config->rope_scaling_factor_config.mscale = rope_scaling.value("mscale", 1.0f);
    // deepseek_v4_config->rope_scaling_factor_config.mscale_all_dim = rope_scaling.value("mscale_all_dim", 1.0f);
    KLLM_LOG_INFO << fmt::format(
        "Using rope_scaling, type: {}, factor: {}, original_max_position_embeddings: {}, beta_fast: {}, beta_slow: {}",
        deepseek_v4_config->rope_scaling_factor_config.type, deepseek_v4_config->rope_scaling_factor_config.factor,
        deepseek_v4_config->rope_scaling_factor_config.original_max_position_embeddings,
        deepseek_v4_config->rope_scaling_factor_config.beta_fast,
        deepseek_v4_config->rope_scaling_factor_config.beta_slow);
  }

  // Parse deepseek_v4 specific common fields (aligned with PrepareDeepSeekV4Attributes)
  deepseek_v4_config->activation_function = config_json.value("hidden_act", "silu");
  deepseek_v4_config->num_hash_layers = config_json.value("num_hash_layers", 3);
  deepseek_v4_config->swiglu_limit = config_json.value("swiglu_limit", 10.0f);

  // 2. Parse moe config (aligned with PrepareDeepSeekV3Attributes, defaults from huggingface_config.json)
  deepseek_v4_config->moe_config.num_experts = config_json.value("n_routed_experts", 256);
  if (deepseek_v4_config->moe_config.num_experts > 1) {
    deepseek_v4_config->moe_config.use_vllm_moe = true;
    deepseek_v4_config->is_moe = true;
    deepseek_v4_config->moe_config.moe_inter_size = config_json.value("moe_intermediate_size", 2048);
    deepseek_v4_config->moe_config.experts_topk = config_json.value("num_experts_per_tok", 6);
    deepseek_v4_config->moe_config.first_k_dense_replace = config_json.value("first_k_dense_replace", 0);
    // For moe group topk config
    deepseek_v4_config->moe_config.num_expert_group = config_json.value("n_group", 1);
    deepseek_v4_config->moe_config.expert_groups_topk = config_json.value("topk_group", 1);
    deepseek_v4_config->moe_config.scoring_func = config_json.value("scoring_func", "sqrtsoftplus");
    deepseek_v4_config->moe_config.topk_method = config_json.value("topk_method", "noaux_tc");
    deepseek_v4_config->moe_config.norm_topk_prob = config_json.value("norm_topk_prob", true);
    deepseek_v4_config->moe_config.routed_scaling_factor = config_json.value("routed_scaling_factor", 1.5f);
    // NOTICE(winminkong): dsv4 没有 e_score_correction_bias 权重，设置为 false
    deepseek_v4_config->moe_config.use_e_score_correction_bias = false;
  }
  deepseek_v4_config->moe_config.num_shared_experts = config_json.value("n_shared_experts", 1);
  if (deepseek_v4_config->moe_config.num_shared_experts > 0) {
    deepseek_v4_config->has_shared_experts = true;
    // NOTICE(winminkong): dsv4 shared_expert_inter_size 不乘以 num_shared_experts
    deepseek_v4_config->moe_config.shared_expert_inter_size = deepseek_v4_config->moe_config.moe_inter_size;
  }
  KLLM_LOG_INFO << fmt::format(
      "Using moe model, num_experts: {}, num_shared_experts: {}, experts_topk: {}, moe_inter_size: {}, "
      "use_e_score_correction_bias: {}, swiglu_limit: {}",
      deepseek_v4_config->moe_config.num_experts, deepseek_v4_config->moe_config.num_shared_experts,
      deepseek_v4_config->moe_config.experts_topk, deepseek_v4_config->moe_config.moe_inter_size,
      deepseek_v4_config->moe_config.use_e_score_correction_bias, deepseek_v4_config->swiglu_limit);

  // 3. Parse deepseek sparse mla (dsa) config (aligned with PrepareDeepSeekV3Attributes)
  deepseek_v4_config->use_dsa = true;
  deepseek_v4_config->dsa_config.index_head_dim = config_json.value("index_head_dim", 128);
  deepseek_v4_config->dsa_config.index_n_heads = config_json.value("index_n_heads", 64);
  deepseek_v4_config->dsa_config.index_topk = config_json.value("index_topk", 512);
  KLLM_LOG_INFO << fmt::format("Using dsa model, index_head_dim: {}, index_n_heads: {}, index_topk: {}",
                               deepseek_v4_config->dsa_config.index_head_dim,
                               deepseek_v4_config->dsa_config.index_n_heads, deepseek_v4_config->dsa_config.index_topk);

  // 5. Parse dsv4 attn config (aligned with PrepareDeepSeekV4Attributes)
  deepseek_v4_config->use_dsv4_attn = true;
  deepseek_v4_config->dsv4_attn_config.o_groups = config_json.value("o_groups", 8);
  deepseek_v4_config->dsv4_attn_config.o_lora_rank = config_json.value("o_lora_rank", 1024);
  deepseek_v4_config->dsv4_attn_config.q_lora_rank = config_json.value("q_lora_rank", 1024);
  deepseek_v4_config->dsv4_attn_config.qk_rope_head_dim = config_json.value("qk_rope_head_dim", 64);
  deepseek_v4_config->dsv4_attn_config.sliding_window = config_json.value("sliding_window", 128);
  deepseek_v4_config->dsv4_attn_config.compress_rope_theta = config_json.value("compress_rope_theta", 160000.0f);
  deepseek_v4_config->dsv4_attn_config.compress_ratios = config_json["compress_ratios"].get<std::vector<uint32_t>>();
  KLLM_LOG_INFO << fmt::format(
      "Using dsv4 attn model, o_groups: {}, o_lora_rank: {}, q_lora_rank: {}, qk_rope_head_dim: {}, sliding_window: "
      "{}, "
      "compress_rope_theta: {}",
      deepseek_v4_config->dsv4_attn_config.o_groups, deepseek_v4_config->dsv4_attn_config.o_lora_rank,
      deepseek_v4_config->dsv4_attn_config.q_lora_rank, deepseek_v4_config->dsv4_attn_config.qk_rope_head_dim,
      deepseek_v4_config->dsv4_attn_config.sliding_window, deepseek_v4_config->dsv4_attn_config.compress_rope_theta);

  // 5. Parse mhc config (aligned with PrepareDeepSeekV4Attributes)
  deepseek_v4_config->use_mhc = true;
  deepseek_v4_config->mhc_config.hc_mult = config_json.value("hc_mult", 4);
  deepseek_v4_config->mhc_config.hc_sinkhorn_iters = config_json.value("hc_sinkhorn_iters", 20);
  deepseek_v4_config->mhc_config.hc_eps = config_json.value("hc_eps", 1e-6);
  KLLM_LOG_INFO << fmt::format("Using mhc model, hc_mult: {}, hc_sinkhorn_iters: {}, hc_eps: {}",
                               deepseek_v4_config->mhc_config.hc_mult, deepseek_v4_config->mhc_config.hc_sinkhorn_iters,
                               deepseek_v4_config->mhc_config.hc_eps);

  // 6. Parse dspark config
  deepseek_v4_config->dspark_config.dspark_block_size = config_json.value("dspark_block_size", 0u);
  deepseek_v4_config->dspark_config.dspark_noise_token_id = config_json.value("dspark_noise_token_id", 0u);
  if (config_json.contains("dspark_target_layer_ids")) {
    auto vec = config_json["dspark_target_layer_ids"].get<std::vector<uint32_t>>();
    deepseek_v4_config->dspark_config.dspark_target_layer_ids =
        std::unordered_set<uint32_t>(vec.begin(), vec.end());
  }
  deepseek_v4_config->dspark_config.dspark_markov_rank = config_json.value("dspark_markov_rank", 256u);
  if (deepseek_v4_config->dspark_config.dspark_block_size > 0) {
    KLLM_CHECK_WITH_INFO(!deepseek_v4_config->dspark_config.dspark_target_layer_ids.empty(),
                         "DSpark needs target layers");
  } else {
    deepseek_v4_config->dspark_config.dspark_target_layer_ids.clear();
  }
  KLLM_LOG_INFO << fmt::format(
      "DSpark config: block_size: {}, noise_token_id: {}, markov_rank: {}, target_layers: {}",
      deepseek_v4_config->dspark_config.dspark_block_size,
      deepseek_v4_config->dspark_config.dspark_noise_token_id,
      deepseek_v4_config->dspark_config.dspark_markov_rank,
      deepseek_v4_config->dspark_config.dspark_target_layer_ids.size());

  // 7. Parse quantization config
  ParseQuantConfig(config_json, deepseek_v4_config, env->GetYamlWeightQuantMethod(), env->GetYamlGptqBackend());
  return Status();
}

Status DeepSeekV4ConfigParser::ParseQuantConfig(const nlohmann::json &config_json,
                                                std::shared_ptr<DeepSeekV4Config> deepseek_v4_config,
                                                const std::string &yaml_weight_quant_method,
                                                const std::string &yaml_gptq_backend) {
  // 解析 quantize_config.json 和 hf_quant_config.json
  nlohmann::json quantize_config = ReadJsonFromFile(deepseek_v4_config->model_dir + "/quantize_config.json");
  nlohmann::json hf_quant_config = ReadJsonFromFile(deepseek_v4_config->model_dir + "/hf_quant_config.json");

  // 解析三种不同的情况
  nlohmann::json quantization_config;
  if (config_json.contains("quantization_config")) {
    // config.json 中有 quantization_config 字段（默认情况）
    deepseek_v4_config->is_quant = true;
    quantization_config = config_json["quantization_config"];
  } else if (!quantize_config.empty()) {
    // config.json 中没有 quantization_config 字段，但有额外文件 quantize_config.json
    KLLM_LOG_INFO << "No quantization_config in config.json, but find quantize_config.json";
    deepseek_v4_config->is_quant = true;
    quantization_config = quantize_config;
  } else if (!hf_quant_config.empty()) {
    // config.json 中没有 quantization_config 字段，没有文件 quantize_config.json，但有 hf_quant_config.json
    KLLM_LOG_INFO << "No quantization_config in config.json, no quantize_config.json, but find hf_quant_config.json";
    deepseek_v4_config->is_quant = true;
    quantization_config = ParseAndConvertQuantConfig(hf_quant_config);
  }

  if (deepseek_v4_config->is_quant) {
    std::string quant_method = quantization_config.at("quant_method");
    if (quant_method == "gptq" || quant_method == "rtn") {
      ParseGPTQQuantConfig(quantization_config, deepseek_v4_config->is_moe, deepseek_v4_config->quant_config);
    } else if (quant_method == "fp8") {
      ParseFP8QuantConfig(quantization_config, deepseek_v4_config->is_moe, deepseek_v4_config->quant_config);
    } else if (quant_method == "compressed-tensors") {
      ParseCompressedTensorsQuantConfig(quantization_config, deepseek_v4_config->is_moe,
                                        deepseek_v4_config->quant_config, deepseek_v4_config->sub_quant_configs);
      deepseek_v4_config->is_compressed_tensors = true;
    } else if (quant_method == "mixed") {
      auto configs = quantization_config["configs"];
      for (auto it = configs.begin(); it != configs.end(); ++it) {
        QuantConfig quant_config;
        quant_method = quantization_config["configs"][it.key()]["method"];
        if (quant_method == "gptq" || quant_method == "rtn") {
          ParseGPTQQuantConfig(quantization_config["configs"][it.key()], deepseek_v4_config->is_moe, quant_config);
        } else if (quant_method == "fp8") {
          ParseFP8QuantConfig(quantization_config["configs"][it.key()], deepseek_v4_config->is_moe, quant_config);
        } else {
          KLLM_THROW(fmt::format("Not support quant_method {}.", quant_method));
        }
        auto layer_mapping = quantization_config["layer_mapping"][it.key()];
        quant_config.pattern_layers = layer_mapping["pattern_layers"].get<std::vector<std::string>>();
        quant_config.ignored_layers = layer_mapping["ignored_layers"].get<std::vector<std::string>>();
        if (layer_mapping["default_config"]) {
          deepseek_v4_config->quant_config = quant_config;
        } else {
          deepseek_v4_config->sub_quant_configs.push_back(quant_config);
        }
      }
      if (deepseek_v4_config->sub_quant_configs.size() == 1 &&
          deepseek_v4_config->sub_quant_configs[0].method == QUANT_GPTQ &&
          deepseek_v4_config->sub_quant_configs[0].pattern_layers.size() == 1 &&
          deepseek_v4_config->sub_quant_configs[0].pattern_layers[0] == ".mlp.experts.") {
        deepseek_v4_config->quant_config.enable_moe_int4 = true;
      }
    } else {
      KLLM_THROW(fmt::format("Not support quant method: {}", quant_method));
    }
  } else if (yaml_weight_quant_method != "auto" && !yaml_gptq_backend.empty()) {
    if (deepseek_v4_config->is_moe) {
      KLLM_THROW(fmt::format("Not support quant_method {} for moe model.", yaml_weight_quant_method));
    }
    if (yaml_weight_quant_method == "fp8_e4m3") {
      // when quantization_config in config.json is null,
      // quant method is decided by quantization_config in yaml.
      deepseek_v4_config->is_quant = true;
      deepseek_v4_config->quant_config.method = QUANT_FP8_E4M3;
      deepseek_v4_config->quant_config.is_checkpoint_fp8_serialized = false;
      deepseek_v4_config->quant_config.is_activation_scheme_static = false;
      KLLM_LOG_INFO << fmt::format(
          "using quant model, quant method: {}, is_checkpoint_fp8_serialized: {}, "
          "is_activation_scheme_static: {}",
          yaml_weight_quant_method, deepseek_v4_config->quant_config.is_checkpoint_fp8_serialized,
          deepseek_v4_config->quant_config.is_activation_scheme_static);
    } else {
      KLLM_THROW(fmt::format("Not support quant_method {}.", yaml_weight_quant_method));
    }
  }
  deepseek_v4_config->quant_config.expert_dtype = config_json.value("expert_dtype", "none");

  // Deepseek only support machete backend
  if (deepseek_v4_config->is_quant && deepseek_v4_config->ContainGptqWeights()) {
    deepseek_v4_config->quant_config.backend = MACHETE_LINEAR_BACKEND;
  } else {
    KLLM_LOG_INFO << "Not using any Quant Backend";
  }

  return Status();
}

}  // namespace ksana_llm
