/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/utils/environment.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include "fmt/core.h"
#include "gflags/gflags.h"
#include "nlohmann/json.hpp"

#include "ksana_llm/models/chatglm/chatglm_config.h"
#include "ksana_llm/models/common/common_config.h"
#include "ksana_llm/models/common_moe/moe_config.h"
#include "ksana_llm/models/gpt/gpt_config.h"
#include "ksana_llm/utils/device_utils.h"
#include "ksana_llm/utils/gguf_file_tensor_loader.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/optional_file.h"
#include "ksana_llm/utils/ret_code.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/string_utils.h"
#include "ksana_llm/utils/yaml_reader.h"

DEFINE_string(config_file, "examples/ksana_llm.yaml", "The config file path");
DEFINE_string(host, "localhost", "HTTP service hostname, default is localhost");
DEFINE_int32(port, 8080, "HTTP service port, default is 8080");

namespace ksana_llm {

DataType GetModelDataType(const nlohmann::json &config_json, ModelConfig &model_config) {
  std::string data_type_raw_str = config_json.value("torch_dtype", "float16");
  std::string unified_data_type_raw_str = data_type_raw_str;
  // unify it to lower case
  std::transform(unified_data_type_raw_str.begin(), unified_data_type_raw_str.end(), unified_data_type_raw_str.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  if (unified_data_type_raw_str == "float16") {
    return DataType::TYPE_FP16;
  } else if (unified_data_type_raw_str == "bfloat16") {
#ifdef ENABLE_BFLOAT16
    return DataType::TYPE_BF16;
#else
    return DataType::TYPE_FP16;
#endif
  } else {
    KLLM_THROW(fmt::format("Not supported model data type: {}.", unified_data_type_raw_str));
  }
}

void ParseModelQuantConfig(const nlohmann::json &config_json, ModelConfig &model_config,
                           std::string &yaml_weight_quant_method, std::string &yaml_gptq_backend) {
  model_config.is_quant = config_json.contains("quantization_config");
  if (model_config.is_quant) {
    std::string quant_method = config_json["quantization_config"].at("quant_method");
    if (quant_method == "gptq") {
      model_config.quant_config.method = QUANT_GPTQ;
      model_config.quant_config.bits = config_json["quantization_config"].at("bits");
      model_config.quant_config.group_size = config_json["quantization_config"].at("group_size");
      model_config.quant_config.desc_act = config_json["quantization_config"].at("desc_act");
      KLLM_LOG_INFO << fmt::format("using quant model, quant method: {}, bits: {}, group_size: {}, desc_act: {}",
                                   quant_method, model_config.quant_config.bits, model_config.quant_config.group_size,
                                   model_config.quant_config.desc_act);
    } else if (quant_method == "awq") {
      model_config.quant_config.method = QUANT_AWQ;
      model_config.quant_config.bits = config_json["quantization_config"].at("bits");
      model_config.quant_config.group_size = config_json["quantization_config"].at("group_size");
      KLLM_LOG_INFO << fmt::format("using quant model, quant method: {}, bits: {}, group_size: {}", quant_method,
                                   model_config.quant_config.bits, model_config.quant_config.group_size);
    } else if (quant_method == "fp8") {
      model_config.quant_config.method = QUANT_FP8_E4M3;
      model_config.quant_config.is_checkpoint_fp8_serialized = true;
      model_config.quant_config.is_activation_scheme_static =
          (config_json["quantization_config"].at("activation_scheme") == "static");
      KLLM_LOG_INFO << fmt::format(
          "using quant model, quant method: {}, is_checkpoint_fp8_serialized: {}, "
          "is_activation_scheme_static: {}",
          quant_method, model_config.quant_config.is_checkpoint_fp8_serialized,
          model_config.quant_config.is_activation_scheme_static);
    } else {
      KLLM_THROW(fmt::format("Not support quant_method {}.", quant_method));
    }
  } else if (yaml_weight_quant_method != "auto") {
    if (yaml_weight_quant_method == "fp8_e4m3") {
      // when quantization_config in config.json is null,
      // quant method is decided by quantization_config in yaml.
      model_config.is_quant = true;
      model_config.quant_config.method = QUANT_FP8_E4M3;
      model_config.quant_config.is_checkpoint_fp8_serialized = false;
      model_config.quant_config.is_activation_scheme_static = false;
      KLLM_LOG_INFO << fmt::format(
          "using quant model, quant method: {}, is_checkpoint_fp8_serialized: {}, "
          "is_activation_scheme_static: {}",
          yaml_weight_quant_method, model_config.quant_config.is_checkpoint_fp8_serialized,
          model_config.quant_config.is_activation_scheme_static);
    } else {
      KLLM_THROW(fmt::format("Not support quant_method {}.", yaml_weight_quant_method));
    }
  }

  if (model_config.quant_config.method == QUANT_GPTQ && model_config.quant_config.desc_act == true) {
    model_config.quant_config.backend = MARLIN_BACKEND;
    KLLM_LOG_INFO << "Using MARLIN Quant Backend, only support MARLIN backend in desc_act mode";
  } else {
    if (yaml_gptq_backend == "cutlass") {
      model_config.quant_config.backend = CUTLASS_BACKEND;
      KLLM_LOG_INFO << "Using CUTLASS Quant Backend";
    } else if (yaml_gptq_backend == "marlin") {
      model_config.quant_config.backend = MARLIN_BACKEND;
      KLLM_LOG_INFO << "Using MARLIN Quant Backend";
    } else {
      KLLM_THROW(fmt::format("Not support quant backend {}.", yaml_gptq_backend));
    }
  }
}

void ParseModelMaxLength(const nlohmann::json &config_json, ModelConfig &model_config) {
  // refer to
  // github vllm-project/vllm/blob vllm/config.py#L1116
  float derived_max_model_len = std::numeric_limits<float>::infinity();
  std::vector<std::string> possible_keys = {/* OPT */ "max_position_embeddings",
                                            /* GPT-2 */ "n_positions",
                                            /* MPT */ "max_seq_len",
                                            /* ChatGLM2 */ "seq_length",
                                            /* Command-R */ "model_max_length",
                                            /* Others */ "max_sequence_length",
                                            "max_seq_length",
                                            "seq_len"};
  for (std::string &key : possible_keys) {
    float max_len = config_json.value(key, std::numeric_limits<float>::infinity());
    derived_max_model_len = std::min(derived_max_model_len, max_len);
  }
  if (derived_max_model_len == std::numeric_limits<float>::infinity()) {
    std::string possible_keys_str = Vector2Str<std::string>(possible_keys);
    KLLM_THROW(
        fmt::format("The model's config.json does not contain any of the following keys to determine"
                    " the original maximum length of the model: {}",
                    possible_keys_str));
  }

  auto rope_scaling_setting = config_json.value("rope_scaling", nlohmann::json());
  if (!rope_scaling_setting.is_null()) {
    model_config.rope_scaling_factor_config.type = rope_scaling_setting.value("type", "default");
    // fit llama3.1 config
    model_config.rope_scaling_factor_config.type =
        rope_scaling_setting.value("rope_type", model_config.rope_scaling_factor_config.type);
    model_config.rope_scaling_factor_config.factor = rope_scaling_setting.value("factor", 1.0f);
    KLLM_LOG_DEBUG << fmt::format("rope_scaling type: {} factor: {}", model_config.rope_scaling_factor_config.type,
                                  model_config.rope_scaling_factor_config.factor);

    std::unordered_set<std::string> possible_rope_types = {"su", "longrope", "llama3"};
    if (possible_rope_types.find(model_config.rope_scaling_factor_config.type) == possible_rope_types.end()) {
      if (model_config.rope_scaling_factor_config.type == "yarn") {
        derived_max_model_len = rope_scaling_setting.value("original_max_position_embeddings", derived_max_model_len);
        model_config.rope_scaling_factor_config.original_max_position_embeddings =
            rope_scaling_setting.value("original_max_position_embeddings", 32768);
      }
      // for dynamic alpha
      if (model_config.rope_scaling_factor_config.type == "dynamic" && rope_scaling_setting.contains("alpha")) {
        model_config.rope_scaling_factor_config.has_alpha = true;
        model_config.rope_scaling_factor_config.scaling_alpha = rope_scaling_setting.value("alpha", 1.0f);
      } else {
        derived_max_model_len *= model_config.rope_scaling_factor_config.factor;
      }
    }

    if (model_config.rope_scaling_factor_config.type == "llama3") {
      model_config.rope_scaling_factor_config.low_freq_factor = rope_scaling_setting.value("low_freq_factor", 1.0f);
      model_config.rope_scaling_factor_config.high_freq_factor = rope_scaling_setting.value("high_freq_factor", 4.0f);
      model_config.rope_scaling_factor_config.original_max_position_embeddings =
          rope_scaling_setting.value("original_max_position_embeddings", 8192);
    }

    if (model_config.rope_scaling_factor_config.type == "mrope") {
      auto &mrope_section = model_config.rope_scaling_factor_config.mrope_section;
      mrope_section = rope_scaling_setting["mrope_section"].get<std::vector<int>>();
      KLLM_CHECK_WITH_INFO(mrope_section.size() == 3,
                           "The length of mrope section used for multimodal rotary embedding must be 3.");
      // Perform a prefix sum to facilitate the MRotaryEmbedding kernel.
      for (int i = 1; i < 3; i++) {
        mrope_section[i] += mrope_section[i - 1];
      }
    }
  }

  model_config.max_token_num = static_cast<int>(derived_max_model_len);
  KLLM_LOG_DEBUG << "Model Max Token Num = " << model_config.max_token_num;
}

void UpdateEndIdFromGeneration(const std::string &model_dir, ModelConfig &model_config) {
  // Priority: `generation_config` argument > `config.json` argument
  // It is recommended to set all generation parameters in `generation_config`
  // Refer to
  // https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py#L1736
  std::filesystem::path abs_model_dir_path = std::filesystem::absolute(model_dir);
  std::string config_file = abs_model_dir_path.u8string() + "/generation_config.json";

  nlohmann::json config_json;
  std::ifstream file(config_file);
  if (!file.is_open()) {
    KLLM_LOG_DEBUG << fmt::format("Gneration config file: {} does not exist.", config_file);
    return;
  } else {
    file >> config_json;
    file.close();
  }

  if (!config_json.contains("eos_token_id")) {
    return;
  }

  std::vector<uint32_t> end_ids;
  if (config_json.at("eos_token_id").is_array()) {
    end_ids = config_json["eos_token_id"].get<std::vector<uint32_t>>();
  } else {
    end_ids = std::vector<uint32_t>{config_json.at("eos_token_id")};
  }
  if (end_ids != model_config.end_ids) {
    KLLM_LOG_WARNING << fmt::format("eos_token_id: [{}] in model config is overwritten by [{}] in generation config",
                                    fmt::join(model_config.end_ids, ", "), fmt::join(end_ids, ", "));
    model_config.end_ids = std::move(end_ids);
  }
}

void PrepareKVScales(const std::string &model_dir, ModelConfig &model_config) {
  // Search for the optional kv_cache_scales.json file
  auto optional_file = Singleton<OptionalFile>::GetInstance();
  // TODO(zhongzhicao): 当前仅尝试从模型文件夹下读取，后续需要从python_dir/kv_scales下读取，并校验模型是否相同
  std::string &kv_scale_path = optional_file->GetOptionalFile(model_dir, "kv_scales", "kv_cache_scales.json");
  if (kv_scale_path == "") {
    KLLM_LOG_WARNING << fmt::format(
        "Loading KV cache scaling factors file error. File not found. Using defalt value 1.0 ");
    return;
  }
  KLLM_LOG_INFO << fmt::format("Found KV cache scaling factors file at {}.", kv_scale_path);

  nlohmann::json kv_scale_json;
  std::ifstream kv_scale_file(kv_scale_path);
  if (!kv_scale_file.is_open()) {
    // TODO(zhongzhicao): load kv scale from model weights
    KLLM_LOG_WARNING << fmt::format("Failed opening KV cache scaling factors file: {}. Using defalt value 1.0 ",
                                    kv_scale_path);
  } else {
    kv_scale_file >> kv_scale_json;
    kv_scale_file.close();
  }

  uint32_t num_layers = kv_scale_json.at("kv_cache").at("scaling_factor").at("0").size();
  // TODO(zhongzhicao): 进行简单校验，后续移除
  if (model_config.num_layer != num_layers) {
    KLLM_LOG_WARNING << fmt::format(
        "Loading KV cache scaling factors error, layer num not aligned. Using default value 1.0.");
    return;
  }

  // TODO(zhongzhicao): load kv scale for tensor_para_size > 1
  int tensor_parallel_size_kv_ = kv_scale_json.at("kv_cache").at("scaling_factor").size();
  if (tensor_parallel_size_kv_ != 1) {
    KLLM_LOG_WARNING << fmt::format(
        "Loading KV cache scaling factors from TP=0. Currently only tp_size = 1 is supported.");
  }
  for (uint32_t i = 0; i < model_config.num_layer; ++i) {
    model_config.k_scales[i] = model_config.v_scales[i] =
        kv_scale_json.at("kv_cache").at("scaling_factor").at("0").at(std::to_string(i));
  }

  KLLM_LOG_INFO << fmt::format(
      "Successfully Loaded KV cache scaling factors. Currently K and V are using the same scaling factors.");
}

Status Environment::ParseConfig(const std::string &config_file) {
  YamlReader yaml_reader;
  Status status = yaml_reader.LoadFile(config_file);
  if (!status.OK()) {
    KLLM_LOG_ERROR << "Load yaml config error." << status.GetMessage();
    return status;
  }

  // Read global setting.
  tensor_parallel_size_ =
      yaml_reader.GetScalar<size_t>(yaml_reader.GetRootNode(), "setting.global.tensor_para_size", 0);
  pipeline_parallel_size_ =
      yaml_reader.GetScalar<size_t>(yaml_reader.GetRootNode(), "setting.global.pipeline_para_size", 1);
  enable_lora_adapter_ =
      yaml_reader.GetScalar<bool>(yaml_reader.GetRootNode(), "setting.global.enable_lora_adapter", false);
  embed_tokens_use_cpu_ =
      yaml_reader.GetScalar<bool>(yaml_reader.GetRootNode(), "setting.global.embed_tokens_use_cpu", false);
  cuda_graph_ = yaml_reader.GetScalar<bool>(yaml_reader.GetRootNode(), "setting.global.enable_cuda_graph", false);
  is_version_report_ = yaml_reader.GetScalar<bool>(yaml_reader.GetRootNode(), "setting.global.is_version_report", true);
  if (tensor_parallel_size_ == 0) {
    int device_size = -1;
    GetDeviceCount(&device_size);
    tensor_parallel_size_ = static_cast<size_t>(device_size);
  }

  if (!(pipeline_parallel_size_ > 0 && tensor_parallel_size_ > 0)) {
    KLLM_THROW(fmt::format("Tensor Para Size {} and Pipeline Para Size {} should > 0", tensor_parallel_size_,
                           pipeline_parallel_size_));
  }

  // Read batch scheduler config.
  batch_scheduler_config_.schedule_strategy = static_cast<ScheduleStrategy>(
      yaml_reader.GetScalar<int>(yaml_reader.GetRootNode(), "setting.batch_scheduler.schedule_strategy", 0));
  batch_scheduler_config_.waiting_timeout_in_ms =
      yaml_reader.GetScalar<size_t>(yaml_reader.GetRootNode(), "setting.batch_scheduler.waiting_timeout_in_ms", 600000);
  batch_scheduler_config_.max_waiting_queue_len =
      yaml_reader.GetScalar<size_t>(yaml_reader.GetRootNode(), "setting.batch_scheduler.max_waiting_queue_len", 1200);
  batch_scheduler_config_.max_token_len =
      yaml_reader.GetScalar<size_t>(yaml_reader.GetRootNode(), "setting.batch_scheduler.max_token_len", 0);
  batch_scheduler_config_.max_step_tokens =
      yaml_reader.GetScalar<size_t>(yaml_reader.GetRootNode(), "setting.batch_scheduler.max_step_tokens", 4096);
  batch_scheduler_config_.max_batch_size =
      yaml_reader.GetScalar<size_t>(yaml_reader.GetRootNode(), "setting.batch_scheduler.max_batch_size", 128);
  batch_scheduler_config_.swapout_block_threshold =
      yaml_reader.GetScalar<float>(yaml_reader.GetRootNode(), "setting.batch_scheduler.swapout_block_threshold", 1.0);
  batch_scheduler_config_.swapin_block_threshold =
      yaml_reader.GetScalar<float>(yaml_reader.GetRootNode(), "setting.batch_scheduler.swapin_block_threshold", 2.0);
  batch_scheduler_config_.launch_block_threshold =
      yaml_reader.GetScalar<float>(yaml_reader.GetRootNode(), "setting.batch_scheduler.launch_block_threshold", 2.0);
  batch_scheduler_config_.preempt_mode = static_cast<PreemptMode>(
      yaml_reader.GetScalar<int>(yaml_reader.GetRootNode(), "setting.batch_scheduler.preempt_mode", 0));
  batch_scheduler_config_.split_fuse_token_num =
      yaml_reader.GetScalar<size_t>(yaml_reader.GetRootNode(), "setting.batch_scheduler.split_fuse_token_num", 0);

  // Read block manager config.
  block_manager_config_.host_allocator_config.block_token_num =
      yaml_reader.GetScalar<size_t>(yaml_reader.GetRootNode(), "setting.block_manager.block_token_num", 16);
  block_manager_config_.device_allocator_config.block_token_num =
      yaml_reader.GetScalar<size_t>(yaml_reader.GetRootNode(), "setting.block_manager.block_token_num", 16);
  block_manager_config_.reserved_device_memory_ratio = yaml_reader.GetScalar<float>(
      yaml_reader.GetRootNode(), "setting.block_manager.reserved_device_memory_ratio", 0.01);
  block_manager_config_.lora_deivce_memory_ratio =
      yaml_reader.GetScalar<float>(yaml_reader.GetRootNode(), "setting.block_manager.lora_deivce_memory_ratio", 0.0);
  block_manager_config_.block_device_memory_ratio =
      yaml_reader.GetScalar<float>(yaml_reader.GetRootNode(), "setting.block_manager.block_device_memory_ratio", -1.0);
  block_manager_config_.lora_host_memory_factor =
      yaml_reader.GetScalar<float>(yaml_reader.GetRootNode(), "setting.block_manager.lora_host_memory_factor", 10.0);
  block_manager_config_.block_host_memory_factor =
      yaml_reader.GetScalar<float>(yaml_reader.GetRootNode(), "setting.block_manager.block_host_memory_factor", 2.0);

  // Load cache manager config
  cache_manager_config_.swap_threadpool_size =
      yaml_reader.GetScalar<size_t>(yaml_reader.GetRootNode(), "setting.batch_scheduler.swap_threadpool_size", 2);
  cache_manager_config_.min_flexible_cache_num =
      yaml_reader.GetScalar<size_t>(yaml_reader.GetRootNode(), "setting.batch_scheduler.min_flexible_cache_num", 0);
  cache_manager_config_.block_token_num = block_manager_config_.device_allocator_config.block_token_num;
  cache_manager_config_.tensor_para_size = tensor_parallel_size_;
  cache_manager_config_.enable_prefix_caching =
      yaml_reader.GetScalar<bool>(yaml_reader.GetRootNode(), "setting.batch_scheduler.enable_auto_prefix_cache", false);
  // TODO(zakwang): Implement support for cases where prefix caching is disabled while split_fuse_token_num is non-zero.
  if (batch_scheduler_config_.split_fuse_token_num != 0 && !cache_manager_config_.enable_prefix_caching) {
    KLLM_LOG_WARNING << "While prefix caching is disabled，split_fuse_token_num will always be disabled. So set "
                        "split_fuse_token_num to 0.";
    batch_scheduler_config_.split_fuse_token_num = 0;
  }

  // Read profiler config.
  profiler_config_.trace_export_url =
      yaml_reader.GetScalar<std::string>(yaml_reader.GetRootNode(), "setting.profiler.trace_export_url", "");
  profiler_config_.metrics_export_url =
      yaml_reader.GetScalar<std::string>(yaml_reader.GetRootNode(), "setting.profiler.metrics_export_url", "");
  profiler_config_.export_interval_millis =
      yaml_reader.GetScalar<uint64_t>(yaml_reader.GetRootNode(), "setting.profiler.export_interval_millis", 60000);
  profiler_config_.export_timeout_millis =
      yaml_reader.GetScalar<uint64_t>(yaml_reader.GetRootNode(), "setting.profiler.export_timeout_millis", 1000);

  auto attributes = yaml_reader.GetMap(yaml_reader.GetRootNode(), "setting.profiler.attributes");
  for (auto it = attributes.begin(); it != attributes.end(); ++it) {
    std::string key = it->first.as<std::string>();
    std::string value = it->second.as<std::string>();
    profiler_config_.resource_attributes[key] = value;
  }
  // quantization_config in yaml takes effect when quantization_config in config.json is null.
  yaml_weight_quant_method_ = yaml_reader.GetScalar<std::string>(
      yaml_reader.GetRootNode(), "setting.quantization_config.weight.quant_method", "auto");

  yaml_gptq_backend_ = yaml_reader.GetScalar<std::string>(yaml_reader.GetRootNode(),
                                                          "setting.quantization_config.gptq_backend", "cutlass");

  // Read base model.
  std::string base_model_dir =
      yaml_reader.GetScalar<std::string>(yaml_reader.GetRootNode(), "model_spec.base_model.model_dir", "");
  std::string tokenizer_dir =
      yaml_reader.GetScalar<std::string>(yaml_reader.GetRootNode(), "model_spec.base_model.tokenizer_dir", "");
  if (tokenizer_dir.empty()) {
    tokenizer_dir = base_model_dir;
  }
  STATUS_CHECK_RETURN(ParseModelConfig(base_model_dir, tokenizer_dir));

  if (model_configs_[""].is_quant == true && model_configs_[""].quant_config.method == QUANT_FP8_E4M3 &&
      model_configs_[""].quant_config.is_checkpoint_fp8_serialized == false) {
    if (block_manager_config_.reserved_device_memory_ratio < 0.02) {
      block_manager_config_.reserved_device_memory_ratio = 0.02;
      KLLM_LOG_INFO
          << "When quant_method is fp8_e4m3, reserved_device_memory_ratio is set to at least 0.02 to prevent oom.";
    }
  } else if (model_configs_[""].is_quant == true && model_configs_[""].quant_config.method == QUANT_GPTQ) {
    if (block_manager_config_.reserved_device_memory_ratio < 0.02) {
      block_manager_config_.reserved_device_memory_ratio = 0.02;
      KLLM_LOG_INFO
          << "When quant_method is gptq, reserved_device_memory_ratio is set to at least 0.02 to prevent oom.";
    }
  }

  auto kv_cache_dtype_str = yaml_reader.GetScalar<std::string>(yaml_reader.GetRootNode(),
                                                               "setting.quantization_config.kv_cache.dtype", "auto");
  DataType kv_cache_dtype = model_configs_[""].weight_data_type;

#ifdef ENABLE_FLASH_ATTN_WITH_CACHE
  if (kv_cache_dtype_str == "fp8_e5m2" || kv_cache_dtype_str == "fp8_e4m3") {
    KLLM_THROW("FlashAttention not support fp8 kv cache");
  }
#else
  if (kv_cache_dtype_str == "fp8_e5m2") {
    kv_cache_dtype = TYPE_FP8_E5M2;
  } else if (kv_cache_dtype_str == "fp8_e4m3") {
    kv_cache_dtype = TYPE_FP8_E4M3;
    PrepareKVScales(base_model_dir, model_configs_[""]);
  }
#endif
  block_manager_config_.host_allocator_config.kv_cache_dtype = kv_cache_dtype;
  block_manager_config_.device_allocator_config.kv_cache_dtype = kv_cache_dtype;

  // Read lora models if needed.
  if (enable_lora_adapter_) {
    auto lora_nodes = yaml_reader.GetSequence(yaml_reader.GetRootNode(), "model_spec.lora_models");
    for (size_t i = 0; i < lora_nodes.size(); ++i) {
      std::string lora_model_name = yaml_reader.GetScalar<std::string>(lora_nodes[i], "model_name", "");
      std::string lora_model_dir = yaml_reader.GetScalar<std::string>(lora_nodes[i], "model_dir", "");
    }
  }

  return Status();
}

void Environment::SetReservedDeviceRatio(float reserved_device_memory_ratio) {
  block_manager_config_.reserved_device_memory_ratio = reserved_device_memory_ratio;
}

// read GGUF CONFIG
Status Environment::ParseModelConfigFromGGUF(const std::string &meta_file_path, ModelConfig &model_config) {
  // load meta data from GGUF file
  GGUFFileTensorLoader gguf_loader(meta_file_path);
  auto context = gguf_loader.GetMetadata();
  auto &metadata_map = context->metadata_map;

  // Helper functions to retrieve metadata values
  auto get_required_value = [&](const std::string &key, const std::string &error_msg) -> std::any {
    auto it = metadata_map.find(key);
    if (it != metadata_map.end()) {
      return it->second.value;
    } else {
      throw std::runtime_error(error_msg);
    }
  };

  auto get_optional_value = [&](const std::string &key, const std::any &default_value) -> std::any {
    auto it = metadata_map.find(key);
    if (it != metadata_map.end()) {
      return it->second.value;
    } else {
      return default_value;
    }
  };

  try {
    model_config.type = std::any_cast<std::string>(
        get_required_value("general.architecture", "Model type is not supported in GGUF format."));
    if (model_config.type != "llama") {
      throw std::runtime_error("Model type is not supported in GGUF format.");
    }

    std::string model_type = model_config.type;
    uint32_t ftype =
        std::any_cast<uint32_t>(get_optional_value("general.file_type", GGUFModelFileType::LLAMA_FTYPE_MOSTLY_F16));
    model_config.weight_data_type = GGUFFileTensorLoader::ConverGGUFModelFileTypeToDataType(ftype);
    model_config.head_num = std::any_cast<uint32_t>(
        get_required_value(model_type + ".attention.head_count", "Model head_num is not supported in GGUF format."));
    model_config.num_key_value_heads = std::any_cast<uint32_t>(get_required_value(
        model_type + ".attention.head_count_kv", "Model num_key_value_heads is not supported in GGUF format."));
    model_config.inter_size = std::any_cast<uint32_t>(
        get_required_value(model_type + ".feed_forward_length", "Model inter_size is not supported in GGUF format."));
    model_config.vocab_size = std::any_cast<uint32_t>(
        get_required_value(model_type + ".vocab_size", "Model vocab_size is not supported in GGUF format."));
    model_config.num_layer = std::any_cast<uint32_t>(
        get_required_value(model_type + ".block_count", "Model num_layer is not supported in GGUF format."));
    model_config.hidden_units = std::any_cast<uint32_t>(
        get_required_value(model_type + ".embedding_length", "Model hidden_units is not supported in GGUF format."));
    model_config.rope_theta = std::any_cast<float>(get_optional_value(model_type + ".rope.freq_base", 10000.0f));
    model_config.layernorm_eps =
        std::any_cast<float>(get_optional_value(model_type + ".attention.layer_norm_rms_epsilon", 1e-6));
    model_config.start_id = std::any_cast<uint32_t>(get_optional_value("tokenizer.ggml.bos_token_id", 1));
    model_config.pad_id = std::any_cast<uint32_t>(get_optional_value("tokenizer.ggml.padding_token_id", (uint32_t)0));
    model_config.max_position_embeddings =
        std::any_cast<uint32_t>(get_optional_value(model_type + ".context_length", 2048));
    model_config.tie_word_embeddings =
        std::any_cast<bool>(get_optional_value(model_type + ".tie_word_embeddings", false));
    model_config.is_visual = metadata_map.count("visual");

    // Handle 'end_ids' which might be a single value or an array
    if (metadata_map.count("tokenizer.ggml.eos_token_id")) {
      auto eos_token_meta = metadata_map["tokenizer.ggml.eos_token_id"];
      if (eos_token_meta.type == GGUFMetaValueType::GGUF_METADATA_VALUE_TYPE_ARRAY) {
        model_config.end_ids = std::any_cast<std::vector<uint32_t>>(eos_token_meta.value);
      } else {
        model_config.end_ids = {std::any_cast<uint32_t>(eos_token_meta.value)};
      }
    } else {
      model_config.end_ids = {2};
    }
    model_config.max_token_num = model_config.max_position_embeddings;

    size_t size_per_head = model_config.hidden_units / model_config.head_num;
    model_config.size_per_head = size_per_head;
    model_config.rotary_embedding = size_per_head;
  } catch (const std::exception &e) {
    return Status(RET_INVALID_ARGUMENT, e.what());
  }

  return Status();
}

Status Environment::ParseModelConfig(const std::string &model_dir, const std::string &tokenizer_dir) {
  std::filesystem::path abs_model_dir_path = std::filesystem::absolute(model_dir);
  std::filesystem::path abs_tokenizer_dir_path = std::filesystem::absolute(tokenizer_dir);
  std::string config_file = abs_model_dir_path.u8string() + "/config.json";
  ModelFileFormat model_file_format;
  ModelConfig model_config;
  Status status;

  model_config.path = abs_model_dir_path.u8string();
  model_config.tokenizer_path = abs_tokenizer_dir_path.u8string();
  model_config.tensor_para_size = tensor_parallel_size_;

  std::vector<std::string> weights_file_list = SearchLocalPath(model_dir, model_file_format);
  model_config.model_file_format = model_file_format;

  if (model_file_format == GGUF) {
    status = ParseModelConfigFromGGUF(weights_file_list[0], model_config);
    if (!status.OK()) {
      return status;
    }
  } else {
    nlohmann::json config_json;
    std::ifstream file(config_file);
    if (!file.is_open()) {
      KLLM_LOG_ERROR << fmt::format("Load model config file: {} error.", config_file);
      return Status(RetCode::RET_INVALID_ARGUMENT, fmt::format("Load model config file: {} error.", config_file));
    } else {
      file >> config_json;
      file.close();
    }

    model_config.weight_data_type = GetModelDataType(config_json, model_config);
    model_config.type = config_json.at("model_type");
    if (model_config.type == "chatglm") {
      PrepareChatglmAttributes(config_json, model_config);
    } else if (model_config.type == "openai-gpt") {  // GPT-1
      // For fairseq transformer, we use the same config as huggingface openai-gpt,
      // and distinguish them by the vocab size.
      if (config_json.at("vocab_size") == 7000) {
        model_config.type = "fairseq-transformer";
        PrepareFairseqTransformerAttributes(config_json, model_config);
      } else {
        PrepareGPT1Attributes(config_json, model_config);
      }
    } else if (model_config.type == "gpt2") {
      PrepareGPT2Attributes(config_json, model_config);
    } else if (model_config.type == "qwen2_moe") {
      PrepareQwen2MoeAttributes(config_json, model_config);
    } else if (model_config.type == "mixtral") {
      PrepareMixtralAttributes(config_json, model_config);
    } else {
      PrepareCommonModelAttributes(config_json, model_config);
    }
    ParseModelMaxLength(config_json, model_config);
    ParseModelQuantConfig(config_json, model_config, yaml_weight_quant_method_, yaml_gptq_backend_);

    UpdateEndIdFromGeneration(model_dir, model_config);
  }

  if (tensor_parallel_size_ > model_config.num_key_value_heads ||
      model_config.num_key_value_heads % tensor_parallel_size_ != 0) {
    KLLM_THROW(
        fmt::format("The size of key_value_heads cannot be evenly divided by the size of tensor_parallel_size_. "
                    "{} % {} != 0 ",
                    model_config.num_key_value_heads, tensor_parallel_size_));
  }

  if (batch_scheduler_config_.max_token_len > 0) {
    if (batch_scheduler_config_.max_token_len > model_config.max_token_num) {
      KLLM_LOG_ERROR << fmt::format(
          "The max_token_num configured in the model's config.json is less than the "
          "max_token_len configured in the ksana yaml file. {} < {}",
          model_config.max_token_num, batch_scheduler_config_.max_token_len);
      return Status(RetCode::RET_INVALID_ARGUMENT,
                    fmt::format("Load model config file: {} error. The max_token_num configured in the model's "
                                "config.json is less than the max_token_len configured in the ksana yaml file."
                                " {} < {}",
                                config_file, batch_scheduler_config_.max_token_len, model_config.max_token_num));
    }
    model_config.max_token_num = batch_scheduler_config_.max_token_len;
  }
  batch_scheduler_config_.max_token_len = model_config.max_token_num;
  batch_scheduler_config_.max_step_tokens =
      std::max(batch_scheduler_config_.max_step_tokens, model_config.max_token_num + 1);
  model_config.block_token_num = block_manager_config_.device_allocator_config.block_token_num;
  model_config.max_batch_size = batch_scheduler_config_.max_batch_size;
  model_config.max_scheduler_token_num = batch_scheduler_config_.max_step_tokens;
  model_config.k_scales = std::vector<float>(model_config.num_layer, 1.0f);  // default k scale value
  model_config.v_scales = std::vector<float>(model_config.num_layer, 1.0f);  // default v scale value
  model_configs_[model_config.name] = model_config;

  KLLM_LOG_DEBUG << fmt::format("Load model {} from config file: {} success.", model_config.name, model_config.path);
  return Status();
}

Status Environment::ParseOptions(int argc, char **argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);

  endpoint_config_.host = FLAGS_host;
  endpoint_config_.port = static_cast<uint32_t>(FLAGS_port);

  endpoint_config_.host = FLAGS_host;
  endpoint_config_.port = static_cast<uint32_t>(FLAGS_port);

  Status status = ParseConfig(FLAGS_config_file);
  if (!status.OK()) {
    KLLM_LOG_ERROR << fmt::format("Parse config file {} error: {}", FLAGS_config_file, status.GetMessage());
    return status;
  }

  return Status();
}

Status Environment::InitializeBlockManagerConfig() {
  KLLM_CHECK_WITH_INFO(model_configs_.size() > 0, "No model configed.");
  const ModelConfig &model_config = model_configs_.begin()->second;

  if (pipeline_config_.lower_layer_idx < 0 || pipeline_config_.upper_layer_idx < 0) {
    pipeline_config_.lower_layer_idx = 0;
    pipeline_config_.upper_layer_idx = model_configs_.begin()->second.num_layer - 1;
  }

  size_t node_layer_num = pipeline_config_.upper_layer_idx - pipeline_config_.lower_layer_idx + 1;
  size_t token_size = (node_layer_num / GetPipeLineParallelSize()) *
                      (model_config.num_key_value_heads / GetTensorParallelSize()) * model_config.size_per_head;
  size_t block_token_num = block_manager_config_.device_allocator_config.block_token_num;
  size_t block_dtype_size = 0ul;
  block_dtype_size = GetTypeSize(block_manager_config_.device_allocator_config.kv_cache_dtype);

  block_manager_config_.host_allocator_config.block_size = token_size * block_token_num * 2 * block_dtype_size;
  block_manager_config_.device_allocator_config.block_size = token_size * block_token_num * 2 * block_dtype_size;

  KLLM_LOG_INFO << fmt::format("Init block num for key or value: ({} / {}) * ({} / {}) * {} = {}", node_layer_num,
                               GetPipeLineParallelSize(), model_config.num_key_value_heads, GetTensorParallelSize(),
                               model_config.size_per_head, token_size);

  KLLM_LOG_INFO << fmt::format("Init token size (bytes) of init block for both key and value: {} * {} * 2 * {} = {}",
                               token_size, block_token_num, block_dtype_size,
                               block_manager_config_.device_allocator_config.block_size);

  block_manager_config_.host_allocator_config.device = MemoryDevice::MEMORY_HOST;
  block_manager_config_.device_allocator_config.device = MemoryDevice::MEMORY_DEVICE;

  // The default block number, will be overwrited through memory usage.
  block_manager_config_.host_allocator_config.blocks_num = 512 * 10;
  block_manager_config_.device_allocator_config.blocks_num = 512;

  return CheckEnvironment();
}

Status Environment::CheckEnvironment() {
  if (block_manager_config_.host_allocator_config.block_size !=
      block_manager_config_.device_allocator_config.block_size) {
    return Status(RET_INVALID_ARGUMENT, fmt::format("block size of device and host is not equal, {} vs {}.",
                                                    block_manager_config_.host_allocator_config.block_size,
                                                    block_manager_config_.device_allocator_config.block_size));
  }

  return Status();
}

Status Environment::GetModelConfigs(std::unordered_map<std::string, ModelConfig> &model_configs) {
  model_configs = model_configs_;
  return Status();
}

Status Environment::GetModelConfig(const std::string &model_name, ModelConfig &model_config) {
  auto it = model_configs_.find(model_name);
  if (it == model_configs_.end()) {
    return Status(RET_INVALID_ARGUMENT, fmt::format("No model named {} found.", model_name));
  }

  model_config = it->second;
  return Status();
}

Status Environment::GetBatchSchedulerConfig(BatchSchedulerConfig &batch_scheduler_config) {
  batch_scheduler_config = batch_scheduler_config_;
  return Status();
}

Status Environment::GetCacheManagerConfig(CacheManagerConfig &cache_manager_config) {
  cache_manager_config = cache_manager_config_;
  return Status();
}

Status Environment::GetBlockManagerConfig(BlockManagerConfig &block_manager_config) {
  block_manager_config = block_manager_config_;
  return Status();
}

Status Environment::GetEndpointConfig(EndpointConfig &endpoint_config) {
  endpoint_config = endpoint_config_;
  return Status();
}

Status Environment::GetProfilerConfig(ProfilerConfig &profiler_config) {
  profiler_config = profiler_config_;
  return Status();
}

bool Environment::IsPrefixCachingEnabled() { return cache_manager_config_.enable_prefix_caching; }

bool Environment::IsFlexibleCachingEnabled() { return cache_manager_config_.min_flexible_cache_num > 0; }

}  // namespace ksana_llm
