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
#include "ksana_llm/models/gpt/gpt_config.h"
#include "ksana_llm/utils/device_utils.h"
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
                           std::string &yaml_weight_quant_method) {
  model_config.is_quant = config_json.contains("quantization_config");
  if (model_config.is_quant) {
    std::string quant_method = config_json["quantization_config"].at("quant_method");
    if (quant_method == "gptq") {
      model_config.quant_config.method = QUANT_GPTQ;
      model_config.quant_config.bits = config_json["quantization_config"].at("bits");
      model_config.quant_config.group_size = config_json["quantization_config"].at("group_size");
      KLLM_LOG_INFO << fmt::format("using quant model, quant method: {}, bits: {}, group_size: {}", quant_method,
                                   model_config.quant_config.bits, model_config.quant_config.group_size);
    } else if (quant_method == "fp8") {
      // TODO(catheywang): support fp8 quantized weight loading.
      KLLM_THROW("Loading of fp8 weights from checkpoint is not supported.");
      model_config.quant_config.method = QUANT_FP8_E4M3;
      model_config.quant_config.is_checkpoint_fp8_serialized = true;
      model_config.quant_config.is_activation_scheme_static =
          (config_json["quantization_config"].at("activation_scheme") == "static");
      KLLM_LOG_INFO << fmt::format(
          "using quant model, quant method: {}, is_checkpoint_fp8_serialized: {}, is_activation_scheme_static: {}",
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
          "using quant model, quant method: {}, is_checkpoint_fp8_serialized: {}, is_activation_scheme_static: {}",
          yaml_weight_quant_method, model_config.quant_config.is_checkpoint_fp8_serialized,
          model_config.quant_config.is_activation_scheme_static);
    } else {
      KLLM_THROW(fmt::format("Not support quant_method {}.", yaml_weight_quant_method));
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
    model_config.rope_scaling_factor_config.factor = rope_scaling_setting.value("factor", 1.0f);
    KLLM_LOG_DEBUG << fmt::format("rope_scaling type: {} factor: {}", model_config.rope_scaling_factor_config.type,
                                  model_config.rope_scaling_factor_config.factor);

    if (model_config.rope_scaling_factor_config.type != "su") {
      if (model_config.rope_scaling_factor_config.type == "yarn") {
        derived_max_model_len = rope_scaling_setting.value("original_max_position_embeddings", derived_max_model_len);
      }
      derived_max_model_len *= model_config.rope_scaling_factor_config.factor;
    }
  }

  model_config.max_token_num = static_cast<int>(derived_max_model_len);
  KLLM_LOG_DEBUG << "Model Max Token Num = " << model_config.max_token_num;
}

void PrepareCommonModelAttributes(const nlohmann::json &config_json, ModelConfig &model_config) {
  model_config.head_num = config_json.at("num_attention_heads");
  model_config.num_key_value_heads = config_json.value("num_key_value_heads", model_config.head_num);
  model_config.inter_size = config_json.at("intermediate_size");
  model_config.vocab_size = config_json.at("vocab_size");
  model_config.num_layer = config_json.at("num_hidden_layers");
  model_config.hidden_units = config_json.at("hidden_size");
  model_config.rope_theta = config_json.value("rope_theta", 10000.0f);
  model_config.layernorm_eps = config_json.value("rms_norm_eps", 1e-6);
  model_config.layernorm_eps = config_json.value("layer_norm_epsilon", model_config.layernorm_eps);
  model_config.start_id = config_json.value("bos_token_id", 1);
  model_config.end_ids = std::vector<int>{config_json.value("eos_token_id", 2)};
  model_config.pad_id = config_json.value("pad_token_id", 0);
  model_config.max_position_embeddings = config_json.value("max_position_embeddings", 2048);
  model_config.tie_word_embeddings = config_json.value("tie_word_embeddings", false);
  model_config.is_visual = config_json.contains("visual");

  size_t size_per_head = model_config.hidden_units / model_config.head_num;
  model_config.size_per_head = size_per_head;
  model_config.rotary_embedding = size_per_head;
}

void UpdateEndIdFromGeneration(const std::string &model_dir, ModelConfig &model_config) {
  // Priority: `generation_config` argument > `model.generation_config`
  // It is recommended to set all generation parameters in `generation_config`
  // Refer to
  // https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py#L1300
  std::filesystem::path raw_model_dir_path = model_dir;
  std::filesystem::path abs_model_dir_path = std::filesystem::absolute(raw_model_dir_path);
  std::string config_file = abs_model_dir_path.u8string() + "/generation_config.json";

  nlohmann::json config_json;
  std::ifstream file(config_file);
  if (!file.is_open()) {
    KLLM_LOG_WARNING << fmt::format("Load generation config file: {} error.", config_file);
    return;
  } else {
    file >> config_json;
    file.close();
  }

  if (!config_json.contains("eos_token_id")) {
    return;
  }

  std::vector<int> end_ids;
  if (config_json.at("eos_token_id").is_array()) {
    for (int end_id : config_json.at("eos_token_id")) {
      if (std::find(end_ids.begin(), end_ids.end(), end_id) == end_ids.end()) {
        end_ids.push_back(end_id);
      }
    }
  } else {
    end_ids.push_back(config_json.at("eos_token_id"));
  }
  if (end_ids != model_config.end_ids) {
    KLLM_LOG_WARNING << fmt::format("eos_token_id: {} in model config is ignored by {} in generation config",
                                    model_config.end_ids.front(), fmt::join(end_ids, ", "));
    model_config.end_ids = end_ids;
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
      yaml_reader.GetScalar<size_t>(yaml_reader.GetRootNode(), "setting.batch_scheduler.max_waiting_queue_len", 256);
  batch_scheduler_config_.max_token_len =
      yaml_reader.GetScalar<size_t>(yaml_reader.GetRootNode(), "setting.batch_scheduler.max_token_len", 0);
  batch_scheduler_config_.max_step_tokens =
      yaml_reader.GetScalar<size_t>(yaml_reader.GetRootNode(), "setting.batch_scheduler.max_step_tokens", 4096);
  batch_scheduler_config_.max_batch_size =
      yaml_reader.GetScalar<size_t>(yaml_reader.GetRootNode(), "setting.batch_scheduler.max_batch_size", 8);
  batch_scheduler_config_.swapout_block_threshold =
      yaml_reader.GetScalar<float>(yaml_reader.GetRootNode(), "setting.batch_scheduler.swapout_block_threshold", 1.0);
  batch_scheduler_config_.swapin_block_threshold =
      yaml_reader.GetScalar<float>(yaml_reader.GetRootNode(), "setting.batch_scheduler.swapin_block_threshold", 2.0);
  batch_scheduler_config_.launch_block_threshold =
      yaml_reader.GetScalar<float>(yaml_reader.GetRootNode(), "setting.batch_scheduler.launch_block_threshold", 2.0);
  batch_scheduler_config_.preempt_mode = static_cast<PreemptMode>(
      yaml_reader.GetScalar<int>(yaml_reader.GetRootNode(), "setting.batch_scheduler.preempt_mode", 0));

  // Read block manager config.
  block_manager_config_.host_allocator_config.block_token_num =
      yaml_reader.GetScalar<size_t>(yaml_reader.GetRootNode(), "setting.block_manager.block_token_num", 16);
  block_manager_config_.device_allocator_config.block_token_num =
      yaml_reader.GetScalar<size_t>(yaml_reader.GetRootNode(), "setting.block_manager.block_token_num", 16);
  block_manager_config_.reserved_device_memory_ratio = yaml_reader.GetScalar<float>(
      yaml_reader.GetRootNode(), "setting.block_manager.reserved_device_memory_ratio", 0.05);
  block_manager_config_.lora_deivce_memory_ratio =
      yaml_reader.GetScalar<float>(yaml_reader.GetRootNode(), "setting.block_manager.lora_deivce_memory_ratio", 0.0);
  block_manager_config_.block_device_memory_ratio =
      yaml_reader.GetScalar<float>(yaml_reader.GetRootNode(), "setting.block_manager.block_device_memory_ratio", -1.0);
  block_manager_config_.lora_host_memory_factor =
      yaml_reader.GetScalar<float>(yaml_reader.GetRootNode(), "setting.block_manager.lora_host_memory_factor", 10.0);
  block_manager_config_.block_host_memory_factor =
      yaml_reader.GetScalar<float>(yaml_reader.GetRootNode(), "setting.block_manager.block_host_memory_factor", 10.0);

  // Load cache manager config
  cache_manager_config_.swap_threadpool_size =
      yaml_reader.GetScalar<size_t>(yaml_reader.GetRootNode(), "setting.batch_scheduler.swap_threadpool_size", 2);
  cache_manager_config_.block_token_num = block_manager_config_.device_allocator_config.block_token_num;
  cache_manager_config_.tensor_para_size = tensor_parallel_size_;
  cache_manager_config_.enable_preifx_caching =
      yaml_reader.GetScalar<bool>(yaml_reader.GetRootNode(), "setting.batch_scheduler.enable_auto_prefix_cache", false);

  // Read profiler config.
  profiler_config_.stat_interval_second =
      yaml_reader.GetScalar<size_t>(yaml_reader.GetRootNode(), "setting.profiler.stat_interval_second", 60);
  profiler_config_.stat_buffer_size =
      yaml_reader.GetScalar<size_t>(yaml_reader.GetRootNode(), "setting.profiler.stat_buffer_size", 1024);
  profiler_config_.report_threadpool_size =
      yaml_reader.GetScalar<size_t>(yaml_reader.GetRootNode(), "setting.profiler.report_threadpool_size", 4);

  // quantization_config in yaml takes effect when quantization_config in config.json is null.
  yaml_weight_quant_method_ = yaml_reader.GetScalar<std::string>(
      yaml_reader.GetRootNode(), "setting.quantization_config.weight.quant_method", "auto");

  // Read base model.
  std::string base_model_dir =
      yaml_reader.GetScalar<std::string>(yaml_reader.GetRootNode(), "model_spec.base_model.model_dir", "");
  status = ParseModelConfig(base_model_dir);
  if (!status.OK()) {
    return status;
  }

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

  if (kv_cache_dtype_str == "fp8_e5m2") {
    kv_cache_dtype = TYPE_FP8_E5M2;
  } else if (kv_cache_dtype_str == "fp8_e4m3") {
    kv_cache_dtype = TYPE_FP8_E4M3;
    PrepareKVScales(base_model_dir, model_configs_[""]);
  }
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

  InitializeBlockManagerConfig();
  return CheckEnvironment();
}

Status Environment::ParseModelConfig(const std::string &model_dir) {
  std::filesystem::path raw_model_dir_path = model_dir;
  std::filesystem::path abs_model_dir_path = std::filesystem::absolute(raw_model_dir_path);
  std::string config_file = abs_model_dir_path.u8string() + "/config.json";

  nlohmann::json config_json;
  std::ifstream file(config_file);
  if (!file.is_open()) {
    KLLM_LOG_ERROR << fmt::format("Load model config file: {} error.", config_file);
    return Status(RetCode::RET_INVALID_ARGUMENT, fmt::format("Load model config file: {} error.", config_file));
  } else {
    file >> config_json;
    file.close();
  }

  ModelConfig model_config;
  model_config.path = abs_model_dir_path.u8string();
  model_config.weight_data_type = GetModelDataType(config_json, model_config);
  model_config.tensor_para_size = tensor_parallel_size_;

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
  } else {
    PrepareCommonModelAttributes(config_json, model_config);
  }
  ParseModelMaxLength(config_json, model_config);
  ParseModelQuantConfig(config_json, model_config, yaml_weight_quant_method_);

  UpdateEndIdFromGeneration(model_dir, model_config);

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
          batch_scheduler_config_.max_token_len, model_config.max_token_num);
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
  gflags::ParseCommandLineFlags(&argc, &argv, true);

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

void Environment::InitializeBlockManagerConfig() {
  KLLM_CHECK_WITH_INFO(model_configs_.size() > 0, "No model configed.");
  const ModelConfig &model_config = model_configs_.begin()->second;

  size_t token_size = (model_config.num_layer / GetPipeLineParallelSize()) *
                      (model_config.num_key_value_heads / GetTensorParallelSize()) * model_config.size_per_head;
  size_t block_token_num = block_manager_config_.device_allocator_config.block_token_num;
  size_t block_dtype_size = 0ul;
  block_dtype_size = GetTypeSize(block_manager_config_.device_allocator_config.kv_cache_dtype);

  block_manager_config_.host_allocator_config.block_size = token_size * block_token_num * 2 * block_dtype_size;
  block_manager_config_.device_allocator_config.block_size = token_size * block_token_num * 2 * block_dtype_size;

  KLLM_LOG_INFO << fmt::format("Init block num for key or value: ({} / {}) * ({} / {}) * {} = {}",
                               model_config.num_layer, GetPipeLineParallelSize(), model_config.num_key_value_heads,
                               GetTensorParallelSize(), model_config.size_per_head, token_size);

  KLLM_LOG_INFO << fmt::format("Init token size (bytes) of init block for both key and value: {} * {} * 2 * {} = {}",
                               token_size, block_token_num, block_dtype_size,
                               block_manager_config_.device_allocator_config.block_size);

  block_manager_config_.host_allocator_config.device = MemoryDevice::MEMORY_HOST;
  block_manager_config_.device_allocator_config.device = MemoryDevice::MEMORY_DEVICE;

  // The default block number, will be overwrited through memory usage.
  block_manager_config_.host_allocator_config.blocks_num = 512 * 10;
  block_manager_config_.device_allocator_config.blocks_num = 512;
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

bool Environment::IsPrefixCachingEnabled() { return cache_manager_config_.enable_preifx_caching; }

}  // namespace ksana_llm
