/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "configure/environment.h"

#include <filesystem>
#include <fstream>
#include <string>

#include "fmt/core.h"
#include "gflags/gflags.h"

#include "utils/memory_utils.h"

#include "device/device_utils.h"
#include "models/common/common_config.h"
#include "models/common_moe/moe_config.h"
#include "utils/attention_backend/attention_backend_manager.h"
#include "utils/gguf_file_tensor_loader.h"
#include "utils/logger.h"
#include "utils/optional_file.h"
#include "utils/ret_code.h"
#include "utils/status.h"
#include "utils/string_utils.h"

DEFINE_string(config_file, "examples/ksana_llm.yaml", "The config file path");
DEFINE_string(host, "localhost", "HTTP service hostname, default is localhost");
DEFINE_int32(port, 8080, "HTTP service port, default is 8080");
namespace ksana_llm {

namespace {

// 把 dsv4_attn_config.compress_ratios 按 override 后的 num_layer 重新对齐:
//   原数组布局: [main(0..N-1), mtp(N..N+M-1)], 其中 N = 原 num_hidden_layers,
//               M = num_nextn_predict_layers。
//   重排后布局: [main(0..n-1), mtp(N..N+M-1)], 其中 n = override 后的 num_layer。
// 这样, 重排后 compress_ratios[i] 对所有 global layer index i 都与运行时实际语义一致:
//   * 0 .. num_layer-1: 截断后的 main 层
//   * num_layer .. num_layer+num_nextn_predict_layers-1: 原始模型尾部的真实 MTP 项
// 仅在 json_model_override_args 实际改小了 num_layer (compress_ratios.size() > num_layer + M)
// 时才会触发重排; 其它情况 (数组长度已对齐, 或无 MTP) 保持原值不变。
void RealignDsv4CompressRatiosAfterOverride(ModelConfig &model_config) {
  auto &compress_ratios = model_config.dsv4_attn_config.compress_ratios;
  const size_t mtp_count = static_cast<size_t>(model_config.num_nextn_predict_layers);
  const size_t target_size = static_cast<size_t>(model_config.num_layer) + mtp_count;

  if (compress_ratios.size() == target_size) {
    return;
  }
  if (compress_ratios.size() < target_size) {
    KLLM_LOG_WARNING << fmt::format(
        "dsv4 compress_ratios size {} smaller than expected num_layer+num_nextn_predict_layers={} "
        "after json_model_override_args; skip realignment.",
        compress_ratios.size(), target_size);
    return;
  }
  if (mtp_count > compress_ratios.size()) {
    KLLM_LOG_WARNING << fmt::format(
        "dsv4 compress_ratios size {} smaller than num_nextn_predict_layers={}; skip realignment.",
        compress_ratios.size(), mtp_count);
    return;
  }

  std::vector<uint32_t> aligned_compress_ratios;
  aligned_compress_ratios.reserve(target_size);
  aligned_compress_ratios.insert(aligned_compress_ratios.end(), compress_ratios.begin(),
                                 compress_ratios.begin() + model_config.num_layer);
  aligned_compress_ratios.insert(aligned_compress_ratios.end(), compress_ratios.end() - mtp_count,
                                 compress_ratios.end());
  compress_ratios = std::move(aligned_compress_ratios);
}

}  // namespace

size_t GetDecodeTokenNumThreshold() {
  static size_t decode_token_num_threshold = 0;
  if (decode_token_num_threshold == 0) {
    const auto env = Singleton<Environment>::GetInstance();
    RuntimeConfig runtime_config;
    ModelConfig model_config;
    if (env->GetRuntimeConfig(runtime_config).OK() && runtime_config.mtp_step_num > 0 &&
        runtime_config.attn_backend_config.enable_blocked_multi_token_forwarding_kv) {
      env->GetModelConfig(model_config);
      // For models with MTP enabled and blocked multi-token forwarding KV (e.g., DeepSeek V3, GLM4-MoE),
      // the threshold is set to mtp_step_num + 1, corresponding to the number of tokens during the
      // validation phase. This allows multi-token decode to use the kPage path instead of falling
      // back to the kFlash prefill path.
      // PTP in steady state needs 2*ptp_step_num (verify + placeholder) + 1 (generated) tokens.
      decode_token_num_threshold = runtime_config.mtp_step_num + 2 * model_config.ptp_step_num + 1;
    } else {
      // Set to one by default
      decode_token_num_threshold = 1;
    }
  }
  return decode_token_num_threshold;
}

Environment::Environment() {}

void Environment::InitializeAttentionBackend() {
  // Initialize attention backend manager using singleton pattern.
  auto attention_manager = AttentionBackendManager::GetInstance();
  if (attention_manager->Initialize()) {
    KLLM_LOG_INFO << "Attention backend initialized.";
  } else {
    KLLM_LOG_ERROR << "Failed to initialize attention backend.";
  }
}

void Environment::Reset() {
  model_config_initialized_ = false;
  model_config_ = {};
  yaml_gptq_backend_ = "";
  yaml_weight_quant_method_ = "";
  is_version_report_ = true;
  profiler_config_ = {};
  schedule_config_parser_.Reset();
}

Status Environment::ParseConfig(const std::string &config_file, const std::string &model_dir_override,
                                const std::string &model_config_filename, const std::string &json_model_override_args) {
  Reset();
  YamlReader yaml_reader;
  const Status status = yaml_reader.LoadFile(config_file);
  if (!status.OK()) {
    KLLM_LOG_ERROR << "Load yaml config error." << status.GetMessage();
    return status;
  }

  // Read profiler config.
  profiler_config_.metrics_export_url =
      yaml_reader.GetScalar<std::string>(yaml_reader.GetRootNode(), "setting.profiler.metrics_export_url", "");
  profiler_config_.export_interval_millis =
      yaml_reader.GetScalar<uint64_t>(yaml_reader.GetRootNode(), "setting.profiler.export_interval_millis", 60000);
  profiler_config_.export_timeout_millis =
      yaml_reader.GetScalar<uint64_t>(yaml_reader.GetRootNode(), "setting.profiler.export_timeout_millis", 1000);

  auto attributes = yaml_reader.GetMap(yaml_reader.GetRootNode(), "setting.profiler.attributes");
  for (auto it = attributes.begin(); it != attributes.end(); ++it) {
    const std::string &key = it->first.as<std::string>();
    const std::string &value = it->second.as<std::string>();
    profiler_config_.resource_attributes[key] = value;
  }

  // Read log config.
  LogConfig log_config;
  log_config.log_dir = yaml_reader.GetScalar<std::string>(yaml_reader.GetRootNode(), "setting.log.log_dir", "log/");
  // 支持人类可读格式，如 "512mb", "1gb"，也支持纯数字（字节）
  std::string max_file_size_str =
      yaml_reader.GetScalar<std::string>(yaml_reader.GetRootNode(), "setting.log.max_file_size", "512mb");
  log_config.max_file_size = ParseFileSize(max_file_size_str);
  log_config.max_backup_count =
      yaml_reader.GetScalar<int>(yaml_reader.GetRootNode(), "setting.log.max_backup_count", 2);
  SetLogConfig(log_config);

  // quantization_config in yaml takes effect when quantization_config in
  // config.json is null.
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
  if (!model_dir_override.empty()) {
    base_model_dir = model_dir_override;
    tokenizer_dir = model_dir_override;
  }

  // Read is_embedding_model from yaml before model config parsing.
  model_config_.is_embedding_model =
      yaml_reader.GetScalar<bool>(yaml_reader.GetRootNode(), "model_spec.base_model.is_embedding_model", false);

  STATUS_CHECK_RETURN(ParseModelAndScheduleConfig(yaml_reader, base_model_dir, tokenizer_dir, model_config_filename,
                                                  json_model_override_args));

  return Status();
}

void Environment::SetReservedDeviceRatio(float reserved_device_memory_ratio) {
  schedule_config_parser_.SetReservedDeviceRatio(reserved_device_memory_ratio);
}

Status Environment::UpdateModelConfig() {
  KLLM_CHECK_WITH_INFO(model_config_initialized_, "model_config not initialized.");
  return schedule_config_parser_.UpdateModelConfig(model_config_);
}

Status Environment::ParseModelAndScheduleConfig(YamlReader &yaml_reader, const std::string &model_dir,
                                                const std::string &tokenizer_dir,
                                                const std::string &model_config_filename,
                                                const std::string &json_model_override_args) {
  KLLM_CHECK_WITH_INFO(!model_config_initialized_, "model_config_initialized_ initialized.");
  EnvModelConfigParser model_config_parser(yaml_weight_quant_method_, yaml_gptq_backend_);
  model_config_parser.ParseModelConfig(model_dir, tokenizer_dir, model_config_filename, model_config_);

  if (!json_model_override_args.empty()) {
    KLLM_LOG_INFO << "Overriding members in model_config_ with json_model_override_args: " << json_model_override_args;
    try {
      nlohmann::json override_json = nlohmann::json::parse(json_model_override_args);
      ApplyJsonModelOverride(override_json, model_config_);
    } catch (const std::exception &e) {
      KLLM_LOG_WARNING << "json_model_override_args: failed to parse JSON, skipping override: " << e.what();
    }

    // num_layer 可能在上面被改小。对 dsv4 而言, compress_ratios 是按"原始模型"读的, 需要
    // 立即重排, 否则后续 cache layout / Compressor / Dsv4Mqa / CompressorMeta 等都会按
    // 错位的 ratio 来初始化 MTP 槽位, 最终触发异步 CUDA 越界。
    if (model_config_.use_dsv4_attn) {
      RealignDsv4CompressRatiosAfterOverride(model_config_);
    }
  }

  schedule_config_parser_.ParseScheduleConfig(yaml_reader, model_config_);
  schedule_config_parser_.UpdateModelConfig(model_config_);

  schedule_config_parser_.UpdateMembers(model_dir, model_config_);

  model_config_initialized_ = true;

  KLLM_LOG_INFO << fmt::format(
      "Load model {} from config file: {} success. num_layer={}, "
      "hidden_units={}, head_num={}, vocab_size={}",
      model_config_.name, model_config_.path + "/" + model_config_filename, model_config_.num_layer,
      model_config_.hidden_units, model_config_.head_num, model_config_.vocab_size);
  return Status();
}

Status Environment::GetModelConfig(ModelConfig &model_config) {
  if (!model_config_initialized_) {
    return Status(RET_MODEL_INVALID, "model config not initialized");
  }
  model_config = model_config_;
  return Status();
}

Status Environment::GetBatchSchedulerConfig(BatchSchedulerConfig &batch_scheduler_config) {
  return schedule_config_parser_.GetBatchSchedulerConfig(batch_scheduler_config);
}

Status Environment::GetCacheManagerConfig(CacheManagerConfig &cache_manager_config) {
  return schedule_config_parser_.GetCacheManagerConfig(cache_manager_config);
}

Status Environment::GetBlockManagerConfig(BlockManagerConfig &block_manager_config) {
  return schedule_config_parser_.GetBlockManagerConfig(block_manager_config);
}

Status Environment::GetRuntimeConfig(RuntimeConfig &runtime_config) {
  return schedule_config_parser_.GetRuntimeConfig(runtime_config);
}

void Environment::SetBlockManagerConfig(const BlockManagerConfig &block_manager_config) {
  schedule_config_parser_.SetBlockManagerConfig(block_manager_config);
}

Status Environment::CalculateBlockNumber(size_t device_free, size_t device_total) {
  return schedule_config_parser_.CalculateBlockNumber(device_free, device_total);
}

Status Environment::ResetPipelineBlockNumber() { return schedule_config_parser_.ResetPipelineBlockNumber(); }

size_t Environment::GetTotalDeviceBlockNum() { return schedule_config_parser_.GetTotalDeviceBlockNum(); }

size_t Environment::GetTotalHostBlockNum() { return schedule_config_parser_.GetTotalHostBlockNum(); }

std::vector<int> Environment::GetDataParaGroupDevices(int dp_id) {
  return schedule_config_parser_.GetDataParaGroupDevices(dp_id);
}

Status Environment::GetProfilerConfig(ProfilerConfig &profiler_config) {
  profiler_config = profiler_config_;
  return Status();
}

size_t Environment::GetTransferLayerChunkSize() { return schedule_config_parser_.GetTransferLayerChunkSize(); }

}  // namespace ksana_llm
