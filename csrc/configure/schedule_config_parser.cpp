/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "configure/schedule_config_parser.h"

#include <algorithm>
#include <fstream>
#include <memory>
#include <stdexcept>

#include "fmt/core.h"
#include "gflags/gflags.h"

#include "configure/environment.h"
#include "utils/memory_utils.h"

#include "cache_manager/cache_layout/base_cache_layout.h"
#include "cache_manager/cache_layout/cache_layout_factory.h"
#include "device/device_utils.h"
#include "models/common/common_config.h"
#include "models/common_moe/moe_config.h"
#include "utils/gguf_file_tensor_loader.h"
#include "utils/logger.h"
#include "utils/optional_file.h"
#include "utils/ret_code.h"
#include "utils/status.h"
#include "utils/string_utils.h"

namespace ksana_llm {

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
        "Loading KV cache scaling factors error, layer num not aligned. Using "
        "default value 1.0.");
    return;
  }

  // TODO(zhongzhicao): load kv scale for tensor_para_size > 1
  size_t tensor_parallel_size_kv_ = kv_scale_json.at("kv_cache").at("scaling_factor").size();
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

ScheduleConfigParser::ScheduleConfigParser() { Reset(); }

void ScheduleConfigParser::Reset() {
  batch_scheduler_config_ = {};
  cache_manager_config_ = {};
  block_manager_config_ = {};
  pipeline_config_ = {};
  expert_parallel_config_ = {};
  connector_config_ = {};
  runtime_config_ = {};
}

Status ScheduleConfigParser::ParseScheduleConfig(YamlReader &yaml_reader, const ModelConfig &model_config) {
  // Read global setting.
  runtime_config_.parallel_basic_config.tensor_parallel_size =
      yaml_reader.GetScalar<size_t>(yaml_reader.GetRootNode(), "setting.global.tensor_para_size", 0);
  runtime_config_.parallel_basic_config.attn_data_parallel_size =
      yaml_reader.GetScalar<size_t>(yaml_reader.GetRootNode(), "setting.global.attn_data_para_size", 1);
  runtime_config_.parallel_basic_config.expert_world_size =
      yaml_reader.GetScalar<size_t>(yaml_reader.GetRootNode(), "setting.global.expert_world_size", 1);
  runtime_config_.parallel_basic_config.expert_parallel_size =
      yaml_reader.GetScalar<size_t>(yaml_reader.GetRootNode(), "setting.global.expert_para_size", 1);
  runtime_config_.enable_full_shared_expert =
      yaml_reader.GetScalar<bool>(yaml_reader.GetRootNode(), "setting.global.enable_full_shared_expert", false);
  if (runtime_config_.parallel_basic_config.tensor_parallel_size == 0) {
    int device_size = -1;
    GetDeviceCount(&device_size);
    runtime_config_.parallel_basic_config.tensor_parallel_size = static_cast<size_t>(device_size);
  }

  // Load EPLB-related environment variable configurations.
  const char *enable_dump_eplb = std::getenv("ENABLE_DUMP_EPLB");
  if (enable_dump_eplb) {
    std::string enable_dump_str(enable_dump_eplb);
    std::transform(enable_dump_str.begin(), enable_dump_str.end(), enable_dump_str.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    runtime_config_.enable_dump_eplb_data = (enable_dump_str == "true" || enable_dump_str == "1");
  }

  const char *dump_eplb_path = std::getenv("DUMP_EPLB_PATH");
  if (dump_eplb_path) {
    runtime_config_.dump_eplb_path = std::string(dump_eplb_path);
  } else {
    // Set default path: ~/.cache/KsanaLLM/EPLB/
    const char *home_dir = std::getenv("HOME");
    if (home_dir) {
      runtime_config_.dump_eplb_path = std::string(home_dir) + "/.cache/KsanaLLM/EPLB/";
    } else {
      runtime_config_.dump_eplb_path = ".cache/KsanaLLM/EPLB/";
    }
  }

  const char *eplb_config_path = std::getenv("EPLB_CONFIG_PATH");
  if (eplb_config_path) {
    runtime_config_.enable_load_eplb_config = true;
    runtime_config_.eplb_config_path = std::string(eplb_config_path);
  }

  KLLM_LOG_INFO << "EPLB Configuration - enable_dump_eplb_data: " << runtime_config_.enable_dump_eplb_data
                << ", dump_eplb_path: " << runtime_config_.dump_eplb_path
                << ", enable_load_eplb_config: " << runtime_config_.enable_load_eplb_config
                << ", eplb_config_path: " << runtime_config_.eplb_config_path;

  // Parsing w4afp8_moe_backend from env
  const char *w4afp8_moe_backend = std::getenv("W4AFP8_MOE_BACKEND");
  if (w4afp8_moe_backend) {
    std::string backend_str(w4afp8_moe_backend);
    if (backend_str == "0" || backend_str == "default") {
      runtime_config_.w4afp8_moe_backend = W4AFP8_MOE_BACKEND::Default;
    } else if (backend_str == "1" || backend_str == "group_triton") {
      runtime_config_.w4afp8_moe_backend = W4AFP8_MOE_BACKEND::GroupTriton;
    } else if (backend_str == "2" || backend_str == "tensor_triton") {
      runtime_config_.w4afp8_moe_backend = W4AFP8_MOE_BACKEND::TensorTriton;
    } else {
      runtime_config_.w4afp8_moe_backend = W4AFP8_MOE_BACKEND::Default;
      KLLM_LOG_WARNING << fmt::format("Unknown W4AFP8_MOE_BACKEND value: {}, and set to Default", backend_str);
    }
  } else {
    // 添加默认值
    runtime_config_.w4afp8_moe_backend = W4AFP8_MOE_BACKEND::Default;
    if (model_config.quant_config.expert_dtype == "fp4") {
      runtime_config_.w4afp8_moe_backend = W4AFP8_MOE_BACKEND::GroupTriton;
      KLLM_LOG_INFO << "Detected FP4 expert model, setting W4AFP8_MOE_BACKEND == GroupTriton";
    } else if (model_config.quant_config.method == QUANT_GPTQ) {
      // 主量化类型是GPTQ说明是纯int4模型
      runtime_config_.w4afp8_moe_backend = W4AFP8_MOE_BACKEND::Default;
      KLLM_LOG_INFO << "Detected pure int4 model, setting W4AFP8_MOE_BACKEND == Default";
    } else if (!model_config.sub_quant_configs.empty() &&
               (model_config.sub_quant_configs[0].method == QUANT_GPTQ ||
                model_config.sub_quant_configs[0].method == QUANT_W4A8_AWQ)) {
      // 混合量化类型
      if (model_config.sub_quant_configs[0].input_scale) {
        // 有input scale说明是w4af8模型
        runtime_config_.w4afp8_moe_backend = W4AFP8_MOE_BACKEND::Default;
        KLLM_LOG_INFO << "Detected w4afp8 model, setting W4AFP8_MOE_BACKEND == Default";
      } else {
        // 没有input scale说明是moe-int4模型 (GPTQ or compressed-tensors)
        runtime_config_.w4afp8_moe_backend = W4AFP8_MOE_BACKEND::GroupTriton;
        KLLM_LOG_INFO << "Detected moe-int4 model, setting W4AFP8_MOE_BACKEND == GroupTriton";
      }
    }
  }

  // Parsing FP4_MOE_BACKEND from env
  const char *fp4_moe_backend_env = std::getenv("FP4_MOE_BACKEND");
  if (fp4_moe_backend_env) {
    std::string backend_str(fp4_moe_backend_env);
    if (backend_str == "marlin") {
      runtime_config_.fp4_moe_backend = FP4_MOE_BACKEND_TYPE::FP4Marlin;
    } else if (backend_str == "humming_w4a16") {
      runtime_config_.fp4_moe_backend = FP4_MOE_BACKEND_TYPE::FP4HummingW4A16;
    } else if (backend_str == "humming_w4a8") {
      runtime_config_.fp4_moe_backend = FP4_MOE_BACKEND_TYPE::FP4HummingW4A8;
    } else {
      runtime_config_.fp4_moe_backend = FP4_MOE_BACKEND_TYPE::FP4Triton;
    }
    KLLM_LOG_INFO << fmt::format("FP4_MOE_BACKEND set to: {}", backend_str);
  }

  KLLM_CHECK_WITH_INFO(
      runtime_config_.parallel_basic_config.tensor_parallel_size >=
          runtime_config_.parallel_basic_config.attn_data_parallel_size,
      fmt::format("Tensor Para Size(tensor_para_size) {} should >= Attention Data Para Size(attn_data_para_size) {}",
                  runtime_config_.parallel_basic_config.tensor_parallel_size,
                  runtime_config_.parallel_basic_config.attn_data_parallel_size));

  KLLM_CHECK_WITH_INFO(
      runtime_config_.parallel_basic_config.tensor_parallel_size %
              runtime_config_.parallel_basic_config.attn_data_parallel_size ==
          0,
      fmt::format("Tensor Para Size(tensor_para_size) {} % Attention Data Para Size(attn_data_para_size) {} != 0",
                  runtime_config_.parallel_basic_config.tensor_parallel_size,
                  runtime_config_.parallel_basic_config.attn_data_parallel_size));

#if (defined(ENABLE_ACL) || defined(ENABLE_TOPS))
  if (runtime_config_.parallel_basic_config.attn_data_parallel_size > 1) {
    KLLM_THROW(
        fmt::format("Huawei Ascend does not support data parallelism, please set attn_data_parallel_size to 1."));
  }
#endif
  if (!(runtime_config_.parallel_basic_config.tensor_parallel_size > 0 &&
        runtime_config_.parallel_basic_config.attn_data_parallel_size > 0)) {
    KLLM_THROW(fmt::format("Tensor Para Size {}, Data Para Size {} should > 0",
                           runtime_config_.parallel_basic_config.tensor_parallel_size,
                           runtime_config_.parallel_basic_config.attn_data_parallel_size));
  }

  int device_num;
  GetDeviceCount(&device_num);
  KLLM_CHECK_WITH_INFO(device_num >= static_cast<int>(runtime_config_.parallel_basic_config.tensor_parallel_size),
                       fmt::format("{} tensor_parallel_size should not bigger than devices num: {}",
                                   runtime_config_.parallel_basic_config.tensor_parallel_size, device_num));

  // Get each atten data parallel group size.
  // NOTE(karlluo): for tp + attn_dp, all gpus consist tensor parallel group, attn_data_parallel_size is the number of
  // attn dp groups and conduct tp in each dp groups. For example, if tp = 4, then gpus = 4 and attn_dp = 2, then each
  // attn dp group size is 2.
  runtime_config_.parallel_basic_config.attn_tensor_parallel_size =
      runtime_config_.parallel_basic_config.tensor_parallel_size /
      runtime_config_.parallel_basic_config.attn_data_parallel_size;

  // NOTE(karlluo): When using PP parallelism (pipeline parallelism), the communication mode is selected, with the
  // default value being "default". The "default" mode is the send-receive mode. When node0 completes the inference of
  // the previous task, device0 on node0 sends data to device0 on node1, and device1 on node0 sends data to device1 on
  // node1. The "scatter" mode is the scatter mode. When node0 completes the inference of the previous task, device0 on
  // node0 sends data to device0, device1, device2, etc., on node1.
  const std::string &pp_comm_type_str = yaml_reader.GetScalar<std::string>(
      yaml_reader.GetRootNode(), "setting.global.pipeline_para_comm_type", "default");
  if (pp_comm_type_str == "scatter") {
    pipeline_config_.pipeline_para_comm_type = DistributedCommunicationType::SCATTER;
  }

  // Read batch scheduler config.
  int scheduler_type_int =
      yaml_reader.GetScalar<int>(yaml_reader.GetRootNode(), "setting.batch_scheduler.scheduler_type", 0);
  if (scheduler_type_int != static_cast<int>(SchedulerType::DEFAULT) &&
      scheduler_type_int != static_cast<int>(SchedulerType::EVENT_DRIVEN)) {
    KLLM_LOG_ERROR << "Unknown scheduler_type: " << scheduler_type_int << ", fallback to DEFAULT.";
    scheduler_type_int = static_cast<int>(SchedulerType::DEFAULT);
  }
  batch_scheduler_config_.scheduler_type = static_cast<SchedulerType>(scheduler_type_int);
  batch_scheduler_config_.schedule_strategy = static_cast<ScheduleStrategy>(
      yaml_reader.GetScalar<int>(yaml_reader.GetRootNode(), "setting.batch_scheduler.schedule_strategy", 0));
  batch_scheduler_config_.pp_multibatch_wb_strategy = static_cast<PPMultibatchWBStrategy>(
      yaml_reader.GetScalar<int>(yaml_reader.GetRootNode(), "setting.batch_scheduler.pp_multibatch_wb_strategy", 0));
  batch_scheduler_config_.waiting_timeout_in_ms =
      yaml_reader.GetScalar<size_t>(yaml_reader.GetRootNode(), "setting.batch_scheduler.waiting_timeout_in_ms", 600000);
  batch_scheduler_config_.max_waiting_queue_len =
      yaml_reader.GetScalar<size_t>(yaml_reader.GetRootNode(), "setting.batch_scheduler.max_waiting_queue_len", 1200);
  batch_scheduler_config_.max_token_len =
      yaml_reader.GetScalar<size_t>(yaml_reader.GetRootNode(), "setting.batch_scheduler.max_token_len", 0);
  batch_scheduler_config_.max_step_token_num =
      yaml_reader.GetScalar<size_t>(yaml_reader.GetRootNode(), "setting.batch_scheduler.max_step_tokens", 4096);
  batch_scheduler_config_.max_chunked_kv_tokens =
      yaml_reader.GetScalar<size_t>(yaml_reader.GetRootNode(), "setting.batch_scheduler.max_chunked_kv_tokens", 0);
  batch_scheduler_config_.max_batch_size =
      yaml_reader.GetScalar<size_t>(yaml_reader.GetRootNode(), "setting.batch_scheduler.max_batch_size", 128);
  batch_scheduler_config_.max_pretransfer_batch_size = yaml_reader.GetScalar<size_t>(
      yaml_reader.GetRootNode(), "setting.batch_scheduler.max_pretransfer_batch_size", 64);
  batch_scheduler_config_.transfer_layer_chunk_size =
      yaml_reader.GetScalar<size_t>(yaml_reader.GetRootNode(), "setting.batch_scheduler.transfer_layer_chunk_size", 1);
  batch_scheduler_config_.max_pp_batch_num =
      yaml_reader.GetScalar<size_t>(yaml_reader.GetRootNode(), "setting.batch_scheduler.max_pp_batch_num", 1);
  batch_scheduler_config_.launch_block_threshold =
      yaml_reader.GetScalar<float>(yaml_reader.GetRootNode(), "setting.batch_scheduler.launch_block_threshold", 2.0);
  batch_scheduler_config_.split_fuse_token_num =
      yaml_reader.GetScalar<size_t>(yaml_reader.GetRootNode(), "setting.batch_scheduler.split_fuse_token_num", 0);
  batch_scheduler_config_.enable_speculative_decoding = yaml_reader.GetScalar<bool>(
      yaml_reader.GetRootNode(), "setting.batch_scheduler.enable_speculative_decoding", false);
  batch_scheduler_config_.speculative_method =
      yaml_reader.GetScalar<std::string>(yaml_reader.GetRootNode(), "setting.batch_scheduler.speculative_method", "");
  if (batch_scheduler_config_.enable_speculative_decoding) {
    if (batch_scheduler_config_.speculative_method.empty()) {
      // Use trie by default
      batch_scheduler_config_.speculative_method = "trie";
    } else if (batch_scheduler_config_.speculative_method != "trie" &&
               batch_scheduler_config_.speculative_method != "asr_custom") {
      KLLM_LOG_WARNING << fmt::format("Unknown speculative_method: {}, disable speculative decoding.",
                                      batch_scheduler_config_.speculative_method);
      batch_scheduler_config_.enable_speculative_decoding = false;
      batch_scheduler_config_.speculative_method.clear();
    }
  } else if (!batch_scheduler_config_.speculative_method.empty()) {
    KLLM_LOG_WARNING << fmt::format("Speculative decoding is disabled, ignore speculative_method: {}",
                                    batch_scheduler_config_.speculative_method);
    batch_scheduler_config_.speculative_method.clear();
  }
  batch_scheduler_config_.mtp_step_num =
      yaml_reader.GetScalar<size_t>(yaml_reader.GetRootNode(), "setting.batch_scheduler.mtp_step_num", 0);
  batch_scheduler_config_.ptp_step_num = model_config.ptp_step_num;
  batch_scheduler_config_.ptp_token_id = model_config.ptp_token_id;

  if (model_config.num_nextn_predict_layers == 0 && batch_scheduler_config_.mtp_step_num != 0) {
    batch_scheduler_config_.mtp_step_num = 0;
    KLLM_LOG_WARNING << "There is no MTP layer in the model, mtp_step_num will be set to 0.";
  }
  if (batch_scheduler_config_.mtp_step_num > 0 && batch_scheduler_config_.ptp_step_num > 0) {
    batch_scheduler_config_.ptp_step_num = 0;
    KLLM_LOG_WARNING << "MTP and PTP cannot be enabled simultaneously, ptp_step_num will be set to 0.";
  }
  if (batch_scheduler_config_.enable_speculative_decoding && batch_scheduler_config_.ptp_step_num > 0) {
    batch_scheduler_config_.ptp_step_num = 0;
    KLLM_LOG_WARNING << "Speculative decoding and PTP cannot be enabled simultaneously, ptp_step_num will be set to 0.";
  }
  KLLM_LOG_INFO << "mtp_step_num: " << batch_scheduler_config_.mtp_step_num
                << ", model MTP layer: " << model_config.num_nextn_predict_layers;

  batch_scheduler_config_.enable_xgrammar =
      yaml_reader.GetScalar<bool>(yaml_reader.GetRootNode(), "setting.batch_scheduler.enable_xgrammar", false);
  batch_scheduler_config_.enable_async =
      yaml_reader.GetScalar<bool>(yaml_reader.GetRootNode(), "setting.batch_scheduler.enable_async", false);

  // Executor 端 build-ahead + device 侧 token 替换总开关（默认关闭）。
  batch_scheduler_config_.enable_executor_build_ahead = yaml_reader.GetScalar<bool>(
      yaml_reader.GetRootNode(), "setting.batch_scheduler.enable_executor_build_ahead", false);

  // Parse ADP Balance Strategy configuration
  batch_scheduler_config_.attention_dp_lb_config.enable_balance = yaml_reader.GetScalar<bool>(
      yaml_reader.GetRootNode(), "setting.batch_scheduler.attention_dp_lb_config.enable_balance", false);
  batch_scheduler_config_.attention_dp_lb_config.max_waiting_steps = yaml_reader.GetScalar<size_t>(
      yaml_reader.GetRootNode(), "setting.batch_scheduler.attention_dp_lb_config.max_waiting_steps", 50);
  batch_scheduler_config_.attention_dp_lb_config.max_waiting_time_in_ms = yaml_reader.GetScalar<size_t>(
      yaml_reader.GetRootNode(), "setting.batch_scheduler.attention_dp_lb_config.max_waiting_time_in_ms", 1000);
  batch_scheduler_config_.attention_dp_lb_config.min_qps_for_waiting = yaml_reader.GetScalar<double>(
      yaml_reader.GetRootNode(), "setting.batch_scheduler.attention_dp_lb_config.min_qps_for_waiting", -1.0);

  KLLM_LOG_INFO << "ADP Balance Strategy - enable_balance: "
                << batch_scheduler_config_.attention_dp_lb_config.enable_balance
                << ", max_waiting_steps: " << batch_scheduler_config_.attention_dp_lb_config.max_waiting_steps
                << ", max_waiting_time_in_ms: " << batch_scheduler_config_.attention_dp_lb_config.max_waiting_time_in_ms
                << ", min_qps_for_waiting: " << batch_scheduler_config_.attention_dp_lb_config.min_qps_for_waiting;

  KLLM_CHECK_WITH_INFO(batch_scheduler_config_.max_pp_batch_num > 0, "max_multi_batch_size should be bigger than 0");

  // When MTP/PTP is enabled, each request requires calculating more tokens while decoding.
  // PTP in steady state has ptp_step_num verify tokens + ptp_step_num placeholder tokens + 1 generated token.
  batch_scheduler_config_.max_decode_tokens_per_req =
      batch_scheduler_config_.mtp_step_num + 2 * batch_scheduler_config_.ptp_step_num + 1;

  if (std::getenv("ENABLE_O_PROJ_OUT_OF_DP") != nullptr) {
    KLLM_CHECK_WITH_INFO(runtime_config_.parallel_basic_config.attn_tensor_parallel_size == 1,
                         "ENABLE_O_PROJ_OUT_OF_DP only support attn_tensor_parallel_size=1");
    runtime_config_.enable_o_proj_out_of_dp = true;
  }

  if (std::getenv("FORCE_NON_CONTIGUOUS_SWA") != nullptr && strcmp(std::getenv("FORCE_NON_CONTIGUOUS_SWA"), "1") == 0) {
    runtime_config_.force_non_contiguous_swa_mode = true;
    KLLM_LOG_WARNING << "FORCE_NON_CONTIGUOUS_SWA=1: all flash batches will use the SWA cache buffer path.";
  }

  // Read block manager config.
  block_manager_config_.device_allocator_config.block_token_num =
      yaml_reader.GetScalar<size_t>(yaml_reader.GetRootNode(), "setting.block_manager.block_token_num", 16);
  // If the model uses MLA, automatically enable Flash MLA and set block_token_num appropriately.
  // DSv4 uses a C128 attention cache that requires block_token_num to be a multiple of 128,
  // so use 256 for DSv4 models instead of the standard Flash MLA value of 64.
  if (model_config.use_mla) {
    if (model_config.use_dsv4_attn) {
      block_manager_config_.device_allocator_config.block_token_num = 256;
      KLLM_LOG_INFO << "Automatically activate Flash MLA for DSv4 models, setting block_token_num to 256";
    } else {
      block_manager_config_.device_allocator_config.block_token_num = 64;
      KLLM_LOG_INFO << "Automatically activate Flash MLA for MLA models, setting block_token_num to 64 for flash_mla";
    }
  }
  block_manager_config_.host_allocator_config.block_token_num =
      block_manager_config_.device_allocator_config.block_token_num;
  runtime_config_.attn_backend_config.block_token_num = block_manager_config_.device_allocator_config.block_token_num;

  block_manager_config_.reserved_device_memory_ratio = yaml_reader.GetScalar<float>(
      yaml_reader.GetRootNode(), "setting.block_manager.reserved_device_memory_ratio", 0.01);
  block_manager_config_.block_device_memory_ratio =
      yaml_reader.GetScalar<float>(yaml_reader.GetRootNode(), "setting.block_manager.block_device_memory_ratio", -1.0);
  block_manager_config_.block_host_memory_factor =
      yaml_reader.GetScalar<float>(yaml_reader.GetRootNode(), "setting.block_manager.block_host_memory_factor", 2.0);
  block_manager_config_.dynamic_reusable_memory_ratio = yaml_reader.GetScalar<float>(
      yaml_reader.GetRootNode(), "setting.block_manager.dynamic_reusable_memory_ratio", 1.0);
  block_manager_config_.host_pool_size_gb =
      yaml_reader.GetScalar<int>(yaml_reader.GetRootNode(), "setting.block_manager.host_pool_size_gb", 0);

  // Load cache manager config
  batch_scheduler_config_.swa_block_num_ratio =
      yaml_reader.GetScalar<float>(yaml_reader.GetRootNode(), "setting.cache_manager.swa_block_num_ratio", 0.1f);
  // Correct invalid values: if the user configured a non-positive ratio, fall back to the default 0.1.
  if (batch_scheduler_config_.swa_block_num_ratio <= 0.0f) {
    batch_scheduler_config_.swa_block_num_ratio = 0.1f;
  }
  cache_manager_config_.min_flexible_cache_num =
      yaml_reader.GetScalar<size_t>(yaml_reader.GetRootNode(), "setting.batch_scheduler.min_flexible_cache_num", 0);
  cache_manager_config_.block_token_num = block_manager_config_.device_allocator_config.block_token_num;
  cache_manager_config_.tensor_para_size = runtime_config_.parallel_basic_config.tensor_parallel_size;
  cache_manager_config_.enable_prefix_caching =
      yaml_reader.GetScalar<bool>(yaml_reader.GetRootNode(), "setting.batch_scheduler.enable_auto_prefix_cache", false);
  cache_manager_config_.enable_non_contiguous_swa_mode = yaml_reader.GetScalar<bool>(
      yaml_reader.GetRootNode(), "setting.cache_manager.enable_non_contiguous_swa_mode", false);
  // Platform/model-specific overrides: disable unsupported features before the generic dependency check below.
#ifdef ENABLE_ACL
  if (cache_manager_config_.enable_prefix_caching) {
    cache_manager_config_.enable_prefix_caching = false;
    KLLM_LOG_WARNING << "prefix caching not support NPU, will change enable_prefix_caching as false";
  }
#endif
  if (model_config.is_visual) {
    if (cache_manager_config_.enable_prefix_caching) {
      KLLM_LOG_WARNING << "PrefixCaching not support Visual Model, will change enable_prefix_caching as false";
      cache_manager_config_.enable_prefix_caching = false;
    }
    if (batch_scheduler_config_.split_fuse_token_num > 0) {
      KLLM_LOG_WARNING << "SplitFuse not support Visual Model, will change split_fuse_token_num as 0";
      batch_scheduler_config_.split_fuse_token_num = 0;
    }
  }

  // DSv4 模型要求 split_fuse_token_num 必须是 block_token_num 的整数倍, 原因:
  // 1) prefix cache commit 单位是 block, 非倍数会让中间 chunk 的尾 block 无法 shareable, 降低命中率
  // 2) c128 attention indexer 按 128 对齐 (block_token_num=256 已是 128 的倍数, 取 256 倍数自然兼容)
  // 直接报错而不是 round-up/down: 让错误配置立即可见, 避免静默改值引入难调试的问题。
  // 仅对 DSv4 生效, 其他模型保持原行为。
  if (model_config.use_dsv4_attn && batch_scheduler_config_.split_fuse_token_num > 0) {
    const size_t block_token_num = block_manager_config_.device_allocator_config.block_token_num;
    KLLM_CHECK_WITH_INFO(
        batch_scheduler_config_.split_fuse_token_num % block_token_num == 0,
        fmt::format("DSv4 requires split_fuse_token_num ({}) to be a multiple of block_token_num ({}); "
                    "set yaml setting.batch_scheduler.split_fuse_token_num to {}, {}, or any multiple.",
                    batch_scheduler_config_.split_fuse_token_num, block_token_num, block_token_num,
                    block_token_num * 2));
  }

  // Read parallel config.
  expert_parallel_config_.expert_world_size =
      yaml_reader.GetScalar<size_t>(yaml_reader.GetRootNode(), "setting.global.expert_world_size", 1);
  expert_parallel_config_.expert_para_size =
      yaml_reader.GetScalar<size_t>(yaml_reader.GetRootNode(), "setting.global.expert_para_size", 1);

  // Read attn backend config.
  runtime_config_.attn_backend_config.enable_blocked_multi_token_forwarding_kv = yaml_reader.GetScalar<bool>(
      yaml_reader.GetRootNode(), "setting.attn_backend.enable_blocked_multi_token_forwarding_kv", false);
  runtime_config_.attn_backend_config.use_flashinfer_for_decode =
      yaml_reader.GetScalar<bool>(yaml_reader.GetRootNode(), "setting.attn_backend.use_flashinfer_for_decode", false);
  if (model_config.use_mla) {
    runtime_config_.attn_backend_config.enable_blocked_multi_token_forwarding_kv = true;
    KLLM_LOG_INFO
        << "Automatically activate Flash MLA for MLA models, setting enable_blocked_multi_token_forwarding_kv to true.";
  }
  // For any model with MTP enabled, blocked multi-token forwarding KV is required to correctly
  // handle multi-token decode (speculative decoding verification phase). Without this, the standard
  // flash attention path uses seqlen_q == seqlen_k which ignores KV cache for multi-token decode.
  // NOTE: runtime_config_.mtp_step_num is not yet assigned at this point (it's copied from
  // batch_scheduler_config_ later in InitializeRuntimeConfig), so we must use batch_scheduler_config_ here.
  if (batch_scheduler_config_.mtp_step_num > 0 &&
      !runtime_config_.attn_backend_config.enable_blocked_multi_token_forwarding_kv) {
    runtime_config_.attn_backend_config.enable_blocked_multi_token_forwarding_kv = true;
    KLLM_LOG_INFO << "Automatically enable blocked multi-token forwarding KV for MTP models (mtp_step_num="
                  << batch_scheduler_config_.mtp_step_num << ").";
  }
  runtime_config_.attn_backend_config.kv_cache_dtype_str = yaml_reader.GetScalar<std::string>(
      yaml_reader.GetRootNode(), "setting.quantization_config.kv_cache.dtype", "auto");
  KLLM_LOG_INFO << fmt::format("enable_blocked_multi_token_forwarding_kv: {}, kv_cache.dtype: {}",
                               runtime_config_.attn_backend_config.enable_blocked_multi_token_forwarding_kv,
                               runtime_config_.attn_backend_config.kv_cache_dtype_str);

  // Parse FlashAttention implementation preference (optional)
  {
    std::string impl =
        yaml_reader.GetScalar<std::string>(yaml_reader.GetRootNode(), "setting.attn_backend.flash_attn_impl", "auto");
    // Trim whitespace and treat empty as auto
    auto is_space = [](unsigned char c) {
      return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
    };
    impl.erase(impl.begin(), std::find_if(impl.begin(), impl.end(), [&](unsigned char ch) { return !is_space(ch); }));
    impl.erase(std::find_if(impl.rbegin(), impl.rend(), [&](unsigned char ch) { return !is_space(ch); }).base(),
               impl.end());
    if (impl.empty()) {
      impl = "auto";
    }
    std::string impl_lower = impl;
    std::transform(impl_lower.begin(), impl_lower.end(), impl_lower.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    using Choice = AttnBackendConfig::FlashAttnImplChoice;
    Choice choice = Choice::AUTO;
    if (impl_lower == "auto") {
      choice = Choice::AUTO;
    } else if (impl_lower == "fa3") {
      choice = Choice::FA3;
    } else if (impl_lower == "vllm_v26" || impl_lower == "vllm") {
      choice = Choice::VLLM_V26;
    } else if (impl_lower == "flashattn_v26" || impl_lower == "fa2_v26" || impl_lower == "flash_attn_v26") {
      choice = Choice::FA2_V26;
    } else if (impl_lower == "flashattn_v25" || impl_lower == "fa2_v25" || impl_lower == "flash_attn_v25") {
      choice = Choice::FA2_V25;
    } else {
      KLLM_LOG_WARNING << "Unknown setting.attn_backend.flash_attn_impl='" << impl << "', fallback to 'auto'.";
      choice = Choice::AUTO;
    }
    runtime_config_.attn_backend_config.flash_attn_impl_choice = choice;

    const char *choice_str = (choice == Choice::AUTO)       ? "auto"
                             : (choice == Choice::FA3)      ? "fa3"
                             : (choice == Choice::VLLM_V26) ? "vllm_v26"
                             : (choice == Choice::FA2_V26)  ? "flashattn_v26"
                                                            : "flashattn_v25";
    KLLM_LOG_INFO << "flash_attn_impl: " << choice_str;
  }

  // Read MOE topk score threshold
  runtime_config_.moe_topk_score_threshold =
      yaml_reader.GetScalar<float>(yaml_reader.GetRootNode(), "setting.global.moe_topk_score_threshold", 0.0f);

  // Determine MOE all-to-all mode from yaml config.
  {
    std::string mode_str =
        yaml_reader.GetScalar<std::string>(yaml_reader.GetRootNode(), "setting.global.moe_all_to_all_mode", "auto");

    // Resolve mode string to enum.
    if (mode_str == "deepep_low_latency") {
      runtime_config_.moe_all_to_all_mode = MoeAllToAllMode::DEEPEP_LOW_LATENCY;
    } else if (mode_str == "deepep_normal") {
      runtime_config_.moe_all_to_all_mode = MoeAllToAllMode::DEEPEP_NORMAL;
    } else if (mode_str == "auto") {
      // Auto: enable DeepEP Normal when expert_world_size > 1, otherwise disabled.
      if (runtime_config_.parallel_basic_config.expert_world_size > 1) {
        runtime_config_.moe_all_to_all_mode = MoeAllToAllMode::DEEPEP_NORMAL;
      } else {
        runtime_config_.moe_all_to_all_mode = MoeAllToAllMode::DISABLED;
      }
    } else {
      KLLM_LOG_ERROR << "Unknown moe_all_to_all_mode: " << mode_str
                     << ", valid values: auto, deepep_normal, deepep_low_latency.";
      KLLM_THROW(fmt::format("Unknown moe_all_to_all_mode: {}", mode_str));
    }
  }
  KLLM_LOG_INFO << "MOE all-to-all mode: "
                << (runtime_config_.moe_all_to_all_mode == MoeAllToAllMode::DISABLED             ? "DISABLED"
                    : runtime_config_.moe_all_to_all_mode == MoeAllToAllMode::DEEPEP_NORMAL      ? "DEEPEP_NORMAL"
                    : runtime_config_.moe_all_to_all_mode == MoeAllToAllMode::DEEPEP_LOW_LATENCY ? "DEEPEP_LOW_LATENCY"
                                                                                                 : "UNKNOWN");

  // DeepEP (Normal / LowLatency) requires TP == DP and enable_full_shared_expert.
  if (runtime_config_.moe_all_to_all_mode == MoeAllToAllMode::DEEPEP_NORMAL ||
      runtime_config_.moe_all_to_all_mode == MoeAllToAllMode::DEEPEP_LOW_LATENCY) {
    KLLM_CHECK_WITH_INFO(
        runtime_config_.parallel_basic_config.attn_data_parallel_size ==
            runtime_config_.parallel_basic_config.tensor_parallel_size,
        fmt::format("DeepEP requires tensor_parallel_size == attn_data_parallel_size, but got tp={}, dp={}",
                    runtime_config_.parallel_basic_config.tensor_parallel_size,
                    runtime_config_.parallel_basic_config.attn_data_parallel_size));
    KLLM_CHECK_WITH_INFO(runtime_config_.enable_full_shared_expert,
                         "DeepEP requires enable_full_shared_expert to be true, please set it in the config yaml.");
  }

  // Read deepep_low_latency_max_tokens from yaml config.
  runtime_config_.deepep_low_latency_max_tokens =
      yaml_reader.GetScalar<size_t>(yaml_reader.GetRootNode(), "setting.global.deepep_low_latency_max_tokens", 256);

  // When DeepEP low_latency is enabled, cap deepep_low_latency_max_tokens
  // to max_batch_size / dp. max_batch_size is the global total across all DPs,
  // so each DP's dispatch buffer only needs to handle its fair share.
  if (runtime_config_.moe_all_to_all_mode == MoeAllToAllMode::DEEPEP_LOW_LATENCY) {
    size_t per_dp_batch =
        batch_scheduler_config_.max_batch_size / runtime_config_.parallel_basic_config.attn_data_parallel_size;
    runtime_config_.deepep_low_latency_max_tokens =
        std::min(runtime_config_.deepep_low_latency_max_tokens, per_dp_batch);
    KLLM_LOG_INFO << fmt::format(
        "DeepEP low_latency: deepep_low_latency_max_tokens capped to {} (max_batch_size={}, dp={})",
        runtime_config_.deepep_low_latency_max_tokens, batch_scheduler_config_.max_batch_size,
        runtime_config_.parallel_basic_config.attn_data_parallel_size);
  }

  InitConnectorConfig(yaml_reader);
  InitGlobalCacheConnectorConfig(yaml_reader);
  // pd_v2 Decode 节点的本地 prefill 算完直接在本地 decode、不向外传 KV, 因此可以安全使用 splitfuse:
  // 把本地大 prefill 切块与 decode 交错执行, 压平单步 latency / TPOT。这正是它需要 splitfuse 的场景。
  // 其余 PD 角色仍不支持: V1 prefill/decode 与 pd_v2 Prefill 节点会把算好的 KV 通过 RDMA 传给对端,
  // splitfuse 的分块 prefill 与"prefill 一次算完才传 KV"的假设冲突 (见原 TODO(zhongzhicao))。
  const bool is_pd_v2_decode = connector_config_.is_pd_v2 && connector_config_.group_role == GroupRole::DECODE;
  if (connector_config_.group_role != GroupRole::NONE && !is_pd_v2_decode &&
      cache_manager_config_.enable_prefix_caching && batch_scheduler_config_.split_fuse_token_num != 0) {
    KLLM_LOG_WARNING << "Split-fuse is not supported for this PD role, setting split_fuse_token_num to 0.";
    batch_scheduler_config_.split_fuse_token_num = 0;
  }
  return Status();
}

void ScheduleConfigParser::UpdateMembers(const std::string &model_dir, ModelConfig &model_config) {
  // Update dtype of kv cache
  auto &kv_cache_dtype_str = runtime_config_.attn_backend_config.kv_cache_dtype_str;
  if (model_config.use_dsa) {
    // If fp8_e4m3 is not specified,
    // DSA/DSV4 prefers the custom fp8 format (Better precision,
    // but larger block size, and computation uses BF16)
    if (kv_cache_dtype_str != "fp8_e4m3") {
      kv_cache_dtype_str = "fp8_ds_mla";
    }
  } else {
    if (!model_config.use_mla && runtime_config_.attn_backend_config.enable_blocked_multi_token_forwarding_kv &&
        IsPrefixCachingEnabled()) {
      kv_cache_dtype_str = "auto";
    } else {
      if (kv_cache_dtype_str == "fp8_e4m3") {
        PrepareKVScales(model_dir, model_config);
      } else if (kv_cache_dtype_str != "fp8_e5m2" || model_config.use_mla) {
        kv_cache_dtype_str = "auto";
      }
    }
  }

  if (runtime_config_.attn_backend_config.use_flashinfer_for_decode) {
    for (size_t layer_idx = 0; layer_idx < model_config.k_scales.size(); ++layer_idx) {
      if (model_config.k_scales[layer_idx] != 1.0f || model_config.v_scales[layer_idx] != 1.0f) {
        KLLM_THROW(fmt::format(
            "FlashInfer only supports k_scale == 1.0f and v_scale == 1.0f, but got k_scale={}, v_scale={} at "
            "layer {}. Please set setting.attn_backend.use_flashinfer_for_decode to false in the yaml config.",
            model_config.k_scales[layer_idx], model_config.v_scales[layer_idx], layer_idx));
      }
    }
  }

  KLLM_LOG_INFO << "Automatically adjust kv_cache.dtype to " << kv_cache_dtype_str;
  // Update kv_cache_dtype based on kv_cache_dtype_str
  auto &kv_cache_dtype = runtime_config_.attn_backend_config.kv_cache_dtype;
  if (kv_cache_dtype_str == "fp8_e4m3") {
    kv_cache_dtype = TYPE_FP8_E4M3;
  } else if (kv_cache_dtype_str == "fp8_e5m2") {
    kv_cache_dtype = TYPE_FP8_E5M2;
  } else if (kv_cache_dtype_str == "fp8_ds_mla") {
    kv_cache_dtype = TYPE_FP8_DS_MLA;
  } else {  // kv_cache_dtype_str == "auto"
    kv_cache_dtype = model_config.weight_data_type;
  }

  // Update state_dtype of compressor
  if (model_config.use_dsv4_attn) {
    // Use BF16 by default
    if (std::getenv("KLLM_USE_FP32_COMPRESSOR")) {
      runtime_config_.compressor_config.state_dtype = TYPE_FP32;
    } else {
      runtime_config_.compressor_config.state_dtype = TYPE_BF16;
    }
    KLLM_LOG_INFO << "Automatically adjust state.dtype of compressor to "
                  << GetTypeString(runtime_config_.compressor_config.state_dtype);
  }

  // Update reserved memory
  if (model_config.is_quant == true && model_config.quant_config.method == QUANT_FP8_E4M3 &&
      model_config.quant_config.is_checkpoint_fp8_serialized == false) {
    if (block_manager_config_.reserved_device_memory_ratio < 0.02) {
      block_manager_config_.reserved_device_memory_ratio = 0.02;
      KLLM_LOG_INFO
          << "When quant_method is fp8_e4m3, reserved_device_memory_ratio is set to at least 0.02 to prevent oom.";
    }
  } else if (model_config.is_quant == true && model_config.quant_config.method == QUANT_GPTQ) {
    if (block_manager_config_.reserved_device_memory_ratio < 0.02) {
      block_manager_config_.reserved_device_memory_ratio = 0.02;
      KLLM_LOG_INFO
          << "When quant_method is gptq, reserved_device_memory_ratio is set to at least 0.02 to prevent oom.";
    }
  }
}

void ScheduleConfigParser::InitConnectorConfig(YamlReader &yaml_reader) {
  // Parse connector role first to check if we should continue parsing
  std::string role_str =
      yaml_reader.GetScalar<std::string>(yaml_reader.GetRootNode(), "setting.connector.group_role", "none");

  const bool decode_node_benchmark =
      (std::getenv("DECODE_NODE_BENCHMARK") != nullptr) && (strcmp(std::getenv("DECODE_NODE_BENCHMARK"), "1") == 0);
  if (decode_node_benchmark) {
    role_str = "decode";
  }

  // Convert to lowercase for case-insensitive comparison
  std::transform(role_str.begin(), role_str.end(), role_str.begin(), [](unsigned char c) { return std::tolower(c); });
  // Check if the role is not None
  if (role_str != "none") {
    // Set role based on parsed string
    if (role_str == "prefill") {
      connector_config_.group_role = GroupRole::PREFILL;
    } else if (role_str == "decode") {
      connector_config_.group_role = GroupRole::DECODE;
    } else if (role_str == "both") {
      connector_config_.group_role = GroupRole::BOTH;
    } else {
      connector_config_.group_role = GroupRole::NONE;
      KLLM_LOG_WARNING << fmt::format("Unknown connector role: {}, defaulting to NONE", role_str);
    }
    pd_v2_runtime_config_.group_role = connector_config_.group_role;

    // Only continue parsing if the role is not NONE
    if (connector_config_.group_role != GroupRole::NONE) {
      // pd_v2 backend selector. "v2" flips is_pd_v2 on; any other value
      // (empty, "v1", etc.) keeps V1 behavior. pd_v2-gated code paths (e.g.
      // ExecutorBlockAllocator contiguous KV pool) read this flag via
      // Environment::GetPdV2RuntimeConfig.
      std::string backend =
          yaml_reader.GetScalar<std::string>(yaml_reader.GetRootNode(), "setting.connector.backend", "");
      std::transform(backend.begin(), backend.end(), backend.begin(), [](unsigned char c) { return std::tolower(c); });
      connector_config_.is_pd_v2 = (backend == "v2");
      pd_v2_runtime_config_.is_pd_v2 = connector_config_.is_pd_v2;

      // Fail-fast on unsupported pd_v2 config combinations (review §P1.4).
      // pd_v2's hooks (PdV2DecodeHook on Decode, PdV2PrefillIntakeWrapper
      // on Prefill) only attach in the EventDriven scheduler path; under
      // the legacy DEFAULT scheduler the connector is created but never
      // wired, requests would silently bypass pd_v2. GroupRole::BOTH is
      // not implemented — pd_v2 splits Engine and Executor processes per
      // node role, "Both on one process" has no test coverage.
      if (connector_config_.is_pd_v2) {
        if (batch_scheduler_config_.scheduler_type != SchedulerType::EVENT_DRIVEN) {
          KLLM_THROW(
              "setting.connector.backend=v2 requires "
              "setting.batch_scheduler.scheduler_type=1 (EVENT_DRIVEN); "
              "got DEFAULT. pd_v2 hooks only register on the event-driven "
              "schedule path.");
        }
        if (connector_config_.group_role == GroupRole::BOTH) {
          KLLM_THROW(
              "setting.connector.backend=v2 with setting.connector.group_role=both "
              "is not supported. pd_v2 cleanly splits Decode and Prefill into "
              "different processes; co-locating both roles in one engine has no "
              "test coverage and isn't validated.");
        }
      }

      // pd_v2 metadata server (mooncake_master). Empty = "P2PHANDSHAKE"
      // single-machine fallback at connector Initialize.
      connector_config_.metadata_addr =
          yaml_reader.GetScalar<std::string>(yaml_reader.GetRootNode(), "setting.connector.metadata_addr", "");
      pd_v2_runtime_config_.metadata_addr = connector_config_.metadata_addr;
      // Redis password for metadata_addr=redis://...; pumped into
      // MC_REDIS_PASSWORD env at connector bootstrap. AUTH issued
      // against Redis server's default user (Redis 6+ ACL with username
      // is not supported in Mooncake v0.3.9 — use default user).
      connector_config_.redis_password =
          yaml_reader.GetScalar<std::string>(yaml_reader.GetRootNode(), "setting.connector.redis_password", "");
      pd_v2_runtime_config_.redis_password = connector_config_.redis_password;

      // pd_v2 transfer chunk size — see ConnectorConfig docstring.
      // Read as int so we can detect / reject negative values rather
      // than silently casting to a huge uint32_t (review §P2.2).
      const int chunk_size_raw =
          yaml_reader.GetScalar<int>(yaml_reader.GetRootNode(), "setting.connector.transfer_layer_chunk_size", 1);
      if (chunk_size_raw < 0) {
        KLLM_THROW(
            fmt::format("setting.connector.transfer_layer_chunk_size must be >= 0, got {}. "
                        "0 disables chunking (whole-model in one batch); positive N triggers "
                        "transfer every N layers.",
                        chunk_size_raw));
      }
      connector_config_.transfer_layer_chunk_size = static_cast<uint32_t>(chunk_size_raw);
      pd_v2_runtime_config_.transfer_layer_chunk_size = connector_config_.transfer_layer_chunk_size;

      // pd_v2 Executor Mooncake RPC base port — per-rank port is assigned
      // as base + executor_rank to avoid collision across executor
      // processes on the same host. Default 20000 keeps single-engine /
      // single-executor dev launches working out of the box.
      connector_config_.pd_v2_executor_rpc_base_port = static_cast<uint32_t>(yaml_reader.GetScalar<int>(
          yaml_reader.GetRootNode(), "setting.connector.pd_v2_executor_rpc_base_port", 20000));
      pd_v2_runtime_config_.pd_v2_executor_rpc_base_port = connector_config_.pd_v2_executor_rpc_base_port;

      // pd_v2 Decode-side "remaining prefill ≤ N stays local" threshold
      // (token count). 0 disables the fast-path. See ConnectorConfig.h
      // for full semantics; PdV2DecodeHook applies it.
      connector_config_.pd_v2_local_complete_threshold =
          yaml_reader.GetScalar<int>(yaml_reader.GetRootNode(), "setting.connector.pd_v2_local_complete_threshold", 0);
      pd_v2_runtime_config_.pd_v2_local_complete_threshold = connector_config_.pd_v2_local_complete_threshold;

      // pd_v2 Decode peer-selection policy (0=kRoundRobin, 1=kLinUCB).
      // Default 0. kLinUCB enables the LOCAL-vs-REMOTE contextual-bandit
      // router; the pd_v2_linucb_* knobs below are only consulted in that mode.
      connector_config_.pd_v2_selector_policy =
          yaml_reader.GetScalar<int>(yaml_reader.GetRootNode(), "setting.connector.pd_v2_selector_policy", 0);
      pd_v2_runtime_config_.pd_v2_selector_policy = connector_config_.pd_v2_selector_policy;

      connector_config_.pd_v2_linucb_alpha =
          yaml_reader.GetScalar<double>(yaml_reader.GetRootNode(), "setting.connector.pd_v2_linucb_alpha", 1.0);
      pd_v2_runtime_config_.pd_v2_linucb_alpha = connector_config_.pd_v2_linucb_alpha;

      connector_config_.pd_v2_linucb_tpot_target_us = yaml_reader.GetScalar<double>(
          yaml_reader.GetRootNode(), "setting.connector.pd_v2_linucb_tpot_target_us", 50000.0);
      pd_v2_runtime_config_.pd_v2_linucb_tpot_target_us = connector_config_.pd_v2_linucb_tpot_target_us;

      connector_config_.pd_v2_linucb_tpot_safety_factor = yaml_reader.GetScalar<double>(
          yaml_reader.GetRootNode(), "setting.connector.pd_v2_linucb_tpot_safety_factor", 0.85);
      pd_v2_runtime_config_.pd_v2_linucb_tpot_safety_factor = connector_config_.pd_v2_linucb_tpot_safety_factor;

      connector_config_.pd_v2_linucb_ttft_threshold_us = yaml_reader.GetScalar<double>(
          yaml_reader.GetRootNode(), "setting.connector.pd_v2_linucb_ttft_threshold_us", 1500000.0);
      pd_v2_runtime_config_.pd_v2_linucb_ttft_threshold_us = connector_config_.pd_v2_linucb_ttft_threshold_us;

      connector_config_.pd_v2_linucb_running_ref =
          yaml_reader.GetScalar<double>(yaml_reader.GetRootNode(), "setting.connector.pd_v2_linucb_running_ref", 16.0);
      pd_v2_runtime_config_.pd_v2_linucb_running_ref = connector_config_.pd_v2_linucb_running_ref;

      connector_config_.pd_v2_linucb_remaining_ref_tokens = yaml_reader.GetScalar<double>(
          yaml_reader.GetRootNode(), "setting.connector.pd_v2_linucb_remaining_ref_tokens", 4096.0);
      pd_v2_runtime_config_.pd_v2_linucb_remaining_ref_tokens = connector_config_.pd_v2_linucb_remaining_ref_tokens;

      connector_config_.pd_v2_linucb_remote_lat_ref_us = yaml_reader.GetScalar<double>(
          yaml_reader.GetRootNode(), "setting.connector.pd_v2_linucb_remote_lat_ref_us", 200000.0);
      pd_v2_runtime_config_.pd_v2_linucb_remote_lat_ref_us = connector_config_.pd_v2_linucb_remote_lat_ref_us;

      // pd_v2 control-stage HostStagePool size (bytes). 0 = use the
      // in-source default (HostStagePool::kDefaultPoolBytes = 2 GiB).
      // See docs/technology/design/pd_v2_control_stage.md §5.1 for
      // sizing guidance; raise for multimodal deploys.
      pd_v2_runtime_config_.control_stage_pool_bytes =
          static_cast<uint64_t>(yaml_reader.GetScalar<long long>(  // NOLINT(runtime/int)
              yaml_reader.GetRootNode(), "setting.connector.control_stage_pool_bytes", 0));

      // pd_v2 control-stage maximum concurrent Decode peers. Default 128
      // matches the design's 1:128 fan-in target.
      pd_v2_runtime_config_.control_stage_max_peers = static_cast<uint32_t>(
          yaml_reader.GetScalar<int>(yaml_reader.GetRootNode(), "setting.connector.control_stage_max_peers", 128));

      // pd_v2 control-stage AcquireSlots timeout (ms). Negative = wait
      // forever (not recommended). Default 5000 ms.
      pd_v2_runtime_config_.control_stage_acquire_timeout_ms = yaml_reader.GetScalar<int>(
          yaml_reader.GetRootNode(), "setting.connector.control_stage_acquire_timeout_ms", 5000);

      // pd_v2 Decode-side async submit worker thread count. The EDS
      // scheduling thread only enqueues each SubmitPrefillRequest; these
      // workers run the (potentially blocking) stage handshake off the
      // scheduling thread so a stuck/full peer never stalls scheduling.
      // Default 4; clamped to >= 1 by the connector config builder.
      pd_v2_runtime_config_.control_stage_async_workers =
          yaml_reader.GetScalar<int>(yaml_reader.GetRootNode(), "setting.connector.control_stage_async_workers", 4);

      // Parse connector type
      std::string router_addr =
          yaml_reader.GetScalar<std::string>(yaml_reader.GetRootNode(), "setting.connector.router_addr", "");
      if (!router_addr.empty() && router_addr.find("://") == std::string::npos) {
        auto colon_pos = router_addr.rfind(':');
        if (colon_pos == std::string::npos || colon_pos == 0 || colon_pos == router_addr.size() - 1) {
          KLLM_LOG_WARNING << fmt::format("router_addr '{}' is not in 'host:port' format, connections may fail",
                                          router_addr);
        } else {
          std::string port_str = router_addr.substr(colon_pos + 1);
          try {
            int port = std::stoi(port_str);
            if (port < 1 || port > 65535) {
              KLLM_LOG_WARNING << fmt::format("router_addr '{}' has port {} outside valid range 1-65535", router_addr,
                                              port);
            }
          } catch (const std::exception &) {
            KLLM_LOG_WARNING << fmt::format("router_addr '{}' has non-numeric port '{}'", router_addr, port_str);
          }
        }
        router_addr = fmt::format("http://{}", router_addr);
      }
      connector_config_.inference_addr =
          yaml_reader.GetScalar<std::string>(yaml_reader.GetRootNode(), "setting.connector.inference_addr", "");
      pd_v2_runtime_config_.inference_addr = connector_config_.inference_addr;
      connector_config_.router_addr = std::move(router_addr);
      pd_v2_runtime_config_.router_addr = connector_config_.router_addr;
      connector_config_.coordinator_addr =
          yaml_reader.GetScalar<std::string>(yaml_reader.GetRootNode(), "setting.connector.coordinator_addr", "");
      pd_v2_runtime_config_.coordinator_addr = connector_config_.coordinator_addr;
      connector_config_.cluster_name =
          yaml_reader.GetScalar<std::string>(yaml_reader.GetRootNode(), "setting.connector.cluster_name", "");
      pd_v2_runtime_config_.cluster_name = connector_config_.cluster_name;
      connector_config_.heartbeat_interval_ms =
          yaml_reader.GetScalar<int>(yaml_reader.GetRootNode(), "setting.connector.heartbeat_interval_ms", 5000);
      pd_v2_runtime_config_.heartbeat_interval_ms = connector_config_.heartbeat_interval_ms;
      connector_config_.transfer_batch =
          yaml_reader.GetScalar<int>(yaml_reader.GetRootNode(), "setting.connector.transfer_batch", 1048576);
      connector_config_.connector_waiting_sec =
          yaml_reader.GetScalar<int>(yaml_reader.GetRootNode(), "setting.connector.connector_waiting_sec", 1800);
      connector_config_.circular_bucket_size =
          yaml_reader.GetScalar<int>(yaml_reader.GetRootNode(), "setting.connector.circular_bucket_size", 8192);
      connector_config_.circular_bucket_num =
          yaml_reader.GetScalar<int>(yaml_reader.GetRootNode(), "setting.connector.circular_bucket_num", 4);
      connector_config_.circular_thread_num =
          yaml_reader.GetScalar<int>(yaml_reader.GetRootNode(), "setting.connector.circular_thread_num", 4);
      connector_config_.send_thread_num =
          yaml_reader.GetScalar<int>(yaml_reader.GetRootNode(), "setting.connector.send_thread_num", 4);
      std::string type_str =
          yaml_reader.GetScalar<std::string>(yaml_reader.GetRootNode(), "setting.connector.communication_type", "");

      // Convert to lowercase for case-insensitive comparison
      std::transform(type_str.begin(), type_str.end(), type_str.begin(),
                     [](unsigned char c) { return std::tolower(c); });

      if (type_str == "nccl") {
        connector_config_.communication_type = CommunicationType::NCCL;
      } else if (type_str == "zmq") {
        connector_config_.communication_type = CommunicationType::ZMQ;
      } else {
        connector_config_.communication_type = CommunicationType::TCP;
      }

      // Log the parsed configuration
      KLLM_LOG_INFO << fmt::format(
          "Connector config parsed: role={}, type={}, router_addr={}, cluster_name={}, "
          "inference_addr={}, heartbeat_interval={}ms, is_pd_v2={}, metadata_addr={}",
          role_str, type_str, connector_config_.router_addr, connector_config_.cluster_name,
          connector_config_.inference_addr, connector_config_.heartbeat_interval_ms, connector_config_.is_pd_v2,
          connector_config_.metadata_addr);
    }
  } else {
    KLLM_LOG_INFO << "Connector role is set to NONE, skipping connector configuration.";
  }

  if (decode_node_benchmark) {
    connector_config_.router_addr = "decode_benchmark";
  }
}

void ScheduleConfigParser::SetReservedDeviceRatio(float reserved_device_memory_ratio) {
  block_manager_config_.reserved_device_memory_ratio = reserved_device_memory_ratio;
}

Status ScheduleConfigParser::UpdateModelConfig(ModelConfig &model_config) {
  if (cache_manager_config_.min_flexible_cache_num != 0 && model_config.use_qk_norm) {
    cache_manager_config_.min_flexible_cache_num = 0;
    KLLM_LOG_WARNING << "flexible cache and qk norm cannot be used together, set min_flexible_cache_num to 0";
  }

  if ((runtime_config_.parallel_basic_config.tensor_parallel_size > model_config.num_key_value_heads ||
       model_config.num_key_value_heads % runtime_config_.parallel_basic_config.tensor_parallel_size != 0) &&
      model_config.num_key_value_heads != 1) {
    KLLM_THROW(
        fmt::format("The size of key_value_heads cannot be evenly divided by the size of "
                    "runtime_config_.parallel_basic_config.tensor_parallel_size. "
                    "{} % {} != 0 ",
                    model_config.num_key_value_heads, runtime_config_.parallel_basic_config.tensor_parallel_size));
  }

  if (runtime_config_.parallel_basic_config.tensor_parallel_size <
          runtime_config_.parallel_basic_config.expert_parallel_size ||
      runtime_config_.parallel_basic_config.tensor_parallel_size %
              runtime_config_.parallel_basic_config.expert_parallel_size !=
          0) {
    KLLM_THROW(
        fmt::format("The size of runtime_config_.parallel_basic_config.tensor_parallel_size cannot be evenly divided "
                    "by the size of "
                    "runtime_config_.parallel_basic_config.expert_parallel_size. "
                    "{} % {} != 0 ",
                    runtime_config_.parallel_basic_config.tensor_parallel_size,
                    runtime_config_.parallel_basic_config.expert_parallel_size));
  }

  if (batch_scheduler_config_.max_token_len > 0) {
    if (batch_scheduler_config_.max_token_len > model_config.max_training_seq_len) {
      KLLM_LOG_WARNING << fmt::format(
          "The max_training_seq_len configured in the model's config.json is less than the "
          "max_token_len configured in the ksana yaml file. {} < {}, use {}",
          model_config.max_training_seq_len, batch_scheduler_config_.max_token_len, model_config.max_training_seq_len);
      runtime_config_.max_seq_len = model_config.max_training_seq_len;
    } else {
      runtime_config_.max_seq_len = batch_scheduler_config_.max_token_len;
    }
  } else {
    runtime_config_.max_seq_len = model_config.max_training_seq_len;
  }
  KLLM_LOG_INFO << "Set max_token_len to " << runtime_config_.max_seq_len;
  batch_scheduler_config_.max_token_len = runtime_config_.max_seq_len;

  if ((!batch_scheduler_config_.split_fuse_token_num ||
       !runtime_config_.attn_backend_config.enable_blocked_multi_token_forwarding_kv) &&
      batch_scheduler_config_.max_step_token_num < batch_scheduler_config_.max_token_len) {
    KLLM_LOG_WARNING << fmt::format(
        "Reset max_step_token_num {} to max_token_len {}, since split fuse is not enabled or chunked KV is not "
        "supported",
        batch_scheduler_config_.max_step_token_num, batch_scheduler_config_.max_token_len);
    batch_scheduler_config_.max_step_token_num = batch_scheduler_config_.max_token_len;
  }

  if (batch_scheduler_config_.max_chunked_kv_tokens == 0) {
    // Set to max_step_token_num by default
    batch_scheduler_config_.max_chunked_kv_tokens = batch_scheduler_config_.max_step_token_num;
  }
  // max_chunked_kv_tokens must be at least max_seq_len so any single request
  // can fit in one chunk
  if (batch_scheduler_config_.max_chunked_kv_tokens < batch_scheduler_config_.max_token_len) {
    KLLM_LOG_WARNING << fmt::format(
        "Reset max_chunked_kv_tokens {} to max_token_len {} to ensure any single request can fit in one chunk",
        batch_scheduler_config_.max_chunked_kv_tokens, batch_scheduler_config_.max_token_len);
    batch_scheduler_config_.max_chunked_kv_tokens = batch_scheduler_config_.max_token_len;
  }

  // Align max_step_token_num to tp_size for even token distribution across ranks
  batch_scheduler_config_.max_step_token_num =
      RoundUp(batch_scheduler_config_.max_step_token_num, runtime_config_.parallel_basic_config.tensor_parallel_size);

  runtime_config_.parallel_basic_config.moe_tensor_para_size =
      runtime_config_.parallel_basic_config.tensor_parallel_size /
      runtime_config_.parallel_basic_config.expert_parallel_size;

  runtime_config_.inter_data_type = model_config.weight_data_type;
  // TODO(robertyuan): These members should be removed from other configs
  runtime_config_.max_batch_size = batch_scheduler_config_.max_batch_size;
  runtime_config_.max_pp_batch_num = batch_scheduler_config_.max_pp_batch_num;
  runtime_config_.max_step_token_num = batch_scheduler_config_.max_step_token_num;
  runtime_config_.max_chunked_kv_tokens = batch_scheduler_config_.max_chunked_kv_tokens;
  runtime_config_.mtp_step_num = batch_scheduler_config_.mtp_step_num;
  runtime_config_.enable_async = batch_scheduler_config_.enable_async;
  runtime_config_.enable_executor_build_ahead = batch_scheduler_config_.enable_executor_build_ahead;
  runtime_config_.enable_speculative_decoding = batch_scheduler_config_.enable_speculative_decoding;

  runtime_config_.separate_prefill_decode = (connector_config_.group_role != GroupRole::NONE);
  runtime_config_.is_decode_only = (connector_config_.group_role == GroupRole::DECODE);
  // pd_v2 Decode is NOT decode-only: the local-bypass fast path
  // (PdV2DecodeHook::local_complete_threshold) runs prefill on this
  // node for short residuals. is_decode_only=true would let
  // SelectEffectiveStepTokenNum shrink the forwarding buffers
  // (mla_hidden_buffer / shared_buffer) to
  // max_batch_size * max_decode_tokens_per_req, which corrupts
  // local-bypass prefill — produces degenerate output where the model
  // echoes input tokens (memory K7). Override here.
  if (pd_v2_runtime_config_.is_pd_v2 && pd_v2_runtime_config_.group_role == GroupRole::DECODE) {
    runtime_config_.is_decode_only = false;
  }
  runtime_config_.enable_prefix_caching =
      cache_manager_config_.enable_prefix_caching || batch_scheduler_config_.split_fuse_token_num > 0;
  runtime_config_.enable_flexible_caching = cache_manager_config_.min_flexible_cache_num > 0;

  // Propagate SWA sliding_window_size from DeepSeek v4 model config.
  if (model_config.use_dsv4_attn) {
    cache_manager_config_.sliding_window_size = model_config.dsv4_attn_config.sliding_window;
  }

  return Status();
}

void ScheduleConfigParser::InitializeKVCacheConfigs(const ModelConfig &model_config,
                                                    const PipelineConfig &pipeline_config,
                                                    std::shared_ptr<BaseModelCacheLayout> block_cache_layout) {
  runtime_config_.attn_backend_config.block_size =
      block_cache_layout->GetBlockCacheLayout("kv-cache")->GetBlockCacheSize();

  // Store SWA block size if the "SWA-cache" layout is registered (DeepSeek-V4 models).
  if (block_cache_layout->IsBlockCacheTypeExist("SWA-cache")) {
    block_manager_config_.swa_device_allocator_config.block_size =
        block_cache_layout->GetBlockCacheLayout("SWA-cache")->GetBlockCacheSize();
    block_manager_config_.swa_device_allocator_config.device = MemoryDevice::MEMORY_DEVICE;
  }
}

Status ScheduleConfigParser::InitializeBlockManagerConfig(const ModelConfig &model_config,
                                                          std::shared_ptr<BaseModelCacheLayout> block_cache_layout) {
  if (pipeline_config_.lower_layer_idx < 0 || pipeline_config_.upper_layer_idx < 0) {
    pipeline_config_.lower_layer_idx = 0;
    pipeline_config_.upper_layer_idx = model_config.num_layer - 1;
    if (model_config.num_nextn_predict_layers != 0 && batch_scheduler_config_.mtp_step_num > 0) {
      pipeline_config_.lower_nextn_layer_idx = model_config.num_layer;
      pipeline_config_.upper_nextn_layer_idx = model_config.num_layer + model_config.num_nextn_predict_layers - 1;
    }
  }

  InitializeKVCacheConfigs(model_config, pipeline_config_, block_cache_layout);

  block_manager_config_.host_allocator_config.block_size = runtime_config_.attn_backend_config.block_size;
  block_manager_config_.device_allocator_config.block_size = runtime_config_.attn_backend_config.block_size;

  block_manager_config_.host_allocator_config.device = MemoryDevice::MEMORY_HOST;
  block_manager_config_.device_allocator_config.device = MemoryDevice::MEMORY_DEVICE;

  // The default block number, will be overwrited through memory usage.
  block_manager_config_.host_allocator_config.blocks_num = 512 * 10;
  block_manager_config_.device_allocator_config.blocks_num = 512;

  return Status();
}

Status ScheduleConfigParser::GetBatchSchedulerConfig(BatchSchedulerConfig &batch_scheduler_config) {
  batch_scheduler_config = batch_scheduler_config_;
  return Status();
}

void ScheduleConfigParser::SetBatchSchedulerConfig(BatchSchedulerConfig &batch_scheduler_config) {
  batch_scheduler_config_ = batch_scheduler_config;
}

Status ScheduleConfigParser::GetCacheManagerConfig(CacheManagerConfig &cache_manager_config) {
  cache_manager_config = cache_manager_config_;
  return Status();
}

void ScheduleConfigParser::SetCacheManagerConfig(CacheManagerConfig &cache_manager_config) {
  cache_manager_config_ = cache_manager_config;
}

Status ScheduleConfigParser::GetBlockManagerConfig(BlockManagerConfig &block_manager_config) {
  block_manager_config = block_manager_config_;
  return Status();
}

void ScheduleConfigParser::SetBlockManagerConfig(const BlockManagerConfig &block_manager_config) {
  block_manager_config_ = block_manager_config;
}

Status ScheduleConfigParser::GetRuntimeConfig(RuntimeConfig &runtime_config) {
  runtime_config = runtime_config_;
  return Status();
}

Status ScheduleConfigParser::CalculateBlockNumber(size_t device_free, size_t device_total) {
  size_t host_total, host_free;
  Status status = GetHostMemoryInfo(&host_free, &host_total);
  if (!status.OK()) {
    return status;
  }

  KLLM_LOG_INFO << "Get memory info, host_total:" << host_total << ", host_free:" << host_free
                << ", device_total:" << device_total << ", device_free:" << device_free
                << ", block_device_memory_ratio:" << block_manager_config_.block_device_memory_ratio
                << ", reserved_device_memory_ratio:" << block_manager_config_.reserved_device_memory_ratio
                << ", block_host_memory_factor:" << block_manager_config_.block_host_memory_factor;

  KLLM_CHECK_WITH_INFO(block_manager_config_.reserved_device_memory_ratio > 0.0,
                       "reserved_device_memory_ratio must be large than 0.0");
  KLLM_CHECK_WITH_INFO(block_manager_config_.block_host_memory_factor >= 0.0, "block_host_memory_factor should >= 0.0");

  const size_t alignment_bytes = 8;
  size_t device_block_memory_size = 0;
  if (block_manager_config_.block_device_memory_ratio >= 0.0) {
    device_block_memory_size =
        DivRoundDown(std::min((static_cast<size_t>(device_total * block_manager_config_.block_device_memory_ratio)),
                              device_free),
                     alignment_bytes) *
        alignment_bytes;
  } else {
    size_t reserved_memory_size = 0;
    float device_reserved_ratio = block_manager_config_.reserved_device_memory_ratio;
    reserved_memory_size = DivRoundUp((device_total * device_reserved_ratio), alignment_bytes) * alignment_bytes;

    device_block_memory_size =
        DivRoundDown((reserved_memory_size < device_free ? device_free - reserved_memory_size : 0ul), alignment_bytes) *
        alignment_bytes;
  }

  const float block_host_memory_ratio = 0.8;
  size_t host_block_memory_size =
      DivRoundDown(
          static_cast<size_t>(std::min(device_block_memory_size * block_manager_config_.block_host_memory_factor,
                                       host_free * block_host_memory_ratio)),
          alignment_bytes) *
      alignment_bytes;

  KLLM_LOG_INFO << "Get block memory info, host_free:" << host_block_memory_size
                << ", device_free:" << device_block_memory_size
                << ", block_size:" << block_manager_config_.host_allocator_config.block_size;

  // When SWA is enabled, each KV block is paired with swa_block_num_ratio SWA blocks.
  // The effective memory cost per KV block is:
  //   kv_block_size + swa_block_num_ratio * swa_block_size
  // so that device_blocks_num = device_block_memory_size / effective_block_size.
  const float swa_block_num_ratio = batch_scheduler_config_.swa_block_num_ratio;
  const size_t swa_block_size = block_manager_config_.swa_device_allocator_config.block_size;
  size_t effective_block_size = block_manager_config_.device_allocator_config.block_size;
  if (swa_block_num_ratio > 0.0f && swa_block_size > 0) {
    effective_block_size += static_cast<size_t>(swa_block_num_ratio * static_cast<float>(swa_block_size));
    KLLM_LOG_INFO << "SWA enabled: swa_block_num_ratio=" << swa_block_num_ratio << ", swa_block_size=" << swa_block_size
                  << ", effective_block_size=" << effective_block_size;
  }

  size_t device_blocks_num = device_block_memory_size / effective_block_size;
  size_t host_blocks_num = host_block_memory_size / block_manager_config_.host_allocator_config.block_size;
  KLLM_LOG_INFO << "Device blocks limit = " << device_blocks_num << "."
                << "Host blocks limit = " << host_blocks_num << ".";
  // Control max device_blocks_num through KLLM_MAX_DEVICE_BLOCKS
  const char *max_blocks_str = std::getenv("KLLM_MAX_DEVICE_BLOCKS");
  if (max_blocks_str != nullptr) {
    try {
      size_t max_device_blocks = std::stoull(max_blocks_str);
      if (max_device_blocks >= 1 && max_device_blocks <= device_blocks_num) {
        device_blocks_num = max_device_blocks;
        KLLM_LOG_INFO << "Using custom max device blocks limit: " << max_device_blocks;
      }
    } catch (const std::exception &e) {
    }
  }
  KLLM_LOG_INFO << "Reset device_blocks_num:" << device_blocks_num << ", host_block_num:" << host_blocks_num;

  // If the number of available device blocks is less than the launch threshold,
  // inference cannot be performed due to insufficient resources.
  if (device_blocks_num <= batch_scheduler_config_.launch_block_threshold) {
    KLLM_THROW("KsanaLLM has insufficient blocks available; unable to perform inference.");
  }

  size_t usable_tokens = (device_blocks_num - batch_scheduler_config_.launch_block_threshold) *
                         block_manager_config_.device_allocator_config.block_token_num;
  if (usable_tokens < batch_scheduler_config_.max_step_token_num) {
    KLLM_LOG_ERROR << fmt::format(
        "Since available device blocks are insufficient, max_step_token_num is reduced from {} to {}",
        batch_scheduler_config_.max_step_token_num, usable_tokens);
    batch_scheduler_config_.max_step_token_num = usable_tokens;
  }
  if (usable_tokens < batch_scheduler_config_.max_token_len) {
    KLLM_LOG_WARNING << fmt::format(
        "Unable to support configured max_token_len ({}), max available context length is {}",
        batch_scheduler_config_.max_token_len, usable_tokens);
  }

  block_manager_config_.device_allocator_config.blocks_num = device_blocks_num;
  block_manager_config_.host_allocator_config.blocks_num = host_blocks_num;

  // Compute and store the SWA device block count.
  if (swa_block_num_ratio > 0.0f && swa_block_size > 0) {
    block_manager_config_.swa_device_allocator_config.blocks_num =
        static_cast<size_t>(static_cast<float>(device_blocks_num) * swa_block_num_ratio);
    KLLM_LOG_INFO << "SWA device block num: " << block_manager_config_.swa_device_allocator_config.blocks_num;
  }

  return Status();
}

Status ScheduleConfigParser::ResetPipelineBlockNumber() {
  // Get block number from pipeline config if in distributed mode.
  PipelineConfig pipeline_config;
  Singleton<Environment>::GetInstance()->GetPipelineConfig(pipeline_config);

  size_t device_blocks_num = pipeline_config.device_block_num;
  size_t host_block_num = pipeline_config.host_block_num;

  KLLM_LOG_INFO << "Reset device_blocks_num:" << device_blocks_num << ", host_block_num:" << host_block_num;

  block_manager_config_.device_allocator_config.blocks_num = device_blocks_num;
  block_manager_config_.host_allocator_config.blocks_num = host_block_num;

  // PP sync only updates main KV block count; SWA pool size must be re-derived from the synced
  // main block count so engine cache manager and executor SWA allocators stay consistent.
  const float swa_block_num_ratio = batch_scheduler_config_.swa_block_num_ratio;
  const size_t swa_block_size = block_manager_config_.swa_device_allocator_config.block_size;
  if (swa_block_num_ratio > 0.0f && swa_block_size > 0) {
    block_manager_config_.swa_device_allocator_config.blocks_num =
        static_cast<size_t>(static_cast<float>(device_blocks_num) * swa_block_num_ratio);
    KLLM_LOG_INFO << "Reset SWA device block num: " << block_manager_config_.swa_device_allocator_config.blocks_num;
  }

  return Status();
}

size_t ScheduleConfigParser::GetTotalDeviceBlockNum() {
  return block_manager_config_.device_allocator_config.blocks_num;
}

size_t ScheduleConfigParser::GetTotalHostBlockNum() { return block_manager_config_.host_allocator_config.blocks_num; }

std::vector<int> ScheduleConfigParser::GetDataParaGroupDevices(int dp_id) {
  size_t device_count = runtime_config_.parallel_basic_config.tensor_parallel_size;
  size_t group_device_count = device_count / runtime_config_.parallel_basic_config.attn_data_parallel_size;

  std::vector<int> group_devices;
  for (size_t i = 0; i < group_device_count; ++i) {
    group_devices.push_back(dp_id * group_device_count + i);
  }

  return group_devices;
}

bool ScheduleConfigParser::IsPrefixCachingEnabled() { return cache_manager_config_.enable_prefix_caching; }

size_t ScheduleConfigParser::GetTransferLayerChunkSize() { return batch_scheduler_config_.transfer_layer_chunk_size; }

void ScheduleConfigParser::InitializeExpertParallelConfig() {
  const char *const expert_master_host = std::getenv("EXPERT_MASTER_HOST");
  const char *const expert_master_port = std::getenv("EXPERT_MASTER_PORT");
  const char *const expert_node_rank = std::getenv("EXPERT_NODE_RANK");
  const char *const use_tcp_data_channel = std::getenv("USE_TCP_DATA_CHANNEL");

  ExpertParallelConfig expert_parallel_config;
  GetExpertParallelConfig(expert_parallel_config);
  expert_parallel_config.expert_node_rank = expert_node_rank ? std::stoi(expert_node_rank) : 0;
  expert_parallel_config.expert_para_size = runtime_config_.parallel_basic_config.expert_parallel_size;
  expert_parallel_config.expert_tensor_para_size = runtime_config_.parallel_basic_config.tensor_parallel_size /
                                                   runtime_config_.parallel_basic_config.expert_parallel_size;
  expert_parallel_config.global_expert_para_size =
      expert_parallel_config.expert_world_size * expert_parallel_config.expert_para_size;
  if (expert_parallel_config.expert_world_size > 1) {
    if (!expert_master_host || !expert_master_port) {
      throw std::runtime_error(
          "The environment variable MASTER_HOST and MASTER_PORT must be set in distributed expert parallel mode.");
    }
  }

  expert_parallel_config.expert_master_host = expert_master_host ? expert_master_host : "";
  expert_parallel_config.expert_master_port = expert_master_port ? std::stoi(expert_master_port) : 0;

  if (use_tcp_data_channel && strcmp(use_tcp_data_channel, "1") == 0) expert_parallel_config_.use_tcp = true;

  KLLM_LOG_INFO << "InferenceService initialize expert parallel config, expert_master_host:"
                << expert_parallel_config.expert_master_host
                << ", expert_master_port:" << expert_parallel_config.expert_master_port
                << ", expert_world_size:" << expert_parallel_config.expert_world_size
                << ", expert_para_size:" << expert_parallel_config.expert_para_size
                << ", global_expert_para_size:" << expert_parallel_config.global_expert_para_size
                << ", expert_node_rank:" << expert_parallel_config.expert_node_rank
                << ", use_tcp: " << expert_parallel_config.use_tcp;
  SetExpertParallelConfig(expert_parallel_config);
}

void ScheduleConfigParser::InitGlobalCacheConnectorConfig(YamlReader &yaml_reader) {
  std::string endpoint = yaml_reader.GetScalar<std::string>(yaml_reader.GetRootNode(),
                                                            "setting.global_cache_connector.service_endpoint", "");
  if (endpoint.empty()) {
    KLLM_LOG_INFO << "Global cache connector endpoint not configured, skipping.";
    return;
  }

  global_cache_connector_config_.service_endpoint = endpoint;
  global_cache_connector_config_.backend_type = yaml_reader.GetScalar<std::string>(
      yaml_reader.GetRootNode(), "setting.global_cache_connector.backend_type", "mooncake");
  global_cache_connector_config_.timeout_ms =
      yaml_reader.GetScalar<int>(yaml_reader.GetRootNode(), "setting.global_cache_connector.timeout_ms", 5000);
  global_cache_connector_config_.max_retry =
      yaml_reader.GetScalar<int>(yaml_reader.GetRootNode(), "setting.global_cache_connector.max_retry", 3);
  global_cache_connector_config_.fetch_thread_num =
      yaml_reader.GetScalar<int>(yaml_reader.GetRootNode(), "setting.global_cache_connector.fetch_thread_num", 4);
  global_cache_connector_config_.role =
      yaml_reader.GetScalar<std::string>(yaml_reader.GetRootNode(), "setting.global_cache_connector.role", "");
  if (!global_cache_connector_config_.role.empty() && global_cache_connector_config_.role != "prefill" &&
      global_cache_connector_config_.role != "decode") {
    KLLM_LOG_WARNING << "Unknown global cache connector role: '" << global_cache_connector_config_.role
                     << "', expected 'prefill', 'decode' or empty. Treating as standalone.";
    global_cache_connector_config_.role = "";
  }

  // 真实 Mooncake Store 配置字段
  global_cache_connector_config_.protocol =
      yaml_reader.GetScalar<std::string>(yaml_reader.GetRootNode(), "setting.global_cache_connector.protocol", "tcp");
  global_cache_connector_config_.rdma_devices =
      yaml_reader.GetScalar<std::string>(yaml_reader.GetRootNode(), "setting.global_cache_connector.rdma_devices", "");
  global_cache_connector_config_.master_server_addr = yaml_reader.GetScalar<std::string>(
      yaml_reader.GetRootNode(), "setting.global_cache_connector.master_server_addr", "127.0.0.1:50051");
  global_cache_connector_config_.global_segment_size = yaml_reader.GetScalar<size_t>(
      yaml_reader.GetRootNode(), "setting.global_cache_connector.global_segment_size", 16 * 1024 * 1024);
  global_cache_connector_config_.local_buffer_size = yaml_reader.GetScalar<size_t>(
      yaml_reader.GetRootNode(), "setting.global_cache_connector.local_buffer_size", 16 * 1024 * 1024);

  // 从已解析的 batch_scheduler / block_manager 配置中填充，用于 connector 提前初始化 staging pool。
  // block_byte_size 依赖模型配置计算，在 engine/executor 创建 connector 时再填充。
  global_cache_connector_config_.max_step_tokens = batch_scheduler_config_.max_step_token_num;
  global_cache_connector_config_.block_token_num =
      static_cast<int>(block_manager_config_.device_allocator_config.block_token_num);

  KLLM_LOG_INFO << "Global cache connector config parsed: " << global_cache_connector_config_.ToString();
}

}  // namespace ksana_llm
