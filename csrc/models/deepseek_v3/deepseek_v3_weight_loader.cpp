/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "models/deepseek_v3/deepseek_v3_weight_loader.h"

#include <regex>

#include "configure/json_config_utils.h"
#include "kernels/trans_layout.h"
#include "model_loader/model_loader_utils.h"
#include "utils/logger.h"
#include "utils/memory_utils.h"
#include "utils/string_utils.h"

#ifdef ENABLE_CUDA
#  include "kernels/nvidia/kernel_wrapper.h"
#endif

namespace ksana_llm {
DeepSeekV3WeightLoader::DeepSeekV3WeightLoader(std::shared_ptr<BaseModelConfig> model_config,
                                               std::shared_ptr<Environment> env, std::shared_ptr<Context> context)
    : BaseModelWeightLoader(model_config, env, context) {
  // Initialize pipeline config, for distributed mode.
  env->GetPipelineConfig(pipeline_config_);
  env->GetRuntimeConfig(runtime_config_);
  weights_to_permute_.resize(runtime_config_.parallel_basic_config.tensor_parallel_size);

  std::shared_ptr<DeepSeekV3Config> deepseek_v3_config = std::dynamic_pointer_cast<DeepSeekV3Config>(model_config_);

  InitWeightLoaderImpl(deepseek_v3_config);
}

DeepSeekV3WeightLoader::~DeepSeekV3WeightLoader() {}

Status DeepSeekV3WeightLoader::FilterWeightNames(std::vector<std::string>& weight_names) {
  const std::vector<std::string> skip_list = {"self_attn.rotary_emb.inv_freq", "vision_tower.", "mm_projector."};

  const int lower_layer_idx = pipeline_config_.lower_layer_idx;
  const int upper_layer_idx = pipeline_config_.upper_layer_idx;
  const int lower_nextn_layer_idx = pipeline_config_.lower_nextn_layer_idx;
  const int upper_nextn_layer_idx = pipeline_config_.upper_nextn_layer_idx;

  for (auto it = weight_names.begin(); it != weight_names.end();) {
    if (CheckWeightNameMatched(*it, skip_list, false)) {
      weight_names.erase(it);
      continue;
    }

    // Skip some layers in distributed mode.
    if (lower_layer_idx >= 0 && upper_layer_idx >= 0) {
      int layer_idx = GetLayerIdxFromName(*it);
      if (layer_idx >= 0 && ((layer_idx < lower_layer_idx || layer_idx > upper_layer_idx) &&
                             (layer_idx < lower_nextn_layer_idx || layer_idx > upper_nextn_layer_idx))) {
        weight_names.erase(it);
        continue;
      }
      // Skip nextn layer in distributed non-sampling node
      if (layer_idx >= 0 && !context_->global->IsSamplingNode() &&
          (layer_idx >= lower_nextn_layer_idx && layer_idx <= upper_nextn_layer_idx)) {
        weight_names.erase(it);
        continue;
      }
    }

    if (!context_->global->IsStandalone()) {
      // Skip embedding on if not master mode.
      if (context_->global->IsDistributedWorker() &&
          !(context_->global->IsSamplingNode() && runtime_config_.mtp_step_num > 0)) {
        if (CheckWeightNameMatched(*it, {"model.embed_tokens.weight"}, false)) {
          weight_names.erase(it);
          continue;
        }
      }

      // Skip lm_head if not tail mode.
      if (!context_->global->IsDistributedLastWorker()) {
        if (CheckWeightNameMatched(*it, {"lm_head.weight"}, false)) {
          weight_names.erase(it);
          continue;
        }
      }
    }
    ++it;
  }
  return Status();
}

Status DeepSeekV3WeightLoader::PostProcessModelWeights(std::unordered_map<std::string, Tensor>& dev_weights_map,
                                                       int dev_rank) {
  std::shared_ptr<DeepSeekV3Config> deepseek_v3_config = std::dynamic_pointer_cast<DeepSeekV3Config>(model_config_);
  std::unordered_set<std::string> post_processed_weights;
  if (deepseek_v3_config->quant_config.is_fp8_blockwise) {
#if defined(ENABLE_FP8) && defined(ENABLE_FP8_TORCH)
    // Post process weight and scale for q_b_proj and kv_b_proj
    weight_impl_->PostProcessFp8E4m3BlockWiseQuantWeights(dev_weights_map, dev_rank, deepseek_v3_config);
#endif
    for (auto& [weight_name, weight_tensor] : dev_weights_map) {
      // for fp8 weight, bind weight scale
      if (weight_tensor.dtype == DataType::TYPE_FP8_E4M3 || weight_tensor.dtype == DataType::TYPE_BLOCK_FP8_E4M3) {
        std::string weight_scale_name = weight_name + "_scale_inv";
        auto scale_iter = dev_weights_map.find(weight_scale_name);
        if (scale_iter != dev_weights_map.end()) {
          weight_tensor.weight_scales = &(scale_iter->second);
          KLLM_LOG_DEBUG << fmt::format("bind {}, shape: {} to {}, shape: {}\n", weight_scale_name,
                                        Vector2Str(std::vector<size_t>(weight_tensor.weight_scales->shape)),
                                        weight_name, Vector2Str(std::vector<size_t>(weight_tensor.shape)));
          post_processed_weights.insert(weight_scale_name);
          post_processed_weights.insert(weight_name);
        } else {
          KLLM_THROW(fmt::format("weight scale inv not found: {}", weight_scale_name));
        }
      }
    }
  }

  for (auto& [weight_name, weight_tensor] : dev_weights_map) {
    if (weights_to_permute_[dev_rank].find(weight_name) != weights_to_permute_[dev_rank].end()) {
      // permute weight tensor
      weight_impl_->PermuteWeight(weight_tensor, {1, 0}, dev_rank);
    }

    if (post_processed_weights.find(weight_name) != post_processed_weights.end()) {
      continue;
    }

    if (!(weight_tensor.dtype == DataType::TYPE_FP32 || weight_tensor.dtype == DataType::TYPE_FP16 ||
          weight_tensor.dtype == DataType::TYPE_BF16)) {
      continue;
    }

    // NOTE: `e_score_correction_bias` need to cast into fp32 for better performance
    if (weight_name.find("gate.e_score_correction_bias") != std::string::npos) {
      KLLM_LOG_DEBUG << fmt::format("Dev_rank-{}, cast weight {}, from {} to {}\n", dev_rank, weight_name,
                                    weight_tensor.dtype, TYPE_FP32);
      CastDeviceTensorType(weight_tensor, TYPE_FP32, dev_rank);
      continue;
    }
    if (weight_tensor.dtype != deepseek_v3_config->weight_data_type) {
      KLLM_LOG_DEBUG << fmt::format("Dev_rank-{}, cast weight {}, from {} to {}\n", dev_rank, weight_name,
                                    weight_tensor.dtype, deepseek_v3_config->weight_data_type);
      CastDeviceTensorType(weight_tensor, deepseek_v3_config->weight_data_type, dev_rank);
    }
  }

  if (deepseek_v3_config->ContainGptqWeights()) {
    KLLM_LOG_DEBUG << "Start gptq post process";
    weight_impl_->PostProcessInt4QuantWeights(dev_weights_map, dev_rank, deepseek_v3_config);
  }

  return Status();
}

Status DeepSeekV3WeightLoader::ProcessModelWeights(const std::unordered_map<std::string, Tensor>& host_model_weights,
                                                   int dev_rank,
                                                   std::unordered_map<std::string, Tensor>& device_model_weights,
                                                   std::unordered_map<std::string, Tensor>& left_host_weights) {
  std::shared_ptr<DeepSeekV3Config> deepseek_v3_config = std::dynamic_pointer_cast<DeepSeekV3Config>(model_config_);

  int32_t layer_idx = -1, expert_idx = -1;
  size_t num_experts = deepseek_v3_config->moe_config.num_experts;
  bool use_vllm_moe = deepseek_v3_config->moe_config.use_vllm_moe;
  // moe combines tensor parallel and moe parallel
  ExpertParallelConfig& expert_parallel_config = deepseek_v3_config->expert_parallel_config;
  size_t expert_node_rank = expert_parallel_config.expert_node_rank;
  size_t global_expert_para_size = expert_parallel_config.expert_world_size * expert_parallel_config.expert_para_size;
  size_t num_experts_per_rank = DivRoundUp(num_experts, global_expert_para_size);

  // init expert map
  size_t rank_expert_offset = expert_node_rank * expert_parallel_config.expert_para_size * num_experts_per_rank;
  size_t expert_offset =
      (global_expert_para_size > 1) ? ((dev_rank % deepseek_v3_config->expert_para_size) * num_experts_per_rank) : 0;
  size_t expert_start_id = rank_expert_offset + expert_offset;
  size_t expert_end_id = std::min(num_experts, expert_start_id + num_experts_per_rank);
  size_t num_layers = deepseek_v3_config->num_layer;
  if (pipeline_config_.lower_nextn_layer_idx >= num_layers) {
    num_layers += pipeline_config_.upper_nextn_layer_idx - pipeline_config_.lower_nextn_layer_idx + 1;
  }
  std::vector<std::vector<int>> expert_map(num_layers, std::vector<int>(num_experts, -1));
  InitExpertMap(num_experts, expert_start_id, expert_end_id, dev_rank, expert_map, device_model_weights);

  KLLM_LOG_INFO << fmt::format("expert_world_size = {}, expert_para_size = {}, global_expert_para_size = {}",
                               expert_parallel_config.expert_world_size, expert_parallel_config.expert_para_size,
                               global_expert_para_size);
  KLLM_LOG_INFO << fmt::format("In rank {} node_rank {}, valid experts is [{}, {})", dev_rank, expert_node_rank,
                               expert_start_id, expert_end_id);

  size_t moe_inter_size_per_rank =
      DivRoundUp(deepseek_v3_config->moe_config.moe_inter_size, deepseek_v3_config->moe_tensor_para_size);
  size_t hidden_units = deepseek_v3_config->hidden_units;
  std::vector<size_t> up_gate_experts_shape = {num_experts_per_rank,
                                               /* up & gate*/ moe_inter_size_per_rank * 2, hidden_units};
  std::vector<size_t> down_experts_shape = {num_experts_per_rank, hidden_units, moe_inter_size_per_rank};

  size_t attn_tp_size = context_->global->GetAttentionTensorParallelSize();
  size_t attn_dev_rank = dev_rank % attn_tp_size;
  size_t kv_lora_rank = deepseek_v3_config->mla_config.kv_lora_rank;
  size_t qk_rope_head_dim = deepseek_v3_config->mla_config.qk_rope_head_dim;
  size_t qk_nope_head_dim = deepseek_v3_config->mla_config.qk_nope_head_dim;
  size_t v_head_dim = deepseek_v3_config->mla_config.v_head_dim;
  size_t q_lora_rank = deepseek_v3_config->mla_config.q_lora_rank;
  size_t head_num = deepseek_v3_config->head_num;
  size_t head_num_tp = static_cast<size_t>(DivRoundUp(deepseek_v3_config->head_num, attn_tp_size));

  if (deepseek_v3_config->model_format == ModelFormat::GGUF) {
    return Status(RET_INVALID_ARGUMENT, "Not support GGUF format yet.");
  }

  // Dequant GPTQ weight
  std::unordered_map<std::string, Tensor> host_gptq_weights;
  for (auto& [origin_file_weight_name, host_weight_tensor] : host_model_weights) {
    std::string file_weight_name = origin_file_weight_name;
    // replace weight name with condition
    if (deepseek_v3_config->ContainGptqWeights() && deepseek_v3_config->IsWeightMatchGptq(file_weight_name)) {
      for (const auto& [pattern, format] : deepseek_v3_config->w4a8_patterns_) {
        if (std::regex_search(origin_file_weight_name, pattern)) {
          file_weight_name = std::regex_replace(origin_file_weight_name, pattern, format);
          break;
        }
      }
    }

    // for GPTQ weight or moe-int4(mixed with fp8) weight, save to host_gptq_weights and load later
    if (deepseek_v3_config->ContainGptqWeights() && deepseek_v3_config->IsWeightMatchGptq(file_weight_name) &&
        deepseek_v3_config->IsGptqContain(file_weight_name)) {
      host_gptq_weights.emplace(file_weight_name, host_weight_tensor);
      continue;
    }

    KLLM_LOG_DEBUG << fmt::format("Dev_rank: {}, processing weight: {}, shape: {}", dev_rank, file_weight_name,
                                  Vector2Str(std::vector<size_t>(host_weight_tensor.shape)));

    // 1. MOE layer
    // Instructions for loading MoE model weights:
    // For each layer of the model, the experts at the same positions
    // of up and gate need to be concatenated and named as up_gate.weight.
    // All experts corresponding to up_gate and down in each layer need to be stacked into one expert weight.
    if (file_weight_name.find(".experts.") != std::string::npos) {
      STATUS_CHECK_RETURN(weight_impl_->GetExpertsIdx(file_weight_name, layer_idx, expert_idx));
      if (layer_idx < 0 || expert_idx < 0) {
        continue;
      }
      int expert_idx_ = expert_map[layer_idx][expert_idx];
      if (expert_idx_ < 0) {
        // Skip load weight when the expert_id will be not used in current rank.
        continue;
      }
#ifdef ENABLE_FP8
      if (deepseek_v3_config->quant_config.is_fp8_blockwise &&
          weight_impl_->LoadMoeFp8E4m3BlockWiseScale(file_weight_name, host_weight_tensor, dev_rank, deepseek_v3_config,
                                                     device_model_weights, expert_idx_)) {
        continue;
      }
#endif
      if (deepseek_v3_config->quant_config.method == QUANT_FP8_E4M3 &&
          (file_weight_name.find("input_scale") != std::string::npos ||
           file_weight_name.find("weight_scale") != std::string::npos)) {
        continue;
      }

      if (file_weight_name.find(".up_proj.") != std::string::npos ||
          file_weight_name.find(".gate_proj.") != std::string::npos) {
        std::string up_gate_experts_name =
            "model.layers." + std::to_string(layer_idx) + ".mlp.experts.up_gate_proj.weight";
        if (device_model_weights.find(up_gate_experts_name) == device_model_weights.end()) {
          device_model_weights[up_gate_experts_name] =
              Tensor(MemoryLocation::LOCATION_DEVICE, host_weight_tensor.dtype, up_gate_experts_shape, nullptr,
                     &(context_->device->GetMemoryManageStream()));
        }

        size_t expert_pitch = moe_inter_size_per_rank * hidden_units * GetTypeSize(host_weight_tensor.dtype);
        size_t double_expert_pitch = expert_pitch * 2;
        size_t src_upgate_offset = deepseek_v3_config->moe_tensor_para_size > 1
                                       ? (dev_rank / expert_parallel_config.expert_para_size) * expert_pitch
                                       : 0;

        Tensor& up_gate_experts_tensor = device_model_weights.at(up_gate_experts_name);
        if (file_weight_name.find(".up_proj.") != std::string::npos) {
          MemcpyAsync(up_gate_experts_tensor.GetPtr<void>() + static_cast<size_t>(expert_idx_) * double_expert_pitch +
                          (use_vllm_moe ? expert_pitch : 0),
                      host_weight_tensor.GetPtr<void>() + src_upgate_offset, expert_pitch, MEMCPY_HOST_TO_DEVICE,
                      context_->device->GetMemoryManageStream());
        } else if (file_weight_name.find(".gate_proj.") != std::string::npos) {
          MemcpyAsync(up_gate_experts_tensor.GetPtr<void>() + static_cast<size_t>(expert_idx_) * double_expert_pitch +
                          (use_vllm_moe ? 0 : expert_pitch),
                      host_weight_tensor.GetPtr<void>() + src_upgate_offset, expert_pitch, MEMCPY_HOST_TO_DEVICE,
                      context_->device->GetMemoryManageStream());
        }
        continue;
      }

      if (file_weight_name.find(".down_proj.") != std::string::npos) {
        std::string down_experts_name = "model.layers." + std::to_string(layer_idx) + ".mlp.experts.down_proj.weight";
        if (device_model_weights.find(down_experts_name) == device_model_weights.end()) {
          device_model_weights[down_experts_name] =
              Tensor(MemoryLocation::LOCATION_DEVICE, host_weight_tensor.dtype, down_experts_shape, nullptr,
                     &(context_->device->GetMemoryManageStream()));
        }

        size_t dst_pitch = moe_inter_size_per_rank * GetTypeSize(host_weight_tensor.dtype);
        size_t src_pitch =
            moe_inter_size_per_rank * deepseek_v3_config->moe_tensor_para_size * GetTypeSize(host_weight_tensor.dtype);
        size_t expert_pitch = moe_inter_size_per_rank * hidden_units * GetTypeSize(host_weight_tensor.dtype);
        size_t src_down_offset = deepseek_v3_config->moe_tensor_para_size > 1
                                     ? (dev_rank / expert_parallel_config.expert_para_size) * dst_pitch
                                     : 0;
        Tensor& down_expert_tensor = device_model_weights.at(down_experts_name);
        Memcpy2DAsync(down_expert_tensor.GetPtr<void>() + static_cast<size_t>(expert_idx_) * expert_pitch, dst_pitch,
                      host_weight_tensor.GetPtr<void>() + src_down_offset, src_pitch, dst_pitch, hidden_units,
                      MEMCPY_HOST_TO_DEVICE, context_->device->GetMemoryManageStream());
        continue;
      }
    }

    // 2. MLA layer
    const std::string fused_lora_a_weight_name = ".fused_lora_a_proj.weight";
    if (file_weight_name.find("self_attn") != std::string::npos &&
        file_weight_name.find("norm.") == std::string::npos) {
#ifdef ENABLE_FP8
      if (deepseek_v3_config->quant_config.is_fp8_blockwise &&
          weight_impl_->LoadMlaFp8E4m3BlockWiseScale(file_weight_name, host_weight_tensor, dev_rank, deepseek_v3_config,
                                                     device_model_weights)) {
        continue;
      }
#endif
      // q_proj is for deepseek v2, q_b_proj is for deepseek v3
      if (file_weight_name.find(".q_proj.weight") != std::string::npos ||
          file_weight_name.find(".q_b_proj.weight") != std::string::npos) {
        // 3072 is deepseek v2
        if (host_weight_tensor.shape[0] != 3072 &&
            (qk_nope_head_dim + qk_rope_head_dim) * head_num != host_weight_tensor.shape[0]) {
          KLLM_THROW(fmt::format(
              "The shape of the 0th dim of the weight named '{} ({})' is not equal to the sum of qk_nope_head_dim {} "
              "and qk_rope_head_dim {}.",
              file_weight_name, host_weight_tensor.shape[0], qk_nope_head_dim, qk_rope_head_dim));
        }

        if (!deepseek_v3_config->quant_config.is_fp8_blockwise) {
          std::vector<size_t> q_b_nope_rope_shape = {
              static_cast<size_t>(DivRoundUp(head_num * (qk_nope_head_dim + qk_rope_head_dim), attn_tp_size)),
              host_weight_tensor.shape[1]};
          Tensor q_b_nope_rope_tensor =
              Tensor(MemoryLocation::LOCATION_DEVICE, host_weight_tensor.dtype, q_b_nope_rope_shape, nullptr,
                     &(context_->device->GetMemoryManageStream()));
          size_t para_pitch = DivRoundUp(head_num, attn_tp_size) * (qk_nope_head_dim + qk_rope_head_dim) *
                              host_weight_tensor.shape[1] * GetTypeSize(host_weight_tensor.dtype);
          size_t tensor_para_offset = attn_dev_rank * para_pitch;
          MemcpyAsync(q_b_nope_rope_tensor.GetPtr<void>(), host_weight_tensor.GetPtr<void>() + tensor_para_offset,
                      para_pitch, MEMCPY_HOST_TO_DEVICE, context_->device->GetMemoryManageStream());

          weight_impl_->PermuteWeight(q_b_nope_rope_tensor, {1, 0}, dev_rank);

          std::string q_b_nope_rope_name =
              file_weight_name.substr(0, file_weight_name.find_first_of('_')) + "_attn.q_b_nope_rope_proj.weight";
          device_model_weights[q_b_nope_rope_name] = q_b_nope_rope_tensor;
        } else {
          // For fp8 blockwise quant, do not split the weights initially, split them after dequantization later.
          std::string q_b_proj_name =
              file_weight_name.substr(0, file_weight_name.find_first_of('_')) + "_attn.q_b_proj.weight";
          std::vector<size_t> q_b_proj_shape = {
              static_cast<size_t>(DivRoundUp(host_weight_tensor.shape[0], attn_tp_size)), host_weight_tensor.shape[1]};
          Tensor q_b_proj_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, host_weight_tensor.dtype, q_b_proj_shape,
                                          nullptr, &(context_->device->GetMemoryManageStream()));

          size_t para_pitch = DivRoundUp(head_num, attn_tp_size) * (qk_nope_head_dim + qk_rope_head_dim) *
                              host_weight_tensor.shape[1] * GetTypeSize(host_weight_tensor.dtype);
          size_t tensor_para_offset = attn_dev_rank * para_pitch;
          MemcpyAsync(q_b_proj_tensor.GetPtr<void>(), host_weight_tensor.GetPtr<void>() + tensor_para_offset,
                      para_pitch, MEMCPY_HOST_TO_DEVICE, context_->device->GetMemoryManageStream());

          device_model_weights[q_b_proj_name] = q_b_proj_tensor;
        }
        continue;
      }
      // q_a_proj is for deepseek v3
      if (file_weight_name.find(".q_a_proj.weight") != std::string::npos) {
        // Weights are not split and are copied to each GPU (fused with kv_a_proj_with_mqa.weight).
        // fp16/bf16: weights needs to transpose
        std::string fused_tensor_name = GetReplacedName(file_weight_name, ".q_a_proj.weight", fused_lora_a_weight_name);
        Tensor fused_tensor;
        if (device_model_weights.find(fused_tensor_name) != device_model_weights.end()) {
          fused_tensor = device_model_weights.at(fused_tensor_name);
        } else {
          fused_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, host_weight_tensor.dtype,
                                {q_lora_rank + kv_lora_rank + qk_rope_head_dim, host_weight_tensor.shape[1]}, nullptr,
                                &(context_->device->GetMemoryManageStream()));
        }
        MemcpyAsync(fused_tensor.GetPtr<void>(), host_weight_tensor.GetPtr<void>(), host_weight_tensor.GetTotalBytes(),
                    MEMCPY_HOST_TO_DEVICE, context_->device->GetMemoryManageStream());
        if (device_model_weights.find(fused_tensor_name) == device_model_weights.end()) {
          device_model_weights[fused_tensor_name] = fused_tensor;
        }
        if (fused_tensor.dtype == DataType::TYPE_FP16 || fused_tensor.dtype == DataType::TYPE_BF16) {
          weights_to_permute_[dev_rank].insert(fused_tensor_name);
        }
        continue;
      }
      if (file_weight_name.find(".kv_a_proj_with_mqa.weight") != std::string::npos) {
        if ((kv_lora_rank + qk_rope_head_dim) != host_weight_tensor.shape[0]) {
          KLLM_THROW(fmt::format(
              "The shape of the 0th dim of the weight named `{}` is not equal to the sum of kv_lora_rank {} "
              "and qk_rope_head_dim {}.",
              file_weight_name, kv_lora_rank, qk_rope_head_dim));
        }
        std::string fused_tensor_name =
            GetReplacedName(file_weight_name, ".kv_a_proj_with_mqa.weight", fused_lora_a_weight_name);
        Tensor fused_tensor;
        if (device_model_weights.find(fused_tensor_name) != device_model_weights.end()) {
          fused_tensor = device_model_weights.at(fused_tensor_name);
        } else {
          fused_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, host_weight_tensor.dtype,
                                {q_lora_rank + kv_lora_rank + qk_rope_head_dim, host_weight_tensor.shape[1]}, nullptr,
                                &(context_->device->GetMemoryManageStream()));
        }
        const int offset = q_lora_rank * host_weight_tensor.shape[1] * host_weight_tensor.GetDTypeSize();
        MemcpyAsync(fused_tensor.GetPtr<void>() + offset, host_weight_tensor.GetPtr<void>(),
                    host_weight_tensor.GetTotalBytes(), MEMCPY_HOST_TO_DEVICE,
                    context_->device->GetMemoryManageStream());
        if (device_model_weights.find(fused_tensor_name) == device_model_weights.end()) {
          device_model_weights[fused_tensor_name] = fused_tensor;
        }
        if (fused_tensor.dtype == DataType::TYPE_FP16 || fused_tensor.dtype == DataType::TYPE_BF16) {
          weights_to_permute_[dev_rank].insert(fused_tensor_name);
        }
        continue;
      }
      if (file_weight_name.find(".kv_b_proj.weight") != std::string::npos) {
        if (head_num * (qk_nope_head_dim + v_head_dim) != host_weight_tensor.shape[0]) {
          KLLM_THROW(fmt::format(
              "The shape of the 0th dim of the weight named '{}' is not equal to the sum of qk_nope_head_dim {} "
              "and v_head_dim {}.",
              file_weight_name, kv_lora_rank, qk_rope_head_dim));
        }

        if (!deepseek_v3_config->quant_config.is_fp8_blockwise) {
          // For kv_b_nope_proj weight load
          std::string kv_b_nope_name =
              file_weight_name.substr(0, file_weight_name.find_first_of('_')) + "_attn.kv_b_nope_proj.weight";
          std::vector<size_t> kv_b_nope_shape = {
              static_cast<size_t>(DivRoundUp(head_num * qk_nope_head_dim, attn_tp_size)), host_weight_tensor.shape[1]};

          size_t para_pitch = DivRoundUp(head_num, attn_tp_size) * (qk_nope_head_dim + v_head_dim) *
                              host_weight_tensor.shape[1] * GetTypeSize(host_weight_tensor.dtype);
          size_t tensor_para_offset = attn_dev_rank * para_pitch;

          Tensor kv_b_nope_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, host_weight_tensor.dtype, kv_b_nope_shape,
                                           nullptr, &(context_->device->GetMemoryManageStream()));
          size_t nope_dst_pitch =
              qk_nope_head_dim * host_weight_tensor.shape[1] * GetTypeSize(host_weight_tensor.dtype);
          size_t src_pitch =
              (qk_nope_head_dim + v_head_dim) * host_weight_tensor.shape[1] * GetTypeSize(host_weight_tensor.dtype);
          Memcpy2DAsync(kv_b_nope_tensor.GetPtr<void>(), nope_dst_pitch,
                        host_weight_tensor.GetPtr<void>() + tensor_para_offset, src_pitch, nope_dst_pitch,
                        DivRoundUp(head_num, attn_tp_size), MEMCPY_HOST_TO_DEVICE,
                        context_->device->GetMemoryManageStream());

          Tensor kv_b_nope_permute = Tensor(MemoryLocation::LOCATION_DEVICE, host_weight_tensor.dtype, kv_b_nope_shape,
                                            nullptr, &(context_->device->GetMemoryManageStream()));
          PermuteDeviceTensor(kv_b_nope_tensor, {1, 0}, dev_rank, kv_b_nope_permute);

          device_model_weights[kv_b_nope_name] = kv_b_nope_permute;

          // For v_head_proj weight load
          std::string v_head_name =
              file_weight_name.substr(0, file_weight_name.find_first_of('_')) + "_attn.v_head_proj.weight";
          std::vector<size_t> v_head_shape = {static_cast<size_t>(DivRoundUp(head_num * v_head_dim, attn_tp_size)),
                                              host_weight_tensor.shape[1]};

          Tensor v_head_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, host_weight_tensor.dtype, v_head_shape,
                                        nullptr, &(context_->device->GetMemoryManageStream()));
          size_t v_head_dst_pitch = v_head_dim * host_weight_tensor.shape[1] * GetTypeSize(host_weight_tensor.dtype);
          Memcpy2DAsync(v_head_tensor.GetPtr<void>(), v_head_dst_pitch,
                        host_weight_tensor.GetPtr<void>() + nope_dst_pitch + tensor_para_offset, src_pitch,
                        v_head_dst_pitch, DivRoundUp(head_num, attn_tp_size), MEMCPY_HOST_TO_DEVICE,
                        context_->device->GetMemoryManageStream());

          Tensor v_head_permute = Tensor(MemoryLocation::LOCATION_DEVICE, host_weight_tensor.dtype, v_head_shape,
                                         nullptr, &(context_->device->GetMemoryManageStream()));
          PermuteDeviceTensor(v_head_tensor, {1, 0}, dev_rank, v_head_permute);

          device_model_weights[v_head_name] = v_head_permute;

          // Copy kv_b_nope_proj to w_uk_t
          Tensor w_uk_t_tensor = kv_b_nope_tensor;
          w_uk_t_tensor.shape = {head_num_tp, qk_nope_head_dim, kv_b_nope_tensor.shape[1]};
          std::string w_uk_t_name = kv_b_nope_name.substr(0, kv_b_nope_name.find_first_of('_')) + "_attn.w_uk_t.weight";
          device_model_weights[w_uk_t_name] = w_uk_t_tensor;

          // Permute vhead_weight_name to w_uv
          v_head_tensor.shape = {head_num_tp, v_head_dim, v_head_shape[1]};
          weight_impl_->PermuteWeight(v_head_tensor, {0, 2, 1}, dev_rank);
          std::string w_uv_name = v_head_name.substr(0, v_head_name.find_first_of('_')) + "_attn.w_uv.weight";
          device_model_weights[w_uv_name] = v_head_tensor;
        } else {
          // For fp8 blockwise quant, do not split the weights initially, split them after dequantization later.
          std::vector<size_t> kv_b_proj_shape = {
              static_cast<size_t>(DivRoundUp(host_weight_tensor.shape[0], attn_tp_size)), host_weight_tensor.shape[1]};

          Tensor kv_b_proj_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, host_weight_tensor.dtype, kv_b_proj_shape,
                                           nullptr, &(context_->device->GetMemoryManageStream()));
          size_t para_pitch = DivRoundUp(head_num, attn_tp_size) * (qk_nope_head_dim + v_head_dim) *
                              host_weight_tensor.shape[1] * GetTypeSize(host_weight_tensor.dtype);
          size_t tensor_para_offset = attn_dev_rank * para_pitch;
          MemcpyAsync(kv_b_proj_tensor.GetPtr<void>(), host_weight_tensor.GetPtr<void>() + tensor_para_offset,
                      para_pitch, MEMCPY_HOST_TO_DEVICE, context_->device->GetMemoryManageStream());

          device_model_weights[file_weight_name] = kv_b_proj_tensor;
        }
        continue;
      }
      if (file_weight_name.find(".o_proj.weight") != std::string::npos) {
        // bf16/fp16: Transpose, then split along axis = 0
        // fp8: Transpose, then split along axis = 0, then transpose
        size_t split_size =
            runtime_config_.enable_o_proj_out_of_dp ? deepseek_v3_config->tensor_para_size : attn_tp_size;
        Tensor dev_tensor;
        bool is_this_weight_quant = deepseek_v3_config->is_quant && !deepseek_v3_config->is_compressed_tensors;
        weight_impl_->TransSplitOptTrans(host_weight_tensor, dev_tensor, dev_rank, deepseek_v3_config, split_size,
                                         is_this_weight_quant);

        device_model_weights[file_weight_name] = dev_tensor;
        continue;
      }
    }

    // 2. dense MLP layer and shared expert layer
    // The parameters of both the mlp (dense) layer and the shared expert layer need to be transposed.
    if (CheckWeightNameMatched(
            file_weight_name,
            {".mlp.gate_proj.", ".mlp.up_proj.", ".mlp.shared_expert.gate_proj", ".mlp.shared_expert.up_proj"},
            false)) {
      // "up && gate proj(bf16/fp16): First split along axis = 0, then transpose."
      // "up && gate proj(fp8 weight/fp32 weight_scale_inv): split along axis = 0."
      Tensor dev_tensor;
      size_t mlp_tensor_para_size =
          runtime_config_.enable_full_shared_expert ? 1 : context_->global->GetTensorParallelSize();
      // For compressed-tensors (moe-int4), dense/shared_expert weights are NOT quantized (bf16/fp16).
      // Use actual weight dtype to determine if transpose is needed, not global is_quant.
      bool is_this_weight_quant = deepseek_v3_config->is_quant && !deepseek_v3_config->is_compressed_tensors;
      weight_impl_->SplitOptTrans(host_weight_tensor, dev_tensor, dev_rank, deepseek_v3_config, mlp_tensor_para_size,
                                  !is_this_weight_quant);
      if (deepseek_v3_config->use_mla && deepseek_v3_config->type != "deepseek_v2") {
        // deepseek v3 need to combine gate & up proj
        weight_impl_->ProcessGateUpProjWeight(file_weight_name, dev_tensor, device_model_weights, dev_rank,
                                              is_this_weight_quant);
      } else {
        device_model_weights[file_weight_name] = dev_tensor;
      }
      continue;
    }

    if (CheckWeightNameMatched(file_weight_name,
                               {
                                   ".mlp.down_proj.",
                                   ".mlp.shared_expert.down_proj",
                               },
                               false)) {
      // down proj(bf16/fp16): transpose first, then split along axis = 0
      // down proj(fp8/fp32): transpose first, then split along axis = 0, and then transpose back.
      Tensor dev_tensor;
      size_t mlp_tensor_para_size =
          runtime_config_.enable_full_shared_expert ? 1 : context_->global->GetTensorParallelSize();
      bool is_this_weight_quant = deepseek_v3_config->is_quant && !deepseek_v3_config->is_compressed_tensors;
      weight_impl_->TransSplitOptTrans(host_weight_tensor, dev_tensor, dev_rank, deepseek_v3_config,
                                       mlp_tensor_para_size, is_this_weight_quant);
      device_model_weights[file_weight_name] = dev_tensor;
      continue;
    }

    // 5. norm layer/`gate.e_score_correction_bias`/indexer fp8 weight
    // Directly load to each device.
    if (file_weight_name.find("norm.") != std::string::npos ||
        file_weight_name.find("gate.e_score_correction_bias") != std::string::npos ||
        file_weight_name.find("self_attn.indexer.wk") != std::string::npos ||
        file_weight_name.find("self_attn.indexer.wq_b") != std::string::npos) {
      Tensor dev_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, host_weight_tensor.dtype, host_weight_tensor.shape,
                                 nullptr, &(context_->device->GetMemoryManageStream()));
      MemcpyAsync(dev_tensor.GetPtr<void>(), host_weight_tensor.GetPtr<void>(), host_weight_tensor.GetTotalBytes(),
                  MEMCPY_HOST_TO_DEVICE, context_->device->GetMemoryManageStream());

      device_model_weights[file_weight_name] = dev_tensor;
      continue;
    }

    // 6. gate weight/indexer bf16 weight
    // Copy to each device and transpose
    if (file_weight_name.find(".mlp.gate.weight") != std::string::npos ||
        file_weight_name.find("self_attn.indexer.weights_proj") != std::string::npos) {
      Tensor dev_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, host_weight_tensor.dtype, host_weight_tensor.shape,
                                 nullptr, &(context_->device->GetMemoryManageStream()));

      MemcpyAsync(dev_tensor.GetPtr<void>(), host_weight_tensor.GetPtr<void>(), host_weight_tensor.GetTotalBytes(),
                  MEMCPY_HOST_TO_DEVICE, context_->device->GetMemoryManageStream());
      weight_impl_->PermuteWeight(dev_tensor, {1, 0}, dev_rank);
      device_model_weights[file_weight_name] = dev_tensor;
      continue;
    }

    // 7. eh_proj.weight (for mtp layer)
    // Split along axis = 0
    if (file_weight_name.find(".eh_proj.weight") != std::string::npos) {
      Tensor dev_tensor;
      weight_impl_->SplitOptTrans(host_weight_tensor, dev_tensor, dev_rank, deepseek_v3_config,
                                  context_->global->GetTensorParallelSize(), true);
      device_model_weights[file_weight_name] = dev_tensor;
      continue;
    }

    // 3. model.embed_tokens.weight;
    // ::FilterWeightNames() filtered this for non master nodes
    // Embedding TP needs to be transposed first, then split, and then transposed back.
    if (CheckWeightNameMatched(file_weight_name, {"model.embed_tokens.weight"}, true)) {
      Tensor dev_tensor;
      weight_impl_->TransSplitOptTrans(host_weight_tensor, dev_tensor, dev_rank, deepseek_v3_config,
                                       context_->global->GetTensorParallelSize(), true);

      device_model_weights[file_weight_name] = dev_tensor;
      continue;
    }

    // lm_head.weight
    // lm_head need to split along axis = 0, and then transpose
    if (CheckWeightNameMatched(file_weight_name, {"lm_head.weight"}, true)) {
      Tensor permute_dev_tensor;
      weight_impl_->SplitOptTrans(host_weight_tensor, permute_dev_tensor, dev_rank, deepseek_v3_config,
                                  context_->global->GetTensorParallelSize(), true);
      device_model_weights[file_weight_name] = permute_dev_tensor;
      continue;
    }
  }

  // Load gptq weights here
  if (deepseek_v3_config->ContainGptqWeights()) {
    weight_impl_->LoadInt4QuantWeight(host_gptq_weights, dev_rank, device_model_weights, deepseek_v3_config,
                                      expert_map);
  }

  return Status();
}

Status DeepSeekV3WeightLoader::InitWeightLoaderImpl(std::shared_ptr<DeepSeekV3Config>& deepseek_v3_config) {
  switch (deepseek_v3_config->weight_data_type) {
    case DataType::TYPE_FP32:
      weight_impl_ = std::make_unique<DeepSeekV3WeightImpl<float>>(context_, runtime_config_);
      break;
    case DataType::TYPE_BF16:
      weight_impl_ = std::make_unique<DeepSeekV3WeightImpl<bfloat16>>(context_, runtime_config_);
      break;
    case DataType::TYPE_FP16:
      weight_impl_ = std::make_unique<DeepSeekV3WeightImpl<float16>>(context_, runtime_config_);
      break;
    default:
      // Handle unexpected data type if needed
      KLLM_THROW(fmt::format("Unexpected data type: {}", deepseek_v3_config->weight_data_type));
      break;
  }
  return Status();
}

Status DeepSeekV3WeightLoader::InitExpertMap(const size_t& num_experts, const size_t& expert_start_id,
                                             const size_t& expert_end_id, int dev_rank,
                                             std::vector<std::vector<int>>& expert_map,
                                             std::unordered_map<std::string, Tensor>& device_model_weights) {
  // Load EPLB mapping configuration from JSON file
  std::unordered_map<int, std::vector<int>> eplb_map;
  if (runtime_config_.enable_load_eplb_config) {
    nlohmann::json json_config = ReadJsonFromFile(runtime_config_.eplb_config_path);
    if (json_config.empty()) {
      KLLM_LOG_ERROR << fmt::format("Failed to load EPLB config file: {}. File not found or empty.",
                                    runtime_config_.eplb_config_path);
    } else {
      for (auto& [key, value] : json_config.items()) {
        if (key.substr(0, 6) == "layer_" && value.is_array()) {
          int layer_idx = std::stoi(key.substr(6));  // 提取layer_后面的数字
          std::vector<int> eplb_data = value.get<std::vector<int>>();
          if (eplb_data.size() != num_experts) {
            KLLM_LOG_ERROR << fmt::format(
                "EPLB config file {} load failed in {}: wrong expert nums {} vs {}(actual model expert nums)",
                runtime_config_.eplb_config_path, key, eplb_data.size(), num_experts);
          } else {
            eplb_map[layer_idx] = eplb_data;
          }
        }
      }
      KLLM_LOG_INFO << fmt::format("Successed to load EPLB map from {}.", runtime_config_.eplb_config_path);
    }
  }

  int layer_num = static_cast<int>(expert_map.size());
  if (!eplb_map.empty()) {
    // 摊平 expert_map 到一维，以拷贝到 Device 空间
    std::vector<int> full_expert_map(static_cast<size_t>(layer_num) * num_experts);
    for (int layer_idx = 0; layer_idx < layer_num; layer_idx++) {
      for (size_t i = 0; i < num_experts; ++i) {
        if (eplb_map.find(layer_idx) != eplb_map.end()) {
          if (i >= expert_start_id && i < expert_end_id) {
            expert_map[layer_idx][eplb_map[layer_idx][i]] = static_cast<int>(i - expert_start_id);
          }
          full_expert_map[layer_idx * num_experts + eplb_map[layer_idx][i]] = static_cast<int>(i);
        } else {
          if (i >= expert_start_id && i < expert_end_id) {
            expert_map[layer_idx][i] = static_cast<int>(i - expert_start_id);
          }
          full_expert_map[layer_idx * num_experts + i] = static_cast<int>(i);
        }
      }
    }
    std::string expert_map_name = "expert_map";
    Tensor dev_tensor =
        Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {static_cast<size_t>(layer_num), num_experts}, nullptr,
               &(context_->device->GetMemoryManageStream()));
    MemcpyAsync(dev_tensor.GetPtr<void>(), full_expert_map.data(), full_expert_map.size() * sizeof(int),
                MEMCPY_HOST_TO_DEVICE, context_->device->GetMemoryManageStream());
    device_model_weights[expert_map_name] = dev_tensor;
  } else {
    // Without eplb map
    for (int layer_idx = 0; layer_idx < layer_num; layer_idx++) {
      for (size_t i = expert_start_id; i < expert_end_id; ++i) {
        expert_map[layer_idx][i] = static_cast<int>(i - expert_start_id);
      }
    }
    Tensor dev_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {0}, nullptr,
                               &(context_->device->GetMemoryManageStream()), "expert_map");
    device_model_weights["expert_map"] = dev_tensor;
  }
  return Status();
}

}  // namespace ksana_llm
