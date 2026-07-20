/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "models/deepseek_v4/deepseek_v4_weight_loader.h"

#include "configure/json_config_utils.h"
#include "kernels/trans_layout.h"
#include "model_loader/model_loader_utils.h"
#include "utils/indexer_skip_config.h"
#include "utils/logger.h"
#include "utils/memory_utils.h"
#include "utils/string_utils.h"

#ifdef ENABLE_CUDA
#  include "csrc/kernels/nvidia/others/humming/launcher/kernel_meta.h"
#  include "csrc/kernels/nvidia/others/humming/preprocess/humming_preprocess.h"
#  include "csrc/kernels/nvidia/others/sglang/main/marlin_moe_wna16/marlin_moe_wna16_preprocess.h"
#  include "kernels/nvidia/kernel_wrapper.h"
#endif

namespace ksana_llm {
DeepSeekV4WeightLoader::DeepSeekV4WeightLoader(std::shared_ptr<BaseModelConfig> model_config,
                                               std::shared_ptr<Environment> env, std::shared_ptr<Context> context)
    : DeepSeekV3WeightLoader(model_config, env, context) {
  // Parent constructor already initializes pipeline_config_, runtime_config_, weights_to_permute_, and weight_impl_.
  deepseek_v4_config_ = std::dynamic_pointer_cast<DeepSeekV4Config>(model_config_);
}

DeepSeekV4WeightLoader::~DeepSeekV4WeightLoader() {}

// Returns true if the raw weight name belongs to an MTP layer.
// DSv4 raw weight maps emit MTP weights with prefix "mtp." (no "model." prefix);
// also accept "model.mtp." in case the weight name has already been remapped (defensive).
static bool IsRawMtpWeightName(const std::string& weight_name) {
  return weight_name.rfind("mtp.", 0) == 0 || weight_name.find("model.mtp.") != std::string::npos;
}

Status DeepSeekV4WeightLoader::FilterWeightNames(std::vector<std::string>& weight_names) {
  const std::vector<std::string> skip_list = {"self_attn.rotary_emb.inv_freq", "vision_tower.", "mm_projector."};

  const int lower_layer_idx = pipeline_config_.lower_layer_idx;
  const int upper_layer_idx = pipeline_config_.upper_layer_idx;
  const int lower_nextn_layer_idx = pipeline_config_.lower_nextn_layer_idx;
  const int upper_nextn_layer_idx = pipeline_config_.upper_nextn_layer_idx;
  KLLM_LOG_DEBUG << fmt::format(
      "V4: pipeline config: lower_layer_idx={}, upper_layer_idx={}, lower_nextn_layer_idx={}, upper_nextn_layer_idx={}",
      lower_layer_idx, upper_layer_idx, lower_nextn_layer_idx, upper_nextn_layer_idx);

  const std::unordered_set<int> indexer_skip_layers =
      GetIndexerSkipLayers(deepseek_v4_config_->num_layer, deepseek_v4_config_->num_nextn_predict_layers,
                           deepseek_v4_config_->dsa_config, deepseek_v4_config_->dsv4_attn_config);

  // NOTE: Weight names are not mapped yet
  for (auto it = weight_names.begin(); it != weight_names.end();) {
    if (CheckWeightNameMatched(*it, skip_list, false)) {
      it = weight_names.erase(it);
      continue;
    }

    const int raw_layer_idx = GetLayerIdxFromName(*it);

    // Skip some layers in distributed mode.
    //
    // 关键: 必须把"原始主干权重 layers.N"和"MTP 权重 mtp.K"分开来分别判定。
    // 否则当 json_model_override_args 把 num_layer 改小到 N == lower_nextn_layer_idx 时,
    // 原始 layers.N (主干) 会因为 layer_idx 落入 nextn range 而被错误保留, 最终与 mtp.K
    // (rename 后 effective_idx == N) 冲突。
    if (lower_layer_idx >= 0 && upper_layer_idx >= 0) {
      if (raw_layer_idx >= 0) {
        const bool is_mtp_weight = IsRawMtpWeightName(*it);

        bool should_keep = false;
        if (is_mtp_weight) {
          // MTP 权重在 ProcessModelWeights 阶段会被 rename 到 layers.<num_layer + mtp_idx>,
          // 因此只能落到 [lower_nextn, upper_nextn] 槽位。
          const int effective_idx = raw_layer_idx + static_cast<int>(deepseek_v4_config_->num_layer);
          should_keep = (lower_nextn_layer_idx >= 0 && effective_idx >= lower_nextn_layer_idx &&
                         effective_idx <= upper_nextn_layer_idx);
          // 仅 sampling node 真正承载 MTP 层的权重。
          if (should_keep && !context_->global->IsSamplingNode()) {
            should_keep = false;
          }
        } else {
          // 原始主干权重必须严格落入主干 range。即便 raw_layer_idx 因为 num_layer override
          // 同时落入 nextn range, 也必须 skip —— 它不是 MTP 权重, 绝不能被错误地装到 MTP 槽位。
          should_keep = (raw_layer_idx >= lower_layer_idx && raw_layer_idx <= upper_layer_idx);
        }

        if (!should_keep) {
          KLLM_LOG_DEBUG << fmt::format("V4: Skip layer (is_mtp={}, raw_layer_idx={}): {}", is_mtp_weight,
                                        raw_layer_idx, *it);
          it = weight_names.erase(it);
          continue;
        }
      }
    }

    // Skip the weights of unused c4 indexers
    if (raw_layer_idx >= 0 && indexer_skip_layers.count(raw_layer_idx) &&
        CheckWeightNameMatched(*it, {fmt::format("layers.{}.attn.indexer.", raw_layer_idx)}, MatchMode::PartialMatch)) {
      KLLM_LOG_DEBUG << fmt::format("V4: indexer skip, layer {}: {}", raw_layer_idx, *it);
      it = weight_names.erase(it);
      continue;
    }

    if (!context_->global->IsStandalone()) {
      // Skip embedding if not master mode.
      if (context_->global->IsDistributedWorker() &&
          !(context_->global->IsSamplingNode() && runtime_config_.mtp_step_num > 0)) {
        if (CheckWeightNameMatched(*it, {"embed.weight"}, true)) {
          it = weight_names.erase(it);
          continue;
        }
      }

      // Skip lm_head/hc_head/final_norm if not tail mode.
      if (!context_->global->IsDistributedLastWorker()) {
        if (CheckWeightNameMatched(*it, {"head.weight"}, true) || CheckWeightNameMatched(*it, {"hc_head_"}, false) ||
            CheckWeightNameMatched(*it, {"norm.weight"}, true)) {
          it = weight_names.erase(it);
          continue;
        }
      }
    }
    ++it;
  }
  return Status();
}

// Helper: simple device copy without any transformation
static void CopyToDevice(const Tensor& host_tensor, Tensor& dev_tensor, std::shared_ptr<Context>& context) {
  dev_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, host_tensor.dtype, host_tensor.shape, nullptr,
                      &(context->device->GetMemoryManageStream()));
  MemcpyAsync(dev_tensor.GetPtr<void>(), host_tensor.GetPtr<void>(), host_tensor.GetTotalBytes(), MEMCPY_HOST_TO_DEVICE,
              context->device->GetMemoryManageStream());
}

// Helper: fuse two weights along dim=0 by matching first_name/second_name in file_weight_name,
// replacing with fused_name. first_name goes to the first half, second_name goes to the second half.
void DeepSeekV4WeightLoader::CombineWeightByRow(const std::string& file_weight_name, const Tensor& host_weight_tensor,
                                                const std::string& first_name, const std::string& second_name,
                                                const std::string& final_name,
                                                std::unordered_map<std::string, Tensor>& device_model_weights,
                                                int dev_rank, size_t fused_dim0) {
  std::string fused_name = file_weight_name;
  auto first_pos = fused_name.find(first_name + ".");
  auto second_pos = fused_name.find(second_name + ".");
  if (first_pos != std::string::npos) {
    fused_name.replace(first_pos, (first_name + ".").size(), final_name + ".");
  } else if (second_pos != std::string::npos) {
    fused_name.replace(second_pos, (second_name + ".").size(), final_name + ".");
  }

  if (device_model_weights.find(fused_name) == device_model_weights.end()) {
    std::vector<size_t> fused_shape = host_weight_tensor.shape;
    fused_shape[0] = (fused_dim0 > 0) ? fused_dim0 : fused_shape[0] * 2;
    device_model_weights[fused_name] = Tensor(MemoryLocation::LOCATION_DEVICE, host_weight_tensor.dtype, fused_shape,
                                              nullptr, &(context_->device->GetMemoryManageStream()));
  }

  Tensor& fused_tensor = device_model_weights.at(fused_name);
  size_t single_size = host_weight_tensor.GetTotalBytes();
  // first_name goes to the beginning, second_name goes after the first part.
  // When fused_dim0 != shape[0]*2, the two parts may have different sizes,
  // so offset for second_name = fused_total - second_size (i.e., first part's size).
  size_t offset =
      (file_weight_name.find(first_name) != std::string::npos) ? 0 : (fused_tensor.GetTotalBytes() - single_size);
  MemcpyAsync(fused_tensor.GetPtr<void>() + offset, host_weight_tensor.GetPtr<void>(), single_size,
              MEMCPY_HOST_TO_DEVICE, context_->device->GetMemoryManageStream());

  if (fused_tensor.dtype == DataType::TYPE_FP16 || fused_tensor.dtype == DataType::TYPE_BF16) {
    weights_to_permute_[dev_rank].insert(fused_name);
  }
  KLLM_LOG_DEBUG << fmt::format("V4: Dev_rank: {}, [CombineWeightByRow] {} -> {}, shape: {}, dtype: {}", dev_rank,
                                file_weight_name, fused_name, Vector2Str(std::vector<size_t>(fused_tensor.shape)),
                                fused_tensor.dtype);
}

// Load FP4 MoE expert weights for DeepSeek V4.
// Original per-expert weight layout:
//   gate weight: [moe_intermediate_size, hidden_size / 2], dtype int8 (FP4 packed)
//   gate scales: [moe_intermediate_size, hidden_size / 32], dtype float8_e8m0fnu
//   up weight:   [moe_intermediate_size, hidden_size / 2], dtype int8
//   up scales:   [moe_intermediate_size, hidden_size / 32], dtype float8_e8m0fnu
//   down weight: [hidden_size, moe_intermediate_size / 2], dtype int8
//   down scales: [hidden_size, moe_intermediate_size / 32], dtype float8_e8m0fnu
//
// GPU layout (all experts merged per layer):
//   up_gate weight: [num_experts, moe_intermediate_size * 2, hidden_size / 2], gate first then up
//   up_gate scales: [num_experts, moe_intermediate_size * 2, hidden_size / 32]
//   down weight:    [num_experts, hidden_size, moe_intermediate_size / 2]
//   down scales:    [num_experts, hidden_size, moe_intermediate_size / 32]
bool DeepSeekV4WeightLoader::LoadFp4MoeWeight(const std::string& file_weight_name, const Tensor& host_weight_tensor,
                                              int dev_rank, int layer_idx, int expert_idx_,
                                              std::unordered_map<std::string, Tensor>& device_model_weights) {
  ExpertParallelConfig& expert_parallel_config = deepseek_v4_config_->expert_parallel_config;
  size_t global_expert_para_size = expert_parallel_config.expert_world_size * expert_parallel_config.expert_para_size;
  size_t num_experts_per_rank = DivRoundUp(deepseek_v4_config_->moe_config.num_experts, global_expert_para_size);
  size_t moe_inter_size = deepseek_v4_config_->moe_config.moe_inter_size;
  size_t moe_inter_size_per_rank = DivRoundUp(moe_inter_size, deepseek_v4_config_->moe_tensor_para_size);
  size_t hidden_units = deepseek_v4_config_->hidden_units;
  size_t moe_tp_rank =
      (deepseek_v4_config_->moe_tensor_para_size > 1) ? (dev_rank / expert_parallel_config.expert_para_size) : 0;

  bool is_weight = file_weight_name.find(".weight") != std::string::npos &&
                   file_weight_name.find("weight_scale_inv") == std::string::npos;
  bool is_scale = file_weight_name.find(".weight_scale_inv") != std::string::npos;

  if (!is_weight && !is_scale) {
    return false;
  }

  std::string suffix = is_weight ? "weight" : "weight_scale_inv";

  // Determine dimensions based on weight vs scale
  // weight: dim1 = hidden_size / 2 (for up_gate) or moe_intermediate_size / 2 (for down)
  // scale:  dim1 = hidden_size / 32 (for up_gate) or moe_intermediate_size / 32 (for down)

  if (file_weight_name.find(".up_proj.") != std::string::npos ||
      file_weight_name.find(".gate_proj.") != std::string::npos) {
    // up_gate: [num_experts, moe_inter_size_per_rank * 2, col_dim]
    // TP splits along moe_intermediate_size (rows), col_dim stays full.
    // For weight: col_dim = hidden_size / 2; for scale: col_dim = hidden_size / 32
    size_t col_dim = host_weight_tensor.shape[1];  // hidden_size/2 or hidden_size/32

    std::string up_gate_name = fmt::format("model.layers.{}.mlp.experts.up_gate_proj.{}", layer_idx, suffix);
    if (device_model_weights.find(up_gate_name) == device_model_weights.end()) {
      device_model_weights[up_gate_name] = Tensor(MemoryLocation::LOCATION_DEVICE, host_weight_tensor.dtype,
                                                  {num_experts_per_rank, moe_inter_size_per_rank * 2, col_dim}, nullptr,
                                                  &(context_->device->GetMemoryManageStream()));
    }

    Tensor& up_gate_tensor = device_model_weights.at(up_gate_name);
    // Each expert half: moe_inter_size_per_rank rows × col_dim columns
    size_t expert_pitch = moe_inter_size_per_rank * col_dim * GetTypeSize(host_weight_tensor.dtype);
    size_t double_expert_pitch = expert_pitch * 2;
    // TP offset: skip to this rank's row slice in the source
    size_t src_offset = (deepseek_v4_config_->moe_tensor_para_size > 1)
                            ? moe_tp_rank * moe_inter_size_per_rank * col_dim * GetTypeSize(host_weight_tensor.dtype)
                            : 0;

    // gate goes first, up goes second
    if (file_weight_name.find(".gate_proj.") != std::string::npos) {
      MemcpyAsync(up_gate_tensor.GetPtr<void>() + static_cast<size_t>(expert_idx_) * double_expert_pitch,
                  host_weight_tensor.GetPtr<void>() + src_offset, expert_pitch, MEMCPY_HOST_TO_DEVICE,
                  context_->device->GetMemoryManageStream());
    } else {  // up_proj
      MemcpyAsync(up_gate_tensor.GetPtr<void>() + static_cast<size_t>(expert_idx_) * double_expert_pitch + expert_pitch,
                  host_weight_tensor.GetPtr<void>() + src_offset, expert_pitch, MEMCPY_HOST_TO_DEVICE,
                  context_->device->GetMemoryManageStream());
    }
    // Mark the combined tensor as FP4 packed type (only for weight, not scale)
    if (up_gate_tensor.dtype != DataType::TYPE_FP4_E2M1FN_x2 && is_weight) {
      up_gate_tensor.dtype = DataType::TYPE_FP4_E2M1FN_x2;
    }
    KLLM_LOG_DEBUG << fmt::format("V4: Dev_rank: {}, [FP4 MoE up_gate] {} -> {}, expert_idx_={}, shape: {}, dtype: {}",
                                  dev_rank, file_weight_name, up_gate_name, expert_idx_,
                                  Vector2Str(std::vector<size_t>(up_gate_tensor.shape)), up_gate_tensor.dtype);
    return true;
  }

  if (file_weight_name.find(".down_proj.") != std::string::npos) {
    // down: [num_experts, hidden_size, col_dim]
    // For weight: col_dim = moe_intermediate_size / 2; for scale: col_dim = moe_intermediate_size / 32
    size_t col_dim = host_weight_tensor.shape[1];  // moe_inter_size/2 or moe_inter_size/32
    size_t col_dim_per_rank = DivRoundUp(col_dim, deepseek_v4_config_->moe_tensor_para_size);

    std::string down_name = fmt::format("model.layers.{}.mlp.experts.down_proj.{}", layer_idx, suffix);
    if (device_model_weights.find(down_name) == device_model_weights.end()) {
      device_model_weights[down_name] = Tensor(MemoryLocation::LOCATION_DEVICE, host_weight_tensor.dtype,
                                               {num_experts_per_rank, hidden_units, col_dim_per_rank}, nullptr,
                                               &(context_->device->GetMemoryManageStream()));
    }

    Tensor& down_tensor = device_model_weights.at(down_name);
    size_t dst_row_bytes = col_dim_per_rank * GetTypeSize(host_weight_tensor.dtype);
    size_t src_row_bytes = col_dim * GetTypeSize(host_weight_tensor.dtype);
    size_t expert_pitch = hidden_units * col_dim_per_rank * GetTypeSize(host_weight_tensor.dtype);
    size_t src_col_offset = (deepseek_v4_config_->moe_tensor_para_size > 1)
                                ? moe_tp_rank * col_dim_per_rank * GetTypeSize(host_weight_tensor.dtype)
                                : 0;

    // Use 2D memcpy: each row of the expert copies col_dim_per_rank elements
    Memcpy2DAsync(down_tensor.GetPtr<void>() + static_cast<size_t>(expert_idx_) * expert_pitch, dst_row_bytes,
                  host_weight_tensor.GetPtr<void>() + src_col_offset, src_row_bytes, dst_row_bytes, hidden_units,
                  MEMCPY_HOST_TO_DEVICE, context_->device->GetMemoryManageStream());
    // Mark the combined tensor as FP4 packed type (only for weight, not scale)
    if (down_tensor.dtype != DataType::TYPE_FP4_E2M1FN_x2 && is_weight) {
      down_tensor.dtype = DataType::TYPE_FP4_E2M1FN_x2;
    }
    KLLM_LOG_DEBUG << fmt::format("V4: Dev_rank: {}, [FP4 MoE down] {} -> {}, expert_idx_={}, shape: {}, dtype: {}",
                                  dev_rank, file_weight_name, down_name, expert_idx_,
                                  Vector2Str(std::vector<size_t>(down_tensor.shape)), down_tensor.dtype);
    return true;
  }

  return false;
}

Status DeepSeekV4WeightLoader::PreProcessModelWeights(
    const std::unordered_map<std::string, Tensor>& host_model_weights) {
  // Convert all weight_scale_inv tensors from TYPE_FP8_E8M0 to TYPE_FP32 in-place.
  // Safe to const_cast here because the actual object passed in is non-const
  // (see model_weight_loader.cpp: *host_model_weights is a shared_ptr-owned map).
  auto& mutable_weights = const_cast<std::unordered_map<std::string, Tensor>&>(host_model_weights);
  for (auto& [weight_name, weight_tensor] : mutable_weights) {
    if ((weight_name.find("scale") != std::string::npos && weight_name.find(".experts.") == std::string::npos) &&
        weight_tensor.dtype == DataType::TYPE_FP8_E8M0) {
      size_t n = weight_tensor.GetTotalBytes();  // 1 byte per element for E8M0
      Tensor fp32_tensor(MemoryLocation::LOCATION_HOST, DataType::TYPE_FP32, weight_tensor.shape);
      const uint8_t* src = static_cast<const uint8_t*>(weight_tensor.GetPtr<void>());
      float* dst = static_cast<float*>(fp32_tensor.GetPtr<void>());
      for (size_t i = 0; i < n; ++i) {
        uint32_t bits = static_cast<uint32_t>(src[i]) << 23;
        memcpy(&dst[i], &bits, sizeof(float));
      }
      weight_tensor = fp32_tensor;
      KLLM_LOG_DEBUG << fmt::format("V4: [PreProcess] converted E8M0->FP32: {}", weight_name);
    }
  }
  return Status();
}

Status DeepSeekV4WeightLoader::ProcessModelWeights(const std::unordered_map<std::string, Tensor>& host_model_weights,
                                                   int dev_rank,
                                                   std::unordered_map<std::string, Tensor>& device_model_weights,
                                                   std::unordered_map<std::string, Tensor>& left_host_weights) {
  if (!deepseek_v4_config_->quant_config.is_fp8_blockwise) {
    KLLM_THROW("DeepSeekV4 only supports fp8_blockwise quantization");
  }
  size_t attn_tp_size = context_->global->GetAttentionTensorParallelSize();
  // For passing to weight_impl_ functions that take std::shared_ptr<DeepSeekV3Config>&
  // (non-const ref cannot bind to implicit conversion temporary).
  std::shared_ptr<DeepSeekV3Config> base_config = deepseek_v4_config_;

  int32_t layer_idx = -1, expert_idx = -1;
  size_t num_experts = deepseek_v4_config_->moe_config.num_experts;
  bool use_vllm_moe = deepseek_v4_config_->moe_config.use_vllm_moe;
  KLLM_LOG_DEBUG << fmt::format("num_experts = {}, use_vllm_moe = {}", num_experts, use_vllm_moe);
  KLLM_LOG_DEBUG << "is_fp8_blockwise: " << deepseek_v4_config_->quant_config.is_fp8_blockwise;
  // Expert parallel config
  ExpertParallelConfig& expert_parallel_config = deepseek_v4_config_->expert_parallel_config;
  size_t expert_node_rank = expert_parallel_config.expert_node_rank;
  size_t global_expert_para_size = expert_parallel_config.expert_world_size * expert_parallel_config.expert_para_size;
  size_t num_experts_per_rank = DivRoundUp(num_experts, global_expert_para_size);

  // Init expert map
  size_t rank_expert_offset = expert_node_rank * expert_parallel_config.expert_para_size * num_experts_per_rank;
  size_t expert_offset =
      (global_expert_para_size > 1) ? ((dev_rank % deepseek_v4_config_->expert_para_size) * num_experts_per_rank) : 0;
  size_t expert_start_id = rank_expert_offset + expert_offset;
  size_t expert_end_id = std::min(num_experts, expert_start_id + num_experts_per_rank);
  size_t num_layers = deepseek_v4_config_->num_layer;
  if (pipeline_config_.lower_nextn_layer_idx >= num_layers) {
    num_layers += pipeline_config_.upper_nextn_layer_idx - pipeline_config_.lower_nextn_layer_idx + 1;
  }
  std::vector<std::vector<int>> expert_map(num_layers, std::vector<int>(num_experts, -1));
  InitExpertMap(num_experts, expert_start_id, expert_end_id, dev_rank, expert_map, device_model_weights);

  KLLM_LOG_DEBUG << fmt::format("expert_world_size = {}, expert_para_size = {}, global_expert_para_size = {}",
                                expert_parallel_config.expert_world_size, expert_parallel_config.expert_para_size,
                                global_expert_para_size);
  KLLM_LOG_DEBUG << fmt::format("In rank {} node_rank {}, valid experts is [{}, {})", dev_rank, expert_node_rank,
                                expert_start_id, expert_end_id);
  KLLM_LOG_DEBUG << fmt::format("V4: num_layers={}, upper_nextn_layer_idx={}, lower_nextn_layer_idx={}", num_layers,
                                pipeline_config_.upper_nextn_layer_idx, pipeline_config_.lower_nextn_layer_idx);

  // MoE shape calculations
  size_t moe_inter_size_per_rank =
      DivRoundUp(deepseek_v4_config_->moe_config.moe_inter_size, deepseek_v4_config_->moe_tensor_para_size);
  size_t hidden_units = deepseek_v4_config_->hidden_units;
  std::vector<size_t> up_gate_experts_shape = {num_experts_per_rank,
                                               /* up & gate*/ moe_inter_size_per_rank * 2, hidden_units};
  std::vector<size_t> down_experts_shape = {num_experts_per_rank, hidden_units, moe_inter_size_per_rank};

  size_t q_lora_rank = static_cast<size_t>(deepseek_v4_config_->dsv4_attn_config.q_lora_rank);
  size_t o_groups = static_cast<size_t>(deepseek_v4_config_->dsv4_attn_config.o_groups);
  size_t o_local_groups = DivRoundUp(o_groups, attn_tp_size);
  size_t o_lora_rank = static_cast<size_t>(deepseek_v4_config_->dsv4_attn_config.o_lora_rank);
  size_t head_dim = static_cast<size_t>(deepseek_v4_config_->size_per_head);
  size_t block_n = deepseek_v4_config_->quant_config.weight_block_size[0];
  size_t block_k = deepseek_v4_config_->quant_config.weight_block_size[1];
  KLLM_LOG_DEBUG << fmt::format("V4: block_n={}, block_k={}", block_n, block_k);
  // Attention config (V4 uses CSA, not MLA — no kv_lora_rank or v_head_dim)
  // size_t attn_tp_size = context_->global->GetAttentionTensorParallelSize();
  // size_t attn_dev_rank = dev_rank % attn_tp_size;
  // size_t qk_rope_head_dim = deepseek_v4_config_->csa_config.qk_rope_head_dim;
  // size_t qk_nope_head_dim = deepseek_v4_config_->size_per_head - qk_rope_head_dim;
  // size_t head_num = deepseek_v4_config_->head_num;
  // size_t head_num_tp = static_cast<size_t>(DivRoundUp(deepseek_v4_config_->head_num, attn_tp_size));
  size_t hc_mult = static_cast<size_t>(deepseek_v4_config_->mhc_config.hc_mult);

  // TODO(winminkong): support GPTQ weight if need
  for (auto& [origin_file_weight_name, host_weight_tensor] : host_model_weights) {
    std::string file_weight_name = origin_file_weight_name;

    // Rewrite MTP weight names: model.mtp.<idx>.xxx → model.layers.<num_layer+idx>.xxx
    // so that MTP weights get unique layer indices and avoid collision with
    // regular layer-0 weights in GetExpertsIdx.
    if (auto mtp_pos = file_weight_name.find("model.mtp."); mtp_pos != std::string::npos) {
      // Skip mtp layers when dsiabled
      if (num_layers == deepseek_v4_config_->num_layer) {
        continue;
      }
      if (auto dot_pos = file_weight_name.find('.', mtp_pos + strlen("model.mtp.")); dot_pos != std::string::npos) {
        int mtp_idx = std::stoi(
            file_weight_name.substr(mtp_pos + strlen("model.mtp."), dot_pos - mtp_pos - strlen("model.mtp.")));
        int mapped_layer = static_cast<int>(deepseek_v4_config_->num_layer) + mtp_idx;
        file_weight_name = file_weight_name.substr(0, mtp_pos) + "model.layers." + std::to_string(mapped_layer) + "." +
                           file_weight_name.substr(dot_pos + 1);
        KLLM_LOG_DEBUG << fmt::format("V4: Dev_rank: {}, MTP rewrite: {} -> {}", dev_rank, origin_file_weight_name,
                                      file_weight_name);
      }
    }

    // Skip weights whose layer_idx >= num_layers
    int extracted_layer_idx = GetLayerIdxFromName(file_weight_name);
    if (extracted_layer_idx >= 0 && static_cast<size_t>(extracted_layer_idx) >= num_layers) {
      KLLM_LOG_DEBUG << fmt::format("V4: Dev_rank: {}, skipping weight (layer_idx={} >= num_layers={}): {}", dev_rank,
                                    extracted_layer_idx, num_layers, file_weight_name);
      continue;
    }

    KLLM_LOG_DEBUG << fmt::format("V4: Dev_rank: {}, processing weight: {}, shape: {}, dtype: {}", dev_rank,
                                  file_weight_name, Vector2Str(std::vector<size_t>(host_weight_tensor.shape)),
                                  host_weight_tensor.dtype);

    // =====================================================================
    // 1. MoE expert weights (gate_proj, up_proj, down_proj + their FP8 scales)
    // =====================================================================
    if (file_weight_name.find(".experts.") != std::string::npos) {
      STATUS_CHECK_RETURN(weight_impl_->GetExpertsIdx(file_weight_name, layer_idx, expert_idx));
      if (layer_idx < 0 || expert_idx < 0) {
        continue;
      }
      int expert_idx_ = expert_map[layer_idx][expert_idx];
      if (expert_idx_ < 0) {
        continue;
      }
      if (deepseek_v4_config_->quant_config.expert_dtype == "fp4") {
        if (LoadFp4MoeWeight(file_weight_name, host_weight_tensor, dev_rank, layer_idx, expert_idx_,
                             device_model_weights)) {
          continue;
        }
      }
#ifdef ENABLE_FP8
      if (deepseek_v4_config_->quant_config.is_fp8_blockwise &&
          weight_impl_->LoadMoeFp8E4m3BlockWiseScale(file_weight_name, host_weight_tensor, dev_rank, base_config,
                                                     device_model_weights, expert_idx_)) {
        continue;
      }
#endif
      if (deepseek_v4_config_->quant_config.method == QUANT_FP8_E4M3 &&
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
        size_t src_upgate_offset = deepseek_v4_config_->moe_tensor_para_size > 1
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
        KLLM_LOG_DEBUG << fmt::format(
            "V4: Dev_rank: {}, [MoE up_gate] stored: {}, shape: {}, dtype: {}", dev_rank, up_gate_experts_name,
            Vector2Str(std::vector<size_t>(up_gate_experts_tensor.shape)), up_gate_experts_tensor.dtype);
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
            moe_inter_size_per_rank * deepseek_v4_config_->moe_tensor_para_size * GetTypeSize(host_weight_tensor.dtype);
        size_t expert_pitch = moe_inter_size_per_rank * hidden_units * GetTypeSize(host_weight_tensor.dtype);
        size_t src_down_offset = deepseek_v4_config_->moe_tensor_para_size > 1
                                     ? (dev_rank / expert_parallel_config.expert_para_size) * dst_pitch
                                     : 0;
        Tensor& down_expert_tensor = device_model_weights.at(down_experts_name);
        Memcpy2DAsync(down_expert_tensor.GetPtr<void>() + static_cast<size_t>(expert_idx_) * expert_pitch, dst_pitch,
                      host_weight_tensor.GetPtr<void>() + src_down_offset, src_pitch, dst_pitch, hidden_units,
                      MEMCPY_HOST_TO_DEVICE, context_->device->GetMemoryManageStream());
        KLLM_LOG_DEBUG << fmt::format("V4: Dev_rank: {}, [MoE down] stored: {}, shape: {}, dtype: {}", dev_rank,
                                      down_experts_name, Vector2Str(std::vector<size_t>(down_expert_tensor.shape)),
                                      down_expert_tensor.dtype);
        continue;
      }
    }

    // =====================================================================
    // 2. Shared expert (gate_proj, up_proj, down_proj)
    //    V4 ALL layers are MoE, so no dense MLP layer handling needed.
    // =====================================================================
    if (CheckWeightNameMatched(file_weight_name, {".mlp.shared_expert.gate_proj", ".mlp.shared_expert.up_proj"},
                               false)) {
      Tensor dev_tensor;
      size_t mlp_tensor_para_size =
          runtime_config_.enable_full_shared_expert ? 1 : context_->global->GetTensorParallelSize();
      bool is_this_weight_quant = deepseek_v4_config_->is_quant && !deepseek_v4_config_->is_compressed_tensors;
      weight_impl_->SplitOptTrans(host_weight_tensor, dev_tensor, dev_rank, mlp_tensor_para_size,
                                  !is_this_weight_quant);
      KLLM_LOG_DEBUG << fmt::format(
          "V4: Dev_rank: {}, [shared expert gate_up fused] processed: {}, shape: {}, dtype: {}", dev_rank,
          file_weight_name, Vector2Str(std::vector<size_t>(dev_tensor.shape)), dev_tensor.dtype);
      // Combine gate & up proj
      weight_impl_->ProcessGateUpProjWeight(file_weight_name, dev_tensor, device_model_weights, dev_rank,
                                            is_this_weight_quant);
      continue;
    }

    if (CheckWeightNameMatched(file_weight_name, {".mlp.shared_expert.down_proj"}, false)) {
      Tensor dev_tensor;
      size_t mlp_tensor_para_size =
          runtime_config_.enable_full_shared_expert ? 1 : context_->global->GetTensorParallelSize();
      bool is_this_weight_quant = deepseek_v4_config_->is_quant && !deepseek_v4_config_->is_compressed_tensors;
      weight_impl_->TransSplitOptTrans(host_weight_tensor, dev_tensor, dev_rank, mlp_tensor_para_size,
                                       is_this_weight_quant);
      device_model_weights[file_weight_name] = dev_tensor;
      KLLM_LOG_DEBUG << fmt::format("V4: Dev_rank: {}, [shared expert down] stored: {}, shape: {}, dtype: {}", dev_rank,
                                    file_weight_name, Vector2Str(std::vector<size_t>(dev_tensor.shape)),
                                    dev_tensor.dtype);
      continue;
    }

    // =====================================================================
    // 3. CSA/HCA attention weights
    //    compressor, indexer, attn_sink, wq_a, wq_b, wo_a, wo_b, wkv
    // =====================================================================
    if (file_weight_name.find("attn.indexer") != std::string::npos ||
        file_weight_name.find("attn.compressor") != std::string::npos ||
        file_weight_name.find("attn_sink") != std::string::npos || file_weight_name.find("norm") != std::string::npos) {
      if (file_weight_name.find(".ape") != std::string::npos) {
        // Apply APE hotfix: equivalent to Python apply_ape_hotfix().
        // ape shape: [ratio, coff * head_dim], where coff = 1 + (ratio == 4).
        // When overlap (ratio == 4, coff == 2):
        //   1. chunk(ape, 2, dim=-1) → left [ratio, head_dim], right [ratio, head_dim]
        //   2. cat([left, right], dim=0) → [2*ratio, head_dim]
        //   3. view(ratio, -1) → [ratio, 2*head_dim]
        // For ratio != 4 (no overlap), just copy as-is.
        const size_t ratio = host_weight_tensor.shape[0];
        const size_t last_dim = host_weight_tensor.shape[1];
        const bool overlap = (ratio == 4);
        if (overlap) {
          const size_t half_dim = last_dim / 2;  // = head_dim
          const size_t elem_size = GetTypeSize(host_weight_tensor.dtype);

          // Create a host buffer for the transformed ape
          Tensor host_ape_transformed(MemoryLocation::LOCATION_HOST, host_weight_tensor.dtype,
                                      host_weight_tensor.shape);
          const char* src = static_cast<const char*>(host_weight_tensor.GetPtr<void>());
          char* dst = static_cast<char*>(host_ape_transformed.GetPtr<void>());

          // orders = [0, 1] (is_model_2604 = true)
          // concat([left, right], dim=0) → [2*ratio, half_dim], then view(ratio, last_dim)
          // Result row i = [concat_row[2*i], concat_row[2*i+1]]
          // concat_row[j] for j < ratio: left[j] = src[j * last_dim ... j * last_dim + half_dim - 1]
          // concat_row[j] for j >= ratio: right[j-ratio] = src[(j-ratio) * last_dim + half_dim ... ]
          for (size_t i = 0; i < ratio; ++i) {
            size_t concat_row_0 = 2 * i;  // index into [2*ratio, half_dim]
            size_t concat_row_1 = 2 * i + 1;

            // First half of dst row i ← concat_row_0
            const char* src_ptr_0;
            if (concat_row_0 < ratio) {
              // from left: src[concat_row_0 * last_dim], length half_dim
              src_ptr_0 = src + concat_row_0 * last_dim * elem_size;
            } else {
              // from right: src[(concat_row_0 - ratio) * last_dim + half_dim], length half_dim
              src_ptr_0 = src + (concat_row_0 - ratio) * last_dim * elem_size + half_dim * elem_size;
            }
            memcpy(dst + i * last_dim * elem_size, src_ptr_0, half_dim * elem_size);

            // Second half of dst row i ← concat_row_1
            const char* src_ptr_1;
            if (concat_row_1 < ratio) {
              src_ptr_1 = src + concat_row_1 * last_dim * elem_size;
            } else {
              src_ptr_1 = src + (concat_row_1 - ratio) * last_dim * elem_size + half_dim * elem_size;
            }
            memcpy(dst + i * last_dim * elem_size + half_dim * elem_size, src_ptr_1, half_dim * elem_size);
          }

          Tensor dev_tensor;
          CopyToDevice(host_ape_transformed, dev_tensor, context_);
          device_model_weights[file_weight_name] = dev_tensor;
        } else {
          // No overlap, just copy as-is
          Tensor dev_tensor;
          CopyToDevice(host_weight_tensor, dev_tensor, context_);
          device_model_weights[file_weight_name] = dev_tensor;
        }
        KLLM_LOG_DEBUG << fmt::format("V4: Dev_rank: {}, [APE hotfix] {}, shape: {}, overlap: {}", dev_rank,
                                      file_weight_name, Vector2Str(std::vector<size_t>(host_weight_tensor.shape)),
                                      overlap);
        continue;
      }
      // Fuse compressor.wkv and compressor.wgate along dim=0 into compressor.wkv_gate
      if (file_weight_name.find("compressor.wkv") != std::string::npos ||
          file_weight_name.find("compressor.wgate") != std::string::npos) {
        CombineWeightByRow(file_weight_name, host_weight_tensor, "compressor.wkv", "compressor.wgate",
                           "compressor.wkv_gate", device_model_weights, dev_rank);
      } else {
        Tensor dev_tensor;
        if (file_weight_name.find("attn_sink") != std::string::npos && attn_tp_size > 1) {
          // Rotate attn_sink so this rank's local heads are at the front.
          // The attention kernel reads the first n_local_heads entries.
          const size_t head_num = host_weight_tensor.shape[0];
          const size_t n_local_heads = head_num / attn_tp_size;
          const size_t attn_dev_rank = dev_rank % attn_tp_size;
          const size_t elem_size = GetTypeSize(host_weight_tensor.dtype);
          KLLM_CHECK_WITH_INFO(
              head_num % attn_tp_size == 0,
              fmt::format("attn_sink head_num {} not divisible by attn_tp_size {}", head_num, attn_tp_size));

          Tensor host_reordered(MemoryLocation::LOCATION_HOST, host_weight_tensor.dtype, host_weight_tensor.shape);
          const char* src = static_cast<const char*>(host_weight_tensor.GetPtr<void>());
          char* dst = static_cast<char*>(host_reordered.GetPtr<void>());

          const size_t local_bytes = n_local_heads * elem_size;
          const size_t rank_offset = attn_dev_rank * local_bytes;
          memcpy(dst, src + rank_offset, local_bytes);
          memcpy(dst + local_bytes, src, rank_offset);
          memcpy(dst + local_bytes + rank_offset, src + rank_offset + local_bytes,
                 head_num * elem_size - local_bytes - rank_offset);

          CopyToDevice(host_reordered, dev_tensor, context_);
          KLLM_LOG_DEBUG << fmt::format(
              "V4: Dev_rank: {}, [attn_sink reorder] head_num={}, n_local_heads={}, attn_dev_rank={}", dev_rank,
              head_num, n_local_heads, attn_dev_rank);
        } else {
          CopyToDevice(host_weight_tensor, dev_tensor, context_);
        }
        if (file_weight_name.find("attn.indexer.weights_proj") != std::string::npos) {
          weight_impl_->PermuteWeight(dev_tensor, {1, 0}, dev_rank);
        }
        device_model_weights[file_weight_name] = dev_tensor;
        KLLM_LOG_DEBUG << fmt::format("V4: Dev_rank: {}, [CSA/HCA *] stored: {}, shape: {}, dtype: {}", dev_rank,
                                      file_weight_name, Vector2Str(std::vector<size_t>(dev_tensor.shape)),
                                      dev_tensor.dtype);
      }
      continue;
    }

    if (file_weight_name.find("attn.wq_a") != std::string::npos ||
        file_weight_name.find("attn.wkv") != std::string::npos) {
      // Fuse attn.wq_a and attn.wkv along dim=0 into attn.wqkv_a
      size_t loaded_first_dim = q_lora_rank + head_dim;
      if (file_weight_name.find("scale") != std::string::npos) {
        loaded_first_dim /= block_n;
      }
      CombineWeightByRow(file_weight_name, host_weight_tensor, "attn.wq_a", "attn.wkv", "attn.wqkv_a",
                         device_model_weights, dev_rank, loaded_first_dim);

      // For DSpark MTP layers (layer_id >= num_layer), after fusing wqkv_a, also expose
      // attn.wkv as a separate entry pointing to the wkv portion inside the fused wqkv_a tensor.
      // This avoids extra device memory allocation — the wkv view shares the same
      // underlying buffer as wqkv_a, offset by the wq_a portion's byte size.
      if (deepseek_v4_config_->dspark_config.dspark_block_size > 0 &&
          file_weight_name.find("attn.wkv") != std::string::npos &&
          extracted_layer_idx >= static_cast<int>(deepseek_v4_config_->num_layer)) {
        // Construct the fused name to look up the already-allocated wqkv_a tensor.
        std::string fused_name = file_weight_name;
        auto wkv_pos = fused_name.find("attn.wkv.");
        if (wkv_pos != std::string::npos) {
          fused_name.replace(wkv_pos, strlen("attn.wkv."), "attn.wqkv_a.");
          auto fused_it = device_model_weights.find(fused_name);
          if (fused_it != device_model_weights.end()) {
            const Tensor& fused_tensor = fused_it->second;
            // wkv portion starts after the wq_a portion: offset in bytes = fused_total - wkv_size
            size_t wkv_bytes = host_weight_tensor.GetTotalBytes();
            size_t fused_total_bytes = fused_tensor.GetTotalBytes();
            size_t offset_bytes = fused_total_bytes - wkv_bytes;
            size_t elem_size = GetTypeSize(host_weight_tensor.dtype);
            size_t offset_elems = offset_bytes / elem_size;
            // Create a view with the original wkv shape, sharing the fused tensor's memory.
            Tensor wkv_view = fused_tensor.GetView(host_weight_tensor.shape, offset_elems);
            // Construct the wkv name from the original file_weight_name (it already contains attn.wkv).
            device_model_weights[file_weight_name] = wkv_view;
            KLLM_LOG_DEBUG << fmt::format(
                "V4: Dev_rank: {}, [DSpark wkv view] stored: {}, shape: {}, offset_bytes: {}, dtype: {}", dev_rank,
                file_weight_name, Vector2Str(std::vector<size_t>(wkv_view.shape)), offset_bytes, wkv_view.dtype);
          }
        }
      }
      continue;
    }

    if (file_weight_name.find("attn.wq_b") != std::string::npos ||
        file_weight_name.find("attn.wo_a") != std::string::npos) {
      Tensor dev_tensor;
      size_t tensor_para_size = context_->global->GetTensorParallelSize();
      bool is_this_weight_quant = deepseek_v4_config_->is_quant && !deepseek_v4_config_->is_compressed_tensors;
      weight_impl_->SplitOptTrans(host_weight_tensor, dev_tensor, dev_rank, tensor_para_size, !is_this_weight_quant);
      if (file_weight_name.find("attn.wo_a") != std::string::npos &&
          (dev_tensor.dtype == DataType::TYPE_FP16 || dev_tensor.dtype == DataType::TYPE_BF16)) {
        KLLM_CHECK_WITH_INFO(o_local_groups * o_lora_rank == dev_tensor.shape[0],
                             "o_local_groups * o_lora_rank != wo_a shape[0]");
        dev_tensor.shape = {o_local_groups, o_lora_rank, dev_tensor.shape[1]};
        weights_to_permute_[dev_rank].insert(file_weight_name);
      }
      device_model_weights[file_weight_name] = dev_tensor;
      KLLM_LOG_DEBUG << fmt::format("V4: Dev_rank: {}, [attn q_b or wo_a fused] processed: {}, shape: {}, dtype: {}",
                                    dev_rank, file_weight_name, Vector2Str(std::vector<size_t>(dev_tensor.shape)),
                                    dev_tensor.dtype);
      continue;
    }

    if (file_weight_name.find("attn.wo_b") != std::string::npos) {
      Tensor dev_tensor;
      bool is_this_weight_quant = deepseek_v4_config_->is_quant && !deepseek_v4_config_->is_compressed_tensors;
      weight_impl_->TransSplitOptTrans(host_weight_tensor, dev_tensor, dev_rank,
                                       context_->global->GetTensorParallelSize(), is_this_weight_quant);
      device_model_weights[file_weight_name] = dev_tensor;
      KLLM_LOG_DEBUG << fmt::format("V4: Dev_rank: {}, [attn wo_b] stored: {}, shape: {}, dtype: {}", dev_rank,
                                    file_weight_name, Vector2Str(std::vector<size_t>(dev_tensor.shape)),
                                    dev_tensor.dtype);
      continue;
    }

    // =====================================================================
    // 4. Gate layers, gate weight, gate bias, gate.tid2eid
    // =====================================================================
    if (file_weight_name.find("gate.weight") != std::string::npos ||
        file_weight_name.find("gate.e_score_correction_bias") != std::string::npos ||
        file_weight_name.find("gate.tid2eid") != std::string::npos) {
      Tensor dev_tensor;
      // tid2eid on disk is int64 shape [vocab, topk], but the hash topk kernel reads it
      // as int32. Expert IDs fit in int32, so narrow at load time. Without this, every
      // other output slot of topk_ids gets the high 32 bits of the int64 (= 0 for small
      // expert IDs), producing a [id, 0, id, 0, ...] pattern.
      const bool is_tid2eid = file_weight_name.find("gate.tid2eid") != std::string::npos;
      if (is_tid2eid && host_weight_tensor.dtype == TYPE_INT64) {
        const int64_t* src = static_cast<const int64_t*>(host_weight_tensor.GetPtr<void>());
        const size_t n = host_weight_tensor.GetTotalBytes() / sizeof(int64_t);
        Tensor host_i32(MemoryLocation::LOCATION_HOST, TYPE_INT32, host_weight_tensor.shape);
        int32_t* dst = static_cast<int32_t*>(host_i32.GetPtr<void>());
        for (size_t i = 0; i < n; ++i) {
          dst[i] = static_cast<int32_t>(src[i]);
        }
        CopyToDevice(host_i32, dev_tensor, context_);
        KLLM_LOG_DEBUG << fmt::format("V4: tid2eid narrowed int64 -> int32 at load ({} elements)", n);
      } else {
        CopyToDevice(host_weight_tensor, dev_tensor, context_);
      }
      if (file_weight_name.find("gate.weight") != std::string::npos) {
        weight_impl_->PermuteWeight(dev_tensor, {1, 0}, dev_rank);
      }
      device_model_weights[file_weight_name] = dev_tensor;
      KLLM_LOG_DEBUG << fmt::format("V4: Dev_rank: {}, [gate] stored: {}, shape: {}, dtype: {}", dev_rank,
                                    file_weight_name, Vector2Str(std::vector<size_t>(dev_tensor.shape)),
                                    dev_tensor.dtype);
      continue;
    }

    // =====================================================================
    // 5. Hyper-Connection weights (hc_attn_*, hc_ffn_*, hc_head_*)
    //    Directly loaded to each device without splitting or transposing.
    // =====================================================================
    // NOTICE(winminkong): mhc中的权重可能也有需要转置的权重
    if (file_weight_name.find("hc_") != std::string::npos) {
      Tensor dev_tensor;
      if (file_weight_name.find("hc_head_fn") != std::string::npos) {
        // For hc_head_fn, allocate a device tensor with shape[0]*2 to reserve space for the second half.
        // Copy the original data into the first half.
        std::vector<size_t> expanded_shape = host_weight_tensor.shape;
        expanded_shape[0] = expanded_shape[0] * 2;
        dev_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, host_weight_tensor.dtype, expanded_shape, nullptr,
                            &(context_->device->GetMemoryManageStream()));
        // Copy original host data to the first half of the expanded device tensor
        MemcpyAsync(dev_tensor.GetPtr<void>(), host_weight_tensor.GetPtr<void>(), host_weight_tensor.GetTotalBytes(),
                    MEMCPY_HOST_TO_DEVICE, context_->device->GetMemoryManageStream());
        if (expanded_shape[0] != hc_mult * 2) {
          KLLM_THROW(fmt::format("V4: hc_head_fn expanded shape[0]={} does not equal hc_mult*2={}, weight: {}",
                                 expanded_shape[0], hc_mult * 2, file_weight_name));
        }
        KLLM_LOG_DEBUG << fmt::format(
            "V4: Dev_rank: {}, [hc_head_fn] expanded: {}, shape: {} (original shape[0]={}), dtype: {}", dev_rank,
            file_weight_name, Vector2Str(std::vector<size_t>(dev_tensor.shape)), host_weight_tensor.shape[0],
            dev_tensor.dtype);
      } else {
        CopyToDevice(host_weight_tensor, dev_tensor, context_);
        KLLM_LOG_DEBUG << fmt::format("V4: Dev_rank: {}, [hc_*] stored: {}, shape: {}, dtype: {}", dev_rank,
                                      file_weight_name, Vector2Str(std::vector<size_t>(dev_tensor.shape)),
                                      dev_tensor.dtype);
      }
      device_model_weights[file_weight_name] = dev_tensor;
      continue;
    }

    // =====================================================================
    // 6. model.embed_tokens.weight: transpose, split by TP, transpose back
    // =====================================================================
    if (CheckWeightNameMatched(file_weight_name, {"model.embed_tokens.weight"}, true)) {
      Tensor dev_tensor;
      weight_impl_->TransSplitOptTrans(host_weight_tensor, dev_tensor, dev_rank,
                                       context_->global->GetTensorParallelSize(), true);
      device_model_weights[file_weight_name] = dev_tensor;
      KLLM_LOG_DEBUG << fmt::format("V4: Dev_rank: {}, [embed_tokens] stored: {}, shape: {}, dtype: {}", dev_rank,
                                    file_weight_name, Vector2Str(std::vector<size_t>(dev_tensor.shape)),
                                    dev_tensor.dtype);
      continue;
    }

    // =====================================================================
    // 7. lm_head.weight: split along axis=0, then transpose
    // =====================================================================
    if (CheckWeightNameMatched(file_weight_name, {"lm_head.weight"}, true)) {
      Tensor permute_dev_tensor;
      weight_impl_->SplitOptTrans(host_weight_tensor, permute_dev_tensor, dev_rank,
                                  context_->global->GetTensorParallelSize(), true);
      device_model_weights[file_weight_name] = permute_dev_tensor;
      KLLM_LOG_DEBUG << fmt::format("V4: Dev_rank: {}, [lm_head] stored: {}, shape: {}, dtype: {}", dev_rank,
                                    file_weight_name, Vector2Str(std::vector<size_t>(permute_dev_tensor.shape)),
                                    permute_dev_tensor.dtype);
      continue;
    }

    // =====================================================================
    // 8. MTP-specific weights
    //     e_proj/h_proj: independent weights, no fuse, no split
    // =====================================================================
    if (file_weight_name.find("e_proj.weight") != std::string::npos ||
        file_weight_name.find("h_proj.weight") != std::string::npos) {
      Tensor dev_tensor;
      CopyToDevice(host_weight_tensor, dev_tensor, context_);
      device_model_weights[file_weight_name] = dev_tensor;
      KLLM_LOG_DEBUG << fmt::format("V4: Dev_rank: {}, [MTP e/h_proj] stored: {}, shape: {}, dtype: {}", dev_rank,
                                    file_weight_name, Vector2Str(std::vector<size_t>(dev_tensor.shape)),
                                    dev_tensor.dtype);
      continue;
    }

    // =====================================================================
    // 8b. DSpark MTP weights (main_proj, markov_head, confidence_head)
    // =====================================================================
    // main_proj: direct copy to each device, no TP split
    if (file_weight_name.find("main_proj.weight") != std::string::npos) {
      Tensor dev_tensor;
      CopyToDevice(host_weight_tensor, dev_tensor, context_);
      device_model_weights[file_weight_name] = dev_tensor;
      KLLM_LOG_DEBUG << fmt::format("V4: Dev_rank: {}, [DSpark main_proj] stored: {}, shape: {}, dtype: {}", dev_rank,
                                    file_weight_name, Vector2Str(std::vector<size_t>(dev_tensor.shape)),
                                    dev_tensor.dtype);
      continue;
    }

    // markov_head.markov_w1: split along dim 1 (TransSplitOptTrans)
    if (file_weight_name.find("markov_head.markov_w1.weight") != std::string::npos) {
      Tensor dev_tensor;
      size_t tensor_para_size = context_->global->GetTensorParallelSize();
      weight_impl_->TransSplitOptTrans(host_weight_tensor, dev_tensor, dev_rank, tensor_para_size, true);
      device_model_weights[file_weight_name] = dev_tensor;
      KLLM_LOG_DEBUG << fmt::format("V4: Dev_rank: {}, [DSpark markov_w1] stored: {}, shape: {}, dtype: {}", dev_rank,
                                    file_weight_name, Vector2Str(std::vector<size_t>(dev_tensor.shape)),
                                    dev_tensor.dtype);
      continue;
    }

    // markov_head.markov_w2: split along dim 0 (SplitOptTrans)
    if (file_weight_name.find("markov_head.markov_w2.weight") != std::string::npos) {
      Tensor dev_tensor;
      size_t tensor_para_size = context_->global->GetTensorParallelSize();
      weight_impl_->SplitOptTrans(host_weight_tensor, dev_tensor, dev_rank, tensor_para_size, true);
      device_model_weights[file_weight_name] = dev_tensor;
      KLLM_LOG_DEBUG << fmt::format("V4: Dev_rank: {}, [DSpark markov_w2] stored: {}, shape: {}, dtype: {}", dev_rank,
                                    file_weight_name, Vector2Str(std::vector<size_t>(dev_tensor.shape)),
                                    dev_tensor.dtype);
      continue;
    }

    // confidence_head.proj: no TP split, copy then transpose
    if (file_weight_name.find("confidence_head.proj.weight") != std::string::npos) {
      Tensor dev_tensor;
      CopyToDevice(host_weight_tensor, dev_tensor, context_);
      weight_impl_->PermuteWeight(dev_tensor, {1, 0}, dev_rank);
      device_model_weights[file_weight_name] = dev_tensor;
      KLLM_LOG_DEBUG << fmt::format("V4: Dev_rank: {}, [DSpark confidence_head] stored: {}, shape: {}, dtype: {}",
                                    dev_rank, file_weight_name, Vector2Str(std::vector<size_t>(dev_tensor.shape)),
                                    dev_tensor.dtype);
      continue;
    }

    // =====================================================================
    // 9. Fallback: any unmatched weight, log warning and direct copy
    // =====================================================================
    KLLM_LOG_WARNING << fmt::format("V4: Unhandled weight: {}, shape: {}, doing direct copy.", file_weight_name,
                                    Vector2Str(std::vector<size_t>(host_weight_tensor.shape)));
    Tensor dev_tensor;
    CopyToDevice(host_weight_tensor, dev_tensor, context_);
    device_model_weights[file_weight_name] = dev_tensor;
  }

  return Status();
}

Status DeepSeekV4WeightLoader::PostProcessModelWeights(std::unordered_map<std::string, Tensor>& dev_weights_map,
                                                       int dev_rank) {
  KLLM_LOG_DEBUG << fmt::format("V4: Dev_rank: {}, [PostProcess] start, total weights: {}", dev_rank,
                                dev_weights_map.size());
  // For passing to weight_impl_ functions that take std::shared_ptr<DeepSeekV3Config>&
  std::shared_ptr<DeepSeekV3Config> base_config = deepseek_v4_config_;

  std::unordered_set<std::string> post_processed_weights;

  if (deepseek_v4_config_->quant_config.is_fp8_blockwise) {
    // Bind FP8 weight scales for all FP8 weights
    for (auto& [weight_name, weight_tensor] : dev_weights_map) {
      if (weight_tensor.dtype == DataType::TYPE_FP8_E4M3 || weight_tensor.dtype == DataType::TYPE_BLOCK_FP8_E4M3) {
        std::string weight_scale_name = weight_name + "_scale_inv";
        // dequant wo_a
        if (weight_name.find("attn.wo_a") != std::string::npos) {
          Tensor& quant_weight = dev_weights_map.at(weight_name);
          Tensor& quant_weight_scale = dev_weights_map.at(weight_scale_name);
          Tensor dev_tensor =
              weight_impl_->DequantFp8E4m3BlockWiseTensor(quant_weight, quant_weight_scale, dev_rank, base_config);
          size_t o_groups = static_cast<size_t>(deepseek_v4_config_->dsv4_attn_config.o_groups);
          size_t attn_tp_size = context_->global->GetAttentionTensorParallelSize();
          size_t o_local_groups = DivRoundUp(o_groups, attn_tp_size);
          size_t o_lora_rank = static_cast<size_t>(deepseek_v4_config_->dsv4_attn_config.o_lora_rank);
          KLLM_CHECK_WITH_INFO(o_local_groups * o_lora_rank == dev_tensor.shape[0],
                               "o_local_groups * o_lora_rank != wo_a shape[0]");
          dev_tensor.shape = {o_local_groups, o_lora_rank, dev_tensor.shape[1]};
          dev_weights_map[weight_name] = dev_tensor;
          dev_weights_map.erase(weight_scale_name);
          weights_to_permute_[dev_rank].insert(weight_name);
          KLLM_LOG_DEBUG << fmt::format("V4: [PostProcess] dequant wo_a: {}, shape: [{}], dtype: {}", weight_name,
                                        Vector2Str(std::vector<size_t>(dev_weights_map[weight_name].shape)),
                                        dev_weights_map[weight_name].dtype);
          continue;
        }
        auto scale_iter = dev_weights_map.find(weight_scale_name);
        if (scale_iter != dev_weights_map.end()) {
          weight_tensor.weight_scales = &(scale_iter->second);
          KLLM_LOG_DEBUG << fmt::format(
              "V4: Dev_rank: {}, [PostProcess] bind fp8 scale {} (shape: {}) -> {} (shape: {})", dev_rank,
              weight_scale_name, Vector2Str(std::vector<size_t>(weight_tensor.weight_scales->shape)), weight_name,
              Vector2Str(std::vector<size_t>(weight_tensor.shape)));
          post_processed_weights.insert(weight_scale_name);
          post_processed_weights.insert(weight_name);
        } else {
          KLLM_THROW(fmt::format("V4: weight scale inv not found: {}", weight_scale_name));
        }
      }
    }
  }

  // Bind FP4 MoE weight scales (E8M0 format stored as weight_scale_inv)
  // NOTE: For Marlin path, scales are only bound here (Marlin block re-processes and re-binds).
  // For Humming path, we MUST NOT mark weights as post-processed here — the humming block
  // needs to run repack_weight_wgmma and process_scales before the weights are usable.
  if (deepseek_v4_config_->quant_config.expert_dtype == "fp4") {
    const bool is_humming_backend = (runtime_config_.fp4_moe_backend == FP4_MOE_BACKEND_TYPE::FP4HummingW4A16 ||
                                     runtime_config_.fp4_moe_backend == FP4_MOE_BACKEND_TYPE::FP4HummingW4A8);
    for (auto& [weight_name, weight_tensor] : dev_weights_map) {
      if (weight_tensor.dtype == DataType::TYPE_FP4_E2M1FN_x2) {
        // up_gate_proj.weight → up_gate_proj.weight_scale_inv
        std::string scale_name = weight_name;
        size_t weight_pos = scale_name.rfind("weight");
        if (weight_pos != std::string::npos) {
          scale_name.replace(weight_pos, 6, "weight_scale_inv");
        }
        auto scale_iter = dev_weights_map.find(scale_name);
        if (scale_iter != dev_weights_map.end()) {
          weight_tensor.weight_scales = &(scale_iter->second);
          KLLM_LOG_DEBUG << fmt::format(
              "V4: Dev_rank: {}, [PostProcess] bind FP4 scale {} (shape: {}) -> {} (shape: {})", dev_rank, scale_name,
              Vector2Str(std::vector<size_t>(scale_iter->second.shape)), weight_name,
              Vector2Str(std::vector<size_t>(weight_tensor.shape)));
          // Only mark as post-processed for non-humming backends.
          // The humming block needs to run its own repack + scale processing.
          if (!is_humming_backend) {
            post_processed_weights.insert(scale_name);
            post_processed_weights.insert(weight_name);
          }
        } else {
          KLLM_LOG_WARNING << fmt::format("V4: FP4 weight scale not found: {}", scale_name);
        }
      }
    }
  }

  // Marlin FP4 weight preprocessing: repack weights + process scales
  if (deepseek_v4_config_->quant_config.expert_dtype == "fp4" &&
      runtime_config_.fp4_moe_backend == FP4_MOE_BACKEND_TYPE::FP4Marlin) {
    cudaStream_t stream = context_->device->GetMemoryManageStream().Get();
    size_t moe_inter_size_per_rank =
        DivRoundUp(deepseek_v4_config_->moe_config.moe_inter_size, deepseek_v4_config_->moe_tensor_para_size);
    size_t hidden_units = deepseek_v4_config_->hidden_units;
    int64_t group_size = 32;  // MXFP4 group size

    for (auto& [weight_name, weight_tensor] : dev_weights_map) {
      if (weight_tensor.dtype != DataType::TYPE_FP4_E2M1FN_x2) continue;

      // Determine dimensions per expert
      int64_t num_experts = weight_tensor.shape[0];
      int64_t size_n, size_k;
      bool is_up_gate = (weight_name.find("up_gate_proj") != std::string::npos);
      if (is_up_gate) {
        // up_gate: [num_experts, inter*2, hidden/2] (FP4 packed, 2 vals/byte)
        size_n = moe_inter_size_per_rank * 2;
        size_k = hidden_units;
      } else {
        // down: [num_experts, hidden, inter/2] (FP4 packed)
        size_n = hidden_units;
        size_k = moe_inter_size_per_rank;
      }

      // ===== Step 1: Transpose weight [E, N, K/2] bytes → [E, K/2, N] bytes =====
      // Viewed as uint32: [E, N, K/8] → [E, K/8, N]
      // The repack kernel indexes as ptr[(k_packed) * size_n + n], i.e. expects K-major (transposed).
      // sglang does: qweight.view(int32).T.contiguous() before gptq_marlin_repack.
      int64_t rows_uint32 = size_n;      // N rows when viewed as uint32
      int64_t cols_uint32 = size_k / 8;  // K/8 columns (32/4=8 elements per uint32)
      size_t weight_bytes = num_experts * rows_uint32 * cols_uint32 * sizeof(uint32_t);

      void* transposed_buf = nullptr;
      cudaMalloc(&transposed_buf, weight_bytes);
      llm_kernels::nvidia::marlin_moe_wna16::transpose_2d_batched(stream, weight_tensor.GetPtr<void>(), transposed_buf,
                                                                  num_experts, rows_uint32, cols_uint32,
                                                                  sizeof(uint32_t));

      // ===== Step 2: Repack transposed weight =====
      // Input: [E, K/8, N] uint32 (K-major), Output: [E, K/16, N*16/8] uint32
      auto repack_shape = llm_kernels::nvidia::marlin_moe_wna16::marlin_moe_repack_meta(size_k, size_n, 4);
      size_t repack_expert_size = repack_shape[0] * repack_shape[1] * sizeof(uint32_t);
      size_t total_repack_size = num_experts * repack_expert_size;

      void* repack_buf = nullptr;
      cudaMalloc(&repack_buf, total_repack_size);

      llm_kernels::nvidia::marlin_moe_wna16::marlin_moe_repack(reinterpret_cast<const uint32_t*>(transposed_buf),
                                                               reinterpret_cast<uint32_t*>(repack_buf), num_experts,
                                                               size_k, size_n, 4, dev_rank, stream);

      cudaFree(transposed_buf);

      // ===== Step 3: Copy repacked weight back =====
      cudaStreamSynchronize(stream);
      KLLM_CHECK_WITH_INFO(
          total_repack_size <= weight_bytes,
          fmt::format("Marlin repack size {} exceeds weight buffer {}", total_repack_size, weight_bytes));
      cudaMemcpy(weight_tensor.GetPtr<void>(), repack_buf, total_repack_size, cudaMemcpyDeviceToDevice);
      cudaFree(repack_buf);

      // Update tensor shape to Marlin layout
      weight_tensor.shape = {static_cast<size_t>(num_experts), static_cast<size_t>(repack_shape[0]),
                             static_cast<size_t>(repack_shape[1])};

      // ===== Step 4: Process scales =====
      // Scale tensor is E8M0 (1 byte/element), shape [E, N, K/32] on GPU.
      // Need: transpose → permute → mxfp4_process_scales, all on uint8 E8M0 data.
      std::string scale_name = weight_name;
      size_t wpos = scale_name.rfind("weight");
      if (wpos != std::string::npos) {
        scale_name.replace(wpos, 6, "weight_scale_inv");
      }
      auto scale_iter = dev_weights_map.find(scale_name);
      if (scale_iter != dev_weights_map.end()) {
        Tensor& scale_tensor = scale_iter->second;
        int64_t scale_rows = size_n;                                       // N
        int64_t scale_cols = size_k / group_size;                          // K/32
        size_t total_scale_bytes = num_experts * scale_rows * scale_cols;  // 1 byte per E8M0 element

        // Step 4a: Transpose scales [E, N, K/32] → [E, K/32, N] (uint8)
        void* scale_transposed = nullptr;
        cudaMalloc(&scale_transposed, total_scale_bytes);
        llm_kernels::nvidia::marlin_moe_wna16::transpose_2d_batched(stream, scale_tensor.GetPtr<void>(),
                                                                    scale_transposed, num_experts, scale_rows,
                                                                    scale_cols, sizeof(uint8_t));

        // Step 4b: Permute scales (on transposed E8M0 uint8 data)
        void* scale_permuted = nullptr;
        cudaMalloc(&scale_permuted, total_scale_bytes);
        llm_kernels::nvidia::marlin_moe_wna16::marlin_moe_permute_scales<uint8_t>(
            stream, reinterpret_cast<const uint8_t*>(scale_transposed), reinterpret_cast<uint8_t*>(scale_permuted),
            num_experts, size_k, size_n, group_size);
        cudaFree(scale_transposed);

        // Step 4c: Apply MXFP4 swizzle [0,2,1,3] + exponent bias +6 (in-place on permuted data)
        llm_kernels::nvidia::marlin_moe_wna16::mxfp4_process_scales(stream, reinterpret_cast<uint8_t*>(scale_permuted),
                                                                    static_cast<int64_t>(total_scale_bytes));

        // Step 4d: Copy processed scales back to scale tensor
        cudaStreamSynchronize(stream);
        cudaMemcpy(scale_tensor.GetPtr<void>(), scale_permuted, total_scale_bytes, cudaMemcpyDeviceToDevice);
        cudaFree(scale_permuted);

        // Update scale shape to [E, K/32, N] after transpose+permute
        scale_tensor.shape = {static_cast<size_t>(num_experts), static_cast<size_t>(scale_cols),
                              static_cast<size_t>(scale_rows)};

        // Bind scale to weight
        weight_tensor.weight_scales = &(scale_iter->second);
        post_processed_weights.insert(scale_name);
      }

      post_processed_weights.insert(weight_name);
      KLLM_LOG_DEBUG << fmt::format("V4: Dev_rank: {}, [PostProcess] Marlin FP4 repack done: {}, new shape: [{},{},{}]",
                                    dev_rank, weight_name, weight_tensor.shape[0], weight_tensor.shape[1],
                                    weight_tensor.shape[2]);
    }
    cudaStreamSynchronize(stream);
  }

  // Humming FP4 weight preprocessing: repack weights + process scales
  if (deepseek_v4_config_->quant_config.expert_dtype == "fp4" &&
      (runtime_config_.fp4_moe_backend == FP4_MOE_BACKEND_TYPE::FP4HummingW4A16 ||
       runtime_config_.fp4_moe_backend == FP4_MOE_BACKEND_TYPE::FP4HummingW4A8)) {
    cudaStream_t stream = context_->device->GetMemoryManageStream().Get();
    auto compute_dtype = (runtime_config_.fp4_moe_backend == FP4_MOE_BACKEND_TYPE::FP4HummingW4A16)
                             ? llm_kernels::nvidia::humming::ComputeDtype::BF16
                             : llm_kernels::nvidia::humming::ComputeDtype::FP8;
    size_t moe_inter_size_per_rank =
        DivRoundUp(deepseek_v4_config_->moe_config.moe_inter_size, deepseek_v4_config_->moe_tensor_para_size);
    size_t hidden_units = deepseek_v4_config_->hidden_units;

    // Collect global_scale entries to insert into map after iteration (avoid invalidating iterators)
    struct GlobalScaleEntry {
      std::string tensor_name;  // key for dev_weights_map
      std::string weight_name;  // corresponding weight tensor to bind to
      float* device_ptr;
      int64_t num_experts;
    };
    std::vector<GlobalScaleEntry> global_scale_entries;

    for (auto& [weight_name, weight_tensor] : dev_weights_map) {
      if (weight_tensor.dtype != DataType::TYPE_FP4_E2M1FN_x2) continue;
      if (post_processed_weights.count(weight_name)) continue;

      int64_t num_experts = weight_tensor.shape[0];
      int64_t size_n, size_k;
      bool is_up_gate = (weight_name.find("up_gate_proj") != std::string::npos);
      if (is_up_gate) {
        size_n = moe_inter_size_per_rank * 2;
        size_k = hidden_units;
      } else {
        size_n = hidden_units;
        size_k = moe_inter_size_per_rank;
      }

      // Step 0 (W4A8 only): Preprocess weights before repacking
      // Normalizes E8M0 scales, adjusts FP4 values, computes per-expert global_scale.
      // Must run BEFORE repack_weight_wgmma since it modifies raw weight values.
      if (runtime_config_.fp4_moe_backend == FP4_MOE_BACKEND_TYPE::FP4HummingW4A8) {
        std::string scale_name = weight_name;
        size_t wpos = scale_name.rfind("weight");
        if (wpos != std::string::npos) {
          scale_name.replace(wpos, 6, "weight_scale_inv");
        }
        auto scale_iter = dev_weights_map.find(scale_name);
        if (scale_iter != dev_weights_map.end()) {
          Tensor& scale_tensor = scale_iter->second;

          // Allocate per-expert global_scale [num_experts] on device
          float* global_scale_ptr = nullptr;
          cudaMalloc(&global_scale_ptr, num_experts * sizeof(float));

          llm_kernels::nvidia::humming::process_mxfp4_w4a8_weight(
              stream,
              weight_tensor.GetPtr<void>(),  // weight (in-place modification)
              scale_tensor.GetPtr<void>(),   // weight_scale (in-place normalization)
              global_scale_ptr,              // global_scale output [E]
              num_experts, size_n, size_k, 32);
          cudaStreamSynchronize(stream);

          // Derive a unique name for this global_scale tensor
          std::string gs_name = weight_name;
          size_t gpos = gs_name.rfind("weight");
          if (gpos != std::string::npos) {
            gs_name.replace(gpos, 6, "weight_global_scale");
          }
          global_scale_entries.push_back({gs_name, weight_name, global_scale_ptr, num_experts});

          KLLM_LOG_DEBUG << fmt::format(
              "V4: Dev_rank: {}, [PostProcess] W4A8 process_mxfp4_w4a8_weight done: {}, global_scale stored as: {}",
              dev_rank, weight_name, gs_name);
        }
      }

      // Step 1: repack_weight_wgmma (with temp buffer)
      // Input: FP4 packed [E, N, K/2] — the repack kernel handles kPackedInput=1
      size_t weight_bytes = weight_tensor.shape[0] * weight_tensor.shape[1] * weight_tensor.shape[2] * sizeof(uint8_t);
      void* repack_buf = nullptr;
      cudaMalloc(&repack_buf, weight_bytes);

      llm_kernels::nvidia::humming::repack_weight_wgmma(stream, weight_tensor.GetPtr<void>(), repack_buf, num_experts,
                                                        size_n, size_k, compute_dtype);

      cudaStreamSynchronize(stream);
      cudaMemcpy(weight_tensor.GetPtr<void>(), repack_buf, weight_bytes, cudaMemcpyDeviceToDevice);
      cudaFree(repack_buf);

      // Step 2: process_scales (libtorch transpose + permute)
      std::string scale_name = weight_name;
      size_t wpos = scale_name.rfind("weight");
      if (wpos != std::string::npos) {
        scale_name.replace(wpos, 6, "weight_scale_inv");
      }
      auto scale_iter = dev_weights_map.find(scale_name);
      if (scale_iter != dev_weights_map.end()) {
        Tensor& scale_tensor = scale_iter->second;
        size_t scale_bytes = num_experts * size_n * (size_k / 32);  // E8M0: 1 byte per group

        void* processed_scales = nullptr;
        cudaMalloc(&processed_scales, scale_bytes);

        llm_kernels::nvidia::humming::process_scales(stream, scale_tensor.GetPtr<void>(), processed_scales, num_experts,
                                                     size_n, size_k, 32, compute_dtype);

        cudaStreamSynchronize(stream);
        cudaMemcpy(scale_tensor.GetPtr<void>(), processed_scales, scale_bytes, cudaMemcpyDeviceToDevice);
        cudaFree(processed_scales);

        // Update scale shape to [E, K/32, N] after transpose+permute
        scale_tensor.shape = {static_cast<size_t>(num_experts), static_cast<size_t>(size_k / 32),
                              static_cast<size_t>(size_n)};

        weight_tensor.weight_scales = &(scale_iter->second);
        post_processed_weights.insert(scale_name);
      }

      post_processed_weights.insert(weight_name);
      KLLM_LOG_DEBUG << fmt::format("V4: Dev_rank: {}, [PostProcess] Humming FP4 repack done: {}", dev_rank,
                                    weight_name);
    }

    // Insert global_scale tensors into the map and bind to corresponding weight tensors.
    // Done after the for-loop to avoid invalidating unordered_map iterators during insertion.
    for (auto& entry : global_scale_entries) {
      // Create a Tensor wrapping the device pointer (is_shared_buffer=true since data_ptr != nullptr)
      auto [iter, inserted] =
          dev_weights_map.emplace(entry.tensor_name, Tensor(MemoryLocation::LOCATION_DEVICE, DataType::TYPE_FP32,
                                                            std::vector<size_t>{static_cast<size_t>(entry.num_experts)},
                                                            static_cast<void*>(entry.device_ptr), entry.tensor_name));
      if (inserted) {
        // Bind global_scale to the weight tensor
        auto weight_iter = dev_weights_map.find(entry.weight_name);
        if (weight_iter != dev_weights_map.end()) {
          weight_iter->second.weight_global_scale = &(iter->second);
        }
        post_processed_weights.insert(entry.tensor_name);
      }
    }
  }

  // Permute weights that need transposition
  for (auto& [weight_name, weight_tensor] : dev_weights_map) {
    if (weights_to_permute_[dev_rank].find(weight_name) != weights_to_permute_[dev_rank].end()) {
      KLLM_LOG_DEBUG << fmt::format("V4: Dev_rank: {}, [PostProcess] permute: {}, shape before: {}, dtype: {}",
                                    dev_rank, weight_name, Vector2Str(std::vector<size_t>(weight_tensor.shape)),
                                    weight_tensor.dtype);
      if (weight_name.find("attn.wo_a") != std::string::npos) {
        KLLM_CHECK_WITH_INFO(weight_tensor.shape.size() == 3, "attn.wo_a shape size must be 3");
        weight_impl_->PermuteWeight(weight_tensor, {0, 2, 1}, dev_rank);
      } else {
        weight_impl_->PermuteWeight(weight_tensor, {1, 0}, dev_rank);
      }
      KLLM_LOG_DEBUG << fmt::format("V4: Dev_rank: {}, [PostProcess] permute done: {}, shape after: {}", dev_rank,
                                    weight_name, Vector2Str(std::vector<size_t>(weight_tensor.shape)));
    }

    if (post_processed_weights.find(weight_name) != post_processed_weights.end()) {
      continue;
    }

    if (!(weight_tensor.dtype == DataType::TYPE_FP32 || weight_tensor.dtype == DataType::TYPE_FP16 ||
          weight_tensor.dtype == DataType::TYPE_BF16)) {
      continue;
    }

    // Gate bias (e_score_correction_bias) should be cast to fp32 for performance
    if (weight_name.find("gate.e_score_correction_bias") != std::string::npos) {
      if (weight_tensor.dtype != TYPE_FP32) {
        KLLM_LOG_DEBUG << fmt::format("V4: Dev_rank: {}, [PostProcess] cast {} from {} to {}", dev_rank, weight_name,
                                      weight_tensor.dtype, TYPE_FP32);
        CastDeviceTensorType(weight_tensor, TYPE_FP32, dev_rank);
      }
      continue;
    }

    // NOTICE(winminkong): 需确定剩余的权重为什么要转换类型
    // Cast to target data type
    // if (weight_tensor.dtype != deepseek_v4_config_->weight_data_type) {
    //   KLLM_LOG_DEBUG << fmt::format("V4: Dev_rank: {}, [PostProcess] cast {} from {} to {}", dev_rank, weight_name,
    //                                 weight_tensor.dtype, deepseek_v4_config_->weight_data_type);
    //   CastDeviceTensorType(weight_tensor, deepseek_v4_config_->weight_data_type, dev_rank);
    // }
  }

  KLLM_LOG_DEBUG << fmt::format("V4: Dev_rank: {}, [PostProcess] complete, total weights: {}", dev_rank,
                                dev_weights_map.size());
  return Status();
}

}  // namespace ksana_llm
