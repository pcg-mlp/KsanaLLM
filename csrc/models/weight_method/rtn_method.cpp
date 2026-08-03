/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "models/weight_method/rtn_method.h"

#include <torch/torch.h>
#include <algorithm>

#include "fmt/core.h"
#include "model_loader/model_loader_utils.h"
#include "models/base/utils.h"
#include "utils/logger.h"
#include "utils/string_utils.h"

namespace ksana_llm {

Status RtnMethod::load_experts_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                                    const std::string& weight_name, const Tensor& weight_tensor, int dev_rank) {
#ifdef ENABLE_CUDA
  // RTN expert weights: .weight (int8, 4-bit packed), .weight_scale (bf16 blockwise), .input_scale (bf16)
  // No transpose needed: direct concat for gate/up, already correct layout for down.
  //
  // Note: weight_map (glm4_moe_weight_map.json) already converts _scale_inv -> weight_scale
  // before reaching here, so we only need to match canonical names.
  if (CheckWeightNameEndMatched(weight_name, {".weight_scale", ".weight"})) {
    torch::Tensor tensor = common_weight_loader_->GetTorchTensorFromTensor(weight_tensor);
    if (tensor.device().type() == torch::kCPU) {
      tensor = tensor.to(torch::Device(torch::kCUDA, dev_rank));
    }
    // Blockwise scales must be bf16 for CUTLASS MoE kernel
    if (CheckWeightNameEndMatched(weight_name, {".weight_scale"}) && tensor.dtype() != torch::kBFloat16) {
      tensor = tensor.to(torch::kBFloat16);
    }
    device_model_weights[weight_name] = common_weight_loader_->GetTensorFromTorchTensor(tensor);
  } else if (CheckWeightNameEndMatched(weight_name, {".input_scale"})) {
    // Per-expert activation scale, just move to device
    device_model_weights[weight_name] = common_weight_loader_->MoveToDevice(weight_tensor, dev_rank);
  } else {
    KLLM_LOG_WARNING << fmt::format(
        "RTN load_experts_proj: unrecognized weight '{}', shape=[{}], dtype={} - not loaded!", weight_name,
        Vector2Str(std::vector<size_t>(weight_tensor.shape)), static_cast<int>(weight_tensor.dtype));
  }
  return Status();
#else
  return Status(RET_RUNTIME_FAILED, "RTN MoE quantization is only supported on CUDA devices");
#endif
}

// Helper function to compute interleave factor for CUTLASS MoE
static size_t GetCutlassMoeInterleave(size_t k_shape) {
  // For SM90 (H100/H200), compute interleave based on k dimension
  if (k_shape % 512 == 0) {
    return 4;
  } else if (k_shape % 256 == 0) {
    return 2;
  } else if (k_shape % 128 == 0) {
    return 1;
  }
  KLLM_LOG_ERROR << fmt::format("K shape {} is not a multiple of 128, cannot compute interleave", k_shape);
  return 1;
}

Status RtnMethod::VerifyWeightsExist(const std::unordered_map<std::string, Tensor>& device_model_weights,
                                     const std::vector<std::string>& weight_names, const std::string& label) {
  std::vector<std::string> missing;
  for (const auto& name : weight_names) {
    if (device_model_weights.find(name) == device_model_weights.end()) {
      missing.push_back(name);
    }
  }
  if (!missing.empty()) {
    std::string msg = fmt::format("RTN: {} missing weights: {}", label, fmt::join(missing, ", "));
    KLLM_LOG_ERROR << msg;
    return Status(RetCode::RET_INVALID_ARGUMENT, msg);
  }
  return Status();
}

void RtnMethod::ApplyScaleInterleave(Tensor& scales_tensor, size_t interleave, int dev_rank) {
  if (scales_tensor.shape.size() != 3 || interleave <= 1) {
    return;
  }
  std::vector<size_t> origin_shape = scales_tensor.shape;
  scales_tensor.shape = {origin_shape[0], origin_shape[1], origin_shape[2] / interleave, interleave};

  // Permute {0, 2, 1, 3}
  common_weight_loader_->PermuteWeight(scales_tensor, {0, 2, 1, 3}, dev_rank);

  // Final reshape: [E, K/(128*interleave), N*interleave]
  scales_tensor.shape = {origin_shape[0], origin_shape[2] / interleave, origin_shape[1] * interleave};
}

void RtnMethod::CreateActScaleAndAlpha(std::unordered_map<std::string, Tensor>& device_model_weights,
                                       const std::string& proj_prefix, float max_input_scale, size_t local_num_experts,
                                       int dev_rank) {
#ifdef ENABLE_CUDA
  auto* stream = &(common_weight_loader_->GetContext()->device->GetMemoryManageStream());
  auto float_option = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, dev_rank);

  // Create act_scale = 1 / max_input_scale
  // IMPORTANT: CUTLASS MoE kernel expects act_scale in bf16 (same as model dtype), NOT fp32
  const std::string act_scale_name = proj_prefix + ".act_scale";
  torch::Tensor act_scale_tensor = torch::full({1}, 1.0f / max_input_scale, float_option).to(torch::kBFloat16);
  device_model_weights[act_scale_name] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16, {1}, nullptr, stream);
  MemcpyAsync(device_model_weights[act_scale_name].GetPtr<void>(), act_scale_tensor.data_ptr(),
              act_scale_tensor.numel() * act_scale_tensor.element_size(), MEMCPY_DEVICE_TO_DEVICE, *stream);

  // Create alpha = max_input_scale for all experts
  const std::string alpha_name = proj_prefix + ".alpha";
  torch::Tensor alpha_tensor = torch::full({static_cast<int64_t>(local_num_experts), 1}, max_input_scale, float_option);
  device_model_weights[alpha_name] =
      Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {local_num_experts, 1}, nullptr, stream);
  MemcpyAsync(device_model_weights[alpha_name].GetPtr<void>(), alpha_tensor.data_ptr(),
              alpha_tensor.numel() * alpha_tensor.element_size(), MEMCPY_DEVICE_TO_DEVICE, *stream);
#endif
}

float RtnMethod::ComputeMaxInputScale(std::unordered_map<std::string, Tensor>& device_model_weights,
                                      const std::string& weight_prefix_name,
                                      const std::vector<int>& local_expert_indices,
                                      const std::vector<std::string>& proj_suffixes) {
  float max_input_scale = 0.0f;
#ifdef ENABLE_CUDA
  for (int expert_idx : local_expert_indices) {
    for (const auto& suffix : proj_suffixes) {
      const std::string scale_name = fmt::format("{}{}.{}.input_scale", weight_prefix_name, expert_idx, suffix);
      auto it = device_model_weights.find(scale_name);
      if (it != device_model_weights.end()) {
        torch::Tensor t = common_weight_loader_->GetTorchTensorFromTensor(it->second);
        float val = torch::max(t.to(torch::kFloat32)).item<float>();
        max_input_scale = std::max(max_input_scale, val);
        device_model_weights.erase(it);
      }
    }
  }
#endif
  return max_input_scale;
}

Status RtnMethod::ProcessExpertProj(std::unordered_map<std::string, Tensor>& device_model_weights,
                                    const std::string& weight_prefix_name, const std::vector<int>& local_expert_indices,
                                    const std::string& proj_name, const std::vector<std::string>& sub_proj_suffixes,
                                    const std::vector<size_t>& reshape_dims, size_t interleave_k_dim,
                                    bool has_input_scale, int dev_rank) {
#ifdef ENABLE_CUDA
  const size_t local_num_experts = local_expert_indices.size();

  // Merge weight tensors for local experts
  std::vector<std::string> expert_weight_names;
  for (int expert_idx : local_expert_indices) {
    for (const auto& suffix : sub_proj_suffixes) {
      expert_weight_names.push_back(fmt::format("{}{}.{}.weight", weight_prefix_name, expert_idx, suffix));
    }
  }

  const std::string merged_weight_name = weight_prefix_name + proj_name + ".weight";
  Status merge_status =
      common_weight_loader_->AutoMergeWeight(expert_weight_names, merged_weight_name, device_model_weights, dev_rank);
  if (!merge_status.OK()) {
    KLLM_LOG_ERROR << fmt::format("RTN: Failed to merge {} weights: {}", proj_name, merge_status.GetMessage());
    return merge_status;
  }
  // Reshape to 3D: [local_num_experts, dim0, dim1_packed]
  device_model_weights[merged_weight_name].shape = reshape_dims;

  // Merge weight_scale tensors (per-group blockwise scales)
  std::vector<std::string> expert_scale_names;
  for (int expert_idx : local_expert_indices) {
    for (const auto& suffix : sub_proj_suffixes) {
      expert_scale_names.push_back(fmt::format("{}{}.{}.weight_scale", weight_prefix_name, expert_idx, suffix));
    }
  }
  // Verify all scale tensors exist before merging
  Status verify_status = VerifyWeightsExist(device_model_weights, expert_scale_names, proj_name + " scale");
  if (!verify_status.OK()) {
    return verify_status;
  }

  const std::string merged_scales_name = weight_prefix_name + proj_name + ".weight_scale";
  Status scale_merge_status =
      common_weight_loader_->AutoMergeWeight(expert_scale_names, merged_scales_name, device_model_weights, dev_rank);
  if (!scale_merge_status.OK()) {
    KLLM_LOG_ERROR << fmt::format("RTN: Failed to merge {} scales: {}", proj_name, scale_merge_status.GetMessage());
    return scale_merge_status;
  }

  Tensor& scales_tensor = device_model_weights[merged_scales_name];
  // Reshape to 3D: [local_num_experts, N, K_groups]
  if (scales_tensor.shape.size() == 2) {
    const size_t k_groups = scales_tensor.shape[1];
    scales_tensor.shape = {local_num_experts, scales_tensor.shape[0] / local_num_experts, k_groups};
  }

  // Apply interleave transformation for CUTLASS MoE
  const size_t interleave = GetCutlassMoeInterleave(interleave_k_dim);
  ApplyScaleInterleave(scales_tensor, interleave, dev_rank);

  // NOTE: Do NOT bind pointers here! Inserting new keys into unordered_map may trigger rehash,
  // invalidating all pointers obtained via &device_model_weights[key].
  // All pointer bindings are deferred to the end of process_experts_proj.

  // Process input_scale and alpha for CutlassGroupWint4Afp8MoeLayer
  if (has_input_scale) {
    float max_input_scale =
        ComputeMaxInputScale(device_model_weights, weight_prefix_name, local_expert_indices, sub_proj_suffixes);
    CreateActScaleAndAlpha(device_model_weights, weight_prefix_name + proj_name, max_input_scale, local_num_experts,
                           dev_rank);
    KLLM_LOG_DEBUG << fmt::format("RTN: {} max_input_scale={}", proj_name, max_input_scale);
  }
#else
  (void)device_model_weights;
  (void)weight_prefix_name;
  (void)local_expert_indices;
  (void)proj_name;
  (void)sub_proj_suffixes;
  (void)reshape_dims;
  (void)interleave_k_dim;
  (void)has_input_scale;
  (void)dev_rank;
#endif
  return Status();
}

void RtnMethod::BindProjPointers(std::unordered_map<std::string, Tensor>& device_model_weights,
                                 const std::string& weight_prefix_name, const std::string& proj_name,
                                 bool has_input_scale) {
  const std::string weight_name = weight_prefix_name + proj_name + ".weight";
  const std::string scales_name = weight_prefix_name + proj_name + ".weight_scale";

  if (device_model_weights.find(weight_name) != device_model_weights.end() &&
      device_model_weights.find(scales_name) != device_model_weights.end()) {
    device_model_weights[weight_name].scales = &device_model_weights[scales_name];
    device_model_weights[weight_name].weight_scales = &device_model_weights[scales_name];
  }

  if (has_input_scale) {
    const std::string act_scale_name = weight_prefix_name + proj_name + ".act_scale";
    const std::string alpha_name = weight_prefix_name + proj_name + ".alpha";

    if (device_model_weights.find(weight_name) != device_model_weights.end()) {
      device_model_weights[weight_name].input_scales = &device_model_weights[act_scale_name];
      device_model_weights[weight_name].input_alpha = &device_model_weights[alpha_name];
    }
  }
}

Status RtnMethod::process_experts_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                                       const std::string& weight_prefix_name, int dev_rank) {
#ifdef ENABLE_CUDA
  // Dynamically detect which experts are present by scanning device_model_weights
  std::vector<int> local_expert_indices;
  const std::string gate_weight_suffix = ".gate_proj.weight";
  for (const auto& [name, tensor] : device_model_weights) {
    // Check if name starts with weight_prefix_name and ends with .gate_proj.weight
    // But exclude .gate_proj.weight_scale (which also contains ".gate_proj.weight")
    if (name.find(weight_prefix_name) == 0 && name.find(gate_weight_suffix) != std::string::npos &&
        !CheckWeightNameEndMatched(name, {".weight_scale"})) {
      // Extract expert index: weight_prefix_name + "X" + ".gate_proj.weight"
      size_t start_pos = weight_prefix_name.length();
      size_t end_pos = name.find(gate_weight_suffix);
      if (end_pos > start_pos) {
        std::string expert_idx_str = name.substr(start_pos, end_pos - start_pos);
        if (!expert_idx_str.empty() && std::all_of(expert_idx_str.begin(), expert_idx_str.end(), ::isdigit)) {
          int expert_idx = std::stoi(expert_idx_str);
          local_expert_indices.push_back(expert_idx);
        }
      }
    }
  }
  // Sort to ensure consistent ordering
  std::sort(local_expert_indices.begin(), local_expert_indices.end());

  if (local_expert_indices.empty()) {
    KLLM_LOG_WARNING << "RTN process_experts_proj: No local experts found, skipping";
    return Status();
  }

  const size_t local_num_experts = local_expert_indices.size();

  // Check if we have input_scale (RTN with input_scale=true)
  const std::string first_gate_input_scale_name =
      fmt::format("{}{}.gate_proj.input_scale", weight_prefix_name, local_expert_indices[0]);
  bool has_input_scale = device_model_weights.find(first_gate_input_scale_name) != device_model_weights.end();

  // ===== Step 1: Process up_gate_proj =====
  const std::string first_gate_proj_name = fmt::format("{}{}.gate_proj.", weight_prefix_name, local_expert_indices[0]);
  if (device_model_weights.find(first_gate_proj_name + "weight") != device_model_weights.end()) {
    Tensor& first_gate_weight = device_model_weights[first_gate_proj_name + "weight"];
    // gate/up shape is [inter_size, hidden_size_packed] e.g. [1536, 2560]
    const size_t inter_size = first_gate_weight.shape[0];
    const size_t hidden_size_packed = first_gate_weight.shape[1];
    // hidden_size = hidden_size_packed * 2 (for int8 packed as int4)
    const size_t hidden_size = hidden_size_packed * 2;

    // Order: [gate, up] for each expert
    // The CUTLASS MoE TMA-warp-specialized kernel (doActivationKernel) expects [gate, up]:
    //   front half (offset 0) = gate_proj (SiLu applied), back half = up_proj (multiplied)
    Status status = ProcessExpertProj(device_model_weights, weight_prefix_name, local_expert_indices, "up_gate_proj",
                                      {"gate_proj", "up_proj"}, {local_num_experts, 2 * inter_size, hidden_size_packed},
                                      hidden_size, has_input_scale, dev_rank);
    if (!status.OK()) {
      return status;
    }
  }

  // ===== Step 2: Process down_proj =====
  const std::string first_down_proj_name = fmt::format("{}{}.down_proj.", weight_prefix_name, local_expert_indices[0]);
  if (device_model_weights.find(first_down_proj_name + "weight") != device_model_weights.end()) {
    Tensor& first_down_weight = device_model_weights[first_down_proj_name + "weight"];
    // down shape is [hidden_size, inter_size_packed] e.g. [5120, 768]
    const size_t down_hidden_units = first_down_weight.shape[0];
    const size_t down_inter_size_packed = first_down_weight.shape[1];
    // inter_size = inter_size_packed * 2 (for int8 packed as int4)
    const size_t inter_size_real = down_inter_size_packed * 2;

    Status status = ProcessExpertProj(device_model_weights, weight_prefix_name, local_expert_indices, "down_proj",
                                      {"down_proj"}, {local_num_experts, down_hidden_units, down_inter_size_packed},
                                      inter_size_real, has_input_scale, dev_rank);
    if (!status.OK()) {
      return status;
    }
  }

  // ===== Step 3: Bind all pointers AFTER all map insertions are done =====
  // This avoids pointer invalidation from unordered_map rehashing.
  BindProjPointers(device_model_weights, weight_prefix_name, "up_gate_proj", has_input_scale);
  BindProjPointers(device_model_weights, weight_prefix_name, "down_proj", has_input_scale);

  KLLM_LOG_DEBUG << fmt::format("RTN process_experts_proj completed for {} local experts", local_num_experts);
  return Status();
#else
  return Status(RET_RUNTIME_FAILED, "RTN MoE quantization is only supported on CUDA devices");
#endif
}

}  // namespace ksana_llm
