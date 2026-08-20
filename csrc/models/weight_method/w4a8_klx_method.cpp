/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "models/weight_method/w4a8_klx_method.h"

#include <torch/torch.h>

#include "model_loader/model_loader_utils.h"
#include "models/base/weight_status.h"
#include "utils/logger.h"

namespace ksana_llm {

W4A8KlxMethod::W4A8KlxMethod(std::shared_ptr<CommonModelWeightLoader> common_weight_loader, int tp)
    : common_weight_loader_(common_weight_loader), tp_(tp) {}

// ============ Static Helpers ============

bool W4A8KlxMethod::IsW4ExpertWeight(const std::string& weight_name) {
  // W4 weights are MoE routed experts (excludes shared experts which are W8).
  // Matches both original per-expert names: *.mlp.experts.{id}.{gate_proj|up_proj|down_proj}.*
  // and stacked names after LoadInt4QuantWeight: *.mlp.experts.{up_gate_proj|down_proj}.*
  static const std::regex expert_pattern(R"(\.mlp\.experts\.(\d+\.)?(gate_proj|up_proj|down_proj|up_gate_proj)\.)");
  return std::regex_search(weight_name, expert_pattern);
}

bool W4A8KlxMethod::IsWeightScale(const std::string& weight_name) {
  static constexpr std::string_view kSuffix = ".weight_scale";
  if (weight_name.size() < kSuffix.size()) return false;
  return weight_name.compare(weight_name.size() - kSuffix.size(), kSuffix.size(), kSuffix) == 0;
}

std::string W4A8KlxMethod::GetScaleName(const std::string& weight_name) {
  // "xxx.weight" -> "xxx.weight_scale"
  return weight_name + "_scale";
}

std::string W4A8KlxMethod::GetWeightName(const std::string& scale_name) {
  // "xxx.weight_scale" -> "xxx.weight"
  // Remove "_scale" suffix
  if (scale_name.size() > 6 && scale_name.substr(scale_name.size() - 6) == "_scale") {
    return scale_name.substr(0, scale_name.size() - 6);
  }
  return scale_name;
}

// ============ Weight Loading ============

Status W4A8KlxMethod::LoadW8Weight(std::unordered_map<std::string, Tensor>& device_model_weights,
                                   const std::string& weight_name, const Tensor& weight_tensor, int dev_rank,
                                   TensorParallelMode tp_mode, size_t tp_size) {
  if (tp_size > 1) {
    device_model_weights[weight_name] =
        common_weight_loader_->TensorParallelSplit(weight_tensor, dev_rank, tp_size, tp_mode);
  } else {
    device_model_weights[weight_name] = common_weight_loader_->MoveToDevice(weight_tensor, dev_rank);
  }
  return Status();
}

Status W4A8KlxMethod::LoadW4ExpertWeight(std::unordered_map<std::string, Tensor>& device_model_weights,
                                         const std::string& weight_name, const Tensor& weight_tensor, int dev_rank) {
  // Expert weights are directly moved to device; stacking is handled by caller
  device_model_weights[weight_name] = common_weight_loader_->MoveToDevice(weight_tensor, dev_rank);
  return Status();
}

// ============ Post-processing ============

Status W4A8KlxMethod::ProcessW8Weight(std::unordered_map<std::string, Tensor>& device_model_weights,
                                      const std::string& weight_name, int dev_rank) {
  // 1. Permute weight: [out_dim, in_dim] -> [in_dim, out_dim] = [k, n]
  auto weight_iter = device_model_weights.find(weight_name);
  if (weight_iter == device_model_weights.end()) {
    return Status(RET_RUNTIME_FAILED, fmt::format("W4A8KlxMethod: weight not found: {}", weight_name));
  }
  Tensor& weight_tensor = weight_iter->second;
  common_weight_loader_->Permute2D(weight_tensor, dev_rank);

  // 2. Process scale: cast BF16 -> FP32, bind to weight
  std::string scale_name = GetScaleName(weight_name);
  auto scale_iter = device_model_weights.find(scale_name);
  if (scale_iter == device_model_weights.end()) {
    return Status(RET_RUNTIME_FAILED, fmt::format("W4A8KlxMethod: scale not found: {}", scale_name));
  }
  Tensor& scale_tensor = scale_iter->second;

  // Cast scale to FP32 if not already (model stores as BF16)
  if (scale_tensor.dtype != DataType::TYPE_FP32) {
    common_weight_loader_->CastDeviceTensorType(scale_tensor, DataType::TYPE_FP32, dev_rank);
  }

  // 3. Bind scale to weight
  weight_tensor.weight_scales = &device_model_weights[scale_name];

  // 4. Set weight status
  WeightStatus weight_status;
  weight_status.quant_mode = QuantMode::QUANT_W4A8_KLX;
  // After permute: weight shape is [k, n]
  weight_status.layout["k"] = weight_tensor.shape[0];
  weight_status.layout["n"] = weight_tensor.shape[1];
  common_weight_loader_->GetContext()->device->SetWeightStatus(weight_name, weight_status);

  KLLM_LOG_DEBUG << fmt::format("W4A8KlxMethod::ProcessW8Weight: {} shape=[{}, {}] scale_shape=[{}]", weight_name,
                                weight_tensor.shape[0], weight_tensor.shape[1], scale_tensor.shape[0]);
  return Status();
}

Status W4A8KlxMethod::ProcessW4Weight(std::unordered_map<std::string, Tensor>& device_model_weights,
                                      const std::string& weight_name, int dev_rank) {
  // 1. Permute weight: [out_dim, in_dim/2] -> [in_dim/2, out_dim] = [k/2, n]
  auto weight_iter = device_model_weights.find(weight_name);
  if (weight_iter == device_model_weights.end()) {
    return Status(RET_RUNTIME_FAILED, fmt::format("W4A8KlxMethod: weight not found: {}", weight_name));
  }
  Tensor& weight_tensor = weight_iter->second;
  common_weight_loader_->Permute2D(weight_tensor, dev_rank);

  // 2. Process scale: cast BF16 -> FP32, multiply by 7 (INT4 range denormalization)
  std::string scale_name = GetScaleName(weight_name);
  auto scale_iter = device_model_weights.find(scale_name);
  if (scale_iter == device_model_weights.end()) {
    return Status(RET_RUNTIME_FAILED, fmt::format("W4A8KlxMethod: scale not found: {}", scale_name));
  }
  Tensor& scale_tensor = scale_iter->second;

  // Cast scale to FP32 if not already
  if (scale_tensor.dtype != DataType::TYPE_FP32) {
    common_weight_loader_->CastDeviceTensorType(scale_tensor, DataType::TYPE_FP32, dev_rank);
  }

  // Multiply scale by 7 (INT4 range: [-8, 7], denormalization factor)
  torch::Tensor scale_torch = common_weight_loader_->GetTorchTensorFromTensor(scale_tensor);
  scale_torch = scale_torch * 7.0f;
  device_model_weights[scale_name] = common_weight_loader_->GetTensorFromTorchTensor(scale_torch);

  // 3. Bind scale to weight
  weight_tensor.weight_scales = &device_model_weights[scale_name];

  // 4. Set weight status
  WeightStatus weight_status;
  weight_status.quant_mode = QuantMode::QUANT_W4A8_KLX;
  // After permute: weight shape is [k/2, n], real k = shape[0] * 2
  weight_status.layout["k"] = weight_tensor.shape[0] * 2;  // unpacked k dimension
  weight_status.layout["n"] = weight_tensor.shape[1];
  common_weight_loader_->GetContext()->device->SetWeightStatus(weight_name, weight_status);

  KLLM_LOG_DEBUG << fmt::format("W4A8KlxMethod::ProcessW4Weight: {} shape=[{}, {}] (packed, real_k={})", weight_name,
                                weight_tensor.shape[0], weight_tensor.shape[1], weight_tensor.shape[0] * 2);
  return Status();
}

Status W4A8KlxMethod::ProcessW4ExpertStackedWeight(std::unordered_map<std::string, Tensor>& device_model_weights,
                                                   const std::string& weight_name, int dev_rank) {
  // For stacked expert weights [num_experts, out_dim, in_dim(/2)]:
  // We don't permute 3D tensors (MoE layer handles transposition internally)
  // But we do process the scale: cast to FP32 and multiply by 7

  std::string scale_name = GetScaleName(weight_name);
  auto scale_iter = device_model_weights.find(scale_name);
  if (scale_iter == device_model_weights.end()) {
    // Scale might not exist yet for stacked weights (individual expert scales merged later)
    KLLM_LOG_DEBUG << fmt::format("W4A8KlxMethod::ProcessW4ExpertStackedWeight: scale not found (may be merged): {}",
                                  scale_name);
    return Status();
  }
  Tensor& scale_tensor = scale_iter->second;

  // Cast scale to FP32
  if (scale_tensor.dtype != DataType::TYPE_FP32) {
    common_weight_loader_->CastDeviceTensorType(scale_tensor, DataType::TYPE_FP32, dev_rank);
  }

  // Multiply scale by 7
  torch::Tensor scale_torch = common_weight_loader_->GetTorchTensorFromTensor(scale_tensor);
  scale_torch = scale_torch * 7.0f;
  device_model_weights[scale_name] = common_weight_loader_->GetTensorFromTorchTensor(scale_torch);

  // Bind scale to weight
  auto weight_iter = device_model_weights.find(weight_name);
  if (weight_iter != device_model_weights.end()) {
    weight_iter->second.weight_scales = &device_model_weights[scale_name];
  }

  // Set weight status
  WeightStatus weight_status;
  weight_status.quant_mode = QuantMode::QUANT_W4A8_KLX;
  if (weight_iter != device_model_weights.end()) {
    auto& ws = weight_iter->second.shape;
    if (ws.size() == 3) {
      weight_status.layout["k"] = ws[2] * 2;  // packed dim * 2
      weight_status.layout["n"] = ws[1];
    }
  }
  common_weight_loader_->GetContext()->device->SetWeightStatus(weight_name, weight_status);

  return Status();
}

}  // namespace ksana_llm
