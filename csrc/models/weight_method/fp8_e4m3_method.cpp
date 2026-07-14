/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "models/weight_method/fp8_e4m3_method.h"

#include <torch/torch.h>

#include "models/base/utils.h"
#include "utils/ret_code.h"

#ifdef ENABLE_CUDA
#  include "kernels/nvidia/kernel_wrapper.h"
#endif

namespace ksana_llm {

Fp8E4m3Method::Fp8E4m3Method(std::shared_ptr<CommonModelWeightLoader> common_weight_loader, int tp,
                             const QuantConfig& quant_config)
    : common_weight_loader_(common_weight_loader),
      tp_(tp),
      is_checkpoint_fp8_serialized_(quant_config.is_checkpoint_fp8_serialized),
      is_activation_scheme_static_(quant_config.is_activation_scheme_static) {}

Status Fp8E4m3Method::LoadWeightWithTPSplit(std::unordered_map<std::string, Tensor>& device_model_weights,
                                            const std::string& weight_name, const Tensor& weight_tensor, int dev_rank,
                                            bool is_row_parallel) {
  TensorParallelMode tp_mode = is_row_parallel ? TensorParallelMode::RowPara : TensorParallelMode::ColPara;
  device_model_weights[weight_name] = common_weight_loader_->TensorParallelSplit(weight_tensor, dev_rank, tp_, tp_mode);
  return Status();
}

Status Fp8E4m3Method::LoadScaleTensor(std::unordered_map<std::string, Tensor>& device_model_weights,
                                      const std::string& weight_name, const Tensor& weight_tensor, int dev_rank) {
  device_model_weights[weight_name] = common_weight_loader_->MoveToDevice(weight_tensor, dev_rank);
  return Status();
}

Status Fp8E4m3Method::load_attn_q_k_v_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                                           const std::string& weight_name, const Tensor& weight_tensor, int dev_rank) {
  if (CheckWeightNameEndMatched(weight_name, {".weight_scale", ".input_scale"})) {
    return LoadScaleTensor(device_model_weights, weight_name, weight_tensor, dev_rank);
  } else if (CheckWeightNameEndMatched(weight_name, {".weight", ".bias"})) {
    return LoadWeightWithTPSplit(device_model_weights, weight_name, weight_tensor, dev_rank, true);
  }
  return Status();
}
Status Fp8E4m3Method::load_attn_o_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                                       const std::string& weight_name, const Tensor& weight_tensor, int dev_rank) {
  if (CheckWeightNameEndMatched(weight_name, {".weight_scale", ".input_scale"})) {
    return LoadScaleTensor(device_model_weights, weight_name, weight_tensor, dev_rank);
  } else if (CheckWeightNameEndMatched(weight_name, {".weight"})) {
    return LoadWeightWithTPSplit(device_model_weights, weight_name, weight_tensor, dev_rank, false);
  }
  return Status();
}

Status Fp8E4m3Method::load_mlp_gate_up_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                                            const std::string& weight_name, const Tensor& weight_tensor, int dev_rank) {
  if (CheckWeightNameEndMatched(weight_name, {".weight_scale", ".input_scale"})) {
    return LoadScaleTensor(device_model_weights, weight_name, weight_tensor, dev_rank);
  } else if (CheckWeightNameEndMatched(weight_name, {".weight"})) {
    return LoadWeightWithTPSplit(device_model_weights, weight_name, weight_tensor, dev_rank, true);
  }
  return Status();
}

Status Fp8E4m3Method::load_mlp_down_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                                         const std::string& weight_name, const Tensor& weight_tensor, int dev_rank) {
  if (CheckWeightNameEndMatched(weight_name, {".weight_scale", ".input_scale"})) {
    return LoadScaleTensor(device_model_weights, weight_name, weight_tensor, dev_rank);
  } else if (CheckWeightNameEndMatched(weight_name, {".weight"})) {
    return LoadWeightWithTPSplit(device_model_weights, weight_name, weight_tensor, dev_rank, false);
  }
  return Status();
}

Status Fp8E4m3Method::ConvertWeightToFp8(std::unordered_map<std::string, Tensor>& device_model_weights,
                                         const std::string& weight_name, int dev_rank) {
  // Replace weight tensor with quantized tensor in device_model_weights,
  // add scale tensor to device_model_weights and bind it to weight tensor
#ifdef ENABLE_CUDA
  if (device_model_weights.find(weight_name) == device_model_weights.end()) {
    KLLM_LOG_WARNING << fmt::format("Weight {} not found", weight_name);
  }

  Tensor& weight_tensor = device_model_weights[weight_name];
  auto weight_shape = std::vector<size_t>(weight_tensor.shape);

  // Create quantized tensor and scale tensor
  std::string scale_name = weight_name.substr(0, weight_name.rfind(".weight")) + ".weight_scale";
  Tensor quant_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP8_E4M3, weight_shape);
  Tensor scale_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {1});

  // Quantize to FP8
  size_t num_elements = weight_shape[0] * weight_shape[1];

  // Determine input type and call appropriate quantize function
  if (weight_tensor.dtype == TYPE_FP16) {
    Fp8E4m3Quantize<half>(1, num_elements, static_cast<const half*>(weight_tensor.GetPtr<void>()),
                          quant_tensor.GetPtr<void>(), static_cast<float*>(scale_tensor.GetPtr<void>()), false,
                          common_weight_loader_->GetContext()->device->GetMemoryManageStream().Get());
  } else if (weight_tensor.dtype == TYPE_BF16) {
    Fp8E4m3Quantize<__nv_bfloat16>(1, num_elements, static_cast<const __nv_bfloat16*>(weight_tensor.GetPtr<void>()),
                                   quant_tensor.GetPtr<void>(), static_cast<float*>(scale_tensor.GetPtr<void>()), false,
                                   common_weight_loader_->GetContext()->device->GetMemoryManageStream().Get());
  } else {
    return Status(RET_RUNTIME_FAILED, "Unsupported weight dtype for FP8 quantization");
  }

  // Replace original weight with quantized weight and bind scale to it
  device_model_weights[weight_name] = quant_tensor;
  device_model_weights[scale_name] = scale_tensor;
  device_model_weights[weight_name].weight_scales = &device_model_weights[scale_name];

  return Status();
#else
  return Status(RET_RUNTIME_FAILED, "FP8 quantization is only supported on CUDA devices");
#endif
}

/*
 * Process projection weights for FP8 quantization.
 * - Dynamic FP8: convert weight to FP8 and bind weight scale to it
 * - Static FP8: bind weight scale and input scale to it
 */
Status Fp8E4m3Method::ProcessProjWeight(std::unordered_map<std::string, Tensor>& device_model_weights,
                                        const std::string& weight_prefix_name, int dev_rank) {
  const std::string weight_name = weight_prefix_name + "weight";
  const std::string weight_scale_name = weight_prefix_name + "weight_scale";
  const std::string input_scale_name = weight_prefix_name + "input_scale";

  if (IsCheckpointFp8Serialized()) {
    // Static FP8: bind weight scale
    if (device_model_weights.find(weight_scale_name) != device_model_weights.end()) {
      device_model_weights[weight_name].weight_scales = &device_model_weights[weight_scale_name];
    } else {
      return Status(RET_MODEL_QUANT_FAILED, "weight_scale not found: " + weight_scale_name);
    }
    // Static FP8: bind input scale
    if (device_model_weights.find(input_scale_name) != device_model_weights.end()) {
      device_model_weights[weight_name].input_scales = &device_model_weights[input_scale_name];
    } else {
      return Status(RET_MODEL_QUANT_FAILED, "input_scale not found: " + input_scale_name);
    }
  } else {
    // Dynamic FP8: convert weight to FP8 and bind weight scale to it
    ConvertWeightToFp8(device_model_weights, weight_name, dev_rank);
  }
  return Status();
}

Status Fp8E4m3Method::process_attn_qkv_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                                            const std::string& weight_prefix_name, int dev_rank) {
  const std::string q_proj_name = weight_prefix_name;
  const std::string k_proj_name = WeightNameReplace(weight_prefix_name, "q_proj", "k_proj");
  const std::string v_proj_name = WeightNameReplace(weight_prefix_name, "q_proj", "v_proj");
  const std::string qkv_proj_name = WeightNameReplace(weight_prefix_name, "q_proj", "query_key_value");

  if (IsCheckpointFp8Serialized()) {
    const std::string q_weight_scale_name = q_proj_name + "weight_scale";
    const std::string k_weight_scale_name = k_proj_name + "weight_scale";
    const std::string v_weight_scale_name = v_proj_name + "weight_scale";
    const std::string qkv_weight_scale_name = qkv_proj_name + "weight_scale";

    const std::string q_input_scale_name = q_proj_name + "input_scale";
    const std::string k_input_scale_name = k_proj_name + "input_scale";
    const std::string v_input_scale_name = v_proj_name + "input_scale";
    const std::string qkv_input_scale_name = qkv_proj_name + "input_scale";
    // Use max of q/k/v_proj weight scales as the merged scale, and rescale q/k/v_proj weights
    if (device_model_weights.find(q_weight_scale_name) != device_model_weights.end() &&
        device_model_weights.find(k_weight_scale_name) != device_model_weights.end() &&
        device_model_weights.find(v_weight_scale_name) != device_model_weights.end()) {
      Tensor qkv_scale(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {1});
      device_model_weights[qkv_weight_scale_name] = qkv_scale;

      // Get max scale using torch
      torch::Tensor q_scale =
          common_weight_loader_->GetTorchTensorFromTensor(device_model_weights[q_weight_scale_name]);
      torch::Tensor k_scale =
          common_weight_loader_->GetTorchTensorFromTensor(device_model_weights[k_weight_scale_name]);
      torch::Tensor v_scale =
          common_weight_loader_->GetTorchTensorFromTensor(device_model_weights[v_weight_scale_name]);
      torch::Tensor qkv_scale_tensor =
          common_weight_loader_->GetTorchTensorFromTensor(device_model_weights[qkv_weight_scale_name]);
      torch::max_out(qkv_scale_tensor, q_scale, k_scale);
      torch::max_out(qkv_scale_tensor, qkv_scale_tensor, v_scale);

      Tensor& q_weight = device_model_weights[q_proj_name + "weight"];
      Tensor& k_weight = device_model_weights[k_proj_name + "weight"];
      Tensor& v_weight = device_model_weights[v_proj_name + "weight"];

      // Rescale q/k/v_proj weights
      RescaleFp8E4m3(q_weight.GetPtr<void>(), q_weight.GetPtr<void>(), q_weight.GetElementNumber(),
                     q_scale.data_ptr<float>(), qkv_scale_tensor.data_ptr<float>(),
                     common_weight_loader_->GetContext()->device->GetMemoryManageStream().Get());
      RescaleFp8E4m3(k_weight.GetPtr<void>(), k_weight.GetPtr<void>(), k_weight.GetElementNumber(),
                     k_scale.data_ptr<float>(), qkv_scale_tensor.data_ptr<float>(),
                     common_weight_loader_->GetContext()->device->GetMemoryManageStream().Get());
      RescaleFp8E4m3(v_weight.GetPtr<void>(), v_weight.GetPtr<void>(), v_weight.GetElementNumber(),
                     v_scale.data_ptr<float>(), qkv_scale_tensor.data_ptr<float>(),
                     common_weight_loader_->GetContext()->device->GetMemoryManageStream().Get());

      // Merge q/k/v_proj weight
      common_weight_loader_->AutoMergeWeight({q_proj_name + "weight", k_proj_name + "weight", v_proj_name + "weight"},
                                             qkv_proj_name + "weight", device_model_weights, dev_rank);

      // Remove q/k/v_proj weight scales from device_model_weights
      device_model_weights.erase(q_weight_scale_name);
      device_model_weights.erase(k_weight_scale_name);
      device_model_weights.erase(v_weight_scale_name);
    }
    // Use max of q/k/v_proj input scales as the merged scale
    if (device_model_weights.find(q_input_scale_name) != device_model_weights.end() &&
        device_model_weights.find(k_input_scale_name) != device_model_weights.end() &&
        device_model_weights.find(v_input_scale_name) != device_model_weights.end()) {
      Tensor qkv_input_scale(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {1});
      device_model_weights[qkv_input_scale_name] = qkv_input_scale;

      torch::Tensor q_scale = common_weight_loader_->GetTorchTensorFromTensor(device_model_weights[q_input_scale_name]);
      torch::Tensor k_scale = common_weight_loader_->GetTorchTensorFromTensor(device_model_weights[k_input_scale_name]);
      torch::Tensor v_scale = common_weight_loader_->GetTorchTensorFromTensor(device_model_weights[v_input_scale_name]);
      torch::Tensor qkv_scale_tensor =
          common_weight_loader_->GetTorchTensorFromTensor(device_model_weights[qkv_input_scale_name]);
      torch::max_out(qkv_scale_tensor, q_scale, k_scale);
      torch::max_out(qkv_scale_tensor, qkv_scale_tensor, v_scale);

      device_model_weights.erase(q_input_scale_name);
      device_model_weights.erase(k_input_scale_name);
      device_model_weights.erase(v_input_scale_name);
    }

    // Bind weight scale and input scale to merged qkv_proj weight
    ProcessProjWeight(device_model_weights, qkv_proj_name, dev_rank);
  } else {
    // Merge q/k/v_proj weight
    common_weight_loader_->AutoMergeWeight({q_proj_name + "weight", k_proj_name + "weight", v_proj_name + "weight"},
                                           qkv_proj_name + "weight", device_model_weights, dev_rank);
    // Dynamic FP8: convert merged qkv_proj weight to FP8 and bind weight scale to it
    ProcessProjWeight(device_model_weights, qkv_proj_name, dev_rank);
  }

  return Status();
}

Status Fp8E4m3Method::process_attn_o_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                                          const std::string& weight_prefix_name, int dev_rank) {
  return ProcessProjWeight(device_model_weights, weight_prefix_name, dev_rank);
}

Status Fp8E4m3Method::process_mlp_gate_up_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                                               const std::string& weight_prefix_name, int dev_rank) {
  const std::string gate_proj_name = weight_prefix_name;
  const std::string up_proj_name = WeightNameReplace(weight_prefix_name, "gate_proj", "up_proj");
  const std::string gate_up_proj_name = WeightNameReplace(weight_prefix_name, "gate_proj", "gate_up_proj");

  if (IsCheckpointFp8Serialized()) {
    const std::string gate_weight_scale_name = gate_proj_name + "weight_scale";
    const std::string up_weight_scale_name = up_proj_name + "weight_scale";
    const std::string gate_up_weight_scale_name = gate_up_proj_name + "weight_scale";

    const std::string gate_input_scale_name = gate_proj_name + "input_scale";
    const std::string up_input_scale_name = up_proj_name + "input_scale";
    const std::string gate_up_input_scale_name = gate_up_proj_name + "input_scale";

    // Use max of gate/up weight scales as the merged weight scale，and rescale gate/up weights
    if (device_model_weights.find(gate_weight_scale_name) != device_model_weights.end() &&
        device_model_weights.find(up_weight_scale_name) != device_model_weights.end() &&
        device_model_weights.find(gate_up_weight_scale_name) == device_model_weights.end()) {
      Tensor gate_up_weight_scale(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {1});
      device_model_weights[gate_up_weight_scale_name] = gate_up_weight_scale;

      torch::Tensor gate_weight_scale =
          common_weight_loader_->GetTorchTensorFromTensor(device_model_weights[gate_weight_scale_name]);
      torch::Tensor up_weight_scale =
          common_weight_loader_->GetTorchTensorFromTensor(device_model_weights[up_weight_scale_name]);
      torch::Tensor gate_up_weight_scale_tensor =
          common_weight_loader_->GetTorchTensorFromTensor(device_model_weights[gate_up_weight_scale_name]);
      torch::max_out(gate_up_weight_scale_tensor, gate_weight_scale, up_weight_scale);

      Tensor& gate_weight = device_model_weights[gate_proj_name + "weight"];
      Tensor& up_weight = device_model_weights[up_proj_name + "weight"];

      // Rescale gate/up weights
      RescaleFp8E4m3(gate_weight.GetPtr<void>(), gate_weight.GetPtr<void>(), gate_weight.GetElementNumber(),
                     gate_weight_scale.data_ptr<float>(), gate_up_weight_scale_tensor.data_ptr<float>(),
                     common_weight_loader_->GetContext()->device->GetMemoryManageStream().Get());
      RescaleFp8E4m3(up_weight.GetPtr<void>(), up_weight.GetPtr<void>(), up_weight.GetElementNumber(),
                     up_weight_scale.data_ptr<float>(), gate_up_weight_scale_tensor.data_ptr<float>(),
                     common_weight_loader_->GetContext()->device->GetMemoryManageStream().Get());

      // Merge gate/up weights
      common_weight_loader_->AutoMergeWeight({gate_proj_name + "weight", up_proj_name + "weight"},
                                             gate_up_proj_name + "weight", device_model_weights, dev_rank);

      device_model_weights.erase(gate_weight_scale_name);
      device_model_weights.erase(up_weight_scale_name);
    }

    // Use max of gate/up input scales as the merged input scale
    if (device_model_weights.find(gate_input_scale_name) != device_model_weights.end() &&
        device_model_weights.find(up_input_scale_name) != device_model_weights.end() &&
        device_model_weights.find(gate_up_input_scale_name) == device_model_weights.end()) {
      Tensor gate_up_input_scale(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {1});
      device_model_weights[gate_up_input_scale_name] = gate_up_input_scale;

      torch::Tensor gate_input_scale =
          common_weight_loader_->GetTorchTensorFromTensor(device_model_weights[gate_input_scale_name]);
      torch::Tensor up_input_scale =
          common_weight_loader_->GetTorchTensorFromTensor(device_model_weights[up_input_scale_name]);
      torch::Tensor gate_up_input_scale_tensor =
          common_weight_loader_->GetTorchTensorFromTensor(device_model_weights[gate_up_input_scale_name]);
      torch::max_out(gate_up_input_scale_tensor, gate_input_scale, up_input_scale);

      device_model_weights.erase(gate_input_scale_name);
      device_model_weights.erase(up_input_scale_name);
    }
    // Bind weight scale and input scale to merged gate_up_proj weight
    ProcessProjWeight(device_model_weights, gate_up_proj_name, dev_rank);
  } else {
    // Merge gate/up weights
    common_weight_loader_->AutoMergeWeight({gate_proj_name + "weight", up_proj_name + "weight"},
                                           gate_up_proj_name + "weight", device_model_weights, dev_rank);
    // Dynamic FP8: convert merged gate_up_proj weight to FP8 and bind weight scale to it
    ProcessProjWeight(device_model_weights, gate_up_proj_name, dev_rank);
  }

  return Status();
}

Status Fp8E4m3Method::process_mlp_down_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                                            const std::string& weight_prefix_name, int dev_rank) {
  return ProcessProjWeight(device_model_weights, weight_prefix_name, dev_rank);
}

// only support static fp8 with both weight scale and input scale.
Status Fp8E4m3Method::process_experts_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                                           const std::string& weight_prefix_name, int dev_rank, int num_experts) {
  // process up_gate_proj.
  // Step 1: Create up_gate input_scale and weight_scale tensors.
  const std::string up_gate_input_scale_name = weight_prefix_name + "up_gate_proj.input_scale";
  const std::string up_gate_weight_scale_name = weight_prefix_name + "up_gate_proj.weight_scale";

  Tensor up_gate_input_scale(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {1});
  Tensor up_gate_weight_scale(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {static_cast<size_t>(num_experts)});
  device_model_weights[up_gate_input_scale_name] = up_gate_input_scale;
  device_model_weights[up_gate_weight_scale_name] = up_gate_weight_scale;

  torch::Tensor up_gate_input_scale_tensor =
      common_weight_loader_->GetTorchTensorFromTensor(device_model_weights[up_gate_input_scale_name]);
  torch::Tensor up_gate_weight_scale_tensor =
      common_weight_loader_->GetTorchTensorFromTensor(device_model_weights[up_gate_weight_scale_name]);

  // Step 2: Process each expert's gate_proj and up_proj.
  // - rescale weights with max value of gate_proj and up_proj's weight_scales.
  // - collect weight names for merging.
  std::vector<std::string> expert_up_gate_names;
  const std::string first_gate_proj_name = fmt::format("{}0.gate_proj.", weight_prefix_name);
  Tensor& first_gate_weight = device_model_weights[first_gate_proj_name + "weight"];
  const size_t inter_size = first_gate_weight.shape[0];
  const size_t hidden_units = first_gate_weight.shape[1];

  for (int expert_idx = 0; expert_idx < num_experts; expert_idx++) {
    const std::string gate_proj_name = fmt::format("{}{}.gate_proj.", weight_prefix_name, expert_idx);
    const std::string up_proj_name = fmt::format("{}{}.up_proj.", weight_prefix_name, expert_idx);

    const std::string gate_weight_scale_name = gate_proj_name + "weight_scale";
    const std::string up_weight_scale_name = up_proj_name + "weight_scale";
    const std::string gate_input_scale_name = gate_proj_name + "input_scale";
    const std::string up_input_scale_name = up_proj_name + "input_scale";

    // Get max value of gate_proj and up_proj's weight_scales for each expert.
    float* gate_weight_scale = static_cast<float*>(device_model_weights[gate_weight_scale_name].GetPtr<void>());
    float* up_weight_scale = static_cast<float*>(device_model_weights[up_weight_scale_name].GetPtr<void>());
    float* expert_weight_scale = up_gate_weight_scale_tensor.data_ptr<float>() + expert_idx;

    Max(expert_weight_scale, gate_weight_scale, up_weight_scale, 1,
        common_weight_loader_->GetContext()->device->GetDeviceId());

    // Rescale gate_proj and up_proj weights.
    Tensor& gate_weight = device_model_weights[gate_proj_name + "weight"];
    Tensor& up_weight = device_model_weights[up_proj_name + "weight"];
    RescaleFp8E4m3(gate_weight.GetPtr<void>(), gate_weight.GetPtr<void>(), gate_weight.GetElementNumber(),
                   gate_weight_scale, expert_weight_scale,
                   common_weight_loader_->GetContext()->device->GetMemoryManageStream().Get());
    RescaleFp8E4m3(up_weight.GetPtr<void>(), up_weight.GetPtr<void>(), up_weight.GetElementNumber(), up_weight_scale,
                   expert_weight_scale, common_weight_loader_->GetContext()->device->GetMemoryManageStream().Get());

    // Collect weight names for merging (order: [up, gate] for each expert).
    expert_up_gate_names.push_back(up_proj_name + "weight");
    expert_up_gate_names.push_back(gate_proj_name + "weight");

    // Get max value of gate_proj and up_proj's input_scales for each layer.
    if (device_model_weights.find(gate_input_scale_name) != device_model_weights.end() &&
        device_model_weights.find(up_input_scale_name) != device_model_weights.end()) {
      float* gate_input_scale = static_cast<float*>(device_model_weights[gate_input_scale_name].GetPtr<void>());
      float* up_input_scale = static_cast<float*>(device_model_weights[up_input_scale_name].GetPtr<void>());
      float* merged_input_scale = up_gate_input_scale_tensor.data_ptr<float>();

      if (expert_idx == 0) {
        Max(merged_input_scale, gate_input_scale, up_input_scale, 1,
            common_weight_loader_->GetContext()->device->GetDeviceId());
      } else {
        Max(merged_input_scale, merged_input_scale, gate_input_scale, 1,
            common_weight_loader_->GetContext()->device->GetDeviceId());
        Max(merged_input_scale, merged_input_scale, up_input_scale, 1,
            common_weight_loader_->GetContext()->device->GetDeviceId());
      }

      device_model_weights.erase(gate_input_scale_name);
      device_model_weights.erase(up_input_scale_name);
    }

    device_model_weights.erase(gate_weight_scale_name);
    device_model_weights.erase(up_weight_scale_name);
  }

  // Step 3: Concat all experts' gate_proj weights and up_proj weights on dim=0.
  const std::string up_gate_weight_name = weight_prefix_name + "up_gate_proj.weight";
  common_weight_loader_->AutoMergeWeight(expert_up_gate_names, up_gate_weight_name, device_model_weights, dev_rank);
  // Reshape from [num_experts * 2 * inter_size, hidden_units] to [num_experts, 2 * inter_size, hidden_units]
  device_model_weights[up_gate_weight_name].shape = {static_cast<size_t>(num_experts), 2 * inter_size, hidden_units};

  // Step 4: Update weight_scale: weight_scale[expert_idx] *= input_scale.
  float* up_gate_weight_scale_base = up_gate_weight_scale_tensor.data_ptr<float>();
  float* merged_input_scale = up_gate_input_scale_tensor.data_ptr<float>();
  for (int expert_idx = 0; expert_idx < num_experts; expert_idx++) {
    InvokeMul(up_gate_weight_scale_base + expert_idx, merged_input_scale, up_gate_weight_scale_base + expert_idx, 1,
              common_weight_loader_->GetContext()->device->GetDeviceId());
  }

  // Step 5: Bind scales to up_gate weight.
  device_model_weights[up_gate_weight_name].weight_scales = &device_model_weights[up_gate_weight_scale_name];
  device_model_weights[up_gate_weight_name].input_scales = &device_model_weights[up_gate_input_scale_name];

  // Process down_proj.
  // Step 1: Create merged input_scale and weight_scale tensors.
  const std::string merged_down_input_scale_name = weight_prefix_name + "down_proj.input_scale";
  const std::string merged_down_weight_scale_name = weight_prefix_name + "down_proj.weight_scale";

  Tensor merged_down_input_scale(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {1});
  Tensor merged_down_weight_scale(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {static_cast<size_t>(num_experts)});
  device_model_weights[merged_down_input_scale_name] = merged_down_input_scale;
  device_model_weights[merged_down_weight_scale_name] = merged_down_weight_scale;

  torch::Tensor merged_down_input_scale_tensor =
      common_weight_loader_->GetTorchTensorFromTensor(device_model_weights[merged_down_input_scale_name]);
  torch::Tensor merged_down_weight_scale_tensor =
      common_weight_loader_->GetTorchTensorFromTensor(device_model_weights[merged_down_weight_scale_name]);

  // Step 2: Process each expert's down_proj.
  // - collect weight names for merging.
  std::vector<std::string> expert_down_names;
  const std::string first_down_proj_name = fmt::format("{}0.down_proj.", weight_prefix_name);
  Tensor& first_down_weight = device_model_weights[first_down_proj_name + "weight"];
  const size_t down_hidden_units = first_down_weight.shape[0];
  const size_t down_inter_size = first_down_weight.shape[1];

  for (int expert_idx = 0; expert_idx < num_experts; expert_idx++) {
    const std::string down_proj_name = fmt::format("{}{}.down_proj.", weight_prefix_name, expert_idx);
    const std::string down_weight_scale_name = down_proj_name + "weight_scale";
    const std::string down_input_scale_name = down_proj_name + "input_scale";

    // Copy down_proj's weight_scale for each expert.
    float* down_weight_scale = static_cast<float*>(device_model_weights[down_weight_scale_name].GetPtr<void>());
    float* expert_weight_scale = merged_down_weight_scale_tensor.data_ptr<float>() + expert_idx;
    cudaMemcpy(expert_weight_scale, down_weight_scale, sizeof(float), cudaMemcpyDeviceToDevice);

    // Collect names for merging.
    expert_down_names.push_back(down_proj_name + "weight");

    // Get max value of down_proj's input_scale for each layer.
    if (device_model_weights.find(down_input_scale_name) != device_model_weights.end()) {
      float* down_input_scale = static_cast<float*>(device_model_weights[down_input_scale_name].GetPtr<void>());
      float* merged_input_scale = merged_down_input_scale_tensor.data_ptr<float>();

      if (expert_idx == 0) {
        cudaMemcpy(merged_input_scale, down_input_scale, sizeof(float), cudaMemcpyDeviceToDevice);
      } else {
        Max(merged_input_scale, merged_input_scale, down_input_scale, 1,
            common_weight_loader_->GetContext()->device->GetDeviceId());
      }

      device_model_weights.erase(down_input_scale_name);
    }

    device_model_weights.erase(down_weight_scale_name);
  }

  // Step 3: Concat all experts' down_proj weights on dim=0.
  const std::string merged_down_name = weight_prefix_name + "down_proj.weight";
  common_weight_loader_->AutoMergeWeight(expert_down_names, merged_down_name, device_model_weights, dev_rank);
  // Reshape from [num_experts * hidden_units, inter_size] to [num_experts, hidden_units, inter_size].
  device_model_weights[merged_down_name].shape = {static_cast<size_t>(num_experts), down_hidden_units, down_inter_size};

  // Step 4: Update weight_scale: weight_scale[expert_idx] *= input_scale.
  float* down_weight_scale = merged_down_weight_scale_tensor.data_ptr<float>();
  float* down_input_scale = merged_down_input_scale_tensor.data_ptr<float>();
  for (int expert_idx = 0; expert_idx < num_experts; expert_idx++) {
    InvokeMul(down_weight_scale + expert_idx, down_input_scale, down_weight_scale + expert_idx, 1,
              common_weight_loader_->GetContext()->device->GetDeviceId());
  }

  // Take reciprocal of down_input_scale.
  Reciprocal(down_input_scale, down_input_scale, 1, common_weight_loader_->GetContext()->device->GetDeviceId());

  // Step 5: Bind scales to merged weight.
  device_model_weights[merged_down_name].weight_scales = &device_model_weights[merged_down_weight_scale_name];
  device_model_weights[merged_down_name].input_scales = &device_model_weights[merged_down_input_scale_name];

  return Status();
}

}  // namespace ksana_llm
