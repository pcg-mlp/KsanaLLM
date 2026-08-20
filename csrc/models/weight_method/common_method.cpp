/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "models/weight_method/common_method.h"

#include <torch/torch.h>

#include "model_loader/model_loader_utils.h"
#include "models/base/utils.h"

namespace ksana_llm {

Status CommonMethod::load_attn_q_k_v_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                                          const std::string& weight_name, const Tensor& weight_tensor, int dev_rank) {
  if (CheckWeightNameEndMatched(weight_name, {".weight"})) {
    device_model_weights[weight_name] =
        common_weight_loader_->TensorParallelSplit(weight_tensor, dev_rank, tp_, TensorParallelMode::RowPara);
  }
  if (CheckWeightNameEndMatched(weight_name, {".bias"})) {
    device_model_weights[weight_name] =
        common_weight_loader_->TensorParallelSplit(weight_tensor, dev_rank, tp_, TensorParallelMode::RowPara);
  }
  return Status();
}

Status CommonMethod::load_attn_o_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                                      const std::string& weight_name, const Tensor& weight_tensor, int dev_rank) {
  if (CheckWeightNameEndMatched(weight_name, {".weight"})) {
    device_model_weights[weight_name] =
        common_weight_loader_->TensorParallelSplit(weight_tensor, dev_rank, tp_, TensorParallelMode::ColPara);
  }
  return Status();
}

Status CommonMethod::load_attn_norm(std::unordered_map<std::string, Tensor>& device_model_weights,
                                    const std::string& weight_name, const Tensor& weight_tensor, int dev_rank) {
  if (CheckWeightNameEndMatched(weight_name, {".weight"})) {
    device_model_weights[weight_name] = common_weight_loader_->MoveToDevice(weight_tensor, dev_rank);
  }
  return Status();
}

Status CommonMethod::load_mlp_gate_up_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                                           const std::string& weight_name, const Tensor& weight_tensor, int dev_rank) {
  if (CheckWeightNameEndMatched(weight_name, {".weight"})) {
    device_model_weights[weight_name] =
        common_weight_loader_->TensorParallelSplit(weight_tensor, dev_rank, tp_, TensorParallelMode::RowPara);
  }
  return Status();
}

Status CommonMethod::load_mlp_down_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                                        const std::string& weight_name, const Tensor& weight_tensor, int dev_rank) {
  if (CheckWeightNameEndMatched(weight_name, {".weight"})) {
    device_model_weights[weight_name] =
        common_weight_loader_->TensorParallelSplit(weight_tensor, dev_rank, tp_, TensorParallelMode::ColPara);
  }
  return Status();
}

Status CommonMethod::load_experts_gate(std::unordered_map<std::string, Tensor>& device_model_weights,
                                       const std::string& weight_name, const Tensor& weight_tensor, int dev_rank) {
  if (CheckWeightNameEndMatched(weight_name, {".weight"})) {
    device_model_weights[weight_name] = common_weight_loader_->MoveToDevice(weight_tensor, dev_rank);
  }
  return Status();
}

Status CommonMethod::load_norm(std::unordered_map<std::string, Tensor>& device_model_weights,
                               const std::string& weight_name, const Tensor& weight_tensor, int dev_rank) {
  if (CheckWeightNameEndMatched(weight_name, {".weight"})) {
    device_model_weights[weight_name] = common_weight_loader_->MoveToDevice(weight_tensor, dev_rank);
  }
  return Status();
}

Status CommonMethod::load_embed_tokens(std::unordered_map<std::string, Tensor>& device_model_weights,
                                       const std::string& weight_name, const Tensor& weight_tensor, int dev_rank) {
  if (CheckWeightNameEndMatched(weight_name, {".weight"})) {
    device_model_weights[weight_name] =
        common_weight_loader_->TensorParallelSplit(weight_tensor, dev_rank, tp_, TensorParallelMode::ColPara);
  }
  return Status();
}

Status CommonMethod::load_lm_head(std::unordered_map<std::string, Tensor>& device_model_weights,
                                  const std::string& weight_name, const Tensor& weight_tensor, int dev_rank) {
  if (CheckWeightNameEndMatched(weight_name, {".weight"})) {
    if (weight_tensor.shape[0] % tp_ == 0) {
      device_model_weights[weight_name] =
          common_weight_loader_->TensorParallelSplit(weight_tensor, dev_rank, tp_, TensorParallelMode::RowPara);
    } else {
      KLLM_LOG_DEBUG << "vocab_size can't be divided by tp_size.";
      device_model_weights[weight_name] =
          common_weight_loader_->LmHeadTensorParallelSplit(weight_tensor, dev_rank, tp_, TensorParallelMode::RowPara);
    }
  }
  return Status();
}

Status CommonMethod::process_attn_qkv_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                                           const std::string& weight_prefix_name, int dev_rank) {
  const std::string q_proj_name = weight_prefix_name;
  const std::string k_proj_name = WeightNameReplace(weight_prefix_name, "q_proj", "k_proj");
  const std::string v_proj_name = WeightNameReplace(weight_prefix_name, "q_proj", "v_proj");
  const std::string qkv_proj_name = WeightNameReplace(weight_prefix_name, "q_proj", "query_key_value");
  // 合并qkv
  {
    const std::vector<std::string> weight_suffixs = {"weight"};
    for (const std::string& weight_suffix : weight_suffixs) {
      common_weight_loader_->AutoMergeWeight(
          {q_proj_name + weight_suffix, k_proj_name + weight_suffix, v_proj_name + weight_suffix},
          qkv_proj_name + weight_suffix, device_model_weights, dev_rank);
    }
  }
  const std::string weight_name = qkv_proj_name + "weight";
  // Kunlun XPU: 保持 weight [n,k] 不转置，由 matmul_layer 设 w_trans=true，
  // 使 xBLAS 内层 k 维归约时 weight 访问 stride=1（对齐 xSGLang I28Args 优化）。
  // 其他平台：Permute2D 转为 [k,n]。
#ifndef ENABLE_KUNLUN
  common_weight_loader_->Permute2D(device_model_weights.at(weight_name), dev_rank);
#endif
  return Status();
}

Status CommonMethod::process_attn_qkv_bias(std::unordered_map<std::string, Tensor>& device_model_weights,
                                           const std::string& weight_prefix_name, int dev_rank) {
  const std::string q_bias_name = weight_prefix_name + "bias";
  const std::string k_bias_name = WeightNameReplace(q_bias_name, "q_proj", "k_proj");
  const std::string v_bias_name = WeightNameReplace(q_bias_name, "q_proj", "v_proj");
  const std::string qkv_bias_name = WeightNameReplace(q_bias_name, "q_proj", "query_key_value");
  if (device_model_weights.find(q_bias_name) != device_model_weights.end()) {
    size_t kv_head_dims = device_model_weights[k_bias_name].shape[0];
    size_t q_head_dims = device_model_weights[q_bias_name].shape[0];
    common_weight_loader_->AutoMergeWeight({q_bias_name, k_bias_name, v_bias_name}, qkv_bias_name, device_model_weights,
                                           dev_rank);
    device_model_weights.at(qkv_bias_name).shape = {1, q_head_dims / kv_head_dims + 2, kv_head_dims};
  }
  return Status();
}

Status CommonMethod::process_attn_fused_qkv_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                                                 const std::string& weight_prefix_name, int dev_rank) {
  const std::string weight_name = weight_prefix_name + "weight";
  common_weight_loader_->Permute2D(device_model_weights.at(weight_name), dev_rank);
  return Status();
}

Status CommonMethod::process_mlp_gate_up_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                                              const std::string& weight_prefix_name, int dev_rank) {
  const std::string gate_proj_name = weight_prefix_name;
  const std::string up_proj_name = WeightNameReplace(weight_prefix_name, "gate_proj", "up_proj");
  const std::string gate_up_proj_name = WeightNameReplace(weight_prefix_name, "gate_proj", "gate_up_proj");
  // 合并gate_up
  {
    const std::vector<std::string> weight_suffixs = {"weight"};
    for (const std::string& weight_suffix : weight_suffixs) {
      common_weight_loader_->AutoMergeWeight({gate_proj_name + weight_suffix, up_proj_name + weight_suffix},
                                             gate_up_proj_name + weight_suffix, device_model_weights, dev_rank);
    }
  }
  const std::string weight_name = gate_up_proj_name + "weight";
#ifndef ENABLE_KUNLUN
  common_weight_loader_->Permute2D(device_model_weights.at(weight_name), dev_rank);
#endif
  return Status();
}

Status CommonMethod::process_mlp_down_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                                           const std::string& weight_prefix_name, int dev_rank) {
  const std::string weight_name = weight_prefix_name + "weight";
#ifndef ENABLE_KUNLUN
  common_weight_loader_->Permute2D(device_model_weights.at(weight_name), dev_rank);
#endif
  return Status();
}

Status CommonMethod::process_experts_gate(std::unordered_map<std::string, Tensor>& device_model_weights,
                                          const std::string& weight_prefix_name, int dev_rank) {
  const std::string weight_name = weight_prefix_name + "weight";
#ifndef ENABLE_KUNLUN
  common_weight_loader_->Permute2D(device_model_weights.at(weight_name), dev_rank);
#endif
  return Status();
}

Status CommonMethod::process_attn_o_proj(std::unordered_map<std::string, Tensor>& device_model_weights,
                                         const std::string& weight_prefix_name, int dev_rank) {
  const std::string weight_name = weight_prefix_name + "weight";
#ifndef ENABLE_KUNLUN
  common_weight_loader_->Permute2D(device_model_weights.at(weight_name), dev_rank);
#endif
  return Status();
}

Status CommonMethod::process_lm_head(std::unordered_map<std::string, Tensor>& device_model_weights,
                                     const std::string& weight_prefix_name, int dev_rank) {
  const std::string weight_name = weight_prefix_name + "weight";
#ifndef ENABLE_KUNLUN
  common_weight_loader_->Permute2D(device_model_weights.at(weight_name), dev_rank);
#endif
  return Status();
}

}  // namespace ksana_llm
