/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/models/common_moe/common_moe_weight.h"
#include <numeric>
#include <regex>

namespace ksana_llm {

template <typename T>
CommonMoeWeight<T>::CommonMoeWeight(const ModelConfig& model_config, int rank, std::shared_ptr<Context> context)
    : CommonWeight<T>(model_config, rank, context) {}

template <typename T>
Status CommonMoeWeight<T>::GetExpertsIdx(const std::string& expert_name) {
  // Get the index of the moe layer and the index of each expert
  std::regex re(R"(\d+)");
  std::sregex_iterator next(expert_name.begin(), expert_name.end(), re);
  std::sregex_iterator end;
  if (next != end) {
    std::smatch match = *next;
    layer_idx_ = std::stoi(match.str());
    next++;
    match = *next;
    expert_idx_ = std::stoi(match.str());
  }
  return Status();
}

template <typename T>
Status CommonMoeWeight<T>::LoadWeightsFromFile(std::shared_ptr<BaseFileTensorLoader>& weights_loader,
                                               std::vector<std::string>& weight_name_list,
                                               std::vector<std::string>& custom_name_list) {
  CommonWeight<T>::LoadWeightsFromFile(weights_loader, weight_name_list, custom_name_list);
  GetBlockManager()->SetDeviceId(rank_);
  int num_experts = model_config_.moe_config.num_experts;
  size_t moe_inter_size_per_rank = DivRoundUp(model_config_.moe_config.moe_inter_size, tensor_para_size_);
  size_t hidden_units = model_config_.hidden_units;
  std::vector<size_t> up_gate_experts_shape = {size_t(num_experts), moe_inter_size_per_rank * 2, hidden_units};
  std::vector<size_t> down_experts_shape = {size_t(num_experts), hidden_units, moe_inter_size_per_rank};

  for (size_t idx = 0; idx < weight_name_list.size(); ++idx) {
    // moe模型权重加载说明:
    // 模型每一层up和gate相同位置的专家需要cat在一起，命名为up_gate.weight，每一层up_gate和down对应的所有专家需要stack为一个专家权重
    std::string& tensor_name = custom_name_list[idx];
    std::string& weight_name = weight_name_list[idx];
    if (tensor_name.find(".experts.") != std::string::npos) {
      if (tensor_name.find(".up_proj.") != std::string::npos || tensor_name.find(".gate_proj.") != std::string::npos) {
        STATUS_CHECK_RETURN(GetExpertsIdx(tensor_name));
        std::string up_gate_experts_name =
            "model.layers." + std::to_string(layer_idx_) + ".mlp.experts.up_gate_proj.weight";
        if (weights_map_.find(up_gate_experts_name) == weights_map_.end()) {
          tensor_manager_->AddWeightTensor(up_gate_experts_name, up_gate_experts_shape, moe_weight_data_type_);
          weights_data_type_map_[up_gate_experts_name] = moe_weight_data_type_;
        }
        // get experts.up_proj and experts.gate_proj weight's data ptr
        void* weight_ptr;
        size_t weight_size;
        std::tie(weight_ptr, weight_size) = weights_loader->GetTensor(weight_name);

        size_t expert_pitch = moe_inter_size_per_rank * hidden_units * GetTypeSize(moe_weight_data_type_);
        size_t double_expert_pitch = expert_pitch * 2;
        size_t src_upgate_offset = rank_;
        src_upgate_offset *= expert_pitch;
        Tensor& up_gate_experts_tensor = weights_map_[up_gate_experts_name];
        if (tensor_name.find(".up_proj.") != std::string::npos) {
          MemcpyAsync(up_gate_experts_tensor.GetPtr<void>() + expert_idx_ * double_expert_pitch,
                      weight_ptr + src_upgate_offset, expert_pitch, MEMCPY_HOST_TO_DEVICE,
                      context_->GetMemoryManageStreams()[rank_]);
        } else if (tensor_name.find(".gate_proj.") != std::string::npos) {
          MemcpyAsync(up_gate_experts_tensor.GetPtr<void>() + expert_pitch + expert_idx_ * double_expert_pitch,
                      weight_ptr + src_upgate_offset, expert_pitch, MEMCPY_HOST_TO_DEVICE,
                      context_->GetMemoryManageStreams()[rank_]);
        }
      }
      if (tensor_name.find(".down_proj.") != std::string::npos) {
        STATUS_CHECK_RETURN(GetExpertsIdx(tensor_name));
        std::string down_experts_name = "model.layers." + std::to_string(layer_idx_) + ".mlp.experts.down_proj.weight";
        if (weights_map_.find(down_experts_name) == weights_map_.end()) {
          tensor_manager_->AddWeightTensor(down_experts_name, down_experts_shape, moe_weight_data_type_);
          weights_data_type_map_[down_experts_name] = moe_weight_data_type_;
        }
        // get experts.down_proj weight's data ptr
        void* down_weight_ptr;
        size_t down_weight_size;
        std::tie(down_weight_ptr, down_weight_size) = weights_loader->GetTensor(weight_name);

        size_t src_down_offset = rank_;
        size_t dst_pitch = moe_inter_size_per_rank * GetTypeSize(moe_weight_data_type_);
        size_t src_pitch = moe_inter_size_per_rank * tensor_para_size_ * GetTypeSize(moe_weight_data_type_);
        size_t expert_pitch = moe_inter_size_per_rank * hidden_units * GetTypeSize(moe_weight_data_type_);
        src_down_offset *= dst_pitch;
        Tensor& down_expert_tensor = weights_map_[down_experts_name];
        Memcpy2DAsync(down_expert_tensor.GetPtr<void>() + expert_idx_ * expert_pitch, dst_pitch,
                      down_weight_ptr + src_down_offset, src_pitch, dst_pitch, hidden_units, MEMCPY_HOST_TO_DEVICE,
                      context_->GetMemoryManageStreams()[rank_]);
      }
    }
  }
  return Status();
}

template <typename T>
Status CommonMoeWeight<T>::PermuteGatingWeight(Tensor& last_gating_tensor, const int num_layer,
                                               const bool is_share_gating) {
  GetBlockManager()->SetDeviceId(rank_);
  for (int layer_idx = 0; layer_idx < num_layer; ++layer_idx) {
    std::string gating_name = "model.layers." + std::to_string(layer_idx) + ".mlp.gate.weight";
    if (is_share_gating) {
      gating_name = "model.layers." + std::to_string(layer_idx) + ".mlp.shared_expert_gate.weight";
    }
    CommonWeight<T>::CommonPermuteWeight(gating_name, last_gating_tensor);
  }
  return Status();
}

template <typename T>
Status CommonMoeWeight<T>::PermuteShareMLPWeight(Tensor& last_share_down_up_tensor, Tensor& last_share_gate_tensor,
                                                 const int num_layer) {
  GetBlockManager()->SetDeviceId(rank_);
  for (int layer_idx = 0; layer_idx < num_layer; ++layer_idx) {
    std::string share_down_proj_name =
        "model.layers." + std::to_string(layer_idx) + ".mlp.shared_expert.down_proj.weight";
    CommonWeight<T>::CommonPermuteWeight(share_down_proj_name, last_share_down_up_tensor);

    std::string share_gate_proj_name =
        "model.layers." + std::to_string(layer_idx) + ".mlp.shared_expert.gate_proj.weight";
    CommonWeight<T>::CommonPermuteWeight(share_gate_proj_name, last_share_gate_tensor);

    std::string share_up_proj_name = "model.layers." + std::to_string(layer_idx) + ".mlp.shared_expert.up_proj.weight";
    // up_proj is optional
    if (weights_map_.find(share_up_proj_name) != weights_map_.end()) {
      CommonWeight<T>::CommonPermuteWeight(share_up_proj_name, last_share_down_up_tensor);
    }
  }
  return Status();
}

template <typename T>
void CommonMoeWeight<T>::ProcessWeights() {
  CommonWeight<T>::ProcessWeights();
  int num_layers = model_config_.num_layer;

  // Permute Gating Weight
  tensor_manager_->CreateTensorWithSameShape("model.layers.0.mlp.gate.weight", "empty_gating_tensor");
  Tensor& last_gating_tensor = weights_map_["empty_gating_tensor"];
  PermuteGatingWeight(last_gating_tensor, num_layers, false);
  GetBlockManager()->FreeContiguous(last_gating_tensor.GetBlockId());
  weights_map_.erase("empty_gating_tensor");

  if (model_config_.has_shared_experts) {
    // Permute  Share Gating Weight
    tensor_manager_->CreateTensorWithSameShape("model.layers.0.mlp.shared_expert_gate.weight",
                                               "empty_share_gating_tensor");
    Tensor& last_share_gating_tensor = weights_map_["empty_share_gating_tensor"];
    PermuteGatingWeight(last_share_gating_tensor, num_layers, true);
    GetBlockManager()->FreeContiguous(last_share_gating_tensor.GetBlockId());
    weights_map_.erase("empty_share_gating_tensor");
    // Permute  Share MLP Weight
    tensor_manager_->CreateTensorWithSameShape("model.layers.0.mlp.shared_expert.down_proj.weight",
                                               "empty_share_down_up_tensor");
    tensor_manager_->CreateTensorWithSameShape("model.layers.0.mlp.shared_expert.gate_proj.weight",
                                               "empty_share_gate_tensor");
    Tensor& last_share_down_up_tensor = weights_map_["empty_share_down_up_tensor"];
    Tensor& last_share_gate_tensor = weights_map_["empty_share_gate_tensor"];
    PermuteShareMLPWeight(last_share_down_up_tensor, last_share_gate_tensor, num_layers);
    GetBlockManager()->FreeContiguous(last_share_down_up_tensor.GetBlockId());
    GetBlockManager()->FreeContiguous(last_share_gate_tensor.GetBlockId());

    weights_map_.erase("empty_share_down_up_tensor");
    weights_map_.erase("empty_share_gate_tensor");
  }
}

template class CommonMoeWeight<float>;
template class CommonMoeWeight<float16>;
#ifdef ENABLE_BFLOAT16
template class CommonMoeWeight<bfloat16>;
#endif

}  // namespace ksana_llm
