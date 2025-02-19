/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/layers/base_layer.h"
#include "ksana_llm/layers/fp8_matmul_layer.h"
#include "ksana_llm/layers/group_matmul_layer.h"
#include "ksana_llm/layers/matmul_layer.h"
#include "ksana_llm/layers/moe_layer.h"
#include "ksana_llm/models/base/base_weight.h"

namespace ksana_llm {

template <typename T>
class MatMulLayerFactory {
 public:
  typedef std::shared_ptr<BaseLayer> (MatMulLayerFactory<T>::*BuildLayerFunc)();
  MatMulLayerFactory(std::shared_ptr<Tensor>& workspace_buffer, const ModelConfig& model_config, const int rank,
                     std::shared_ptr<Context> context) {
    context_ = context;
    rank_ = rank;
    model_config_ = model_config;
    workspace_buffer_ = workspace_buffer;

    builder_map_[{TYPE_FP32, TYPE_FP32, TYPE_FP32, QUANT_NONE}] = &MatMulLayerFactory<T>::BuildLayer<MatMulLayer<T>>;
    builder_map_[{TYPE_FP16, TYPE_FP16, TYPE_FP16, QUANT_NONE}] = &MatMulLayerFactory<T>::BuildLayer<MatMulLayer<T>>;
    builder_map_[{TYPE_BF16, TYPE_BF16, TYPE_BF16, QUANT_NONE}] = &MatMulLayerFactory<T>::BuildLayer<MatMulLayer<T>>;
#ifdef ENABLE_CUDA
    if (model_config_.is_moe) {
      builder_map_[{TYPE_FP32, TYPE_FP32, TYPE_FP32, MOE_QUANT_NONE}] = &MatMulLayerFactory<T>::BuildLayer<MoeLayer<T>>;
      builder_map_[{TYPE_FP16, TYPE_FP16, TYPE_FP16, MOE_QUANT_NONE}] = &MatMulLayerFactory<T>::BuildLayer<MoeLayer<T>>;
      builder_map_[{TYPE_BF16, TYPE_BF16, TYPE_BF16, MOE_QUANT_NONE}] = &MatMulLayerFactory<T>::BuildLayer<MoeLayer<T>>;
      // TODO(winminkong): support moe layer for quant
    }
#endif
#ifdef ENABLE_FP8
    builder_map_[{TYPE_FP8_E4M3, TYPE_FP32, TYPE_FP32, QUANT_FP8_E4M3}] =
        &MatMulLayerFactory<T>::BuildLayer<Fp8MatMulLayer<T>>;
    builder_map_[{TYPE_FP8_E4M3, TYPE_FP16, TYPE_FP16, QUANT_FP8_E4M3}] =
        &MatMulLayerFactory<T>::BuildLayer<Fp8MatMulLayer<T>>;
    builder_map_[{TYPE_FP8_E4M3, TYPE_BF16, TYPE_BF16, QUANT_FP8_E4M3}] =
        &MatMulLayerFactory<T>::BuildLayer<Fp8MatMulLayer<T>>;
#endif
    builder_map_[{TYPE_I4_GROUP, TYPE_FP16, TYPE_FP16, QUANT_GPTQ}] =
        &MatMulLayerFactory<T>::BuildLayer<GroupMatMulLayer<T, TYPE_I4_GROUP>>;
    builder_map_[{TYPE_I4_GROUP, TYPE_FP16, TYPE_FP16, QUANT_AWQ}] =
        &MatMulLayerFactory<T>::BuildLayer<GroupMatMulLayer<T, TYPE_I4_GROUP>>;
  }
  ~MatMulLayerFactory() {
    if (workspace_buffer_) {
      Tensor& tensor = *workspace_buffer_;
      DestroyTensor(tensor, rank_);
    }
  }
  template <typename ClassT>
  std::shared_ptr<BaseLayer> BuildLayer() {
    return std::make_shared<ClassT>();
  }
  std::shared_ptr<BaseLayer> AutoCreateLayer(std::shared_ptr<BaseWeight> base_weight, std::string weight_name,
                                             DataType weight_type, DataType input_type, DataType output_type,
                                             const std::vector<std::any>& init_params) {
    // gptq layer
    if (model_config_.is_quant &&
        (model_config_.quant_config.method == QUANT_GPTQ || model_config_.quant_config.method == QUANT_AWQ)) {
      size_t hidden_size = model_config_.hidden_units;
      size_t inter_size = model_config_.inter_size;
      // The inter size in config.json for the qwen1 model is twice the true inter size.
      if (model_config_.type == "qwen") {
        inter_size /= 2;
      }
      size_t tp = model_config_.tensor_para_size;
      size_t qkv_size = model_config_.size_per_head * (model_config_.head_num + 2 * model_config_.num_key_value_heads);
      // Because the layout convertion, we can't get n/k from weight shape, and have to calculate it.
      std::map<std::string, std::tuple<size_t, size_t, bool>> kn_pairs;
      kn_pairs["query_key_value"] = std::make_tuple(hidden_size, qkv_size / tp, true);
      kn_pairs["o_proj"] = std::make_tuple(hidden_size / tp, hidden_size, false);
      kn_pairs["gate_proj"] = std::make_tuple(hidden_size, inter_size / tp, true);
      kn_pairs["up_proj"] = kn_pairs["gate_proj"];
      kn_pairs["down_proj"] = std::make_tuple(inter_size / tp, hidden_size, false);
      for (const auto& kn : kn_pairs) {
        if (weight_name.find(kn.first) != std::string::npos) {
          std::vector<std::any> group_matmul_param;
          group_matmul_param.push_back(static_cast<bool>(model_config_.quant_config.method == QUANT_AWQ));
          group_matmul_param.push_back(static_cast<bool>(model_config_.quant_config.desc_act));
          group_matmul_param.push_back(static_cast<bool>(std::get<2>(kn.second)));
          group_matmul_param.push_back(static_cast<GroupQuantBackend>(model_config_.quant_config.backend));
          group_matmul_param.push_back(model_config_.max_scheduler_token_num);
          group_matmul_param.push_back(std::get<1>(kn.second));
          group_matmul_param.push_back(std::get<0>(kn.second));
          group_matmul_param.push_back(model_config_.quant_config.group_size);
          group_matmul_param.push_back(true);
          return CreateLayer(TYPE_I4_GROUP, input_type, output_type, group_matmul_param, QUANT_GPTQ);
        }
      }
    }
    // fp8 layer
    if (base_weight->GetModelWeights(weight_name).dtype == TYPE_FP8_E4M3) {
      std::vector<std::any> fp8_matmul_params;
      // max_m_
      fp8_matmul_params.push_back(model_config_.max_scheduler_token_num);
      // weight is [n, k], k is shape[1]
      fp8_matmul_params.push_back(base_weight->GetModelWeights(weight_name).shape[1]);
      return CreateLayer(TYPE_FP8_E4M3, input_type, output_type, fp8_matmul_params, QUANT_FP8_E4M3);
    }
    // default layer
    return CreateLayer(base_weight, weight_name, input_type, output_type, init_params, QUANT_NONE);
  }

  std::shared_ptr<BaseLayer> AutoCreateLayer(std::shared_ptr<BaseWeight> base_weight,
                                             std::vector<std::string> weight_names, DataType weight_type,
                                             DataType input_type, DataType output_type,
                                             const std::vector<std::any>& init_params) {
    // moe layer   (weight_names[0]: up_gate_experts, weight_names[1]: down_experts)
    std::vector<std::any> moe_matmul_param = init_params;
    moe_matmul_param.push_back(model_config_.max_scheduler_token_num);
    size_t up_gate_experts_num = base_weight->GetModelWeights(weight_names[0]).shape[0];
    size_t down_experts_num = base_weight->GetModelWeights(weight_names[1]).shape[0];
    if (up_gate_experts_num != down_experts_num) {
      KLLM_THROW(fmt::format("Moe Weights Load Error: up_gate experts {} and down_experts {} should should be equal",
                             up_gate_experts_num, down_experts_num));
    }
    moe_matmul_param.push_back(up_gate_experts_num);  // num_experts
    size_t up_gate_hidden_size = base_weight->GetModelWeights(weight_names[0]).shape[2];
    size_t down_hidden_size = base_weight->GetModelWeights(weight_names[1]).shape[1];
    if (up_gate_hidden_size != down_hidden_size) {
      KLLM_THROW(
          fmt::format("Moe Weights Load Error: up_gate_experts hidden_size {} and down_experts hidden_size {} should "
                      "should be equal",
                      up_gate_hidden_size, down_hidden_size));
    }
    moe_matmul_param.push_back(base_weight->GetModelWeights(weight_names[0]).shape[2]);  // hidden_size
    moe_matmul_param.push_back(base_weight->GetModelWeights(weight_names[1]).shape[2]);  // Inter_size
    moe_matmul_param.push_back(model_config_.moe_config.experts_topk);                   // experts topk
    moe_matmul_param.push_back(model_config_.tensor_para_size);                          // TP_size
    weight_type = base_weight->GetModelWeights(weight_names[0]).dtype;
    DataType down_weight_type = base_weight->GetModelWeights(weight_names[1]).dtype;
    if (down_weight_type != weight_type) {
      KLLM_THROW(fmt::format(
          "Moe Weights Load Error: down_experts dtype {} and up_gate_experts dtype {} should have same dtype",
          down_weight_type, weight_type));
    }
    return CreateLayer(weight_type, input_type, output_type, moe_matmul_param, MOE_QUANT_NONE);
  }

  std::shared_ptr<BaseLayer> CreateLayer(std::shared_ptr<BaseWeight> base_weight, std::string weight_name,
                                         DataType input_type, DataType output_type,
                                         const std::vector<std::any>& init_params, QuantMode quant_mode = QUANT_NONE) {
    DataType weight_type = base_weight->GetModelWeights(weight_name).dtype;
    return CreateLayer(weight_type, input_type, output_type, init_params, quant_mode);
  }
  std::shared_ptr<BaseLayer> CreateLayer(DataType weight_type, DataType input_type, DataType output_type,
                                         const std::vector<std::any>& init_params, QuantMode quant_mode = QUANT_NONE) {
    auto it = builder_map_.find({weight_type, input_type, output_type, quant_mode});
    if (it != builder_map_.end()) {
      std::shared_ptr<BaseLayer> layer = (this->*(it->second))();
      layer->Init(init_params, context_, rank_);
      size_t workspace_size = layer->GetWorkSpaceSize();
      if (workspace_buffer_ == nullptr && workspace_size > 0) {
        KLLM_LOG_DEBUG << fmt::format("Create WorkSpace Buffer: {}", workspace_size);
        workspace_buffer_ = std::make_shared<Tensor>();
        Tensor& tensor = *workspace_buffer_;
        CreateTensor(tensor, {workspace_size}, DataType::TYPE_INT8, rank_, MemoryDevice::MEMORY_DEVICE);
      } else if (workspace_buffer_ && workspace_buffer_->GetTotalBytes() < workspace_size) {
        KLLM_LOG_DEBUG << fmt::format("Increase WorkSpace Buffer from: {} to: {}", workspace_buffer_->GetTotalBytes(),
                                      workspace_size);
        Tensor& tensor = *workspace_buffer_;
        DestroyTensor(tensor, rank_);
        CreateTensor(tensor, {workspace_size}, DataType::TYPE_INT8, rank_, MemoryDevice::MEMORY_DEVICE);
      } else {
        KLLM_LOG_DEBUG << "WorkSpace Buffer is big enough.";
      }
      layer->SetWorkSpaceBuffer(workspace_buffer_);

      layer->Preprocess(model_config_);

      return layer;
    } else {
      KLLM_THROW(fmt::format("Not support weight_type {}, input_type {}, output_type {}, quant_mode {}.", weight_type,
                             input_type, output_type, quant_mode));
    }
  }

 private:
  std::shared_ptr<Context> context_;
  int rank_;
  std::shared_ptr<Tensor> workspace_buffer_ = nullptr;
  ModelConfig model_config_;

  // std::map<std::tuple<weight_type, input_type, output_type, quant_mode>, BuildLayerFunc>
  std::map<std::tuple<DataType, DataType, DataType, QuantMode>, BuildLayerFunc> builder_map_;
};

}  // namespace ksana_llm
