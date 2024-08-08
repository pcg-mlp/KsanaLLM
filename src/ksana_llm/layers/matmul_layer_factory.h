/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/layers/base_layer.h"
#include "ksana_llm/layers/fp8_matmul_layer.h"
#include "ksana_llm/layers/group_matmul_layer.h"
#include "ksana_llm/layers/matmul_layer.h"
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
    if (model_config_.is_quant && model_config_.quant_config.method == QUANT_GPTQ) {
      if (weight_name.find("lm_head") == std::string::npos) {
        std::vector<std::any> group_matmul_param;
        group_matmul_param.push_back(model_config_.max_scheduler_token_num);
        group_matmul_param.push_back(base_weight->GetModelWeights(weight_name).shape[1] * 2);
        group_matmul_param.push_back(base_weight->GetModelWeights(weight_name).shape[0]);
        group_matmul_param.push_back(model_config_.quant_config.group_size);
        return CreateLayer(TYPE_I4_GROUP, input_type, output_type, group_matmul_param, QUANT_GPTQ);
      }
    }
    // fp8 layer
    if (base_weight->GetModelWeights(weight_name).dtype == TYPE_FP8_E4M3) {
      std::vector<std::any> fp8_matmul_params;
      fp8_matmul_params.push_back(model_config_.max_scheduler_token_num);
      size_t inter_size = model_config_.inter_size;
      size_t head_num = model_config_.head_num;
      size_t num_kv_heads = model_config_.num_key_value_heads;
      size_t size_per_head = model_config_.size_per_head;
      fp8_matmul_params.push_back(
          std::max(std::max(inter_size, (head_num + 2 * num_kv_heads) * size_per_head), head_num * size_per_head * 2));
      return CreateLayer(TYPE_FP8_E4M3, input_type, output_type, fp8_matmul_params, QUANT_FP8_E4M3);
    }
    // default layer
    return CreateLayer(base_weight, weight_name, input_type, output_type, init_params, QUANT_NONE);
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
      throw std::runtime_error(fmt::format("Not support weight_type {}, input_type {}, output_type {}, quant_mode {}.",
                                           weight_type, input_type, output_type, quant_mode));
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
