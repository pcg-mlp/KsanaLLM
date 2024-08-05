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
  MatMulLayerFactory() {
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
    builder_map_[{TYPE_I4_G128, TYPE_FP16, TYPE_FP16, QUANT_GPTQ}] =
        &MatMulLayerFactory<T>::BuildLayer<GroupMatMulLayer<T, TYPE_I4_G128>>;
  }
  template <typename ClassT>
  std::shared_ptr<BaseLayer> BuildLayer() {
    return std::make_shared<ClassT>();
  }
  std::shared_ptr<BaseLayer> AutoCreateLayer(std::shared_ptr<BaseWeight> base_weight, std::string weight_name,
                                             DataType weight_type, DataType input_type, DataType output_type,
                                             ModelConfig& model_config_, const std::vector<std::any>& init_params,
                                             std::shared_ptr<Context> context, int rank) {
    // gptq layer
    if (model_config_.is_quant && model_config_.quant_config.method == QUANT_GPTQ) {
      const size_t max_m = model_config_.max_scheduler_token_num;
      const size_t hidden_size = model_config_.hidden_units;
      const size_t inter_size = model_config_.inter_size;
      const size_t num_q_heads = model_config_.head_num;
      const size_t num_qkv_heads = model_config_.head_num + 2 * model_config_.num_key_value_heads;
      const size_t tp = model_config_.tensor_para_size;

      std::map<std::string, std::array<size_t, 3>> mnk_pairs;
      mnk_pairs["query_key_value"] = {max_m, (hidden_size * num_qkv_heads) / (tp * num_q_heads), hidden_size};
      mnk_pairs["o_proj"] = {max_m, hidden_size, hidden_size / tp};
      mnk_pairs["gate_proj"] = {max_m, inter_size / tp, hidden_size};
      mnk_pairs["up_proj"] = mnk_pairs["gate_proj"];
      mnk_pairs["down_proj"] = {max_m, hidden_size, inter_size / tp};

      for (const auto& pair : mnk_pairs) {
        if (weight_name.find(pair.first) != std::string::npos) {
          std::vector<std::any> gptq_matmul_param;
          gptq_matmul_param.push_back(std::get<0>(pair.second));
          gptq_matmul_param.push_back(std::get<1>(pair.second));
          gptq_matmul_param.push_back(std::get<2>(pair.second));
          gptq_matmul_param.push_back(model_config_.quant_config.group_size);
          return CreateLayer(TYPE_I4_G128, input_type, output_type, gptq_matmul_param, context, rank, QUANT_GPTQ);
        }
      }
      return CreateLayer(base_weight, weight_name, input_type, output_type, init_params, context, rank, QUANT_NONE);
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
      return CreateLayer(TYPE_FP8_E4M3, input_type, output_type, fp8_matmul_params, context, rank, QUANT_FP8_E4M3);
    }
    // default layer
    return CreateLayer(base_weight, weight_name, input_type, output_type, init_params, context, rank, QUANT_NONE);
  }
  std::shared_ptr<BaseLayer> CreateLayer(std::shared_ptr<BaseWeight> base_weight, std::string weight_name,
                                         DataType input_type, DataType output_type,
                                         const std::vector<std::any>& init_params, std::shared_ptr<Context> context,
                                         int rank, QuantMode quant_mode = QUANT_NONE) {
    DataType weight_type = base_weight->GetModelWeights(weight_name).dtype;
    return CreateLayer(weight_type, input_type, output_type, init_params, context, rank, quant_mode);
  }
  std::shared_ptr<BaseLayer> CreateLayer(DataType weight_type, DataType input_type, DataType output_type,
                                         const std::vector<std::any>& init_params, std::shared_ptr<Context> context,
                                         int rank, QuantMode quant_mode = QUANT_NONE) {
    auto it = builder_map_.find({weight_type, input_type, output_type, quant_mode});
    if (it != builder_map_.end()) {
      std::shared_ptr<BaseLayer> layer = (this->*(it->second))();
      layer->Init(init_params, context, rank);
      return layer;
    } else {
      throw std::runtime_error(fmt::format("Not support weight_type {}, input_type {}, output_type {}, quant_mode {}.",
                                           weight_type, input_type, output_type, quant_mode));
    }
  }

 private:
  // std::map<std::tuple<weight_type, input_type, output_type, quant_mode>, BuildLayerFunc>
  std::map<std::tuple<DataType, DataType, DataType, QuantMode>, BuildLayerFunc> builder_map_;
};

}  // namespace ksana_llm
