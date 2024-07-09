/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/layers/base_layer.h"
#include "ksana_llm/layers/fp8_matmul_layer.h"
#include "ksana_llm/layers/gptq_matmul_layer.h"
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
    builder_map_[{TYPE_FP8_E4M3, TYPE_FP32, TYPE_FP32, QUANT_NONE}] =
        &MatMulLayerFactory<T>::BuildLayer<Fp8MatMulLayer<T>>;
    builder_map_[{TYPE_FP8_E4M3, TYPE_FP16, TYPE_FP16, QUANT_NONE}] =
        &MatMulLayerFactory<T>::BuildLayer<Fp8MatMulLayer<T>>;
    builder_map_[{TYPE_FP8_E4M3, TYPE_BF16, TYPE_BF16, QUANT_NONE}] =
        &MatMulLayerFactory<T>::BuildLayer<Fp8MatMulLayer<T>>;
    builder_map_[{TYPE_I4_G128, TYPE_FP16, TYPE_FP16, QUANT_GPTQ}] =
        &MatMulLayerFactory<T>::BuildLayer<GPTQMatMulLayer<T, TYPE_I4_G128>>;
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
    if (model_config_.is_quant && model_config_.quant_config.method == "gptq") {
      if (weight_name != "lm_head.weight") {  // skip lm_head
        std::vector<std::any> gptq_matmul_param;
        gptq_matmul_param.push_back(model_config_.max_token_num);
        gptq_matmul_param.push_back(std::max(model_config_.inter_size, 3 * model_config_.hidden_units));
        gptq_matmul_param.push_back(model_config_.inter_size);
        gptq_matmul_param.push_back(model_config_.quant_config.group_size);
        return CreateLayer(TYPE_I4_G128, input_type, output_type, gptq_matmul_param, context, rank, QUANT_GPTQ);
      }
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
