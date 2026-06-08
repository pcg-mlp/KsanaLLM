/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include "layers/matmul_base_layer.h"
#include "models/base/base_weight.h"
#include "utils/tensor.h"

namespace ksana_llm {

class MatMulLayerFactory {
 public:
  typedef std::shared_ptr<MatMulBaseLayer> (MatMulLayerFactory::*BuildLayerFunc)();
  MatMulLayerFactory(const ModelConfig& model_config, const RuntimeConfig& runtime_config, const int rank,
                     std::shared_ptr<Context> context);

  ~MatMulLayerFactory() {}

  template <typename ClassT>
  std::shared_ptr<MatMulBaseLayer> BuildLayer() {
    return std::make_shared<ClassT>();
  }
  std::shared_ptr<MatMulBaseLayer> AutoCreateLayer(std::shared_ptr<BaseWeight> base_weight, std::string weight_name,
                                                   DataType weight_type, DataType input_type, DataType output_type,
                                                   LinearComputeBackend backend,
                                                   const std::vector<std::any>& init_params);

  std::shared_ptr<MatMulBaseLayer> CreateLayer(std::shared_ptr<BaseWeight> base_weight, std::string weight_name,
                                               DataType input_type, DataType output_type,
                                               const std::vector<std::any>& init_params, QuantMode quant_mode,
                                               LinearComputeBackend backend);

  std::shared_ptr<MatMulBaseLayer> CreateLayer(DataType weight_type, DataType input_type, DataType output_type,
                                               const std::vector<std::any>& init_params, QuantMode quant_mode,
                                               LinearComputeBackend backend);

 private:
  std::shared_ptr<Context> context_;
  int rank_;
  ModelConfig model_config_;
  RuntimeConfig runtime_config_;

  std::map<std::tuple<DataType, DataType, DataType, QuantMode, LinearComputeBackend>, BuildLayerFunc> builder_map_;
};

}  // namespace ksana_llm
