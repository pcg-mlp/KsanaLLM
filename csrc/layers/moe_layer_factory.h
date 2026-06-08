/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include "layers/base_moe_layer.h"
#include "models/base/base_weight.h"
#include "utils/tensor.h"

namespace ksana_llm {

// Forward declaration
class ExpertParallelManager;

class MoeLayerFactory {
 public:
  typedef std::shared_ptr<BaseMoeLayer> (MoeLayerFactory::*BuildLayerFunc)();
  MoeLayerFactory(const ModelConfig& model_config, const RuntimeConfig& runtime_config, const int rank,
                  std::shared_ptr<Context> context);

  ~MoeLayerFactory() {}

  template <typename ClassT>
  std::shared_ptr<BaseMoeLayer> BuildLayer() {
    return std::make_shared<ClassT>();
  }

  std::shared_ptr<BaseMoeLayer> AutoCreateMoeLayer(std::shared_ptr<BaseWeight> base_weight,
                                                   std::vector<std::string> weight_names, DataType weight_type,
                                                   DataType input_type, DataType output_type, int layer_idx,
                                                   const std::vector<std::any>& init_params);

  std::shared_ptr<BaseMoeLayer> CreateLayer(DataType weight_type, DataType input_type, DataType output_type,
                                            const std::vector<std::any>& init_params, QuantMode quant_mode,
                                            MoeComputeBackend backend);

  // 获取 ExpertParallelManager
  std::shared_ptr<ExpertParallelManager> GetExpertParallelManager() const { return expert_parallel_manager_; }

 private:
  std::shared_ptr<Context> context_;
  int rank_;
  ModelConfig model_config_;
  RuntimeConfig runtime_config_;

  // map key: {weight_type, input_type, output_type, quant_mode, backend}
  std::map<std::tuple<DataType, DataType, DataType, QuantMode, MoeComputeBackend>, BuildLayerFunc> builder_map_;

  // Expert Parallel Manager (Exchange data with ExecutorRuntime)
  std::shared_ptr<ExpertParallelManager> expert_parallel_manager_ = nullptr;
};

}  // namespace ksana_llm
