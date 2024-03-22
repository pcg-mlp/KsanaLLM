/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/models/base/base_weight.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/utils.h"

namespace ksana_llm {

template <typename T>
class LlamaWeight : public BaseWeight {
 public:
  LlamaWeight() {}
  ~LlamaWeight();
  explicit LlamaWeight(const ModelConfig& model_config, int rank, std::shared_ptr<Context> context);

  Tensor GetModelWeights(const std::string& weight_name);

 private:
  Status PermuteTensor(int hidden_units, int inter_size, int num_layer, int vocab_size);

  std::vector<std::string> SearchLocalPath(const std::string& model_path, bool& is_safetensors);

  Status LoadLlamaWeightsMap(const ModelConfig& model_config);

  std::string ConcatLayerName(std::string layer_flag, int& layer_index);

  Status AddWeightTensor(std::string weight_name, std::vector<size_t> shapes, DataType dtype);

  Status LoadWeightsFromFile(std::shared_ptr<BaseFileTensorLoader>& weights_loader);

  bool IsLoaded();
  bool weights_had_loaded_ = false;

  std::unordered_map<std::string, Tensor> weights_map_;
  std::unordered_map<std::string, DataType> weights_data_type_map_;

  std::string model_path_ = "";
  int rank_ = 0;
  int tensor_para_size_ = 1;

  std::shared_ptr<Context> context_{nullptr};

  ModelConfig model_config_;
};

}  // namespace ksana_llm
