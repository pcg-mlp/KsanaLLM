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
  // Status LoadWeightsFromBin(const std::string& file_name);
  Status PermuteTensor(int hidden_units, int inter_size, int num_layer, int vocab_size, int tensor_para_size);

  std::vector<std::string> SearchLocalPath(const std::string& model_path);

  Status LoadLlamaWeightsMap(const ModelConfig& model_config);

  std::string ConcatLayerName(std::string layer_flag, int& layer_index);

  Status AddWeightTensor(std::string weight_name, std::vector<size_t> shapes, DataType dtype);

  Status LoadWeightsFromBin(const std::string& file_name);

  bool IsLoaded();
  bool weights_had_loaded_ = false;

  std::unordered_map<std::string, Tensor> weights_map_;

  std::string model_path_ = "";
  int rank_ = 0;

  std::shared_ptr<Context> context_{nullptr};
};

}  // namespace ksana_llm
