/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "numerous_llm/models/base/base_weight.h"
#include "numerous_llm/utils/environment.h"
#include "numerous_llm/utils/utils.h"

namespace numerous_llm {

class LlamaWeight : public BaseWeight {
 public:
  LlamaWeight() {}
  ~LlamaWeight();
  explicit LlamaWeight(const ModelConfig& model_config);

  Tensor GetModelWeights(std::string& weight_name);

 private:
  Status LoadWeightFromBin(Tensor tensor, std::string binfile);

  Status LoadLlamaWeightsMap();

  std::string ConcatLayerName(std::string layer_flag, int& layer_index);

  std::string GetBinfileName(std::string weight_name);

  Status AddWeightTensor(std::string weight_name, std::vector<size_t> shapes, DataType dtype);

  std::unordered_map<std::string, Tensor> weights_map_;

  std::unordered_map<std::string, std::string> binfile_map_ = {
    {"gather_embedding", "model.wte.weight.bin"},
    {"norm", "model.final_layernorm.weight.bin"},
    {"lm_head", "model.lm_head.weight.bin"}
  };

  std::string model_path_ = "";
  DataType weight_data_type_;
  int head_num_ = 0;
  int size_per_head_ = 0;
  int hidden_units_ = 0;
  int inter_size_ = 0;
  int num_layer_ = 0;
  int rotary_embedding_ = 0;
  int vocab_size_ = 0;
  int rank_ = 0;
  int tensor_para_size_ = 0;
};

}  // namespace numerous_llm
