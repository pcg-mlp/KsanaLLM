/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/models/base/base_weight.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/utils.h"

namespace ksana_llm {

template <typename T>
class CommonWeight : public BaseWeight {
  public:
    CommonWeight() {}
    ~CommonWeight();
    explicit CommonWeight(const ModelConfig& model_config, int rank, std::shared_ptr<Context> context);

    Tensor GetModelWeights(const std::string& weight_name);

  private:
    Status PermuteTensor(int hidden_units, int inter_size, int num_layer, int vocab_size);

    std::vector<std::string> SearchLocalPath(const std::string& model_path, bool& is_safetensors);

    Status LoadLlamaWeightsMap(const ModelConfig& model_config);

    std::string ConcatLayerName(std::string layer_flag, int& layer_index, bool is_bias = false);

    Status AddWeightTensor(std::string weight_name, std::vector<size_t> shapes, DataType dtype);

    Status CreateTensorWithSameShape(const std::string& origin_tensor_name, const std::string& copy_tensor_name);

    Status LoadWeightsFromFile(std::shared_ptr<BaseFileTensorLoader>& weights_loader);

    Status PermuteSingleTensorOfQKVWeight(void* src, void* dst, Tensor& q_in_tensor, Tensor& q_out_tensor,
                                          std::vector<size_t>& data_shape, std::vector<size_t>& qkv_dst_shape);
    Status PermuteQKVWeight(Tensor& last_qkv_tensor, Tensor& q_in_tensor, Tensor& q_out_tensor, const int num_layer);

    Status PermuteMLPWeight(Tensor& last_mlp_tensor, const int num_layer);

    Status PermuteOutputProjectWeight(Tensor& last_o_proj_tensor, const int num_layer);

    Status GetOptionalWeight(std::vector<std::string>& tensor_name_list,
                             std::vector<std::string>& optional_weight_list);

    Status PrepareLoadOpMeta(size_t& tensor_para_offset, std::vector<size_t>& weight_shape, bool& transpose_first,
                             const std::string& tensor_name);

    bool IsLoaded();
    bool weights_had_loaded_ = false;

    std::unordered_map<std::string, Tensor> weights_map_;
    std::unordered_map<std::string, DataType> weights_data_type_map_;

    std::string model_path_ = "";
    int rank_ = 0;
    int tensor_para_size_ = 1;
    std::string model_name_ = "";
    DataType weight_data_type_ = TYPE_FP16;

    std::shared_ptr<Context> context_{nullptr};

    ModelConfig model_config_;
};

}  // namespace ksana_llm
