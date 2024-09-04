/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/models/base/base_weight.h"
#include "ksana_llm/models/quant/quant_weight.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/tensor_manager.h"
#include "ksana_llm/utils/utils.h"

namespace ksana_llm {

template <typename T>
class CommonWeight : public BaseWeight {
 public:
  CommonWeight() {}
  ~CommonWeight() override;
  explicit CommonWeight(const ModelConfig& model_config, int rank, std::shared_ptr<Context> context);

  Tensor GetModelWeights(const std::string& weight_name) override;

  void ProcessWeights() override;

  Status LoadWeightsFromFile(std::shared_ptr<BaseFileTensorLoader>& weights_loader,
                             std::vector<std::string>& weight_name_list,
                             std::vector<std::string>& custom_name_list) override;

 protected:
  Status ConvertCommonTensor(int hidden_units, int inter_size, int num_layer, int vocab_size);

  Status ReshapeQkvTensor(int num_layer);
  Status ConvertLmheadTensor();

  Status GetModelInfo(const ModelConfig& model_config);

  std::string ConcatLayerName(std::string layer_flag, int& layer_index, bool is_bias = false);

  Status LoadRegularTensor(void* weight_ptr, std::string tensor_name, std::vector<size_t>& weight_shape,
                           DataType& weight_data_type, bool transpose_first, size_t tensor_para_offset,
                           size_t& weight_size);

  Status PermuteSingleTensorOfQKVWeight(void* src, void* dst, Tensor& q_in_tensor, Tensor& q_out_tensor,
                                        std::vector<size_t>& data_shape, std::vector<size_t>& qkv_dst_shape);
  Status PermuteQKVWeight(Tensor& last_qkv_tensor, Tensor& q_in_tensor, Tensor& q_out_tensor, const int num_layer);

  Status CommonPermuteWeight(const std::string& origin_tensor_name, Tensor& swap_tensor);

  Status PermuteMLPWeight(Tensor& last_down_up_tensor, Tensor& last_gate_tensor, const int num_layer);

  Status PermuteOutputProjectWeight(Tensor& last_o_proj_tensor, const int num_layer);

  Status PrepareLoadOpMeta(size_t& tensor_para_offset, std::vector<size_t>& weight_shape, bool& transpose_first,
                           const std::string& tensor_name);

  void ChunkGateWeight(const int num_layer);

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

  std::shared_ptr<TensorManager> tensor_manager_;

  std::shared_ptr<QuantWeight<T>> quant_weight_slover_;
};

}  // namespace ksana_llm
