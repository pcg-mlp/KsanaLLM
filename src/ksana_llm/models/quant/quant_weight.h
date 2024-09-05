/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/models/base/base_weight.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/tensor_manager.h"
#include "ksana_llm/utils/utils.h"

#ifdef ENABLE_CUDA
#  include "csrc/kernels/nvidia/asymmetric_gemm/cutlass_preprocessors.h"
#endif

namespace ksana_llm {

// Load quantized weights, used together with CommonWeight
template <typename T>
class QuantWeight {
 public:
  QuantWeight(const ModelConfig& model_config, int rank, std::shared_ptr<Context> context,
              std::unordered_map<std::string, Tensor>& weights_map);
  ~QuantWeight();

  // Enable quantized loading if the model is a quantized model
  bool IsEnable();

  // Determine if weights need to be filtered, e.g. in gptq model, "*.g_idx" is unnecessary and should be filtered out.
  bool FilterOutQuantWeight(const std::string& tensor_name);

  // Group weight conversion for weight transformation, partitioning, transposition, etc.
  Status ConvertGroupTensor(int hidden_units, int inter_size, int num_layer);

  // Load the weight if it is a quantized weight.
  // Currently, for q/k/v, they are loaded separately first,
  // then merged into qkv in ConvertGroupTensor, and finally q/k/v are deleted.
  // This is because weight layout conversion requires individual weight processing.
  bool LoadQuantWeight(std::string& tensor_name, std::vector<size_t>& weight_shape, DataType& weight_data_type,
                       void* weight_ptr);

#ifdef ENABLE_FP8
  // Copy scale from weights_loader_ to weights_map_
  bool LoadFp8E4m3Scale(std::string& tensor_name, std::vector<size_t>& weight_shape, DataType& weight_data_type,
                        void* weight_ptr);
  // Bind scale to weight
  Status BindFp8E4m3Scale(const int num_layer, const int num_heads, const int num_kv_heads);
  Status BindFp8E4m3ScaleOfProjWeight(std::string name, const int num_layer);
  Status BindFp8E4m3ScaleOfQkvWeight(std::string name, const int num_layer, const int num_heads,
                                     const int num_kv_heads);
  Status GetMaxScaleOfQkv(float* q_scale, float* k_scale, float* v_scale, float* qkv_scale);

  Status ConvertFp8E4m3Tensor(std::string& weight_name, DataType quant_type);

  Status ConvertFp8E4m3(const int num_layer);
#endif

 private:
#ifdef ENABLE_CUDA
  torch::Tensor AutoUnpack(const std::string& tensor_name, torch::Tensor& tensor);

  torch::Tensor UnpackAWQ(const torch::Tensor& qweight, int bits, int group_size);

  torch::Tensor GetReverseOrder(const torch::Tensor& iweights, int bits);

  torch::Tensor UnpackQWeight(const torch::Tensor& qtensor, int bits);

  torch::Tensor UnpackGPTQ(const torch::Tensor& qweight);

  torch::Tensor PackInt8ToPackedInt4(torch::Tensor weight);

  torch::Tensor PreprocessWeightsForMixedGemmWarpper(torch::Tensor row_major_quantized_weight,
                                                     llm_kernels::nvidia::QuantType quant_type);

  Status AddWeightFromTorchTensor(const std::string& name, torch::Tensor& tensor);
#endif

  // Check if the model is a quantized model
  bool CheckQuantModel();

  // Weigth list for storing model weights, it needs to come from CommonWeight
  std::unordered_map<std::string, Tensor>& weights_map_;

  // TensorManager for adding weights, it needs to come from CommonWeight
  std::shared_ptr<TensorManager> tensor_manager_;

  int tensor_para_size_ = 1;

  // weight is quantized in checkpoint
  bool enable_ = false;

  int rank_ = 0;
  std::shared_ptr<Context> context_{nullptr};
  ModelConfig model_config_;
};

}  // namespace ksana_llm
