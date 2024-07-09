/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/models/base/base_weight.h"
#include "ksana_llm/models/tensor_manager.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/utils.h"

#ifdef ENABLE_CUDA
#  include "csrc/kernels/nvidia/asymmetric_gemm/cutlass_preprocessors.h"
#endif

namespace ksana_llm {

template <typename T>
class QuantWeight {
 public:
  QuantWeight(const ModelConfig& model_config, int rank, std::shared_ptr<Context> context,
              std::unordered_map<std::string, Tensor>& weights_map);
  ~QuantWeight();

  bool IsEnable();

  bool FilterOutQuantWeight(const std::string& tensor_name);

  Status ConvertGPTQTensor(int hidden_units, int inter_size, int num_layer);

  bool LoadQuantWeight(std::string& tensor_name, std::vector<size_t>& weight_shape, DataType& weight_data_type,
                       void* weight_ptr);

 private:
#ifdef ENABLE_CUDA
  torch::Tensor UnpackInt32IntoInt8(const torch::Tensor& w_packed);

  torch::Tensor PackInt8TensorToPackedInt4(torch::Tensor weight);

  torch::Tensor PreprocessWeightsForMixedGemmWarpper(torch::Tensor row_major_quantized_weight,
                                                     llm_kernels::nvidia::QuantType quant_type);

  torch::Tensor ConvertGPTQLayout(torch::Tensor qweight_int32);
#endif

  bool CheckQuantModel();

  std::unordered_map<std::string, Tensor>& weights_map_;

  std::shared_ptr<TensorManager> tensor_manager_;

  int tensor_para_size_ = 1;

  bool enable_ = false;

  int rank_ = 0;
  std::shared_ptr<Context> context_{nullptr};
  ModelConfig model_config_;
};

}  // namespace ksana_llm
