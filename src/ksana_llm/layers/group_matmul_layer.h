/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/layers/base_layer.h"
#include "ksana_llm/models/base/base_weight.h"

namespace ksana_llm {

template <typename T, DataType WT>
class GroupMatMulLayer : public BaseLayer {
 public:
  virtual Status Init(const std::vector<std::any>& parameters, std::shared_ptr<Context> context, int rank) override;

  virtual size_t GetWorkSpaceSize() override;

  virtual Status Preprocess(const ModelConfig& model_config_) override;

  virtual Status Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) override;

 private:
  bool is_awq_;
  bool is_gptq_desc_;
  // Whether the K-dimension of the weights is complete. If the weights split in the K dimension, is_k_full_ = false
  bool is_k_full_;
  GroupQuantBackend backend_;

  size_t max_m_, max_n_, max_k_;
  size_t groupsize_;

  // Marlin parameter
  size_t marlin_workspace_size_;
  size_t marlin_input_tmp_size_;
  size_t marlin_output_tmp_size_;
  size_t marlin_workspace_offset_;
  size_t marlin_input_tmp_offset_;
  size_t marlin_output_tmp_offset_;

  // Cutlass parameter
  bool cutlass_use_gemv_cuda_core_;
  std::vector<size_t> cutlass_config_map_;
};

}  // namespace ksana_llm
