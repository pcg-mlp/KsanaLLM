/* Copyright 2024 Tencent Inc.  All rights reserved.
==============================================================================*/
#pragma once
#include "csrc/utils/nvidia/workspace.h"
#include "ksana_llm/layers/base_layer.h"
#include "ksana_llm/models/base/base_weight.h"
#include "ksana_llm/models/common_moe/moe_config.h"

namespace ksana_llm {

struct WorkspaceInfo {
  void* workspace{};
  void* scale_probs{};
  void* fc2_output{};
  void* src_to_dest_map{};
  void* selected_experts{};
  void* lora_workspace{};
  size_t size{};
};
#ifdef ENABLE_CUDA
template <typename T>
class MoeLayer : public BaseLayer {
 public:
  virtual Status Init(const std::vector<std::any>& parameters, std::shared_ptr<Context> context, int rank) override;

  virtual size_t GetWorkSpaceSize() override;

  virtual Status SetWorkSpaceBuffer(const std::shared_ptr<Tensor>& workspace_buffer) override;

  virtual Status Preprocess(const ModelConfig& model_config_) override;

  virtual Status Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) override;

 private:
  MoeScaleNormMode moe_scale_norm_mode_;
  size_t max_ws_bytes_;
  size_t max_token_num_;
  size_t expert_num_;
  size_t expert_hidden_size_;
  size_t expert_inter_size_;
  size_t expert_topk_;
  int tp_size_;
  bool use_lora_ = false;

  // The vector of the best config index for every tokens number
  std::vector<size_t> config_map_;
  WorkspaceInfo workspace_info_;
};
#endif
}  // namespace ksana_llm