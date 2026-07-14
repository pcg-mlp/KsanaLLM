/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "device/common_device.h"
#include "layers/base_layer.h"

namespace ksana_llm {

struct GroupedTopkLayerParameters : public BaseLayerParameters {
  int topk{0};
  bool renormalize{false};
  int num_expert_group{0};
  int topk_group{0};
  std::string scoring_func{""};
  float routed_scaling_factor{1.0f};
  bool use_e_score_correction_bias{false};
  float score_threshold{0.f};

  std::string ToString() const {
    return fmt::format(
        "topk:{}, renormalize:{}, num_expert_group:{}, topk_group:{}, scoring_func:{}, "
        "routed_scaling_factor:{}, use_e_score_correction_bias:{}, score_threshold:{}",
        topk, renormalize, num_expert_group, topk_group, scoring_func, routed_scaling_factor,
        use_e_score_correction_bias, score_threshold);
  }
};

#ifdef ENABLE_CUDA
class GroupedTopkLayer : public BaseLayer {
 public:
  virtual Status Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                      std::shared_ptr<Context> context, int rank) override;

  virtual Status Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) override;

 private:
  template <typename T>
  Status ForwardT(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors);

 private:
  struct GroupedTopkLayerParameters params_;

  size_t expert_para_size_;
  size_t expert_world_size_;

  bool is_profile_mode_;
};
#endif

}  // namespace ksana_llm
