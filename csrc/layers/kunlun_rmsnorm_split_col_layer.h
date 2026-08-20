/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "layers/base_layer.h"

namespace ksana_llm {

class KunlunRmsnormSplitColLayer : public BaseLayer {
 public:
  Status Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
              std::shared_ptr<Context> context, int rank) override;

  Status Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors);

 private:
  template <typename T>
  Status ForwardT(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors);

 protected:
  float rms_norm_eps_;
  int64_t q_dim_;
  int64_t kv_dim_;
  int64_t rot_dim_;
};

}  // namespace ksana_llm
