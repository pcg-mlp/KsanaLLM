/* Copyright 2024 Tencent Inc.  All rights reserved.
 *  *  * ==============================================================================*/
#pragma once
#include "csrc/utils/nvidia/workspace.h"
#include "layers/moe_layer.h"
#include "models/base/base_weight.h"
#include "models/common_moe/moe_config.h"

namespace ksana_llm {

#ifdef ENABLE_CUDA

class Fp8MoeLayer : public MoeLayer {
 public:
  virtual Status Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                      std::shared_ptr<Context> context, int rank) override;

  virtual size_t GetWorkspaceSize() override;

  virtual Status SetWorkspaceBuffer(const std::shared_ptr<Tensor>& workspace_buffer) override;

  virtual Status Preprocess(const ModelConfig& model_config_, const RuntimeConfig& runtime_config) override;

  // eplb_map/e_score_bias/input_ids are nullptr when unused.
  virtual Status Forward(const Tensor* hidden_states, const Tensor* gating_output, const Tensor* w1, const Tensor* w2,
                         const Tensor* eplb_map, Tensor* output, Tensor* workspace,
                         const Tensor* e_score_bias = nullptr, const Tensor* input_ids = nullptr,
                         const LayerExecContext& exec_ctx = {}) override;

 private:
  template <typename T>
  size_t GetWorkspaceSizeT();

  template <typename T>
  Status PreprocessT(const ModelConfig& model_config_, const RuntimeConfig& runtime_config);

  template <typename T>
  Status ForwardT(const Tensor* hidden_states, const Tensor* gating_output, const Tensor* w1, const Tensor* w2,
                  const Tensor* eplb_map, Tensor* output, const Tensor* e_score_bias, const Tensor* input_ids);

 private:
  size_t quant_buffer_size_;
  void* quant_buffer_ptr_;
  size_t best_config_index_;
};
#endif
}  // namespace ksana_llm
