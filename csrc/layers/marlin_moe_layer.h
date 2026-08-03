/* Copyright 2024 Tencent Inc.  All rights reserved.
 * ==============================================================================*/
#pragma once
#include "csrc/utils/nvidia/workspace.h"
#include "layers/base_moe_layer.h"
#include "models/base/base_weight.h"
#include "models/common_moe/moe_config.h"

namespace ksana_llm {

#ifdef ENABLE_CUDA
class MarlinMoeLayer : public BaseMoeLayer {
 public:
  virtual Status Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                      std::shared_ptr<Context> context, int rank) override;

  virtual size_t GetWorkspaceSize() override;

  virtual Status Preprocess(const ModelConfig& model_config_, const RuntimeConfig& runtime_config) override;

  // BaseMoeLayer typed Forward override; eplb_map/workspace/e_score_bias/input_ids unused in Marlin path.
  virtual Status Forward(const Tensor* hidden_states, const Tensor* gating_output, const Tensor* w1, const Tensor* w2,
                         const Tensor* eplb_map, Tensor* output, Tensor* workspace,
                         const Tensor* e_score_bias = nullptr, const Tensor* input_ids = nullptr,
                         const LayerExecContext& exec_ctx = {}) override;

 private:
  template <typename T>
  size_t GetWorkspaceSizeT();

 private:
  bool use_lora_ = false;
  size_t num_bits_ = 4;

  size_t max_ws_bytes_;
  size_t max_gating_size_;
};
#endif
}  // namespace ksana_llm
