/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "distributed/expert_parallel/expert_parallel_manager.h"
#include "layers/base_layer.h"

namespace ksana_llm {

#ifdef ENABLE_CUDA
// Abstract base class for all MoE layer implementations.
// Provides a unified typed Forward interface so callers can invoke it without
// knowing the concrete subtype.
class BaseMoeLayer : public BaseLayer {
 public:
  // Unified typed Forward for all MoE layer variants.
  // eplb_map, e_score_bias, input_ids may be nullptr when unused (e.g., Marlin path).
  virtual Status Forward(const Tensor* hidden_states, const Tensor* gating_output, const Tensor* w1, const Tensor* w2,
                         const Tensor* eplb_map, Tensor* output, Tensor* workspace,
                         const Tensor* e_score_bias = nullptr, const Tensor* input_ids = nullptr,
                         const LayerExecContext& exec_ctx = {}) = 0;

  // Default no-op. Subclasses with expert parallel support override this.
  virtual void SetExpertParallelManager(std::shared_ptr<ExpertParallelManager> expert_parallel_manager) {}
};
#endif

}  // namespace ksana_llm
