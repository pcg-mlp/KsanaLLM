/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "layers/base_layer.h"
#ifdef ENABLE_ACL
#  include "3rdparty/LLM_kernels/csrc/utils/ascend/atb_executor.h"
#  include "3rdparty/LLM_kernels/csrc/utils/ascend/common.h"
#endif

namespace ksana_llm {

class EmbLookupLayer : public BaseLayer {
 public:
  virtual Status Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                      std::shared_ptr<Context> context, int rank) override;

  // steps==nullptr when not using step-based lookup.
  Status Forward(const Tensor* input_ids, const Tensor* ids_offsets, const Tensor* prefix_offsets,
                 const Tensor* emb_weight, Tensor* output, const Tensor* steps = nullptr,
                 const LayerExecContext& exec_ctx = {});

 private:
  template <typename T>
  Status ForwardT(const Tensor* input_ids, const Tensor* ids_offsets, const Tensor* prefix_offsets,
                  const Tensor* emb_weight, Tensor* output, const Tensor* steps);

 protected:
  // Whether to use emb_scale for word embedding.
  bool use_emb_scale_{false};
  // The scale for word embedding.
  float emb_scale_ = 1.f;

  // The position encoding weight.
  void* pos_weight_{nullptr};

#ifdef ENABLE_ACL
  llm_kernels::utils::ATBOperationExecutor atb_op_executor_;
#endif  // ENABLE_ACL
};

class CpuEmbLookupLayer : public BaseLayer {
 public:
  Status Forward(const Tensor* input_ids, const Tensor* emb_table, const Tensor* emb_weight, Tensor* output,
                 const LayerExecContext& exec_ctx = {});

 private:
  template <typename T>
  Status ForwardT(const Tensor* input_ids, const Tensor* emb_table, const Tensor* emb_weight, Tensor* output);
};

}  // namespace ksana_llm
