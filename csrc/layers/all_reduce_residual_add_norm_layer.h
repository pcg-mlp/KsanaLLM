/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "layers/add_norm_layer.h"
#include "layers/base_layer.h"
#include "models/base/forwarding_context.h"
#include "models/communicator/tp_communicator.h"

namespace ksana_llm {

/**
 * This layer is used to fuse the allreduce and residual add norm operation
 * into one kernel operation for dtype: fp16, bf16
 * Step 1: tp_comm_->AllReduce(reduce_buffer_tensors, hidden_buffer_tensors_0)
 * Step 2: add_->Forward(hidden_buffer_tensors_0[0], residual_buffer[0], residual_buffer)
 * Step 3 (optioanl): layernorm_layer_->Forward(residual_buffer, hidden_buffer_tensors_0)
 */
class AllReduceResidualAddNormLayer : public BaseLayer {
 public:
  virtual ~AllReduceResidualAddNormLayer() = default;

  virtual Status Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                      std::shared_ptr<Context> context, int rank) override;

  // AllReduce + residual add only (residual updated in-place). residual==nullptr skips the add.
  Status Forward(const Tensor* input, Tensor* residual_inout, const LayerExecContext& exec_ctx = {});
  // AllReduce + residual add + RMSNorm: norm result written to output.
  Status Forward(const Tensor* input, const Tensor* residual, Tensor* output, const LayerExecContext& exec_ctx = {});

 private:
  // residual_in==nullptr means 1-input path (allreduce + residual update in-place).
  template <typename T>
  Status ForwardT(const Tensor& input, const Tensor* residual_in, Tensor& output);

  bool is_init_ = false;
  float rms_norm_eps_;
  Tensor rms_norm_weight_;
  std::shared_ptr<TpCommunicator> tp_comm_;
  std::shared_ptr<AddNormLayer> add_norm_layer_;
#ifdef ENABLE_MULTICAST
  NvlsMcastMemory* nvls_mcast_memory_ = nullptr;
#endif
};

}  // namespace ksana_llm
