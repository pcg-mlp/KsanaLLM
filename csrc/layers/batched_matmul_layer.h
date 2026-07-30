/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "layers/matmul_base_layer.h"

namespace ksana_llm {

class BatchedMatMulLayer : public MatMulBaseLayer {
 public:
  virtual Status Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                      std::shared_ptr<Context> context, int rank) override;

  virtual size_t GetWorkspaceSize() override;

  // a: [m, bs, k], b: [bs, k, n] -> output: [m, bs, n]
  virtual Status Forward(const Tensor* a, const Tensor* b, Tensor* output,
                         const LayerExecContext& exec_ctx = {}) override;

 private:
#ifdef ENABLE_CUDA
  void* cublas_workspace_ptr_{nullptr};
  int cublas_workspace_size_{-1};
#endif

 private:
  template <typename T>
  Status ForwardT(const Tensor& a, const Tensor& b, Tensor& output, const LayerExecContext& exec_ctx);
};

}  // namespace ksana_llm
