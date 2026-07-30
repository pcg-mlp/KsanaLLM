/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "layers/batched_matmul_layer.h"

#include "3rdparty/LLM_kernels/csrc/kernels/nvidia/gemm_wrapper/gemm_wrapper.h"
#include "kernels/nvidia/kernel_wrapper.h"

namespace ksana_llm {

Status BatchedMatMulLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                                std::shared_ptr<Context> context, int rank) {
  context_ = context;
  rank_ = rank;
  inter_data_type_ = runtime_config.inter_data_type;
  return Status();
}

size_t BatchedMatMulLayer::GetWorkspaceSize() {
  size_t workspace_size = InvokeGetCublasWorkspaceSize();
  KLLM_LOG_DEBUG << fmt::format("Rank[{}] Request {} for BatchedMatMulLayer", rank_, workspace_size);
  return workspace_size;
}

template <typename T>
cudaDataType_t GetCublasDataType();
#define GET_Cublas_DATA_TYPE(T, Cublas_TYPE) \
  template <>                                \
  cudaDataType_t GetCublasDataType<T>() {    \
    return Cublas_TYPE;                      \
  }
GET_Cublas_DATA_TYPE(float, CUDA_R_32F);
GET_Cublas_DATA_TYPE(half, CUDA_R_16F);
GET_Cublas_DATA_TYPE(__nv_bfloat16, CUDA_R_16BF);
#undef GET_Cublas_DATA_TYPE

template <typename T>
Status BatchedMatMulLayer::ForwardT(const Tensor& a, const Tensor& b, Tensor& output,
                                    const LayerExecContext& exec_ctx) {
  KLLM_CHECK_WITH_INFO(a.shape.size() == 3, "input tensors shape size should be 3.");
  KLLM_CHECK_WITH_INFO(b.shape.size() == 3, "input tensors shape size should be 3.");
  KLLM_CHECK_WITH_INFO(a.shape[1] == b.shape[0], "input batch size should be equal.");
  KLLM_CHECK_WITH_INFO(a.shape[2] == b.shape[1], "input k dim should be equal.");

  size_t bs = a.shape[1];
  size_t m = a.shape[0];
  size_t n = b.shape[2];
  size_t k = a.shape[2];

  cublas_workspace_ptr_ = nullptr;
  cublas_workspace_size_ = 0;
  cublasLtMatmulAlgo_t* algo_ptr = nullptr;
  if (workspace_buffer_ == nullptr || workspace_buffer_->GetTotalBytes() == 0) {
    KLLM_LOG_DEBUG << "No workspace can be reused for batched matmul layer.";
  } else {
    cublas_workspace_ptr_ = workspace_buffer_->GetPtr<void>();
    cublas_workspace_size_ = workspace_buffer_->GetTotalBytes();
  }

  InvokeBatchedMatMul<T>(context_->device->ext->GetCublasHandle(), context_->device->ext->GetCublasLtHandle(), bs, m, n,
                         k, /*lda*/ bs * k, /*ldb*/ n, /*ldc*/ bs * n, /*batch_offset_a*/ k, /*batch_offset_b*/ n * k,
                         /*batch_offset_c*/ n, reinterpret_cast<const void*>(a.GetPtr<void>()),
                         reinterpret_cast<const void*>(b.GetPtr<void>()), output.GetPtr<void>(),
                         GetComputeStream(exec_ctx), cublas_workspace_ptr_, cublas_workspace_size_, algo_ptr);
  output.shape = {m, bs, n};
  output.dtype = a.dtype;
  return Status();
}

Status BatchedMatMulLayer::Forward(const Tensor* a, const Tensor* b, Tensor* output, const LayerExecContext& exec_ctx) {
  DISPATCH_BY_3_DTYPE(inter_data_type_, ForwardT, *a, *b, *output, exec_ctx);
}

}  // namespace ksana_llm
