/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "layers/matmul_layer.h"

#include "3rdparty/LLM_kernels/csrc/kernels/nvidia/gemm_wrapper/gemm_wrapper.h"
#include "kernels/nvidia/kernel_wrapper.h"

namespace ksana_llm {

Status MatMulLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                         std::shared_ptr<Context> context, int rank) {
  context_ = context;
  rank_ = rank;
  cublas_workspace_ptr_ = nullptr;
  cublaslt_algo_ptr_ = nullptr;
  inter_data_type_ = runtime_config.inter_data_type;

  return Status();
}

size_t MatMulLayer::GetWorkspaceSize() {
  if (context_->device->ext->GetGPUGemmAlgoHelper().IsInit() || std::getenv("MATMUL_LAYER_USE_WORKSPACE")) {
    cublas_workspace_size_ = llm_kernels::nvidia::GetCublasWorkspaceSize();
  }
  return cublas_workspace_size_;
}

template <typename T>
Status MatMulLayer::ForwardT(const Tensor& input, const Tensor& weight, Tensor& output,
                             const LayerExecContext& exec_ctx) {
  if (cublas_workspace_size_ > 0) {
    cublas_workspace_ptr_ = workspace_buffer_->GetPtr<void>();
  }
  if (context_->device->ext->GetGPUGemmAlgoHelper().IsInit()) {
    uint32_t sm = context_->device->ext->GetComputeCapacity();
    uint32_t cuda_ver = context_->device->ext->GetCudaVersion();
    // NOTE(karlluo): for continue batching, there is not batch size here.
    constexpr uint64_t BATCH_SIZE = 1ull;
    // NOTE(karlluo): using an impossible type to make sure no algo can be search for an unregistered type.
    cudaDataType_t dtype = CUDA_R_64F;
    if (std::is_same<T, float>::value) {
      dtype = CUDA_R_32F;
    } else if (std::is_same<T, half>::value) {
      dtype = CUDA_R_16F;
    } else if (std::is_same<T, __nv_bfloat16>::value) {
      dtype = CUDA_R_16BF;
    }
    llm_kernels::nvidia::GemmAlgoInfo gemm_algo_info = context_->device->ext->GetGPUGemmAlgoHelper().GetGemmAlgo(
        sm, cuda_ver, BATCH_SIZE, input.shape[0], weight.shape[1], input.shape[1], dtype, dtype, dtype, CUDA_R_32F,
        CUBLAS_OP_N, CUBLAS_OP_N);
    if (gemm_algo_info.gemm_op_type != llm_kernels::nvidia::DEFAULT_GEMM_ALGO) {
      cublaslt_algo_ptr_ = &(gemm_algo_info.cublaslt_algo);
    }
  }

  cudaStream_t compute_stream = GetComputeStream(exec_ctx);
  InvokeMatMul<T>(context_->device->ext->GetCublasHandle(), context_->device->ext->GetCublasLtHandle(),
                  static_cast<int>(input.shape[0]), static_cast<int>(weight.shape[1]), static_cast<int>(input.shape[1]),
                  reinterpret_cast<const void*>(input.GetPtr<void>()),
                  reinterpret_cast<const void*>(weight.GetPtr<void>()), output.GetPtr<void>(), compute_stream,
                  cublas_workspace_ptr_, cublaslt_algo_ptr_, cublas_workspace_size_);

  output.shape = {input.shape[0], weight.shape[1]};
  output.dtype = input.dtype;
  cublaslt_algo_ptr_ = nullptr;
  return Status();
}

Status MatMulLayer::Forward(const Tensor* input, const Tensor* weight, Tensor* output,
                            const LayerExecContext& exec_ctx) {
  DISPATCH_BY_3_DTYPE(inter_data_type_, ForwardT, *input, *weight, *output, exec_ctx);
}

}  // namespace ksana_llm
