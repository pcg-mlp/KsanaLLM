/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#ifdef ENABLE_FP8
#  include "layers/fp8_matmul_layer.h"

#  include "kernels/nvidia/kernel_wrapper.h"

namespace ksana_llm {

Status Fp8MatMulLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                            std::shared_ptr<Context> context, int rank) {
  STATUS_CHECK_FAILURE(BaseLayer::Init(parameters, runtime_config, context, rank));
  int parameter_index = 0;
  max_m_ = std::any_cast<const size_t>(parameters[parameter_index++]);
  max_k_ = std::any_cast<const size_t>(parameters[parameter_index++]);
  cublas_workspace_size_ = InvokeGetCublasWorkspaceSize();

  return Status();
}

size_t Fp8MatMulLayer::GetWorkspaceSize(const int m, const int k) {
  size_t input_size = m * k * GetTypeSize(TYPE_FP8_E4M3);
  size_t scale_size = GetTypeSize(TYPE_FP32);
  size_t workspace_size = input_size + scale_size + cublas_workspace_size_;
  return workspace_size;
}

size_t Fp8MatMulLayer::GetWorkspaceSize() {
  size_t workspace_size = GetWorkspaceSize(max_m_, max_k_);
  KLLM_LOG_DEBUG << fmt::format("Rank[{}] Request {} for Fp8MatMulLayer", rank_, workspace_size);
  return workspace_size;
}

template <typename T>
Status Fp8MatMulLayer::ForwardT(const Tensor& input, const Tensor& weight, Tensor& output,
                                const LayerExecContext& exec_ctx) {
  int m = input.shape[0];
  int k = input.shape[1];
  int n = weight.shape[0];
  const T* input_ptr = static_cast<const T*>(input.GetPtr<void>());
  size_t workspace_size = GetWorkspaceSize(m, k);
  if (workspace_size > workspace_buffer_->GetTotalBytes()) {
    KLLM_THROW(fmt::format("workspace size {} > buffer size {}", workspace_size, workspace_buffer_->GetTotalBytes()));
  }
  void* cublas_workspace = workspace_buffer_->GetPtr<void>();
  void* input_quant = reinterpret_cast<uint8_t*>(cublas_workspace) + cublas_workspace_size_;
  const void* weight_quant = weight.GetPtr<const void>();
  const void* weight_scale = weight.weight_scales->GetPtr<const void>();
  if (weight_scale == nullptr) {
    KLLM_THROW("Cannot load weight_scale. Weight_scale is nullptr.");
  }
  T* output_ptr = static_cast<T*>(output.GetPtr<void>());
  output.shape = {static_cast<size_t>(m), static_cast<size_t>(n)};
  output.dtype = input.dtype;
  float* input_scale = nullptr;
  if (weight.input_scales) {
    input_scale = static_cast<float*>(weight.input_scales->GetPtr<void>());
    Fp8E4m3Quantize<T>(1, m * k, input_ptr, input_quant, input_scale, true, GetComputeStream(exec_ctx));
  } else {
    input_scale =
        reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(input_quant) + GetTypeSize(TYPE_FP8_E4M3) * m * k);
    Fp8E4m3Quantize<T>(1, m * k, input_ptr, input_quant, input_scale, false, GetComputeStream(exec_ctx));
  }

  if (context_->device->ext->GetGPUGemmAlgoHelper().IsInit()) {
    uint32_t sm = context_->device->ext->GetComputeCapacity();
    uint32_t cuda_ver = context_->device->ext->GetCudaVersion();
    // NOTE(karlluo): for continue batching, there is not batch size here.
    constexpr uint64_t BATCH_SIZE = 1ull;
    // NOTE(karlluo): using an impossible type to make sure no algo can be search for an unregistered type.
    cudaDataType_t c_dtype = CUDA_R_32F;
    if (std::is_same<T, float>::value) {
      c_dtype = CUDA_R_32F;
    } else if (std::is_same<T, half>::value) {
      c_dtype = CUDA_R_16F;
    } else if (std::is_same<T, __nv_bfloat16>::value) {
      c_dtype = CUDA_R_16BF;
    }
    llm_kernels::nvidia::GemmAlgoInfo gemm_algo_info = context_->device->ext->GetGPUGemmAlgoHelper().GetGemmAlgo(
        sm, cuda_ver, BATCH_SIZE, input.shape[0], weight.shape[1], input.shape[1], CUDA_R_8F_E4M3, CUDA_R_8F_E4M3,
        c_dtype, CUDA_R_32F, CUBLAS_OP_T, CUBLAS_OP_N);
    if (gemm_algo_info.gemm_op_type != llm_kernels::nvidia::DEFAULT_GEMM_ALGO) {
      cublaslt_algo_ptr_ = &(gemm_algo_info.cublaslt_algo);
    }
  }

  Fp8QuantizedMatMul<T>(context_->device->ext->GetCublasHandle(), context_->device->ext->GetCublasLtHandle(), m, n, k,
                        input_quant, input_scale, weight_quant, weight_scale, output_ptr, GetComputeStream(exec_ctx),
                        cublaslt_algo_ptr_, cublas_workspace, cublas_workspace_size_);
  cublaslt_algo_ptr_ = nullptr;
  return Status();
}

Status Fp8MatMulLayer::Forward(const Tensor* input, const Tensor* weight, Tensor* output,
                               const LayerExecContext& exec_ctx) {
  DISPATCH_BY_3_DTYPE(inter_data_type_, ForwardT, *input, *weight, *output, exec_ctx);
}

}  // namespace ksana_llm
#endif
