/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include "csrc/kernels/nvidia/gemm_wrapper/gemm_wrapper.h"

#include <iostream>
#include <ostream>

#include "csrc/kernels/nvidia/gemm_wrapper/cublas_gemm_algo_map.h"
#include "csrc/utils/nvidia/cuda_utils.h"

using namespace llm_kernels::utils;

namespace llm_kernels {
namespace nvidia {

cublasStatus_t InvokeCublasGemmEx(cublasHandle_t cublas_handle, cublasOperation_t transa, cublasOperation_t transb,
                                  const int32_t m, const int32_t n, const int32_t k, const void* alpha,
                                  const void* a_ptr, cudaDataType_t a_type, int32_t lda, const void* b_ptr,
                                  cudaDataType_t b_type, int32_t ldb, const void* beta, void* c_ptr,
                                  cudaDataType_t c_type, int32_t ldc, cudaDataType_t compute_type,
                                  cublasGemmAlgo_t algo) {
  return cublasGemmEx(cublas_handle, transa, transb, m, n, k, alpha, a_ptr, a_type, lda, b_ptr, b_type, ldb, beta,
                      c_ptr, c_type, ldc, compute_type, algo);
}

cublasStatus_t InvokeCublasGemm(cublasHandle_t cublas_handle, cublasLtHandle_t cublaslt_handle,
                                cublasOperation_t transa, cublasOperation_t transb, const int32_t m, const int32_t n,
                                const int32_t k, const void* a_ptr, const int32_t lda, cudaDataType_t a_type,
                                const void* b_ptr, const int32_t ldb, cudaDataType_t b_type, void* c_ptr,
                                const int32_t ldc, cudaDataType_t c_type, cudaDataType_t compute_type,
                                cudaStream_t& stream, void* workspace_ptr) {
  cublasLtMatmulAlgo_t* cublaslt_algo = nullptr;
  return InvokeCublasGemm(cublas_handle, cublaslt_handle, transa, transb, m, n, k, a_ptr, lda, a_type, b_ptr, ldb,
                          b_type, c_ptr, ldc, c_type, 1.0f, 0.0f, compute_type, stream, workspace_ptr, cublaslt_algo);
}

cublasStatus_t InvokeCublasGemm(cublasHandle_t cublas_handle, cublasLtHandle_t cublaslt_handle,
                                cublasOperation_t transa, cublasOperation_t transb, const int32_t m, const int32_t n,
                                const int32_t k, const void* a_ptr, const int32_t lda, cudaDataType_t a_type,
                                const void* b_ptr, const int32_t ldb, cudaDataType_t b_type, void* c_ptr,
                                const int32_t ldc, cudaDataType_t c_type, float f_alpha, float f_beta,
                                cudaDataType_t compute_type, cudaStream_t& stream, void* workspace_ptr,
                                cublasLtMatmulAlgo_t* cublaslt_algo) {
  // NOTE(karlluo): half no static cast in regular c_ptr++
  half h_alpha = (half)(f_alpha);
  half h_beta = (half)(f_beta);

  // TODO(karlluo): will invoke accuraccy problem
  int32_t is_fp16_compute_type = compute_type == CUDA_R_16F ? 1 : 0;
  // fp32 use cublas as default
  // fp16 use cublasLt as default
  const void* alpha = is_fp16_compute_type ? reinterpret_cast<void*>(&h_alpha) : reinterpret_cast<void*>(&f_alpha);
  const void* beta = is_fp16_compute_type ? reinterpret_cast<void*>(&h_beta) : reinterpret_cast<void*>(&f_beta);

  // prepare description
  cublasLtMatmulDesc_t operation_desc = nullptr;
  cublasLtMatrixLayout_t a_desc = nullptr;
  cublasLtMatrixLayout_t b_desc = nullptr;
  cublasLtMatrixLayout_t c_desc = nullptr;
  cudaDataType_t scale_type = CUDA_R_32F;
  cublasComputeType_t inner_compute_type;
  // for heuristic search
  cublasLtMatmulHeuristicResult_t heuristic_result = {};
  int returned_result = 0;

  if (is_fp16_compute_type) {
    // TODO(karlluo): support CUBLAS_COMPUTE_32F_FAST_TF32
    inner_compute_type = CUBLAS_COMPUTE_16F;
  } else {
    inner_compute_type = CUBLAS_COMPUTE_32F;
  }

  // Create descriptors for the original matrices
  RETURN_NVIDIA_CUBLAS_ERROR(
      cublasLtMatrixLayoutCreate(&a_desc, a_type, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda));
  RETURN_NVIDIA_CUBLAS_ERROR(
      cublasLtMatrixLayoutCreate(&b_desc, b_type, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb));
  RETURN_NVIDIA_CUBLAS_ERROR(cublasLtMatrixLayoutCreate(&c_desc, c_type, m, n, ldc));
  RETURN_NVIDIA_CUBLAS_ERROR(cublasLtMatmulDescCreate(&operation_desc, inner_compute_type, scale_type));
  RETURN_NVIDIA_CUBLAS_ERROR(
      cublasLtMatmulDescSetAttribute(operation_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(cublasOperation_t)));
  RETURN_NVIDIA_CUBLAS_ERROR(
      cublasLtMatmulDescSetAttribute(operation_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(cublasOperation_t)));

  // TODO(karlluo): get from GetCublasWorkspaceSize()
  size_t workspace_size = workspace_ptr == nullptr ? 0ul : DEFAULT_CUBLAS_WORKSPACE_SIZE;
  bool is_use_cublaslt_algo = (cublaslt_algo != nullptr) && (workspace_size > 0);

  RETURN_NVIDIA_CUBLAS_ERROR(
      cublasLtMatmul(cublaslt_handle, operation_desc, alpha, a_ptr, a_desc, b_ptr, b_desc, beta, c_ptr, c_desc, c_ptr,
                     c_desc, is_use_cublaslt_algo ? nullptr : cublaslt_algo, workspace_ptr, workspace_size, stream));

  RETURN_NVIDIA_CUBLAS_ERROR(cublasLtMatmulDescDestroy(operation_desc));
  RETURN_NVIDIA_CUBLAS_ERROR(cublasLtMatrixLayoutDestroy(c_desc));
  RETURN_NVIDIA_CUBLAS_ERROR(cublasLtMatrixLayoutDestroy(b_desc));
  RETURN_NVIDIA_CUBLAS_ERROR(cublasLtMatrixLayoutDestroy(a_desc));
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t InvokeCublasStridedBatchedGemm(cublasHandle_t cublas_handle, cublasLtHandle_t cublaslt_handle,
                                              cublasOperation_t transa, cublasOperation_t transb, const int32_t m,
                                              const int32_t n, const int32_t k, const void* a_ptr, const int32_t lda,
                                              const int64_t strideA, cudaDataType_t a_type, const void* b_ptr,
                                              const int32_t ldb, const int64_t strideB, cudaDataType_t b_type,
                                              void* c_ptr, const int32_t ldc, const int64_t strideC,
                                              cudaDataType_t c_type, const int32_t batch_count,
                                              cudaDataType_t compute_type, const float f_alpha, const float f_beta) {
  half h_alpha = (half)f_alpha;
  half h_beta = (half)f_beta;

  int32_t is_fp16_compute_type = compute_type == CUDA_R_16F ? true : false;
  const void* alpha =
      is_fp16_compute_type ? reinterpret_cast<void*>(&h_alpha) : reinterpret_cast<const void*>(&f_alpha);
  const void* beta = is_fp16_compute_type ? reinterpret_cast<void*>(&h_beta) : reinterpret_cast<const void*>(&f_beta);

  return cublasGemmStridedBatchedEx(cublas_handle, transa, transb, m, n, k, alpha, a_ptr, a_type, lda, strideA, b_ptr,
                                    b_type, ldb, strideB, beta, c_ptr, c_type, ldc, strideC, batch_count, compute_type,
                                    CUBLAS_GEMM_DEFAULT);
}

cublasStatus_t InvokeCublasBatchedGemm(cublasHandle_t cublas_handle, cublasLtHandle_t cublaslt_handle,
                                       cublasOperation_t transa, cublasOperation_t transb, const int32_t m,
                                       const int32_t n, const int32_t k, const void* const* a_ptr, const int32_t lda,
                                       cudaDataType_t AType, const void* const* b_ptr, const int32_t ldb,
                                       cudaDataType_t BType, void* const* c_ptr, const int32_t ldc,
                                       cudaDataType_t CType, cudaDataType_t compute_type, const int32_t batch_count) {
  float f_alpha = static_cast<float>(1.0f);
  float f_beta = static_cast<float>(0.0f);

  half h_alpha = (half)1.0f;
  half h_beta = (half)0.0f;

  int32_t is_fp16_compute_type = compute_type == CUDA_R_16F ? true : false;
  const void* alpha = is_fp16_compute_type ? reinterpret_cast<void*>(&h_alpha) : reinterpret_cast<void*>(&f_alpha);
  const void* beta = is_fp16_compute_type ? reinterpret_cast<void*>(&h_beta) : reinterpret_cast<void*>(&f_beta);

  return cublasGemmBatchedEx(cublas_handle, transa, transb, m, n, k, alpha, a_ptr, AType, lda, b_ptr, BType, ldb, beta,
                             c_ptr, CType, ldc, batch_count, compute_type, CUBLAS_GEMM_DEFAULT);
}

}  // namespace nvidia
}  // namespace llm_kernels