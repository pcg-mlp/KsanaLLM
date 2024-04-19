/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include "csrc/kernels/nvidia/gemm_wrapper/cublas_gemm_algo_map.h"

#include <iostream>
#include <ostream>

#include "csrc/utils/nvidia/cuda_utils.h"

using namespace llm_kernels::utils;

namespace llm_kernels {
namespace nvidia {

uint32_t GetAlignment(uintptr_t address) {
  // alignment are in bytes
  uint32_t alignment = 256;
  for (;; alignment /= 2) {
    if (!(address % alignment)) {
      return alignment;
    }
  }
}

bool CublasLtGemmAlgoMap::IsGemmAlgoExist(const int batch_count, const int m, const int n, const int k,
                                          const cudaDataType_t data_type, const cublasOperation_t trans_a,
                                          const cublasOperation_t trans_b) {
  CublasLtGemmAlgoConfig tmp{batch_count, n, m, k, trans_a, trans_b, data_type};
  return algo_map_.find(tmp) != algo_map_.end();
}

const CublasLtGemmAlgoInfo CublasLtGemmAlgoMap::GetOrCreateCublasLtGemmAlgoInfo(const int batch_count, const int m,
                                                                                const int n, const int k,
                                                                                const cudaDataType_t data_type,
                                                                                const cublasOperation_t trans_a,
                                                                                const cublasOperation_t trans_b) {
  CublasLtGemmAlgoConfig tmp{batch_count, n, m, k, trans_a, trans_b, data_type};
  if (algo_map_.find(tmp) != algo_map_.end()) {
    return algo_map_[tmp];
  } else {
    CublasLtGemmAlgoInfo tmp_algo;
    tmp_algo.custom_opt = -1;
    tmp_algo.tile = -1;
    tmp_algo.num_splits_k = -1;
    tmp_algo.swizzle = -1;
    tmp_algo.reduction_scheme = -1;
    tmp_algo.workspace_size = -1;
    tmp_algo.stages = -1;
    tmp_algo.exec_time = -1.0f;
    return tmp_algo;
  }
}

const std::unordered_map<CublasLtGemmAlgoConfig, CublasLtGemmAlgoInfo, CublasLtGemmAlgoConfigHasher>&
CublasLtGemmAlgoMap::GetAlgoMap() {
  return algo_map_;
}

cublasLtMatmulAlgo_t HeuristicSearchCublasAlgo(cublasLtHandle_t cublaslt_handle, cublasOperation_t transa,
                                               cublasOperation_t transb, const int32_t m, const int32_t n,
                                               const int32_t k, const void* a_ptr, const int32_t lda,
                                               cudaDataType_t a_type, const void* b_ptr, const int32_t ldb,
                                               cudaDataType_t b_type, void* c_ptr, const int32_t ldc,
                                               cudaDataType_t c_type, float f_alpha, float f_beta,
                                               cudaDataType_t compute_type, const size_t workspace_size) {
  // TODO(karlluo): will invoke accuraccy problem
  int32_t is_fp16_compute_type = compute_type == CUDA_R_16F ? 1 : 0;

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
  CHECK_NVIDIA_CUDA_ERROR(
      cublasLtMatrixLayoutCreate(&a_desc, a_type, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda));
  CHECK_NVIDIA_CUDA_ERROR(
      cublasLtMatrixLayoutCreate(&b_desc, b_type, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb));
  CHECK_NVIDIA_CUDA_ERROR(cublasLtMatrixLayoutCreate(&c_desc, c_type, m, n, ldc));
  CHECK_NVIDIA_CUDA_ERROR(cublasLtMatmulDescCreate(&operation_desc, inner_compute_type, scale_type));
  CHECK_NVIDIA_CUDA_ERROR(
      cublasLtMatmulDescSetAttribute(operation_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(cublasOperation_t)));
  CHECK_NVIDIA_CUDA_ERROR(
      cublasLtMatmulDescSetAttribute(operation_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(cublasOperation_t)));
  cublasLtMatmulPreference_t preference_desc = nullptr;
  CHECK_NVIDIA_CUDA_ERROR(cublasLtMatmulPreferenceCreate(&preference_desc));
  CHECK_NVIDIA_CUDA_ERROR(cublasLtMatmulPreferenceInit(preference_desc));
  CHECK_NVIDIA_CUDA_ERROR(cublasLtMatmulPreferenceSetAttribute(
      preference_desc, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size, sizeof(workspace_size)));
#if (CUBLAS_VERSION) <= 12000
  uint32_t pointer_mode_mask = 0;
  CHECK_NVIDIA_CUDA_ERROR(cublasLtMatmulPreferenceSetAttribute(preference_desc, CUBLASLT_MATMUL_PREF_EPILOGUE_MASK,
                                                               &pointer_mode_mask, sizeof(pointer_mode_mask)));
#endif
  uint32_t a_alignment = GetAlignment(reinterpret_cast<uintptr_t>(a_ptr));
  uint32_t b_alignment = GetAlignment(reinterpret_cast<uintptr_t>(b_ptr));
  uint32_t c_alignment = GetAlignment(reinterpret_cast<uintptr_t>(c_ptr));
  // TODO(karlluo): support bias
  uint32_t d_alignment = GetAlignment(reinterpret_cast<uintptr_t>(/*bias*/ nullptr));
  CHECK_NVIDIA_CUDA_ERROR(cublasLtMatmulPreferenceSetAttribute(
      preference_desc, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, &a_alignment, sizeof(uint32_t)));
  CHECK_NVIDIA_CUDA_ERROR(cublasLtMatmulPreferenceSetAttribute(
      preference_desc, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, &b_alignment, sizeof(uint32_t)));
  CHECK_NVIDIA_CUDA_ERROR(cublasLtMatmulPreferenceSetAttribute(
      preference_desc, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, &c_alignment, sizeof(uint32_t)));
  CHECK_NVIDIA_CUDA_ERROR(cublasLtMatmulPreferenceSetAttribute(
      preference_desc, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES, &d_alignment, sizeof(uint32_t)));
  CHECK_NVIDIA_CUDA_ERROR(cublasLtMatmulAlgoGetHeuristic(cublaslt_handle, operation_desc, a_desc, b_desc, c_desc,
                                                         c_desc, preference_desc, /*requestedAlgoCount*/ 1,
                                                         &heuristic_result, &returned_result));
  if (returned_result == 0) {
    CHECK_NVIDIA_CUDA_ERROR(CUBLAS_STATUS_NOT_SUPPORTED);
  } else {
    return heuristic_result.algo;
  }
}

}  // namespace nvidia
}  // namespace llm_kernels
