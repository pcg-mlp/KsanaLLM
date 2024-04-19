/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#pragma once

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace llm_kernels {
namespace nvidia {

struct CublasLtGemmAlgoConfig {
  int batch_count;
  int m;
  int n;
  int k;
  cublasOperation_t trans_a;
  cublasOperation_t trans_b;
  cudaDataType_t data_type;

  bool operator==(CublasLtGemmAlgoConfig const& config) const {
    return (batch_count == config.batch_count) && (m == config.m) && (n == config.n) && (k == config.k) &&
           (data_type == config.data_type) && (trans_a == config.trans_a) && (trans_b == config.trans_b);
  }
};

class CublasLtGemmAlgoConfigHasher {
 public:
  std::size_t operator()(CublasLtGemmAlgoConfig const& config) const {
    // mutiply prime number and bitwise xor can be fast hash in low conflict range
    return config.batch_count * 98317ull ^ config.m * 49157ull ^ config.n * 24593ull ^ config.k * 196613ull ^
           static_cast<int>(config.data_type) * 6151ull ^ static_cast<int>(config.trans_a) * 3571ull ^
           static_cast<int>(config.trans_b) * 1907ull;
  }
};

typedef struct {
  cublasLtMatmulAlgo_t cublaslt_algo;
  int custom_opt;
  int tile;
  int num_splits_k;
  int swizzle;
  int reduction_scheme;
  int workspace_size;
  // only used in cublasLt >= 11.0
  int stages;
#if (CUBLAS_VER_MAJOR >= 11 && CUBLAS_VER_MINOR >= 11 && CUBLAS_VER_PATCH >= 3)
  uint16_t inner_shape_id;
  uint16_t cluster_shape_id;
#elif (CUBLAS_VER_MAJOR == 11 && CUBLAS_VER_MINOR == 11 && CUBLAS_VER_PATCH < 3)
  uint16_t mma_shape_id;
  uint16_t cga_shape_id;
  uint16_t sche_mode;
#endif
  float exec_time;
} CublasLtGemmAlgoInfo;

class CublasLtGemmAlgoMap {
 public:
  CublasLtGemmAlgoMap(){};
  //   explicit CublasLtGemmAlgoMap(const std::string config_filename);
  CublasLtGemmAlgoMap(const CublasLtGemmAlgoMap& algo_map);
  ~CublasLtGemmAlgoMap();

  bool IsGemmAlgoExist(const int batch_count, const int m, const int n, const int k, const cudaDataType_t data_type,
                       const cublasOperation_t trans_a, const cublasOperation_t trans_b);

  const CublasLtGemmAlgoInfo GetOrCreateCublasLtGemmAlgoInfo(const int batch_count, const int m, const int n,
                                                             const int k, const cudaDataType_t data_type,
                                                             const cublasOperation_t trans_a,
                                                             const cublasOperation_t trans_b);

  const std::unordered_map<CublasLtGemmAlgoConfig, CublasLtGemmAlgoInfo, CublasLtGemmAlgoConfigHasher>& GetAlgoMap();

 private:
  std::unordered_map<CublasLtGemmAlgoConfig, CublasLtGemmAlgoInfo, CublasLtGemmAlgoConfigHasher> algo_map_;
};

cublasLtMatmulAlgo_t HeuristicSearchCublasAlgo(cublasLtHandle_t cublaslt_handle, cublasOperation_t transa,
                                               cublasOperation_t transb, const int32_t m, const int32_t n,
                                               const int32_t k, const void* a_ptr, const int32_t lda,
                                               cudaDataType_t a_type, const void* b_ptr, const int32_t ldb,
                                               cudaDataType_t b_type, void* c_ptr, const int32_t ldc,
                                               cudaDataType_t c_type, float f_alpha, float f_beta,
                                               cudaDataType_t compute_type, const size_t workspace_size);

}  // namespace nvidia
}  // namespace llm_kernels