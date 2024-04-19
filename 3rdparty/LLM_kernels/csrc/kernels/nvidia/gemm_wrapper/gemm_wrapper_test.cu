/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include <gtest/gtest.h>

#include "csrc/kernels/nvidia/gemm_wrapper/cublas_gemm_algo_map.h"
#include "csrc/kernels/nvidia/gemm_wrapper/gemm_wrapper.h"
#include "tests/kernels/nvidia/utils/testsuit_base.h"

namespace llm_kernels {
namespace nvidia {
namespace test {

struct LlamaNvidiaGemmCublasTestOpPair {
  cublasOperation_t transa;
  cublasOperation_t transb;
};

class LlamaNvidiaGemmWrapperTestSuit : public NvidiaTestSuitBase {
 public:
  void SetUp() override {
    NvidiaTestSuitBase::SetUp();
    CHECK_NVIDIA_CUDA_ERROR(cublasCreate(&cublas_handle));
    CHECK_NVIDIA_CUDA_ERROR(cublasLtCreate(&cublaslt_handle));
    CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&cublas_workspace_buffer_ptr, GetCublasWorkspaceSize()));
  }

  void TearDown() override {
    NvidiaTestSuitBase::TearDown();
    CHECK_NVIDIA_CUDA_ERROR(cublasDestroy(cublas_handle));
    CHECK_NVIDIA_CUDA_ERROR(cublasLtDestroy(cublaslt_handle));
    CHECK_NVIDIA_CUDA_ERROR(cudaFree(cublas_workspace_buffer_ptr));
  }

 protected:
  using NvidiaTestSuitBase::stream;

  void* cublas_workspace_buffer_ptr{nullptr};

  cublasHandle_t cublas_handle;
  cublasLtHandle_t cublaslt_handle;
  const std::vector<LlamaNvidiaGemmCublasTestOpPair> cublas_op_pairs{
      {CUBLAS_OP_N, CUBLAS_OP_N}, {CUBLAS_OP_N, CUBLAS_OP_T}, {CUBLAS_OP_T, CUBLAS_OP_N}, {CUBLAS_OP_T, CUBLAS_OP_T}};

  template <typename T>
  void ComputeReference(const cublasOperation_t transa, const cublasOperation_t transb, const void* a_ptr,
                        const void* b_ptr, void* c_ptr, size_t m, size_t n, size_t k, float alpha = 1.0f,
                        float beta = 0.0f) {
    size_t lda = (transa == CUBLAS_OP_N) ? k : m;
    size_t ldb = (transb == CUBLAS_OP_N) ? n : k;
    size_t ldc = n;

    cudaDataType_t atype;
    cudaDataType_t btype;
    cudaDataType_t ctype;
    cudaDataType_t compute_type;
    if (std::is_same<T, float>::value) {
      atype = CUDA_R_32F;
      btype = CUDA_R_32F;
      ctype = CUDA_R_32F;
      compute_type = CUDA_R_32F;
    } else if (std::is_same<T, half>::value) {
      atype = CUDA_R_16F;
      btype = CUDA_R_16F;
      ctype = CUDA_R_16F;
      compute_type = CUDA_R_32F;
    } else {
      throw std::runtime_error("Unknown test type in ComputeReference. Only support float, half and __nv_bfloat16.");
    }

    CHECK_NVIDIA_CUDA_ERROR(cublasGemmEx(cublas_handle, transb, transa, n, m, k, (const void*)&alpha, b_ptr, btype, ldb,
                                         a_ptr, atype, lda, (const void*)&beta, c_ptr, ctype, ldc, compute_type,
                                         CUBLAS_GEMM_DEFAULT));
    CHECK_NVIDIA_CUDA_ERROR(cudaDeviceSynchronize());
  }

  template <typename T>
  std::string GenerateCublasGemmTestName(const std::string& task_name, const cublasOperation_t transa,
                                         const cublasOperation_t transb, const size_t batch_size, const size_t m,
                                         const size_t n, const size_t k) {
    std::string result = task_name;
    result += "_b_" + std::to_string(batch_size);
    result += "_m_" + std::to_string(m);
    result += "_n_" + std::to_string(n);
    result += "_k_" + std::to_string(k);

    if (std::is_same<T, float>::value) {
      result += "_float";
    } else if (std::is_same<T, half>::value) {
      result += "_half";
    } else {
      throw std::runtime_error(
          "Unknown test type in GenerateCublasGemmTestName. Only support float, half and __nv_bfloat16.");
    }

    result += (transa == CUBLAS_OP_N) ? "_AN_" : "_AT_";
    result += (transb == CUBLAS_OP_N) ? "BN" : "BT";

    return result;
  }

  template <typename T>
  void TestCublasGemmCorrectnessMatmul(size_t m, size_t n, size_t k) {
    BufferMeta a_buffer = CreateBuffer<T>(MemoryType::MEMORY_GPU, {m, k}, /*is_random_init*/ true);
    BufferMeta b_buffer = CreateBuffer<T>(MemoryType::MEMORY_GPU, {k, n}, /*is_random_init*/ true);
    BufferMeta c_buffer = CreateBuffer<T>(MemoryType::MEMORY_GPU, {m, n}, /*is_random_init*/ false);
    BufferMeta expected_buffer = CreateBuffer<T>(MemoryType::MEMORY_GPU, {m, n}, /*is_random_init*/ false);
    cudaDataType_t atype;
    cudaDataType_t btype;
    cudaDataType_t ctype;
    cudaDataType_t compute_type;
    float miss_match_rate = 0.01f;
    if (std::is_same<T, float>::value) {
      atype = CUDA_R_32F;
      btype = CUDA_R_32F;
      ctype = CUDA_R_32F;
      compute_type = CUDA_R_32F;
    } else if (std::is_same<T, half>::value) {
      atype = CUDA_R_16F;
      btype = CUDA_R_16F;
      ctype = CUDA_R_16F;
      compute_type = CUDA_R_32F;
    } else {
      throw std::runtime_error("Unknown test type. Only support float and half.");
    }

    for (auto& op_pair : cublas_op_pairs) {
      int32_t lda = (op_pair.transa == CUBLAS_OP_N) ? k : m;
      int32_t ldb = (op_pair.transb == CUBLAS_OP_N) ? n : k;
      int32_t ldc = n;
      float alpha = 1.0f;
      float beta = 0.0f;

      std::string test_name = GenerateCublasGemmTestName<T>(std::string("TestCublasGemmCorrectnessMatmul"),
                                                            op_pair.transa, op_pair.transb, 1ul, m, n, k);
      // compute the reference
      ComputeReference<T>(op_pair.transa, op_pair.transb, (const void*)a_buffer.data_ptr,
                          (const void*)b_buffer.data_ptr, expected_buffer.data_ptr, m, n, k);
      CHECK_NVIDIA_CUDA_ERROR(cudaDeviceSynchronize());

      CHECK_NVIDIA_CUDA_ERROR(InvokeCublasGemmEx(cublas_handle, op_pair.transb, op_pair.transa, n, m, k,
                                                 (const void*)&alpha, b_buffer.data_ptr, btype, ldb, a_buffer.data_ptr,
                                                 atype, lda, (const void*)&beta, c_buffer.data_ptr, ctype, ldc,
                                                 compute_type, CUBLAS_GEMM_DEFAULT));
      CHECK_NVIDIA_CUDA_ERROR(cudaDeviceSynchronize());
      EXPECT_TRUE(
          CheckResult<T>(test_name + "_invokeCublasGemmEx", expected_buffer, c_buffer, 1e-4f, 1e-5f, miss_match_rate));

      CHECK_NVIDIA_CUDA_ERROR(InvokeCublasGemm(cublas_handle, cublaslt_handle, op_pair.transb, op_pair.transa, n, m, k,
                                               b_buffer.data_ptr, ldb, btype, a_buffer.data_ptr, lda, atype,
                                               c_buffer.data_ptr, ldc, ctype, compute_type, stream));
      CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
      CHECK_NVIDIA_CUDA_ERROR(cudaDeviceSynchronize());
      EXPECT_TRUE(
          CheckResult<T>(test_name + "_invokeCublasGemm_1", expected_buffer, c_buffer, 1e-4f, 1e-5f, miss_match_rate));

      CHECK_NVIDIA_CUDA_ERROR(InvokeCublasGemm(cublas_handle, cublaslt_handle, op_pair.transb, op_pair.transa, n, m, k,
                                               b_buffer.data_ptr, ldb, btype, a_buffer.data_ptr, lda, atype,
                                               c_buffer.data_ptr, ldc, ctype, alpha, beta, compute_type, stream,
                                               nullptr, nullptr));
      CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
      CHECK_NVIDIA_CUDA_ERROR(cudaDeviceSynchronize());
      EXPECT_TRUE(
          CheckResult<T>(test_name + "_invokeCublasGemm_2", expected_buffer, c_buffer, 1e-4f, 1e-5f, miss_match_rate));

      cublasLtMatmulAlgo_t cublaslt_algo = HeuristicSearchCublasAlgo(
          cublaslt_handle, op_pair.transb, op_pair.transa, n, m, k, b_buffer.data_ptr, ldb, btype, a_buffer.data_ptr,
          lda, atype, c_buffer.data_ptr, ldc, ctype, alpha, beta, compute_type, GetCublasWorkspaceSize());
      CHECK_NVIDIA_CUDA_ERROR(InvokeCublasGemm(cublas_handle, cublaslt_handle, op_pair.transb, op_pair.transa, n, m, k,
                                               b_buffer.data_ptr, ldb, btype, a_buffer.data_ptr, lda, atype,
                                               c_buffer.data_ptr, ldc, ctype, alpha, beta, compute_type, stream,
                                               cublas_workspace_buffer_ptr, &cublaslt_algo));
      CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
      CHECK_NVIDIA_CUDA_ERROR(cudaDeviceSynchronize());
      EXPECT_TRUE(
          CheckResult<T>(test_name + "_invokeCublasGemm_3", expected_buffer, c_buffer, 1e-4f, 1e-5f, miss_match_rate));
    }
  }

  template <typename T>
  void TestCublasBatchGemmCorrectnessMatmul(size_t batch_size, size_t m, size_t n, size_t k) {
    BufferMeta a_buffer = CreateBuffer<T>(MemoryType::MEMORY_GPU, {batch_size, m, k}, /*is_random_init*/ true);
    BufferMeta b_buffer = CreateBuffer<T>(MemoryType::MEMORY_GPU, {batch_size, k, n}, /*is_random_init*/ true);
    BufferMeta c_buffer = CreateBuffer<T>(MemoryType::MEMORY_GPU, {batch_size, m, n}, /*is_random_init*/ false);
    BufferMeta expected_buffer = CreateBuffer<T>(MemoryType::MEMORY_GPU, {batch_size, m, n}, /*is_random_init*/ false);
    cudaDataType_t atype;
    cudaDataType_t btype;
    cudaDataType_t ctype;
    cudaDataType_t compute_type;
    if (std::is_same<T, float>::value) {
      atype = CUDA_R_32F;
      btype = CUDA_R_32F;
      ctype = CUDA_R_32F;
      compute_type = CUDA_R_32F;
    } else if (std::is_same<T, half>::value) {
      atype = CUDA_R_16F;
      btype = CUDA_R_16F;
      ctype = CUDA_R_16F;
      compute_type = CUDA_R_32F;
    } else {
      throw std::runtime_error("Unknown test type. Only support float and half.");
    }

    for (auto& op_pair : cublas_op_pairs) {
      int32_t lda = (op_pair.transa == CUBLAS_OP_N) ? k : m;
      int32_t ldb = (op_pair.transb == CUBLAS_OP_N) ? n : k;
      int32_t ldc = n;
      int64_t stridea = m * k;
      int64_t strideb = k * n;
      int64_t stridec = m * n;
      float alpha = 1.0f;
      float beta = 0.0f;

      std::string test_name = GenerateCublasGemmTestName<T>(std::string("TestCublasBatchGemmCorrectnessMatmul"),
                                                            op_pair.transa, op_pair.transb, batch_size, m, n, k);

      for (size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        ComputeReference<T>(op_pair.transa, op_pair.transb,
                            (const void*)(((T*)a_buffer.data_ptr) + stridea * batch_idx),
                            (const void*)(((T*)b_buffer.data_ptr) + strideb * batch_idx),
                            (void*)(((T*)expected_buffer.data_ptr) + stridec * batch_idx), m, n, k);
      }
      CHECK_NVIDIA_CUDA_ERROR(cudaDeviceSynchronize());

      CHECK_NVIDIA_CUDA_ERROR(InvokeCublasStridedBatchedGemm(
          cublas_handle, cublaslt_handle, op_pair.transb, op_pair.transa, n, m, k, b_buffer.data_ptr, ldb, strideb,
          btype, a_buffer.data_ptr, lda, stridea, atype, c_buffer.data_ptr, ldc, stridec, ctype, batch_size,
          compute_type, alpha, beta));
      CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
      CHECK_NVIDIA_CUDA_ERROR(cudaDeviceSynchronize());
      // TODO(karlluo): investigate why bfloat16 b=7 m=1041 n=2047 k=999 miss match rate can be 34.77%
      if (std::is_same<T, __nv_bfloat16>::value) {
        EXPECT_TRUE(CheckResult<T>(test_name + "_invokeCublasStridedBatchedGemm_1", expected_buffer, c_buffer, 1e-4f,
                                   1e-5f, 0.4f));
      } else {
        EXPECT_TRUE(
            CheckResult<T>(test_name + "_invokeCublasStridedBatchedGemm_1", expected_buffer, c_buffer, 1e-4f, 1e-5f));
      }
    }
  }

  template <typename T>
  void TestCublasGemmPerformance(size_t m, size_t n, size_t k) {
    BufferMeta a_buffer = CreateBuffer<T>(MemoryType::MEMORY_GPU, {m, k}, /*is_random_init*/ true);
    BufferMeta b_buffer = CreateBuffer<T>(MemoryType::MEMORY_GPU, {k, n}, /*is_random_init*/ true);
    BufferMeta c_buffer = CreateBuffer<T>(MemoryType::MEMORY_GPU, {m, n}, /*is_random_init*/ false);
    BufferMeta expected_buffer = CreateBuffer<T>(MemoryType::MEMORY_GPU, {m, n}, /*is_random_init*/ false);
    cudaDataType_t atype;
    cudaDataType_t btype;
    cudaDataType_t ctype;
    cudaDataType_t compute_type;
    cudaEvent_t start;
    cudaEvent_t stop;
    float time_elapsed_ms = 0.f;
    CHECK_NVIDIA_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_NVIDIA_CUDA_ERROR(cudaEventCreate(&stop));
    constexpr int warmup_rounds = 5;
    constexpr int tested_rounds = 100;
    float miss_match_rate = 0.01f;
    if (std::is_same<T, float>::value) {
      atype = CUDA_R_32F;
      btype = CUDA_R_32F;
      ctype = CUDA_R_32F;
      compute_type = CUDA_R_32F;
    } else if (std::is_same<T, half>::value) {
      atype = CUDA_R_16F;
      btype = CUDA_R_16F;
      ctype = CUDA_R_16F;
      compute_type = CUDA_R_32F;
    } else {
      throw std::runtime_error("Unknown test type. Only support float and half.");
    }

    for (auto& op_pair : cublas_op_pairs) {
      int32_t lda = (op_pair.transa == CUBLAS_OP_N) ? k : m;
      int32_t ldb = (op_pair.transb == CUBLAS_OP_N) ? n : k;
      int32_t ldc = n;
      float alpha = 1.0f;
      float beta = 0.0f;

      std::string test_name = GenerateCublasGemmTestName<T>(std::string("TestCublasGemmPerformance"), op_pair.transa,
                                                            op_pair.transb, 1ul, m, n, k);

      float InvokeCublasGemmEx_time_elapsed_ms = 0.f;
      // warmup InvokeCublasGemmEx
      for (int i = 0; i < warmup_rounds; ++i) {
        CHECK_NVIDIA_CUDA_ERROR(InvokeCublasGemmEx(cublas_handle, op_pair.transb, op_pair.transa, n, m, k,
                                                   (const void*)&alpha, b_buffer.data_ptr, btype, ldb,
                                                   a_buffer.data_ptr, atype, lda, (const void*)&beta, c_buffer.data_ptr,
                                                   ctype, ldc, compute_type, CUBLAS_GEMM_DEFAULT));
        CHECK_NVIDIA_CUDA_ERROR(cudaDeviceSynchronize());
        CHECK_NVIDIA_CUDA_ERROR(cudaEventRecord(start));
      }
      for (int i = 0; i < tested_rounds; ++i) {
        CHECK_NVIDIA_CUDA_ERROR(InvokeCublasGemmEx(cublas_handle, op_pair.transb, op_pair.transa, n, m, k,
                                                   (const void*)&alpha, b_buffer.data_ptr, btype, ldb,
                                                   a_buffer.data_ptr, atype, lda, (const void*)&beta, c_buffer.data_ptr,
                                                   ctype, ldc, compute_type, CUBLAS_GEMM_DEFAULT));
      }
      CHECK_NVIDIA_CUDA_ERROR(cudaEventRecord(stop));
      CHECK_NVIDIA_CUDA_ERROR(cudaEventSynchronize(stop));
      CHECK_NVIDIA_CUDA_ERROR(cudaDeviceSynchronize());
      CHECK_NVIDIA_CUDA_ERROR(cudaEventElapsedTime(&InvokeCublasGemmEx_time_elapsed_ms, start, stop));

      float InvokeCublasGemm_1_time_elapsed_ms = 0.f;
      // warmup InvokeCublasGemm_1
      for (int i = 0; i < warmup_rounds; ++i) {
        CHECK_NVIDIA_CUDA_ERROR(InvokeCublasGemm(cublas_handle, cublaslt_handle, op_pair.transb, op_pair.transa, n, m,
                                                 k, b_buffer.data_ptr, ldb, btype, a_buffer.data_ptr, lda, atype,
                                                 c_buffer.data_ptr, ldc, ctype, compute_type, stream));
      }
      CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
      CHECK_NVIDIA_CUDA_ERROR(cudaEventRecord(start));
      for (int i = 0; i < tested_rounds; ++i) {
        CHECK_NVIDIA_CUDA_ERROR(InvokeCublasGemm(cublas_handle, cublaslt_handle, op_pair.transb, op_pair.transa, n, m,
                                                 k, b_buffer.data_ptr, ldb, btype, a_buffer.data_ptr, lda, atype,
                                                 c_buffer.data_ptr, ldc, ctype, compute_type, stream));
      }
      CHECK_NVIDIA_CUDA_ERROR(cudaEventRecord(stop));
      CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
      CHECK_NVIDIA_CUDA_ERROR(cudaEventSynchronize(stop));
      CHECK_NVIDIA_CUDA_ERROR(cudaEventElapsedTime(&InvokeCublasGemm_1_time_elapsed_ms, start, stop));

      float InvokeCublasGemm_2_time_elapsed_ms = 0.f;
      // warmup InvokeCublasGemm_2
      for (int i = 0; i < warmup_rounds; ++i) {
        CHECK_NVIDIA_CUDA_ERROR(InvokeCublasGemm(cublas_handle, cublaslt_handle, op_pair.transb, op_pair.transa, n, m,
                                                 k, b_buffer.data_ptr, ldb, btype, a_buffer.data_ptr, lda, atype,
                                                 c_buffer.data_ptr, ldc, ctype, alpha, beta, compute_type, stream,
                                                 nullptr, nullptr));
      }
      CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
      CHECK_NVIDIA_CUDA_ERROR(cudaEventRecord(start));
      for (int i = 0; i < tested_rounds; ++i) {
        CHECK_NVIDIA_CUDA_ERROR(InvokeCublasGemm(cublas_handle, cublaslt_handle, op_pair.transb, op_pair.transa, n, m,
                                                 k, b_buffer.data_ptr, ldb, btype, a_buffer.data_ptr, lda, atype,
                                                 c_buffer.data_ptr, ldc, ctype, alpha, beta, compute_type, stream,
                                                 nullptr, nullptr));
      }
      CHECK_NVIDIA_CUDA_ERROR(cudaEventRecord(stop));
      CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
      CHECK_NVIDIA_CUDA_ERROR(cudaEventSynchronize(stop));
      CHECK_NVIDIA_CUDA_ERROR(cudaEventElapsedTime(&InvokeCublasGemm_2_time_elapsed_ms, start, stop));

      float InvokeCublasGemm_3_time_elapsed_ms = 0.f;
      cublasLtMatmulAlgo_t cublaslt_algo = HeuristicSearchCublasAlgo(
          cublaslt_handle, op_pair.transb, op_pair.transa, n, m, k, b_buffer.data_ptr, ldb, btype, a_buffer.data_ptr,
          lda, atype, c_buffer.data_ptr, ldc, ctype, alpha, beta, compute_type, GetCublasWorkspaceSize());
      // warmup InvokeCublasGemm_3 and prerun
      for (int i = 0; i < warmup_rounds; ++i) {
        CHECK_NVIDIA_CUDA_ERROR(InvokeCublasGemm(cublas_handle, cublaslt_handle, op_pair.transb, op_pair.transa, n, m,
                                                 k, b_buffer.data_ptr, ldb, btype, a_buffer.data_ptr, lda, atype,
                                                 c_buffer.data_ptr, ldc, ctype, alpha, beta, compute_type, stream,
                                                 cublas_workspace_buffer_ptr, &cublaslt_algo));
      }
      CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
      CHECK_NVIDIA_CUDA_ERROR(cudaEventRecord(start));
      for (int i = 0; i < tested_rounds; ++i) {
        CHECK_NVIDIA_CUDA_ERROR(InvokeCublasGemm(cublas_handle, cublaslt_handle, op_pair.transb, op_pair.transa, n, m,
                                                 k, b_buffer.data_ptr, ldb, btype, a_buffer.data_ptr, lda, atype,
                                                 c_buffer.data_ptr, ldc, ctype, alpha, beta, compute_type, stream,
                                                 cublas_workspace_buffer_ptr, &cublaslt_algo));
      }
      CHECK_NVIDIA_CUDA_ERROR(cudaEventRecord(stop));
      CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
      CHECK_NVIDIA_CUDA_ERROR(cudaEventSynchronize(stop));
      CHECK_NVIDIA_CUDA_ERROR(cudaEventElapsedTime(&InvokeCublasGemm_3_time_elapsed_ms, start, stop));
    }
  }
};

TEST_F(LlamaNvidiaGemmWrapperTestSuit, CublasTest) {
  using testcase_t = std::tuple<size_t, size_t, size_t>;

  std::vector<testcase_t> testcases = {{16, 32, 64},   {255, 255, 255},    {1041, 1, 9999},
                                       {1041, 999, 1}, {1041, 2047, 9999}, {256, 256, 256}};

  // Computation correctness tests
  for (testcase_t& tc : testcases) {
    size_t m = std::get<0>(tc);
    size_t n = std::get<1>(tc);
    size_t k = std::get<2>(tc);

    TestCublasGemmCorrectnessMatmul<float>(m, n, k);
    TestCublasGemmCorrectnessMatmul<half>(m, n, k);

    TestCublasBatchGemmCorrectnessMatmul<float>(7, m, n, k);
    TestCublasBatchGemmCorrectnessMatmul<half>(7, m, n, k);

    TestCublasGemmPerformance<float>(m, n, k);
    TestCublasGemmPerformance<half>(m, n, k);
  }
}
}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels