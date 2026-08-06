/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <type_traits>

#include "csrc/utils/nvidia/cuda_utils.h"
#include "tests/kernels/nvidia/utils/testsuit_base.h"

#include "csrc/kernels/nvidia/activation/activation.h"

namespace llm_kernels {
namespace nvidia {
namespace test {

class LlamaNvidiaActivationTestSuit : public NvidiaTestSuitBase {
 public:
  void SetUp() override { NvidiaTestSuitBase::SetUp(); }

  void TearDown() override { NvidiaTestSuitBase::TearDown(); }

 protected:
  using NvidiaTestSuitBase::stream;
  const std::vector<std::pair<int, int>> m_n_pairs = {{1, 2048}, {2, 4096}, {64000, 8}};

 protected:
  template <template <typename> class Activation, typename T>
  std::string GetActivationName() {
    if constexpr (std::is_same_v<Activation<T>, GeluActivation<T>>) {
      return "gelu";
    } else if constexpr (std::is_same_v<Activation<T>, ReluActivation<T>>) {
      return "relu";
    } else {  // std::is_same_v<Activation<T>, SiluActivation<T>>
      return "silu";
    }
  }

  template <template <typename T> class Activation, typename T>
  void RunActivationRef(const std::string& kernel_name) {
    std::string type_str = "float";
    if (std::is_same<T, half>::value) {
      type_str = "half";
    } else if (std::is_same<T, __nv_bfloat16>::value) {
      type_str = "bfloat16";
    }

    std::stringstream ss;
    ss << "python activation_test.py --activation_kernel=" << kernel_name << " --type=" << type_str << " --activation=";
    ss << GetActivationName<Activation, T>();
    system(ss.str().c_str());
  }

  template <template <typename T> class Activation, typename T>
  void RunActivation(cudaStream_t stream, BufferMeta& input_meta, BufferMeta& gated_weight_meta,
                     BufferMeta& output_meta, BufferMeta& output_ref_meta, size_t m, size_t n,
                     const std::string& type_str) {
    const int* ia3_tasks = nullptr;
    const T* bias = nullptr;
    const T* ia3_weights = nullptr;
    const T* gated_bias = nullptr;
    const int int8_mode = 0;
    const int* padding_offset = nullptr;
    const int seq_len = 0;
    const float* activation_in = nullptr;
    const float* activation_out = nullptr;
    CHECK_NVIDIA_CUDA_ERROR(cudaMemcpyAsync(output_meta.data_ptr, input_meta.data_ptr, sizeof(T) * m * n,
                                            cudaMemcpyDeviceToDevice, stream));
    InvokeGenericActivation<Activation, T, T>(reinterpret_cast<T*>(output_meta.data_ptr), bias,
                                              reinterpret_cast<const T*>(gated_weight_meta.data_ptr), gated_bias,
                                              ia3_tasks, ia3_weights, m, n, int8_mode, activation_in, activation_out,
                                              padding_offset, seq_len, stream);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
    CHECK_NVIDIA_CUDA_ERROR(cudaDeviceSynchronize());

    EXPECT_TRUE(CheckResult<T>("activation_" + type_str + "_m_" + std::to_string(m) + "_n_" + std::to_string(n),
                               output_ref_meta, output_meta, 1e-5f, 1e-5f));

    auto cuda_run = [&]() {
      InvokeGenericActivation<Activation, T, T>(reinterpret_cast<T*>(output_meta.data_ptr), bias,
                                                reinterpret_cast<const T*>(gated_weight_meta.data_ptr), gated_bias,
                                                ia3_tasks, ia3_weights, m, n, int8_mode, activation_in, activation_out,
                                                padding_offset, seq_len, stream);
    };
    float milliseconds = MeasureCudaExecutionTime(cuda_run, stream, 10, 100);
    std::cout << std::left << "Activation " << GetActivationName<Activation, T>() << " m=" << std::setw(6) << m
              << " n=" << std::setw(6) << n << " execution 1 times " << std::setw(10) << milliseconds << " ms"
              << std::endl;
  }

  template <typename T>
  void TestActivation(cudaStream_t stream) {
    std::string type_str = "float";
    if (std::is_same<T, half>::value) {
      type_str = "half";
    } else if (std::is_same<T, __nv_bfloat16>::value) {
      type_str = "bfloat16";
    }
    for (const auto& m_n_pair : m_n_pairs) {
      const size_t m = static_cast<size_t>(m_n_pair.first);
      const size_t n = static_cast<size_t>(m_n_pair.second);
      BufferMeta input_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {m, n},
                                              /*is_random_init*/ true);
      BufferMeta gated_weight_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {m, n},
                                                     /*is_random_init*/ true);
      BufferMeta output_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {m, n},
                                               /*is_random_init*/ false);
      BufferMeta output_ref_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {m, n},
                                                   /*is_random_init*/ false);

      input_meta.SaveToNpy<T>("activation_test_input.npy");
      gated_weight_meta.SaveToNpy<T>("activation_test_gated_weight.npy");
      system(("python activation_test.py --activation_kernel=InvokeGenericActivation --type=" + type_str +
              " --activation=all")
                 .c_str());

      output_ref_meta.LoadNpy<T>("activation_test_output_silu.npy", MemoryType::MEMORY_GPU);
      RunActivation<SiluActivation, T>(stream, input_meta, gated_weight_meta, output_meta, output_ref_meta, m, n,
                                       type_str);
      output_ref_meta.LoadNpy<T>("activation_test_output_gelu.npy", MemoryType::MEMORY_GPU);
      RunActivation<GeluActivation, T>(stream, input_meta, gated_weight_meta, output_meta, output_ref_meta, m, n,
                                       type_str);
      output_ref_meta.LoadNpy<T>("activation_test_output_relu.npy", MemoryType::MEMORY_GPU);
      RunActivation<ReluActivation, T>(stream, input_meta, gated_weight_meta, output_meta, output_ref_meta, m, n,
                                       type_str);

      DeleteBuffer(output_ref_meta);
      DeleteBuffer(output_meta);
      DeleteBuffer(gated_weight_meta);
      DeleteBuffer(input_meta);
    }
  }

  template <template <typename T> class Activation, typename T>
  void TestRowBasedActivation(cudaStream_t stream) {
    for (const auto& m_n_pair : m_n_pairs) {
      const size_t m = static_cast<size_t>(m_n_pair.first);
      const size_t n = static_cast<size_t>(m_n_pair.second);
      BufferMeta input_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {m, n},
                                              /*is_random_init*/ true);
      BufferMeta output_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {m, n / 2},
                                               /*is_random_init*/ false);
      BufferMeta output_ref_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {m, n / 2},
                                                   /*is_random_init*/ false);

      std::string type_str = "float";
      if (std::is_same<T, half>::value) {
        type_str = "half";
      } else if (std::is_same<T, __nv_bfloat16>::value) {
        type_str = "bfloat16";
      }
      input_meta.SaveToNpy<T>("row_based_activation_test_input.npy");

      RunActivationRef<Activation, T>("InvokeRowBasedActivation");

      output_ref_meta.LoadNpy<T>("row_based_activation_test_output.npy", MemoryType::MEMORY_GPU);

      InvokeRowBasedActivation<Activation, T, /*HAS_SWIGLU_LIMIT*/ false>(reinterpret_cast<T*>(output_meta.data_ptr),
                                                                          reinterpret_cast<T*>(input_meta.data_ptr), m,
                                                                          n, /*swiglu_limit=*/0.0f, stream);
      CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
      CHECK_NVIDIA_CUDA_ERROR(cudaDeviceSynchronize());

      EXPECT_TRUE(
          CheckResult<T>("row_based_activation_" + type_str + "_m_" + std::to_string(m) + "_n_" + std::to_string(n),
                         output_ref_meta, output_meta, 1e-5f, 1e-5f));

      auto cuda_run = [&]() {
        InvokeRowBasedActivation<Activation, T, /*HAS_SWIGLU_LIMIT*/ false>(reinterpret_cast<T*>(output_meta.data_ptr),
                                                                            reinterpret_cast<T*>(input_meta.data_ptr),
                                                                            m, n, /*swiglu_limit=*/0.0f, stream);
      };
      float milliseconds = MeasureCudaExecutionTime(cuda_run, stream, 10, 100);
      std::cout << std::left << "Row-based Activation " << GetActivationName<Activation, T>() << " m=" << std::setw(6)
                << m << " n=" << std::setw(6) << n << " execution 1 times " << std::setw(10) << milliseconds << " ms"
                << std::endl;
      DeleteBuffer(output_ref_meta);
      DeleteBuffer(output_meta);
      DeleteBuffer(input_meta);
    }
  }

  template <template <typename T> class Activation, typename T>
  void TestRowBasedActivationWithLimit(cudaStream_t stream) {
    const size_t m = 64, n = 4096;
    const float swiglu_limit = 10.0f;

    BufferMeta input_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {m, n}, /*is_random_init*/ true);
    BufferMeta output_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {m, n / 2}, false);
    BufferMeta output_ref_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {m, n / 2}, false);

    std::string type_str = "float";
    if (std::is_same<T, half>::value)
      type_str = "half";
    else if (std::is_same<T, __nv_bfloat16>::value)
      type_str = "bfloat16";

    input_meta.SaveToNpy<T>("row_based_activation_test_input.npy");

    // Extended Python command: includes --swiglu_limit
    std::stringstream ss;
    ss << "python activation_test.py --activation_kernel=InvokeRowBasedActivation"
       << " --type=" << type_str << " --activation=silu"
       << " --swiglu_limit=" << swiglu_limit;
    system(ss.str().c_str());

    output_ref_meta.LoadNpy<T>("row_based_activation_test_output.npy", MemoryType::MEMORY_GPU);

    InvokeRowBasedActivation<Activation, T, /*HAS_SWIGLU_LIMIT*/ true>(reinterpret_cast<T*>(output_meta.data_ptr),
                                                                       reinterpret_cast<T*>(input_meta.data_ptr), m, n,
                                                                       swiglu_limit, stream);
    cudaStreamSynchronize(stream);
    cudaDeviceSynchronize();

    EXPECT_TRUE(CheckResult<T>(
        "row_based_activation_swiglu_" + type_str + "_m_" + std::to_string(m) + "_n_" + std::to_string(n),
        output_ref_meta, output_meta, 1e-2f, 1e-2f));

    DeleteBuffer(output_ref_meta);
    DeleteBuffer(output_meta);
    DeleteBuffer(input_meta);
  }

  template <typename T>
  static const char* GenericActivationTypeName() {
    if constexpr (std::is_same_v<T, half>) {
      return "half";
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      return "BF16";
    } else {
      return "float";
    }
  }

  template <typename T>
  static T FloatToActivationType(float v) {
    if constexpr (std::is_same_v<T, half>) {
      return __float2half(v);
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      return __float2bfloat16(v);
    } else {
      return static_cast<T>(v);
    }
  }

  template <typename T>
  static float ActivationTypeToFloat(T v) {
    if constexpr (std::is_same_v<T, half>) {
      return __half2float(v);
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      return __bfloat162float(v);
    } else {
      return static_cast<float>(v);
    }
  }

  // N=10944 hits the else launch path (n/4/packed_elems > 1024) for float/half/BF16.
  // Historical int32 grid-stride overflow on the second iteration:
  //   packed_elems=2 (half/BF16): ~1.5*m*N > INT32_MAX => m >= 130817
  //   packed_elems=1 (float):     ~2*m*N   > INT32_MAX => m >= 98113
  // Cover last safe-ish border, first overflow border, and a production crash scale (163828).
  template <typename T>
  void TestGenericActivationInt32OverflowBoundary(cudaStream_t stream) {
    constexpr int32_t kN = 10944;  // DeepSeek-V2-Lite intermediate_size
    constexpr int32_t packed_elems = ElemsNum<typename PackTypeAlign<T>::type>::value;
    const char* type_name = GenericActivationTypeName<T>();

    std::vector<int32_t> overflow_ms;
    if constexpr (packed_elems == 1) {
      // 98112: 2*m*N still <= INT32_MAX; 98113: first m where 2*m*N > INT32_MAX.
      overflow_ms = {98112, 98113, 163828};
    } else {
      // 130816: 1.5*m*N still <= INT32_MAX; 130817: first m where 1.5*m*N > INT32_MAX.
      overflow_ms = {130816, 130817, 163828};
    }

    size_t free_bytes = 0;
    size_t total_bytes = 0;
    CHECK_NVIDIA_CUDA_ERROR(cudaMemGetInfo(&free_bytes, &total_bytes));

    int ran_cases = 0;
    for (const int32_t m : overflow_ms) {
      const size_t elems = static_cast<size_t>(m) * static_cast<size_t>(kN);
      // input/gated/output + ~256MB slack for allocator fragmentation.
      const size_t need_bytes = elems * sizeof(T) * 3 + (256ull << 20);
      if (free_bytes < need_bytes) {
        std::cout << "Skip m=" << m << " n=" << kN << " " << type_name
                  << " overflow boundary: need≈" << (need_bytes >> 20) << "MB free, have≈" << (free_bytes >> 20)
                  << "MB" << std::endl;
        continue;
      }
      ++ran_cases;

      BufferMeta input_meta = CreateBuffer<T>(MemoryType::MEMORY_GPU, {static_cast<size_t>(m), static_cast<size_t>(kN)},
                                              /*is_random_init*/ false);
      BufferMeta gated_weight_meta =
          CreateBuffer<T>(MemoryType::MEMORY_GPU, {static_cast<size_t>(m), static_cast<size_t>(kN)},
                          /*is_random_init*/ false);
      BufferMeta output_meta =
          CreateBuffer<T>(MemoryType::MEMORY_GPU, {static_cast<size_t>(m), static_cast<size_t>(kN)},
                          /*is_random_init*/ false);

      // Seed a few elements so we can spot-check correctness after a successful launch.
      const T gate_seed = FloatToActivationType<T>(1.0f);
      const T up_seed = FloatToActivationType<T>(2.0f);
      const size_t last = elems - 1;
      CHECK_NVIDIA_CUDA_ERROR(cudaMemcpyAsync(reinterpret_cast<T*>(input_meta.data_ptr), &gate_seed, sizeof(T),
                                              cudaMemcpyHostToDevice, stream));
      CHECK_NVIDIA_CUDA_ERROR(cudaMemcpyAsync(reinterpret_cast<T*>(input_meta.data_ptr) + last, &gate_seed, sizeof(T),
                                              cudaMemcpyHostToDevice, stream));
      CHECK_NVIDIA_CUDA_ERROR(cudaMemcpyAsync(reinterpret_cast<T*>(gated_weight_meta.data_ptr), &up_seed, sizeof(T),
                                              cudaMemcpyHostToDevice, stream));
      CHECK_NVIDIA_CUDA_ERROR(cudaMemcpyAsync(reinterpret_cast<T*>(gated_weight_meta.data_ptr) + last, &up_seed,
                                              sizeof(T), cudaMemcpyHostToDevice, stream));

      CHECK_NVIDIA_CUDA_ERROR(cudaMemcpyAsync(output_meta.data_ptr, input_meta.data_ptr, sizeof(T) * elems,
                                              cudaMemcpyDeviceToDevice, stream));

      const int* ia3_tasks = nullptr;
      const T* bias = nullptr;
      const T* ia3_weights = nullptr;
      const T* gated_bias = nullptr;
      InvokeGenericActivation<SiluActivation, T, T>(
          reinterpret_cast<T*>(output_meta.data_ptr), bias, reinterpret_cast<const T*>(gated_weight_meta.data_ptr),
          gated_bias, ia3_tasks, ia3_weights, m, kN, /*int8_mode=*/0, /*activation_in=*/nullptr,
          /*activation_out=*/nullptr, /*padding_offset=*/nullptr, /*seq_len=*/0, stream);

      // Must synchronize and surface sticky illegal-memory-access if int32 indexing regresses.
      ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess)
          << "InvokeGenericActivation " << type_name << " overflow-boundary failed at m=" << m << " n=" << kN
          << " (likely int32 grid-stride index overflow)";
      ASSERT_EQ(cudaGetLastError(), cudaSuccess)
          << "CUDA sticky error after InvokeGenericActivation at m=" << m << " n=" << kN;

      T out_first{};
      T out_last{};
      CHECK_NVIDIA_CUDA_ERROR(cudaMemcpyAsync(&out_first, reinterpret_cast<T*>(output_meta.data_ptr), sizeof(T),
                                              cudaMemcpyDeviceToHost, stream));
      CHECK_NVIDIA_CUDA_ERROR(cudaMemcpyAsync(&out_last, reinterpret_cast<T*>(output_meta.data_ptr) + last, sizeof(T),
                                              cudaMemcpyDeviceToHost, stream));
      CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

      const float expected = (1.0f / (1.0f + std::exp(-1.0f))) * 2.0f;  // silu(1) * 2
      const float tol = std::is_same_v<T, float> ? 1e-5f : 2e-2f;
      EXPECT_NEAR(ActivationTypeToFloat(out_first), expected, tol) << "first element mismatch at m=" << m;
      EXPECT_NEAR(ActivationTypeToFloat(out_last), expected, tol) << "last element mismatch at m=" << m;

      DeleteBuffer(output_meta);
      DeleteBuffer(gated_weight_meta);
      DeleteBuffer(input_meta);

      CHECK_NVIDIA_CUDA_ERROR(cudaMemGetInfo(&free_bytes, &total_bytes));
    }

    if (ran_cases == 0) {
      GTEST_SKIP() << "Insufficient GPU memory to run any " << type_name
                   << " GenericActivation int32 overflow boundary case (n=" << kN << ")";
    }
  }
};

TEST_F(LlamaNvidiaActivationTestSuit, HalfActivationCommonTest) { TestActivation<half>(stream); }

TEST_F(LlamaNvidiaActivationTestSuit, FloatActivationCommonTest) { TestActivation<float>(stream); }

TEST_F(LlamaNvidiaActivationTestSuit, Bf16ActivationCommonTest) { TestActivation<__nv_bfloat16>(stream); }

TEST_F(LlamaNvidiaActivationTestSuit, HalfRowBasedActivationCommonTest) {
  TestRowBasedActivation<SiluActivation, half>(stream);
}

TEST_F(LlamaNvidiaActivationTestSuit, FloatRowBasedActivationCommonTest) {
  TestRowBasedActivation<SiluActivation, float>(stream);
}

TEST_F(LlamaNvidiaActivationTestSuit, Bf16RowBasedActivationCommonTest) {
  TestRowBasedActivation<SiluActivation, __nv_bfloat16>(stream);
}

TEST_F(LlamaNvidiaActivationTestSuit, HalfRowBasedActivationSwigluLimitTest) {
  TestRowBasedActivationWithLimit<SiluActivation, half>(stream);
}
TEST_F(LlamaNvidiaActivationTestSuit, FloatRowBasedActivationSwigluLimitTest) {
  TestRowBasedActivationWithLimit<SiluActivation, float>(stream);
}
TEST_F(LlamaNvidiaActivationTestSuit, Bf16RowBasedActivationSwigluLimitTest) {
  TestRowBasedActivationWithLimit<SiluActivation, __nv_bfloat16>(stream);
}

// Regression for int32 grid-stride overflow in InvokeGenericActivationKernel.
// half/BF16 (packed_elems=2): second-iter id ≈ 1.5*m*N crosses INT32_MAX at m >= 130817.
// float (packed_elems=1): second-iter id ≈ 2*m*N crosses INT32_MAX at m >= 98113.
// Existing common cases only go to m=64000 and cannot catch a regression to int32 indices.
TEST_F(LlamaNvidiaActivationTestSuit, Bf16GenericActivationInt32OverflowBoundaryTest) {
  TestGenericActivationInt32OverflowBoundary<__nv_bfloat16>(stream);
}

TEST_F(LlamaNvidiaActivationTestSuit, HalfGenericActivationInt32OverflowBoundaryTest) {
  TestGenericActivationInt32OverflowBoundary<half>(stream);
}

TEST_F(LlamaNvidiaActivationTestSuit, FloatGenericActivationInt32OverflowBoundaryTest) {
  TestGenericActivationInt32OverflowBoundary<float>(stream);
}

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels