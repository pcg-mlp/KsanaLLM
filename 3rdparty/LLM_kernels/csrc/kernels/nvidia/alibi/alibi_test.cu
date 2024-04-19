/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include <gtest/gtest.h>

#include "csrc/utils/nvidia/cuda_utils.h"
#include "tests/kernels/nvidia/utils/testsuit_base.h"

#include "csrc/kernels/nvidia/alibi/alibi.h"

namespace llm_kernels {
namespace nvidia {
namespace test {

class LlamaNvidiaAlibiTestSuit : public NvidiaTestSuitBase {
 public:
  void SetUp() override { NvidiaTestSuitBase::SetUp(); }

  void TearDown() override { NvidiaTestSuitBase::TearDown(); }

 protected:
  using NvidiaTestSuitBase::stream;
  const std::vector<int> num_heads_list = {40, 80};

 protected:
  void GetAlibiSlopesRef(float *slopes, int total_num_heads) {
    int closest_power_of_2 = pow(2, floor(log2(total_num_heads)));
    float base = pow(2, -(pow(2, -(log2(closest_power_of_2) - 3))));
    for (int i = 0; i < closest_power_of_2; i++) {
      slopes[i] = pow(base, i + 1);
    }

    if (closest_power_of_2 != total_num_heads) {
      float extra_base = pow(2, -(pow(2, -(log2(2 * closest_power_of_2) - 3))));
      int num_remaining_heads = min(closest_power_of_2, total_num_heads - closest_power_of_2);
      for (int i = 0; i < num_remaining_heads; i++) {
        slopes[closest_power_of_2 + i] = pow(extra_base, 2 * i + 1);
      }
    }
  }

  void TestAlibi(const int num_heads) {
    BufferMeta output_meta = CreateBuffer<float>(MemoryType::MEMORY_GPU, {static_cast<size_t>(num_heads)},
                                                 /*is_random_init*/ false);
    BufferMeta output_cpu_meta = CopyToHost<float>(output_meta);

    GetAlibiSlopesRef(reinterpret_cast<float *>(output_cpu_meta.data_ptr), num_heads);
    BufferMeta output_ref_meta = CopyToDevice<float>(output_cpu_meta);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    GetAlibiSlopesCuda(reinterpret_cast<float *>(output_meta.data_ptr), num_heads, stream);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    EXPECT_TRUE(CheckResult<float>("get_alibi_slopes_float_num_heads_" + std::to_string(num_heads), output_ref_meta,
                                   output_meta, 1e-5f, 1e-5f));

    DeleteBuffer(output_ref_meta);
    DeleteBuffer(output_cpu_meta);
    DeleteBuffer(output_meta);
  }
};

TEST_F(LlamaNvidiaAlibiTestSuit, FloatAlibiTest) {
  for (int num_heads : num_heads_list) {
    TestAlibi(num_heads);
  }
}

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels
