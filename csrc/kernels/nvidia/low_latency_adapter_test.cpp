/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#include <cstring>
#include <vector>

#include "cuda_runtime.h"
#include "test.h"

#include "csrc/kernels/nvidia/moe/cutlass_moe/low_latency_adapter.h"
#include "device/nvidia/cuda_utils.h"

namespace ksana_llm {

class LowLatencyAdapterTest : public testing::Test {
 public:
  void SetUp() override {
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaStreamCreate(&stream));
  }

  void TearDown() override {
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (stream) {
      CUDA_CHECK(cudaStreamDestroy(stream));
    }
  }

 protected:
  int device{-1};
  cudaStream_t stream{nullptr};
};

// 验证 InvokeAdaptLowLatencyDispatch 生成正确的 slot IDs 和 weights
TEST_F(LowLatencyAdapterTest, BasicSlotAssignment) {
  constexpr int num_experts = 2;
  constexpr int tokens_per_expert = 4;
  constexpr int ep_rank = 0;
  constexpr int num_slots = 8;
  constexpr int total_tokens = num_experts * tokens_per_expert;

  // recv_expert_count: expert 0 有 2 个有效 token, expert 1 有 3 个
  std::vector<int32_t> h_recv_count = {2, 3};

  int32_t* d_recv_count;
  int32_t* d_slots;
  float* d_weights;
  CUDA_CHECK(cudaMalloc(&d_recv_count, num_experts * sizeof(int32_t)));
  CUDA_CHECK(cudaMalloc(&d_slots, total_tokens * sizeof(int32_t)));
  CUDA_CHECK(cudaMalloc(&d_weights, total_tokens * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_recv_count, h_recv_count.data(), num_experts * sizeof(int32_t), cudaMemcpyHostToDevice));

  llm_kernels::nvidia::InvokeAdaptLowLatencyDispatch(d_recv_count, d_slots, d_weights, num_experts, tokens_per_expert,
                                                      ep_rank, num_slots, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::vector<int32_t> h_slots(total_tokens);
  std::vector<float> h_weights(total_tokens);
  CUDA_CHECK(cudaMemcpy(h_slots.data(), d_slots, total_tokens * sizeof(int32_t), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_weights.data(), d_weights, total_tokens * sizeof(float), cudaMemcpyDeviceToHost));

  // expert 0: slot=0 for valid, -1 for invalid
  EXPECT_EQ(h_slots[0], 0);   // expert 0, token 0: valid
  EXPECT_EQ(h_slots[1], 0);   // expert 0, token 1: valid
  EXPECT_EQ(h_slots[2], -1);  // expert 0, token 2: invalid
  EXPECT_EQ(h_slots[3], -1);  // expert 0, token 3: invalid

  // expert 1: slot=1 for valid, -1 for invalid
  EXPECT_EQ(h_slots[4], 1);   // expert 1, token 0: valid
  EXPECT_EQ(h_slots[5], 1);   // expert 1, token 1: valid
  EXPECT_EQ(h_slots[6], 1);   // expert 1, token 2: valid
  EXPECT_EQ(h_slots[7], -1);  // expert 1, token 3: invalid

  // 所有 weights 应为 1.0
  for (int i = 0; i < total_tokens; ++i) {
    EXPECT_FLOAT_EQ(h_weights[i], 1.0f) << "Weight mismatch at index " << i;
  }

  CUDA_CHECK(cudaFree(d_recv_count));
  CUDA_CHECK(cudaFree(d_slots));
  CUDA_CHECK(cudaFree(d_weights));
}

}  // namespace ksana_llm
