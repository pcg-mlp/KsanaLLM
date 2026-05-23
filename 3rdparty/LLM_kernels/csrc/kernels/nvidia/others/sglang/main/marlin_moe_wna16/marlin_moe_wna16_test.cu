/*
 * Copyright 2026 Tencent Inc. All rights reserved.
 *
 * Unit test for the marlin_moe_wna16 operator.
 * This is a smoke test that verifies the kernel runs without crashing.
 */

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <cstring>
#include <vector>

#include "csrc/kernels/nvidia/others/sglang/main/marlin_moe_wna16/marlin_moe_wna16_wrapper.h"
#include "csrc/kernels/nvidia/others/sglang/main/marlin_moe_wna16/ops.h"
#include "csrc/utils/nvidia/cuda_utils.h"
#include "tests/kernels/nvidia/utils/testsuit_base.h"

namespace llm_kernels {
namespace nvidia {
namespace test {

class MarlinMoeWna16TestSuit : public NvidiaTestSuitBase {
 public:
  void SetUp() override { NvidiaTestSuitBase::SetUp(); }
  void TearDown() override { NvidiaTestSuitBase::TearDown(); }
};

// ==========================================================================
// MXFP4 (W4A16) smoke test — this is the DSv4 inference path.
// Uses kFE2M1f weight type + kFE8M0fnu scale type, group_size=32, bf16 compute.
// ==========================================================================
TEST_F(MarlinMoeWna16TestSuit, SmokeTestMXFP4_BF16) {
  // Problem dimensions (multiples of 64 required)
  const int64_t size_m = 16;
  const int64_t size_k = 128;   // must be multiple of 64; group_size=32 → num_groups=4
  const int64_t size_n = 128;
  const int64_t num_experts = 2;
  const int64_t top_k = 1;
  const int64_t moe_block_size = 16;

  // MXFP4: group_size=32 → num_groups = size_k / 32 = 4
  const int64_t group_size = 32;
  const int64_t num_groups = size_k / group_size;  // 4

  // No act_order, no zero points
  const int64_t g_idx_len = 0;
  const int64_t perm_len = 0;
  const int64_t b_zeros_len = 0;

  // MoE routing: all tokens go to expert 0
  const int64_t sorted_token_ids_len = size_m;  // = moe_block_size

  // MXFP4: kFE2M1f weight + kFE8M0fnu scale (scale_dtype=3)
  const ::marlin_moe_wna16::ScalarTypeId b_q_type_id = ::marlin_moe_wna16::kFE2M1f.id();
  const int scale_dtype = 3;    // fp8_e8m0fnu
  const int compute_dtype = 1;  // bf16

  const bool is_k_full = true;
  const bool use_atomic_add = false;
  const bool use_fp32_reduce = false;
  const bool is_zp_float = false;
  const bool mul_topk_weights = false;
  const bool is_ep = false;

  // -----------------------------------------------------------------------
  // Weight tensor in marlin format for FP4 (kFE2M1f):
  //   pack_factor = 32 / 4 = 8
  //   tile_size = 16
  //   shape = [num_experts, size_k / tile_size, size_n * tile_size / pack_factor]
  //         = [2, 128/16, 128*16/8] = [2, 8, 256]
  //
  // For MXFP4 (E2M1): each 4-bit value encodes a sign+exp+mantissa float.
  // Value 0b0000 = +0.0, so fill with zeros.
  // -----------------------------------------------------------------------
  const int64_t tile_size = marlin_moe_wna16::kTileSize;
  const int64_t pack_factor = 8;
  const int64_t weight_dim1 = size_k / tile_size;                        // 8
  const int64_t weight_dim2 = size_n * tile_size / pack_factor;          // 256
  const int64_t weight_elems = num_experts * weight_dim1 * weight_dim2;  // 4096

  // All zeros = fp4 value 0.0
  std::vector<int32_t> h_weight(weight_elems, 0);

  // -----------------------------------------------------------------------
  // Scales: [num_experts, num_groups, size_n] = [2, 4, 128]
  // Format: fp8_e8m0fnu (unsigned 8-bit exponent-only float)
  // E8M0 encoding: value = 2^(stored_value - 127)
  // For scale=1.0: stored_value = 127 = 0x7F
  // -----------------------------------------------------------------------
  const int64_t scales_elems = num_experts * num_groups * size_n;  // 1024
  // Each scale is 1 byte (fp8_e8m0fnu), stored as uint8_t
  std::vector<uint8_t> h_scales(scales_elems, 0x7F);  // 127 → scale = 2^0 = 1.0

  // -----------------------------------------------------------------------
  // Activations: [size_m, size_k] = [16, 128] in bf16
  // Use small random-ish values so output is non-trivial
  // -----------------------------------------------------------------------
  const int64_t a_elems = size_m * size_k;
  std::vector<__nv_bfloat16> h_a(a_elems);
  for (int64_t i = 0; i < a_elems; ++i) {
    h_a[i] = __float2bfloat16(0.01f * (i % 7 - 3));  // small values in [-0.03, 0.03]
  }

  // Output: [size_m * top_k, size_n] = [16, 128] in bf16
  const int64_t c_elems = size_m * top_k * size_n;

  // -----------------------------------------------------------------------
  // MoE routing: sorted_token_ids = [0..15], expert_ids = [0], num_tokens_past_padded = 16
  // -----------------------------------------------------------------------
  std::vector<int32_t> h_sorted_token_ids(sorted_token_ids_len);
  for (int32_t i = 0; i < sorted_token_ids_len; ++i) {
    h_sorted_token_ids[i] = i;
  }
  const int64_t num_blocks = sorted_token_ids_len / moe_block_size;
  std::vector<int32_t> h_expert_ids(num_blocks, 0);
  std::vector<int32_t> h_num_tokens_past_padded(1, static_cast<int32_t>(sorted_token_ids_len));
  std::vector<float> h_topk_weights(size_m * top_k, 1.0f);

  // -----------------------------------------------------------------------
  // Workspace
  // -----------------------------------------------------------------------
  int64_t workspace_elems = marlin_moe_wna16::get_workspace_size(size_n, sorted_token_ids_len, moe_block_size);
  if (workspace_elems < 1) workspace_elems = 1;

  int64_t c_tmp_elems = marlin_moe_wna16::get_c_tmp_size(size_n, sorted_token_ids_len, moe_block_size, b_q_type_id);
  if (c_tmp_elems < 1) c_tmp_elems = 1;

  int64_t a_tmp_elems = marlin_moe_wna16::get_a_tmp_size(size_m, size_k, top_k);
  if (a_tmp_elems < 1) a_tmp_elems = 1;

  // -----------------------------------------------------------------------
  // Allocate GPU buffers
  // -----------------------------------------------------------------------
  void* d_a = nullptr;
  void* d_c = nullptr;
  void* d_c_tmp = nullptr;
  void* d_b_q_weight = nullptr;
  void* d_b_scales = nullptr;
  void* d_a_tmp = nullptr;
  int32_t* d_sorted_token_ids = nullptr;
  int32_t* d_expert_ids = nullptr;
  int32_t* d_num_tokens_past_padded = nullptr;
  float* d_topk_weights = nullptr;
  int* d_workspace = nullptr;

  CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&d_a, a_elems * sizeof(__nv_bfloat16)));
  CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&d_c, c_elems * sizeof(__nv_bfloat16)));
  CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&d_c_tmp, c_tmp_elems * sizeof(float)));
  CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&d_b_q_weight, weight_elems * sizeof(int32_t)));
  CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&d_b_scales, scales_elems * sizeof(uint8_t)));
  CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&d_a_tmp, a_tmp_elems * sizeof(__nv_bfloat16)));
  CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&d_sorted_token_ids, sorted_token_ids_len * sizeof(int32_t)));
  CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&d_expert_ids, num_blocks * sizeof(int32_t)));
  CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&d_num_tokens_past_padded, sizeof(int32_t)));
  CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&d_topk_weights, size_m * top_k * sizeof(float)));
  CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&d_workspace, workspace_elems * sizeof(int)));

  allocated_gpu_buffers.insert(reinterpret_cast<uintptr_t>(d_a));
  allocated_gpu_buffers.insert(reinterpret_cast<uintptr_t>(d_c));
  allocated_gpu_buffers.insert(reinterpret_cast<uintptr_t>(d_c_tmp));
  allocated_gpu_buffers.insert(reinterpret_cast<uintptr_t>(d_b_q_weight));
  allocated_gpu_buffers.insert(reinterpret_cast<uintptr_t>(d_b_scales));
  allocated_gpu_buffers.insert(reinterpret_cast<uintptr_t>(d_a_tmp));
  allocated_gpu_buffers.insert(reinterpret_cast<uintptr_t>(d_sorted_token_ids));
  allocated_gpu_buffers.insert(reinterpret_cast<uintptr_t>(d_expert_ids));
  allocated_gpu_buffers.insert(reinterpret_cast<uintptr_t>(d_num_tokens_past_padded));
  allocated_gpu_buffers.insert(reinterpret_cast<uintptr_t>(d_topk_weights));
  allocated_gpu_buffers.insert(reinterpret_cast<uintptr_t>(d_workspace));

  // -----------------------------------------------------------------------
  // Upload to device
  // -----------------------------------------------------------------------
  CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(d_a, h_a.data(), a_elems * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
  CHECK_NVIDIA_CUDA_ERROR(cudaMemset(d_c, 0, c_elems * sizeof(__nv_bfloat16)));
  CHECK_NVIDIA_CUDA_ERROR(cudaMemset(d_c_tmp, 0, c_tmp_elems * sizeof(float)));
  CHECK_NVIDIA_CUDA_ERROR(cudaMemset(d_workspace, 0, workspace_elems * sizeof(int)));
  CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(d_b_q_weight, h_weight.data(), weight_elems * sizeof(int32_t), cudaMemcpyHostToDevice));
  CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(d_b_scales, h_scales.data(), scales_elems * sizeof(uint8_t), cudaMemcpyHostToDevice));
  CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(d_sorted_token_ids, h_sorted_token_ids.data(), sorted_token_ids_len * sizeof(int32_t), cudaMemcpyHostToDevice));
  CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(d_expert_ids, h_expert_ids.data(), num_blocks * sizeof(int32_t), cudaMemcpyHostToDevice));
  CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(d_num_tokens_past_padded, h_num_tokens_past_padded.data(), sizeof(int32_t), cudaMemcpyHostToDevice));
  CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(d_topk_weights, h_topk_weights.data(), size_m * top_k * sizeof(float), cudaMemcpyHostToDevice));

  // -----------------------------------------------------------------------
  // Call the MXFP4 kernel
  // -----------------------------------------------------------------------
  llm_kernels::nvidia::marlin_moe_wna16::moe_wna16_marlin_gemm(
      /*a=*/d_a,
      /*c=*/d_c,
      /*c_tmp=*/d_c_tmp,
      /*b_q_weight=*/d_b_q_weight,
      /*b_bias=*/nullptr,
      /*b_scales=*/d_b_scales,
      /*global_scale=*/nullptr,
      /*b_zeros=*/nullptr,
      /*g_idx=*/nullptr,
      /*perm=*/nullptr,
      /*a_tmp=*/d_a_tmp,
      /*sorted_token_ids=*/d_sorted_token_ids,
      /*expert_ids=*/d_expert_ids,
      /*num_tokens_past_padded=*/d_num_tokens_past_padded,
      /*topk_weights=*/d_topk_weights,
      /*workspace=*/d_workspace,
      /*moe_block_size=*/moe_block_size,
      /*top_k=*/top_k,
      /*mul_topk_weights=*/mul_topk_weights,
      /*is_ep=*/is_ep,
      /*b_q_type_id=*/b_q_type_id,
      /*scale_dtype=*/scale_dtype,
      /*size_m=*/size_m,
      /*size_n=*/size_n,
      /*size_k=*/size_k,
      /*num_groups=*/num_groups,
      /*sorted_token_ids_len=*/sorted_token_ids_len,
      /*g_idx_len=*/g_idx_len,
      /*perm_len=*/perm_len,
      /*b_zeros_len=*/b_zeros_len,
      /*is_k_full=*/is_k_full,
      /*use_atomic_add=*/use_atomic_add,
      /*use_fp32_reduce=*/use_fp32_reduce,
      /*is_zp_float=*/is_zp_float,
      /*compute_dtype=*/compute_dtype,
      /*stream=*/stream);

  // -----------------------------------------------------------------------
  // Verify: no crash, no CUDA errors
  // -----------------------------------------------------------------------
  CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
  CHECK_NVIDIA_CUDA_ERROR(cudaGetLastError());

  // With weights=0.0 (FP4 zero) and scale=1.0, output should be all zeros
  // regardless of input activations. Verify a sample.
  std::vector<__nv_bfloat16> h_c(c_elems);
  CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(h_c.data(), d_c, c_elems * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

  // Check that output is all zeros (W=0 → C = A * 0 = 0)
  int non_zero_count = 0;
  for (int64_t i = 0; i < c_elems; ++i) {
    float val = __bfloat162float(h_c[i]);
    if (std::abs(val) > 1e-5f) {
      non_zero_count++;
    }
  }
  // With zero weights, output should be zero (or very close)
  EXPECT_EQ(non_zero_count, 0) << "Expected all-zero output with zero weights, got "
                                << non_zero_count << " non-zero elements";
}

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels
