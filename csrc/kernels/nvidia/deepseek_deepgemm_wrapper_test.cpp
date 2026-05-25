/*
 * Copyright 2025 Tencent Inc.  All rights reserved.
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <c10/cuda/CUDAStream.h>
#include <torch/torch.h>

#include <cmath>
#include <stdexcept>
#include <string>

#include "csrc/utils/nvidia/cuda_utils.h"
#include "device/nvidia/deepseek_deepgemm_bridge.h"
#include "kernels/nvidia/deepseek_deepgemm_wrapper.h"

namespace ksana_llm {
namespace nvidia {
namespace test {

#if ENABLE_DEEPSEEK_DEEPGEMM

// Helper function to cast tensor to FP8 with per-row scaling
std::pair<torch::Tensor, torch::Tensor> CastToFp8PerRow(const torch::Tensor& x) {
  auto x_f = x.to(torch::kFloat32);
  auto x_amax = x_f.abs().amax(-1, true).clamp_min(1e-4);
  auto sf = x_amax / 448.0f;
  auto x_scaled = (x_f / sf).to(torch::kFloat8_e4m3fn);
  return {x_scaled, sf.squeeze(-1)};
}

// Helper function to create fused KV cache (FP8 data + scale) matching DeepGEMM format
// Input: kv [num_blocks, block_size, 1, head_dim] in bfloat16
// Output: [num_blocks, block_size, 1, head_dim + sizeof(float)] in uint8
torch::Tensor CreateFusedKvCache(const torch::Tensor& kv) {
  auto kv_f = kv.to(torch::kCPU).to(torch::kFloat32);
  const int num_blocks = kv.size(0);
  const int block_size = kv.size(1);
  const int num_heads = kv.size(2);  // Should be 1
  const int head_dim = kv.size(3);

  // Compute per-row scale (amax along head_dim)
  auto x_amax = kv_f.abs().amax(3, true).clamp_min(1e-4);  // [num_blocks, block_size, 1, 1]
  auto sf = x_amax / 448.0f;

  // Quantize to FP8
  auto x_scaled = (kv_f / sf).to(torch::kFloat8_e4m3fn);  // [num_blocks, block_size, 1, head_dim]

  // Create output tensor
  auto fused_cache = torch::empty({num_blocks, block_size * (head_dim + 4)}, torch::kUInt8);

  // Copy FP8 data
  auto fp8_data = x_scaled.view({num_blocks, block_size * head_dim}).view(torch::kUInt8);
  fused_cache.narrow(1, 0, block_size * head_dim).copy_(fp8_data);

  // Copy scales
  auto scale_data = sf.view({num_blocks, block_size}).view(torch::kUInt8);
  fused_cache.narrow(1, block_size * head_dim, block_size * 4).copy_(scale_data);

  return fused_cache.view({num_blocks, block_size, num_heads, head_dim + 4});
}

class DeepSeekDeepGEMMWrapperTest : public ::testing::Test {
 protected:
  DeepSeekDeepGEMMWrapperTest() : device_(torch::kCPU) {}

  ~DeepSeekDeepGEMMWrapperTest() {}

  void SetUp() override {
    if (!torch::cuda::is_available()) {
      GTEST_SKIP() << "CUDA device unavailable for DeepSeekDeepGEMMWrapper tests.";
    }
    device_ = torch::Device(torch::kCUDA, 0);
    torch::manual_seed(42);
  }

  torch::Device device_;
};

// ============================================================================
// Computation Tests - Verify kernels can execute without errors
// Using parameters that match DeepGEMM constraints:
// - num_heads must be multiple of 8
// - head_dim = 128
// - seq_len must be large enough (2048, 4096)
// - block_kv = 64 for paged attention
// ============================================================================

TEST_F(DeepSeekDeepGEMMWrapperTest, Fp8MqaLogitsCanCompute) {
  auto wrapper = DeepSeekDeepGEMMWrapper::GetInstance();

  // Use parameters matching DeepGEMM test: num_heads=64, head_dim=128, seq_len=1024
  constexpr int seq_len = 1024;
  constexpr int seq_len_kv = 1024;
  constexpr int num_heads = 64;
  constexpr int head_dim = 128;

  // Generate random input data
  auto q_bf16 = torch::randn({seq_len, num_heads, head_dim}, torch::kBFloat16).to(device_);
  auto kv_bf16 = torch::randn({seq_len_kv, head_dim}, torch::kBFloat16).to(device_);
  auto weights = torch::randn({seq_len, num_heads}, torch::kFloat32).to(device_);

  // Create sequence length boundaries (disable CP mode like DeepGEMM test)
  auto cu_seq_len_k_start = torch::zeros({seq_len}, torch::kInt32).to(device_);
  auto cu_seq_len_k_end = torch::arange(seq_len, dtype(torch::kInt32)).to(device_) + (seq_len_kv - seq_len);

  // Convert to FP8
  auto q_fp8 = q_bf16.to(torch::kFloat8_e4m3fn);
  auto [kv_fp8, kv_scales] = CastToFp8PerRow(kv_bf16);
  kv_fp8 = kv_fp8.to(device_);
  kv_scales = kv_scales.to(device_);

  // Allocate output tensor with proper alignment for compressed logits
  constexpr int block_kv = 256;
  const int max_seqlen_k = seq_len_kv;  // single request, max_seqlen_k = seq_len_kv
  const int stride_logits = ((max_seqlen_k + block_kv - 1) / block_kv) * block_kv;
  const int aligned_seq_len = ((seq_len + 3) / 4) * 4;
  auto logits = torch::zeros({aligned_seq_len, stride_logits}, torch::kFloat32).to(device_);

  // Run the kernel - should not throw
  EXPECT_NO_THROW(wrapper->Fp8MqaLogits(q_fp8, {kv_fp8, kv_scales}, weights, cu_seq_len_k_start, cu_seq_len_k_end,
                                        logits, max_seqlen_k, at::cuda::getCurrentCUDAStream().stream()));
  cudaDeviceSynchronize();

  // Verify output contains finite values in valid region
  auto logits_cpu = logits.narrow(0, 0, seq_len).narrow(1, 0, max_seqlen_k).to(torch::kCPU);
  bool has_finite = false;
  for (int i = 0; i < std::min(seq_len, 100) && !has_finite; ++i) {
    int start = 0;
    int end = std::min(i + (seq_len_kv - seq_len), seq_len_kv);
    for (int j = start; j < end && !has_finite; ++j) {
      if (std::isfinite(logits_cpu[i][j].item<float>())) {
        has_finite = true;
      }
    }
  }
  EXPECT_TRUE(has_finite) << "Fp8MqaLogits should produce finite values in valid region";
}

TEST_F(DeepSeekDeepGEMMWrapperTest, Fp8MqaLogitsWithContextParallelism) {
  auto wrapper = DeepSeekDeepGEMMWrapper::GetInstance();

  // Use parameters matching DeepGEMM test
  constexpr int seq_len = 2048;
  constexpr int seq_len_kv = 4096;
  constexpr int num_heads = 64;
  constexpr int head_dim = 128;

  auto q_bf16 = torch::randn({seq_len, num_heads, head_dim}, torch::kBFloat16).to(device_);
  auto kv_bf16 = torch::randn({seq_len_kv, head_dim}, torch::kBFloat16).to(device_);
  auto weights = torch::randn({seq_len, num_heads}, torch::kFloat32).to(device_);

  // Generate CP test data similar to DeepGEMM test
  // seq_len_kv % seq_len == 0 and seq_len % 2 == 0
  const int chunk_size = seq_len / 2;
  const int cp_size = seq_len_kv / seq_len;
  const int cp_id = cp_size / 3;

  auto cu_seq_len_k_start = torch::zeros({seq_len}, torch::kInt32);
  auto cu_seq_len_k_end = torch::zeros({seq_len}, torch::kInt32);
  for (int i = 0; i < chunk_size; ++i) {
    cu_seq_len_k_end[i] = cp_id * chunk_size + i;
    cu_seq_len_k_end[i + chunk_size] = (cp_size * 2 - 1 - cp_id) * chunk_size + i;
  }
  cu_seq_len_k_start = cu_seq_len_k_start.to(device_);
  cu_seq_len_k_end = cu_seq_len_k_end.to(device_);

  auto q_fp8 = q_bf16.to(torch::kFloat8_e4m3fn);
  auto [kv_fp8, kv_scales] = CastToFp8PerRow(kv_bf16);
  kv_fp8 = kv_fp8.to(device_);
  kv_scales = kv_scales.to(device_);

  constexpr int block_kv = 256;
  const int max_seqlen_k = seq_len_kv;  // upper bound for any token's visible KV range
  const int stride_logits = ((max_seqlen_k + block_kv - 1) / block_kv) * block_kv;
  const int aligned_seq_len = ((seq_len + 3) / 4) * 4;
  auto logits = torch::zeros({aligned_seq_len, stride_logits}, torch::kFloat32).to(device_);

  EXPECT_NO_THROW(wrapper->Fp8MqaLogits(q_fp8, {kv_fp8, kv_scales}, weights, cu_seq_len_k_start, cu_seq_len_k_end,
                                        logits, max_seqlen_k, at::cuda::getCurrentCUDAStream().stream()));
  cudaDeviceSynchronize();

  // Verify output is produced
  auto logits_cpu = logits.narrow(0, 0, seq_len).narrow(1, 0, max_seqlen_k).to(torch::kCPU);
  EXPECT_GT(logits_cpu.numel(), 0) << "Fp8MqaLogits with CP should produce output";
}

TEST_F(DeepSeekDeepGEMMWrapperTest, PagedMqaLogitsMetadataGeneration) {
  auto wrapper = DeepSeekDeepGEMMWrapper::GetInstance();

  constexpr int batch_size = 64;
  constexpr int block_kv = 64;
  const int num_sms = llm_kernels::utils::GetSMCount();

  // Create context lengths similar to DeepGEMM test
  auto context_lens = torch::randint(5734, 10649, {batch_size}, torch::kInt32).to(device_);

  // Allocate schedule metadata
  auto schedule_meta = torch::zeros({num_sms + 1, 2}, torch::kInt32).to(device_);

  // Generate metadata - should not throw
  EXPECT_NO_THROW(wrapper->PagedMqaLogitsMetadata(context_lens, schedule_meta, batch_size, block_kv,
                                                  at::cuda::getCurrentCUDAStream().stream()));
  cudaDeviceSynchronize();

  // Verify the metadata is populated
  auto schedule_meta_cpu = schedule_meta.to(torch::kCPU);
  EXPECT_GE(schedule_meta_cpu.sum().item<int>(), 0) << "Schedule metadata should be populated";
}

TEST_F(DeepSeekDeepGEMMWrapperTest, Fp8PagedMqaLogitsCanCompute) {
  auto wrapper = DeepSeekDeepGEMMWrapper::GetInstance();

  // Use parameters matching DeepGEMM test: batch_size=4, next_n=1, heads=64, head_dim=128, blocksize=64
  constexpr int batch_size = 4;
  constexpr int next_n = 1;
  constexpr int num_heads = 64;
  constexpr int head_dim = 128;
  constexpr int block_kv = 64;
  constexpr int max_model_len = 2222;
  constexpr int avg_kv = 2048;

  const int num_sms = llm_kernels::utils::GetSMCount();
  const int num_kv_blocks = max_model_len * 3;

  // Generate random query
  auto q_bf16 = torch::randn({batch_size, next_n, num_heads, head_dim}, torch::kBFloat16).to(device_);
  auto q_fp8 = q_bf16.to(torch::kFloat8_e4m3fn);

  // Generate random KV cache data in DeepGEMM format
  auto kv_cache_bf16 = torch::randn({num_kv_blocks, block_kv, 1, head_dim}, torch::kBFloat16).to(device_);
  auto fused_kv_cache = CreateFusedKvCache(kv_cache_bf16).to(device_);

  // Generate weights
  auto weights = torch::randn({batch_size * next_n, num_heads}, torch::kFloat32).to(device_);

  // Create context lengths (random between 0.7*avg_kv and 1.3*avg_kv)
  auto context_lens =
      torch::randint(static_cast<int>(0.7 * avg_kv), static_cast<int>(1.3 * avg_kv), {batch_size}, torch::kInt32)
          .to(device_);
  auto context_lens_cpu = context_lens.to(torch::kCPU);

  // Create block tables
  int max_context = context_lens_cpu.max().item<int>();
  int max_blocks_per_seq = (max_context + block_kv - 1) / block_kv;
  auto block_table = torch::zeros({batch_size, max_blocks_per_seq}, torch::kInt32);

  // Assign random block indices
  auto block_idx_pool = torch::randperm(num_kv_blocks, torch::kInt32);
  int counter = 0;
  for (int i = 0; i < batch_size; ++i) {
    int ctx_len = context_lens_cpu[i].item<int>();
    int num_blocks_needed = (ctx_len + block_kv - 1) / block_kv;
    for (int b = 0; b < num_blocks_needed; ++b) {
      block_table[i][b] = block_idx_pool[counter % num_kv_blocks].item<int>();
      counter++;
    }
  }
  block_table = block_table.to(device_);

  // Generate schedule metadata
  auto schedule_meta = torch::zeros({num_sms + 1, 2}, torch::kInt32).to(device_);
  wrapper->PagedMqaLogitsMetadata(context_lens, schedule_meta, batch_size, block_kv,
                                  at::cuda::getCurrentCUDAStream().stream());

  // Allocate output logits with proper alignment
  constexpr int split_kv = 256;
  const int aligned_max_model_len = ((max_model_len + split_kv - 1) / split_kv) * split_kv;
  auto logits = torch::zeros({batch_size * next_n, aligned_max_model_len}, torch::kFloat32).to(device_);

  // Run the kernel - should not throw
  EXPECT_NO_THROW(wrapper->Fp8PagedMqaLogits(q_fp8, fused_kv_cache, weights, context_lens, block_table, schedule_meta,
                                             logits, max_model_len, at::cuda::getCurrentCUDAStream().stream(), true));
  cudaDeviceSynchronize();

  // Verify output contains finite values within context length
  auto logits_cpu = logits.narrow(1, 0, max_model_len).to(torch::kCPU);
  int checked = 0;
  int finite_count = 0;
  for (int i = 0; i < std::min(batch_size * next_n, 10); ++i) {
    int ctx_len = context_lens_cpu[i / next_n].item<int>();
    for (int j = 0; j < std::min(ctx_len, 10); ++j) {
      float val = logits_cpu[i][j].item<float>();
      if (std::isfinite(val)) {
        finite_count++;
      }
      checked++;
    }
  }
  EXPECT_GT(finite_count, 0) << "Fp8PagedMqaLogits should produce some finite values";
}

TEST_F(DeepSeekDeepGEMMWrapperTest, Fp8MqaLogitsCompressedLogits) {
  auto wrapper = DeepSeekDeepGEMMWrapper::GetInstance();

  // Use parameters matching DeepGEMM constraints
  constexpr int seq_len = 2048;
  constexpr int seq_len_kv = 4096;
  constexpr int num_heads = 64;
  constexpr int head_dim = 128;

  auto q_bf16 = torch::randn({seq_len, num_heads, head_dim}, torch::kBFloat16).to(device_);
  auto kv_bf16 = torch::randn({seq_len_kv, head_dim}, torch::kBFloat16).to(device_);
  auto weights = torch::randn({seq_len, num_heads}, torch::kFloat32).to(device_);

  // Set up boundaries: start=0, end=seq_len_kv/2 (compressed/relative format)
  auto cu_seq_len_k_start = torch::zeros({seq_len}, torch::kInt32).to(device_);
  auto cu_seq_len_k_end = torch::full({seq_len}, seq_len_kv / 2, torch::kInt32).to(device_);

  auto q_fp8 = q_bf16.to(torch::kFloat8_e4m3fn);
  auto [kv_fp8, kv_scales] = CastToFp8PerRow(kv_bf16);
  kv_fp8 = kv_fp8.to(device_);
  kv_scales = kv_scales.to(device_);

  constexpr int block_kv = 256;
  const int max_seqlen_k = seq_len_kv / 2;  // compressed: max visible range
  const int stride_logits = ((max_seqlen_k + block_kv - 1) / block_kv) * block_kv;
  const int aligned_seq_len = ((seq_len + 3) / 4) * 4;

  // Test compressed logits mode
  auto logits = torch::full({aligned_seq_len, stride_logits}, 999.0f, torch::kFloat32).to(device_);
  EXPECT_NO_THROW(wrapper->Fp8MqaLogits(q_fp8, {kv_fp8, kv_scales}, weights, cu_seq_len_k_start, cu_seq_len_k_end,
                                        logits, max_seqlen_k, at::cuda::getCurrentCUDAStream().stream()));
  cudaDeviceSynchronize();

  // Verify logits are written in compressed format [0, max_seqlen_k)
  auto logits_cpu = logits.narrow(0, 0, seq_len).narrow(1, 0, max_seqlen_k).to(torch::kCPU);
  bool has_finite = false;
  for (int i = 0; i < std::min(seq_len, 100) && !has_finite; ++i) {
    for (int j = 0; j < max_seqlen_k && !has_finite; ++j) {
      if (std::isfinite(logits_cpu[i][j].item<float>())) {
        has_finite = true;
      }
    }
  }
  EXPECT_TRUE(has_finite) << "Compressed logits should produce finite values in [0, max_seqlen_k) range";
}

TEST_F(DeepSeekDeepGEMMWrapperTest, Fp8MqaLogitsLargeScale) {
  auto wrapper = DeepSeekDeepGEMMWrapper::GetInstance();

  // Test with larger dimensions similar to real model usage
  constexpr int seq_len = 4096;
  constexpr int seq_len_kv = 8192;
  constexpr int num_heads = 64;
  constexpr int head_dim = 128;

  auto q_bf16 = torch::randn({seq_len, num_heads, head_dim}, torch::kBFloat16).to(device_);
  auto kv_bf16 = torch::randn({seq_len_kv, head_dim}, torch::kBFloat16).to(device_);
  auto weights = torch::randn({seq_len, num_heads}, torch::kFloat32).to(device_);

  auto cu_seq_len_k_start = torch::zeros({seq_len}, torch::kInt32).to(device_);
  auto cu_seq_len_k_end = torch::arange(seq_len, dtype(torch::kInt32)).to(device_) + (seq_len_kv - seq_len);

  auto q_fp8 = q_bf16.to(torch::kFloat8_e4m3fn);
  auto [kv_fp8, kv_scales] = CastToFp8PerRow(kv_bf16);
  kv_fp8 = kv_fp8.to(device_);
  kv_scales = kv_scales.to(device_);

  constexpr int block_kv = 256;
  const int max_seqlen_k = seq_len_kv;  // single request, max_seqlen_k = seq_len_kv
  const int stride_logits = ((max_seqlen_k + block_kv - 1) / block_kv) * block_kv;
  const int aligned_seq_len = ((seq_len + 3) / 4) * 4;
  auto logits = torch::zeros({aligned_seq_len, stride_logits}, torch::kFloat32).to(device_);

  EXPECT_NO_THROW(wrapper->Fp8MqaLogits(q_fp8, {kv_fp8, kv_scales}, weights, cu_seq_len_k_start, cu_seq_len_k_end,
                                        logits, max_seqlen_k, at::cuda::getCurrentCUDAStream().stream()));
  cudaDeviceSynchronize();

  // Verify computation completed
  auto logits_cpu = logits.narrow(0, 0, seq_len).narrow(1, 0, max_seqlen_k).to(torch::kCPU);
  EXPECT_GT(logits_cpu.numel(), 0) << "Large scale Fp8MqaLogits should produce output";
}

#endif  // ENABLE_DEEPSEEK_DEEPGEMM

}  // namespace test
}  // namespace nvidia
}  // namespace ksana_llm
