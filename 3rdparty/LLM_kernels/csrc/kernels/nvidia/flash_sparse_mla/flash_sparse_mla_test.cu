/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 */

#include <gtest/gtest.h>

#include "csrc/kernels/nvidia/flash_sparse_mla/flash_sparse_mla.h"
#include "csrc/utils/nvidia/cuda_utils.h"
#include "params.h"
#include "tests/kernels/nvidia/utils/testsuit_base.h"

namespace llm_kernels {
namespace nvidia {
namespace test {

using namespace llm_kernels::utils;

class LlamaNvidiaFlashSparseMlaTestSuit : public NvidiaTestSuitBase {
 public:
  void SetUp() override { NvidiaTestSuitBase::SetUp(); }

  void TearDown() override { NvidiaTestSuitBase::TearDown(); }

 protected:
  void PrepareData(const size_t batch_size, const size_t seqlen_q, const size_t seqlen_k, const size_t num_heads_q) {
    // Prepare meta data
    d_tile_scheduler_metadata =
        CreateBuffer<int>(MemoryType::MEMORY_GPU, {static_cast<size_t>(GetNumSmParts(num_heads_q, seqlen_q)),
                                                   static_cast<size_t>(GetDecodingSchedMetaSize() / sizeof(int))});
    d_num_splits = CreateBuffer<int>(MemoryType::MEMORY_GPU, {batch_size + 1});
    // Prepare indices
    h_indices = CreateBuffer<int>(MemoryType::MEMORY_CPU, {batch_size * seqlen_q, topk});
    int* current_h_indices_ptr = reinterpret_cast<int*>(h_indices.data_ptr);
    num_blocks = 0;
    for (size_t i = 0; i < batch_size; i++) {
      for (size_t j = 0; j < seqlen_q; j++) {
        for (size_t k = 0; k < topk; k++) {
          if (k <= seqlen_k - seqlen_q + j) {
            *current_h_indices_ptr++ = num_blocks * block_size + k;
          } else {
            *current_h_indices_ptr++ = -1;
          }
        }
      }
      num_blocks += (seqlen_k + block_size - 1) / block_size;
    }
    d_indices = CopyToDevice<int>(h_indices);
    // Prepare qkv
    d_q = CreateBuffer<T>(MemoryType::MEMORY_GPU, {batch_size, seqlen_q, num_heads_q, head_size_k},
                          /*is_random_init*/ true);
    d_q_fp8 = CreateBuffer<__nv_fp8_e4m3>(MemoryType::MEMORY_GPU, {batch_size, seqlen_q, num_heads_q, head_size_k},
                                          /*is_random_init*/ true);
    d_kcache = CreateBuffer<__nv_fp8_e4m3>(MemoryType::MEMORY_GPU, {num_blocks, kcache_batch_stride_large},
                                           /*is_random_init*/ true);
    d_kcache_fp8 = CreateBuffer<__nv_fp8_e4m3>(MemoryType::MEMORY_GPU, {num_blocks, kcache_batch_stride_large_fp8},
                                               /*is_random_init*/ true);
    // Prepare large enough workspace
    d_workspace = CreateBuffer<float>(MemoryType::MEMORY_GPU, {1 << 28});
    // Prepare output
    d_out = CreateBuffer<T>(MemoryType::MEMORY_GPU, {batch_size, seqlen_q, num_heads_q, head_size_v});
  }

  void FreeData() {
    DeleteBuffer(d_tile_scheduler_metadata);
    DeleteBuffer(d_num_splits);
    DeleteBuffer(h_indices);
    DeleteBuffer(d_indices);
    DeleteBuffer(d_q);
    DeleteBuffer(d_q_fp8);
    DeleteBuffer(d_kcache);
    DeleteBuffer(d_kcache_fp8);
    DeleteBuffer(d_workspace);
    DeleteBuffer(d_out);
  }

  using NvidiaTestSuitBase::stream;
  using T = __nv_bfloat16;

  // Config of DeepSeek-V32
  const size_t num_heads_k = 1;
  const size_t head_size_k = 576;
  const size_t head_size_v = 512;
  const size_t block_size = 64;
  const size_t kcache_batch_stride_large =
      block_size * (656 + 132);  // dim of fp8_ds_mla is 656, extended by 132 to simulate mixed cache block with indexer
  const size_t kcache_batch_stride_large_fp8 =
      block_size * (576 + 132);  // dim of fp8_e4m3 is 576, extended by 132 to simulate mixed cache block with indexer
  const float softmax_scale = 0.1147213867929261;
  const size_t topk = 2048;

  size_t num_blocks = 0;

  // Data buffers for test
  BufferMeta d_tile_scheduler_metadata;
  BufferMeta d_num_splits;
  BufferMeta h_indices;
  BufferMeta d_indices;
  BufferMeta d_q;
  BufferMeta d_q_fp8;
  BufferMeta d_kcache;
  BufferMeta d_kcache_fp8;
  BufferMeta d_workspace;
  BufferMeta d_out;
};

TEST_F(LlamaNvidiaFlashSparseMlaTestSuit, FlashSparseMlaKernelAccTest) {
  const size_t batch_size = 2;
  const size_t seqlen_q = 2;
  const size_t seqlen_k = 100;
  const size_t num_heads_q = 128;

  PrepareData(batch_size, seqlen_q, seqlen_k, num_heads_q);

  auto IsBf16NaN = [](const T x) -> bool {
    uint16_t u16 = 0;
    static_assert(sizeof(uint16_t) == sizeof(T));
    std::memcpy(&u16, &x, sizeof(u16));
    constexpr uint16_t kExpMask = 0x7F80u;   // bits 14..7
    constexpr uint16_t kMantMask = 0x007Fu;  // bits 6..0
    uint16_t exp = u16 & kExpMask;
    uint16_t mant = u16 & kMantMask;
    // exp all ones and mantissa non-zero -> NaN
    return (exp == kExpMask) && (mant != 0);
  };

  // Invoke bf16 kernel
  InvokeFlashSparseMlaWithKVCache<T, uint8_t, KVCacheType::kFp8DsMla>(
      d_q.data_ptr, reinterpret_cast<uint8_t*>(d_kcache.data_ptr), reinterpret_cast<int*>(d_indices.data_ptr),
      reinterpret_cast<int*>(d_tile_scheduler_metadata.data_ptr), reinterpret_cast<int*>(d_num_splits.data_ptr),
      /*sched_meta_initialized*/ false, batch_size, seqlen_q, num_heads_q, head_size_k, num_blocks, block_size,
      num_heads_k, kcache_batch_stride_large, topk, head_size_v, softmax_scale, stream,
      reinterpret_cast<float*>(d_workspace.data_ptr), reinterpret_cast<T*>(d_out.data_ptr));
  CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

  // Check no NaNs in the output
  BufferMeta h_out = CopyToHost<T>(d_out);
  for (size_t i = 0; i < batch_size * seqlen_q * num_heads_q * head_size_v; i++) {
    EXPECT_FALSE(IsBf16NaN(reinterpret_cast<T*>(h_out.data_ptr)[i]));
  }
  DeleteBuffer(h_out);

  // Invoke fp8 kernel
  InvokeFlashSparseMlaWithKVCache<T, uint8_t, KVCacheType::kFp8E4M3>(
      d_q_fp8.data_ptr, reinterpret_cast<uint8_t*>(d_kcache_fp8.data_ptr), reinterpret_cast<int*>(d_indices.data_ptr),
      reinterpret_cast<int*>(d_tile_scheduler_metadata.data_ptr), reinterpret_cast<int*>(d_num_splits.data_ptr),
      /*sched_meta_initialized*/ false, batch_size, seqlen_q, num_heads_q, head_size_k, num_blocks, block_size,
      num_heads_k, kcache_batch_stride_large_fp8, topk, head_size_v, softmax_scale, stream,
      reinterpret_cast<float*>(d_workspace.data_ptr), reinterpret_cast<T*>(d_out.data_ptr));
  CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

  // Check no NaNs in the output
  h_out = CopyToHost<T>(d_out);
  for (size_t i = 0; i < batch_size * seqlen_q * num_heads_q * head_size_v; i++) {
    EXPECT_FALSE(IsBf16NaN(reinterpret_cast<T*>(h_out.data_ptr)[i]));
  }
  DeleteBuffer(h_out);

  FreeData();
}

// Performance test is disabled by default
TEST_F(LlamaNvidiaFlashSparseMlaTestSuit, DISABLED_FlashSparseMlaKernelPerfTest) {
  const size_t batch_size = 64;

  for (const size_t num_heads_q : {16, 128}) {  // TP8/DP8
    for (const size_t seqlen_k : {512, 1024, 2048, 4096, 8192, 16384, 32768, 65535}) {
      const size_t seqlen_q = 1;
      PrepareData(batch_size, seqlen_q, seqlen_k, num_heads_q);

      BufferMeta d_expanded_q;
      BufferMeta d_expanded_q_fp8 = d_q_fp8;
      BufferMeta d_expanded_out;
      if (num_heads_q < 64) {
        d_expanded_q = CreateBuffer<T>(MemoryType::MEMORY_GPU, {batch_size, seqlen_q, 64, head_size_k});
        d_expanded_out = CreateBuffer<T>(MemoryType::MEMORY_GPU, {batch_size, seqlen_q, 64, head_size_v});
      } else {
        d_expanded_q = d_q;
        d_expanded_out = d_out;
      }

      auto invoke_kernel_bf16 = [&](const bool sched_meta_initialized) {
        if (num_heads_q < 64) {
          cudaMemcpy2DAsync(d_expanded_q.data_ptr, 64 * head_size_k * sizeof(T), d_q.data_ptr,
                            num_heads_q * head_size_k * sizeof(T), num_heads_q * head_size_k * sizeof(T),
                            batch_size * seqlen_q, cudaMemcpyDeviceToDevice, stream);
        }
        InvokeFlashSparseMlaWithKVCache<T, uint8_t, KVCacheType::kFp8DsMla>(
            d_expanded_q.data_ptr, reinterpret_cast<uint8_t*>(d_kcache.data_ptr),
            reinterpret_cast<int*>(d_indices.data_ptr), reinterpret_cast<int*>(d_tile_scheduler_metadata.data_ptr),
            reinterpret_cast<int*>(d_num_splits.data_ptr), sched_meta_initialized, batch_size, seqlen_q,
            std::max(num_heads_q, 64ul), head_size_k, num_blocks, block_size, num_heads_k, kcache_batch_stride_large,
            topk, head_size_v, softmax_scale, stream, reinterpret_cast<float*>(d_workspace.data_ptr),
            reinterpret_cast<T*>(d_expanded_out.data_ptr));
        if (num_heads_q < 64) {
          cudaMemcpy2DAsync(d_out.data_ptr, num_heads_q * head_size_v * sizeof(T), d_expanded_out.data_ptr,
                            64 * head_size_v * sizeof(T), num_heads_q * head_size_v * sizeof(T), batch_size * seqlen_q,
                            cudaMemcpyDeviceToDevice, stream);
        }
      };

      auto invoke_kernel_fp8 = [&](const bool sched_meta_initialized) {
        InvokeFlashSparseMlaWithKVCache<T, uint8_t, KVCacheType::kFp8E4M3>(
            d_expanded_q_fp8.data_ptr, reinterpret_cast<uint8_t*>(d_kcache_fp8.data_ptr),
            reinterpret_cast<int*>(d_indices.data_ptr), reinterpret_cast<int*>(d_tile_scheduler_metadata.data_ptr),
            reinterpret_cast<int*>(d_num_splits.data_ptr), sched_meta_initialized, batch_size, seqlen_q, num_heads_q,
            head_size_k, num_blocks, block_size, num_heads_k, kcache_batch_stride_large_fp8, topk, head_size_v,
            softmax_scale, stream, reinterpret_cast<float*>(d_workspace.data_ptr),
            reinterpret_cast<T*>(d_expanded_out.data_ptr));
      };

      const int warmups = 5;
      const int iterations = 10;

      // Initialize the sched_meta
      invoke_kernel_bf16(false);
      // Performance run
      auto cuda_run_bf16 = [&]() { invoke_kernel_bf16(true); };
      const float elapsed_ms_bf16 = MeasureCudaExecutionTime(cuda_run_bf16, stream, warmups, iterations);
      std::cout << "Num heads q: " << num_heads_q << ", Seq len k: " << seqlen_k
                << ", Execution time of flash sparse mla bf16: " << elapsed_ms_bf16 << " ms" << std::endl;

      // FP8 and BF16 share the same sched_meta
      // Performance run
      auto cuda_run_fp8 = [&]() { invoke_kernel_fp8(true); };
      const float elapsed_ms_fp8 = MeasureCudaExecutionTime(cuda_run_fp8, stream, warmups, iterations);
      std::cout << "Num heads q: " << num_heads_q << ", Seq len k: " << seqlen_k
                << ", Execution time of flash sparse mla fp8: " << elapsed_ms_fp8 << " ms" << std::endl;

      FreeData();

      if (num_heads_q < 64) {
        DeleteBuffer(d_expanded_q);
        DeleteBuffer(d_expanded_out);
      }
    }
  }
}

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels
