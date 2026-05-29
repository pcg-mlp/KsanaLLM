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

enum class ModelType { kV32, kMODEL1 };

struct ModelConfig {
  ModelType type;
  const char* name;
  size_t head_size_k;
  size_t head_size_v;
  size_t kcache_batch_stride;  // for fp8_ds_mla
  size_t topk;

  bool IsModel1() const { return type == ModelType::kMODEL1; }
};

static constexpr size_t kBlockSize = 64;
static constexpr size_t kCacheBlockPadding = 132;

static const ModelConfig kModelConfigs[] = {
    {ModelType::kV32, "V32",
     /*head_size_k=*/576, /*head_size_v=*/512,
     /*kcache_batch_stride=*/kBlockSize * (656 + kCacheBlockPadding),
     /*topk=*/2048},
    {ModelType::kMODEL1, "MODEL1",
     /*head_size_k=*/512, /*head_size_v=*/512,
     /*kcache_batch_stride=*/kBlockSize * (584 + kCacheBlockPadding),
     /*topk=*/1024},
};

class LlamaNvidiaFlashSparseMlaTestSuit : public NvidiaTestSuitBase {
 public:
  void SetUp() override { NvidiaTestSuitBase::SetUp(); }

  void TearDown() override { NvidiaTestSuitBase::TearDown(); }

 protected:
  void PrepareData(const ModelConfig& cfg, const size_t batch_size, const size_t seqlen_q, const size_t seqlen_k,
                   const size_t num_heads_q) {
    current_cfg = &cfg;
    kcache_batch_stride_fp8 = kBlockSize * (cfg.head_size_k + kCacheBlockPadding);

    d_tile_scheduler_metadata =
        CreateBuffer<int>(MemoryType::MEMORY_GPU, {static_cast<size_t>(GetNumSmParts(num_heads_q, seqlen_q)),
                                                   static_cast<size_t>(GetDecodingSchedMetaSize() / sizeof(int))});
    d_num_splits = CreateBuffer<int>(MemoryType::MEMORY_GPU, {batch_size + 1});
    // Prepare indices
    h_indices = CreateBuffer<int>(MemoryType::MEMORY_CPU, {batch_size * seqlen_q, cfg.topk});
    int* current_h_indices_ptr = reinterpret_cast<int*>(h_indices.data_ptr);
    num_blocks = 0;
    for (size_t i = 0; i < batch_size; i++) {
      for (size_t j = 0; j < seqlen_q; j++) {
        for (size_t k = 0; k < cfg.topk; k++) {
          if (k <= seqlen_k - seqlen_q + j) {
            *current_h_indices_ptr++ = num_blocks * kBlockSize + k;
          } else {
            *current_h_indices_ptr++ = -1;
          }
        }
      }
      num_blocks += (seqlen_k + kBlockSize - 1) / kBlockSize;
    }
    d_indices = CopyToDevice<int>(h_indices);

    // Prepare topk_length (MODEL1)
    if (cfg.IsModel1()) {
      h_topk_length = CreateBuffer<int>(MemoryType::MEMORY_CPU, {batch_size * seqlen_q});
      int* topk_length_data = reinterpret_cast<int*>(h_topk_length.data_ptr);
      for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < seqlen_q; j++) {
          topk_length_data[i * seqlen_q + j] = std::min(cfg.topk, seqlen_k - seqlen_q + j + 1);
        }
      }
      d_topk_length = CopyToDevice<int>(h_topk_length);
    }

    // Prepare extra cache (MODEL1)
    if (cfg.IsModel1()) {
      extra_num_blocks = num_blocks;
      d_extra_kcache = CreateBuffer<__nv_fp8_e4m3>(MemoryType::MEMORY_GPU, {extra_num_blocks, cfg.kcache_batch_stride},
                                                   /*is_random_init*/ true);
      d_extra_kcache_fp8 =
          CreateBuffer<__nv_fp8_e4m3>(MemoryType::MEMORY_GPU, {extra_num_blocks, kcache_batch_stride_fp8},
                                      /*is_random_init*/ true);
      // extra_indices: copy from h_indices (same topk, same layout)
      d_extra_indices = CopyToDevice<int>(h_indices);
      // extra_topk_length: copy from h_topk_length
      d_extra_topk_length = CopyToDevice<int>(h_topk_length);
      // attn_sink: fixed 64 floats to avoid pad issues
      d_attn_sink = CreateBuffer<float>(MemoryType::MEMORY_GPU, {64}, /*is_random_init*/ true);
    }

    // Prepare qkv
    d_q = CreateBuffer<T>(MemoryType::MEMORY_GPU, {batch_size, seqlen_q, num_heads_q, cfg.head_size_k},
                          /*is_random_init*/ true);
    d_q_fp8 = CreateBuffer<__nv_fp8_e4m3>(MemoryType::MEMORY_GPU, {batch_size, seqlen_q, num_heads_q, cfg.head_size_k},
                                          /*is_random_init*/ true);
    d_kcache = CreateBuffer<__nv_fp8_e4m3>(MemoryType::MEMORY_GPU, {num_blocks, cfg.kcache_batch_stride},
                                           /*is_random_init*/ true);
    d_kcache_fp8 = CreateBuffer<__nv_fp8_e4m3>(MemoryType::MEMORY_GPU, {num_blocks, kcache_batch_stride_fp8},
                                               /*is_random_init*/ true);
    // Prepare large enough workspace
    d_workspace = CreateBuffer<float>(MemoryType::MEMORY_GPU, {1 << 28});
    // Prepare output
    d_out = CreateBuffer<T>(MemoryType::MEMORY_GPU, {batch_size, seqlen_q, num_heads_q, cfg.head_size_v});
  }

  void FreeData() {
    DeleteBuffer(d_tile_scheduler_metadata);
    DeleteBuffer(d_num_splits);
    DeleteBuffer(h_indices);
    DeleteBuffer(d_indices);
    if (current_cfg->IsModel1()) {
      DeleteBuffer(h_topk_length);
      DeleteBuffer(d_topk_length);
      DeleteBuffer(d_extra_kcache);
      DeleteBuffer(d_extra_kcache_fp8);
      DeleteBuffer(d_extra_indices);
      DeleteBuffer(d_extra_topk_length);
      DeleteBuffer(d_attn_sink);
    }
    DeleteBuffer(d_q);
    DeleteBuffer(d_q_fp8);
    DeleteBuffer(d_kcache);
    DeleteBuffer(d_kcache_fp8);
    DeleteBuffer(d_workspace);
    DeleteBuffer(d_out);
  }

  int* GetTopkLengthPtr() const {
    return current_cfg->IsModel1() ? reinterpret_cast<int*>(d_topk_length.data_ptr) : nullptr;
  }
  float* GetAttnSinkPtr() const {
    return current_cfg->IsModel1() ? reinterpret_cast<float*>(d_attn_sink.data_ptr) : nullptr;
  }
  uint8_t* GetExtraKcachePtr() const {
    return current_cfg->IsModel1() ? reinterpret_cast<uint8_t*>(d_extra_kcache.data_ptr) : nullptr;
  }
  uint8_t* GetExtraKcacheFp8Ptr() const {
    return current_cfg->IsModel1() ? reinterpret_cast<uint8_t*>(d_extra_kcache_fp8.data_ptr) : nullptr;
  }
  int* GetExtraIndicesPtr() const {
    return current_cfg->IsModel1() ? reinterpret_cast<int*>(d_extra_indices.data_ptr) : nullptr;
  }
  int* GetExtraTopkLengthPtr() const {
    return current_cfg->IsModel1() ? reinterpret_cast<int*>(d_extra_topk_length.data_ptr) : nullptr;
  }

  using NvidiaTestSuitBase::stream;
  using T = __nv_bfloat16;

  const size_t num_heads_k = 1;
  const float softmax_scale = 0.1147213867929261;

  const ModelConfig* current_cfg = nullptr;
  size_t kcache_batch_stride_fp8 = 0;
  size_t num_blocks = 0;
  size_t extra_num_blocks = 0;

  BufferMeta d_tile_scheduler_metadata;
  BufferMeta d_num_splits;
  BufferMeta h_indices;
  BufferMeta d_indices;
  BufferMeta h_topk_length;
  BufferMeta d_topk_length;
  BufferMeta d_extra_kcache;
  BufferMeta d_extra_kcache_fp8;
  BufferMeta d_extra_indices;
  BufferMeta d_extra_topk_length;
  BufferMeta d_attn_sink;
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

  auto IsBf16NaN = [](const T x) -> bool {
    uint16_t u16 = 0;
    static_assert(sizeof(uint16_t) == sizeof(T));
    std::memcpy(&u16, &x, sizeof(u16));
    constexpr uint16_t kExpMask = 0x7F80u;
    constexpr uint16_t kMantMask = 0x007Fu;
    uint16_t exp = u16 & kExpMask;
    uint16_t mant = u16 & kMantMask;
    return (exp == kExpMask) && (mant != 0);
  };

  for (const auto& cfg : kModelConfigs) {
    SCOPED_TRACE(cfg.name);
    PrepareData(cfg, batch_size, seqlen_q, seqlen_k, num_heads_q);
    int* topk_length_ptr = GetTopkLengthPtr();
    float* attn_sink_ptr = GetAttnSinkPtr();
    int* extra_indices_ptr = GetExtraIndicesPtr();
    int* extra_topk_length_ptr = GetExtraTopkLengthPtr();
    int cur_extra_num_blocks = cfg.IsModel1() ? static_cast<int>(extra_num_blocks) : 0;
    int cur_extra_block_size = cfg.IsModel1() ? static_cast<int>(kBlockSize) : 0;
    int cur_extra_topk = cfg.IsModel1() ? static_cast<int>(cfg.topk) : 0;
    int cur_extra_kcache_batch_stride = cfg.IsModel1() ? static_cast<int>(cfg.kcache_batch_stride) : -1;

    // Invoke fp8_ds_mla kernel
    InvokeFlashSparseMlaWithKVCache<T, uint8_t, KVCacheType::kFp8DsMla>(
        d_q.data_ptr, reinterpret_cast<uint8_t*>(d_kcache.data_ptr), reinterpret_cast<int*>(d_indices.data_ptr),
        reinterpret_cast<int*>(d_tile_scheduler_metadata.data_ptr), reinterpret_cast<int*>(d_num_splits.data_ptr),
        /*sched_meta_initialized*/ false, batch_size, seqlen_q, num_heads_q, cfg.head_size_k, num_blocks, kBlockSize,
        num_heads_k, cfg.kcache_batch_stride, cfg.topk, cfg.head_size_v, softmax_scale, stream,
        reinterpret_cast<float*>(d_workspace.data_ptr), reinterpret_cast<T*>(d_out.data_ptr), topk_length_ptr,
        attn_sink_ptr, GetExtraKcachePtr(), extra_indices_ptr, extra_topk_length_ptr, cur_extra_num_blocks,
        cur_extra_block_size, cur_extra_topk, cur_extra_kcache_batch_stride);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    BufferMeta h_out = CopyToHost<T>(d_out);
    for (size_t i = 0; i < batch_size * seqlen_q * num_heads_q * cfg.head_size_v; i++) {
      EXPECT_FALSE(IsBf16NaN(reinterpret_cast<T*>(h_out.data_ptr)[i]));
    }
    DeleteBuffer(h_out);

    // Invoke fp8_e4m3 kernel
    int cur_extra_kcache_batch_stride_fp8 = cfg.IsModel1() ? static_cast<int>(kcache_batch_stride_fp8) : -1;
    InvokeFlashSparseMlaWithKVCache<T, uint8_t, KVCacheType::kFp8E4M3>(
        d_q_fp8.data_ptr, reinterpret_cast<uint8_t*>(d_kcache_fp8.data_ptr), reinterpret_cast<int*>(d_indices.data_ptr),
        reinterpret_cast<int*>(d_tile_scheduler_metadata.data_ptr), reinterpret_cast<int*>(d_num_splits.data_ptr),
        /*sched_meta_initialized*/ false, batch_size, seqlen_q, num_heads_q, cfg.head_size_k, num_blocks, kBlockSize,
        num_heads_k, kcache_batch_stride_fp8, cfg.topk, cfg.head_size_v, softmax_scale, stream,
        reinterpret_cast<float*>(d_workspace.data_ptr), reinterpret_cast<T*>(d_out.data_ptr), topk_length_ptr,
        attn_sink_ptr, GetExtraKcacheFp8Ptr(), extra_indices_ptr, extra_topk_length_ptr, cur_extra_num_blocks,
        cur_extra_block_size, cur_extra_topk, cur_extra_kcache_batch_stride_fp8);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    h_out = CopyToHost<T>(d_out);
    for (size_t i = 0; i < batch_size * seqlen_q * num_heads_q * cfg.head_size_v; i++) {
      EXPECT_FALSE(IsBf16NaN(reinterpret_cast<T*>(h_out.data_ptr)[i]));
    }
    DeleteBuffer(h_out);

    FreeData();
  }
}

// Performance test is disabled by default
TEST_F(LlamaNvidiaFlashSparseMlaTestSuit, DISABLED_FlashSparseMlaKernelPerfTest) {
  const size_t batch_size = 64;

  for (const auto& cfg : kModelConfigs) {
    for (const size_t num_heads_q : {16, 64}) {
      for (const size_t seqlen_k : {512, 1024, 2048, 4096, 8192, 16384, 32768, 65536}) {
        const size_t seqlen_q = 1;
        PrepareData(cfg, batch_size, seqlen_q, seqlen_k, num_heads_q);
        int* topk_length_ptr = GetTopkLengthPtr();
        float* attn_sink_ptr = GetAttnSinkPtr();
        int* extra_indices_ptr = GetExtraIndicesPtr();
        int* extra_topk_length_ptr = GetExtraTopkLengthPtr();
        int cur_extra_num_blocks = cfg.IsModel1() ? static_cast<int>(extra_num_blocks) : 0;
        int cur_extra_block_size = cfg.IsModel1() ? static_cast<int>(kBlockSize) : 0;
        int cur_extra_topk = cfg.IsModel1() ? static_cast<int>(cfg.topk) : 0;
        int cur_extra_kcache_batch_stride = cfg.IsModel1() ? static_cast<int>(cfg.kcache_batch_stride) : -1;
        int cur_extra_kcache_batch_stride_fp8 = cfg.IsModel1() ? static_cast<int>(kcache_batch_stride_fp8) : -1;

        BufferMeta d_expanded_q;
        BufferMeta d_expanded_q_fp8 = d_q_fp8;
        BufferMeta d_expanded_out;
        const bool need_expand = num_heads_q < 64;
        if (need_expand) {
          d_expanded_q = CreateBuffer<T>(MemoryType::MEMORY_GPU, {batch_size, seqlen_q, 64, cfg.head_size_k});
          d_expanded_out = CreateBuffer<T>(MemoryType::MEMORY_GPU, {batch_size, seqlen_q, 64, cfg.head_size_v});
        } else {
          d_expanded_q = d_q;
          d_expanded_out = d_out;
        }

        auto invoke_kernel_bf16 = [&](const bool sched_meta_initialized) {
          if (need_expand) {
            cudaMemcpy2DAsync(d_expanded_q.data_ptr, 64 * cfg.head_size_k * sizeof(T), d_q.data_ptr,
                              num_heads_q * cfg.head_size_k * sizeof(T), num_heads_q * cfg.head_size_k * sizeof(T),
                              batch_size * seqlen_q, cudaMemcpyDeviceToDevice, stream);
          }
          InvokeFlashSparseMlaWithKVCache<T, uint8_t, KVCacheType::kFp8DsMla>(
              d_expanded_q.data_ptr, reinterpret_cast<uint8_t*>(d_kcache.data_ptr),
              reinterpret_cast<int*>(d_indices.data_ptr), reinterpret_cast<int*>(d_tile_scheduler_metadata.data_ptr),
              reinterpret_cast<int*>(d_num_splits.data_ptr), sched_meta_initialized, batch_size, seqlen_q,
              need_expand ? 64ul : num_heads_q, cfg.head_size_k, num_blocks, kBlockSize, num_heads_k,
              cfg.kcache_batch_stride, cfg.topk, cfg.head_size_v, softmax_scale, stream,
              reinterpret_cast<float*>(d_workspace.data_ptr), reinterpret_cast<T*>(d_expanded_out.data_ptr),
              topk_length_ptr, attn_sink_ptr, GetExtraKcachePtr(), extra_indices_ptr, extra_topk_length_ptr,
              cur_extra_num_blocks, cur_extra_block_size, cur_extra_topk, cur_extra_kcache_batch_stride);
          if (need_expand) {
            cudaMemcpy2DAsync(d_out.data_ptr, num_heads_q * cfg.head_size_v * sizeof(T), d_expanded_out.data_ptr,
                              64 * cfg.head_size_v * sizeof(T), num_heads_q * cfg.head_size_v * sizeof(T),
                              batch_size * seqlen_q, cudaMemcpyDeviceToDevice, stream);
          }
        };

        auto invoke_kernel_fp8 = [&](const bool sched_meta_initialized) {
          InvokeFlashSparseMlaWithKVCache<T, uint8_t, KVCacheType::kFp8E4M3>(
              d_expanded_q_fp8.data_ptr, reinterpret_cast<uint8_t*>(d_kcache_fp8.data_ptr),
              reinterpret_cast<int*>(d_indices.data_ptr), reinterpret_cast<int*>(d_tile_scheduler_metadata.data_ptr),
              reinterpret_cast<int*>(d_num_splits.data_ptr), sched_meta_initialized, batch_size, seqlen_q, num_heads_q,
              cfg.head_size_k, num_blocks, kBlockSize, num_heads_k, kcache_batch_stride_fp8, cfg.topk, cfg.head_size_v,
              softmax_scale, stream, reinterpret_cast<float*>(d_workspace.data_ptr),
              reinterpret_cast<T*>(d_expanded_out.data_ptr), topk_length_ptr, attn_sink_ptr, GetExtraKcacheFp8Ptr(),
              extra_indices_ptr, extra_topk_length_ptr, cur_extra_num_blocks, cur_extra_block_size, cur_extra_topk,
              cur_extra_kcache_batch_stride_fp8);
        };

        const int warmups = 5;
        const int iterations = 10;

        // Initialize sched_meta + perf run fp8_ds_mla
        invoke_kernel_bf16(false);
        auto cuda_run_bf16 = [&]() { invoke_kernel_bf16(true); };
        const float elapsed_ms_bf16 = MeasureCudaExecutionTime(cuda_run_bf16, stream, warmups, iterations);
        std::cout << "[" << cfg.name << "] Num heads q: " << num_heads_q << ", Seq len k: " << seqlen_k
                  << ", Execution time fp8_ds_mla: " << elapsed_ms_bf16 << " ms" << std::endl;

        // Perf run fp8_e4m3 (shares sched_meta)
        auto cuda_run_fp8 = [&]() { invoke_kernel_fp8(true); };
        const float elapsed_ms_fp8 = MeasureCudaExecutionTime(cuda_run_fp8, stream, warmups, iterations);
        std::cout << "[" << cfg.name << "] Num heads q: " << num_heads_q << ", Seq len k: " << seqlen_k
                  << ", Execution time fp8_e4m3: " << elapsed_ms_fp8 << " ms" << std::endl;

        FreeData();
        if (need_expand) {
          DeleteBuffer(d_expanded_q);
          DeleteBuffer(d_expanded_out);
        }
      }
    }
  }
}

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels
