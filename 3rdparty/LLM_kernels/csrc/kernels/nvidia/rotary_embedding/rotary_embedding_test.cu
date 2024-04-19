// Copyright 2024 Tencent Inc.  All rights reserved.

#include <gtest/gtest.h>

#include "csrc/kernels/nvidia/rotary_embedding/rotary_embedding.h"
#include "tests/kernels/nvidia/utils/testsuit_base.h"

namespace llm_kernels {
namespace nvidia {
namespace test {
class LlamaNvidiaRotaryEmbeddingTestSuit : public NvidiaTestSuitBase {
 public:
  void SetUp() override { NvidiaTestSuitBase::SetUp(); }

  void TearDown() override { NvidiaTestSuitBase::TearDown(); }

 protected:
  using NvidiaTestSuitBase::stream;

  // following config is loaded from llama-13B
  int rotary_dim{128};
  int max_position_embeddings{2048};
  int num_tokens{512};
  int num_kv_heads{40};
  int num_heads{40};
  int head_size{128};
  float base{10000};
  bool is_neox{true};
  size_t batch_size{1ul};
  RotaryEmbeddingType rotary_embedding_type{RotaryEmbeddingType::DEFAULT};
  float scaling_factor{1.0f};

 protected:
  template <typename T>
  void ForwardLlamaRotaryEmbedding() {
    using DataType = T;
    // create kernel's buffer
    int query_stride = num_heads * head_size;
    int key_stride = num_kv_heads * head_size;
    // [num_tokens]
    BufferMeta positions_cpu_meta = CreateBuffer<int64_t>(MemoryType::MEMORY_CPU, {static_cast<size_t>(num_tokens)},
                                                          true, 0, max_position_embeddings);
    int cpu_rotary_pos_idx = 0;
    for (size_t idx = 0; idx < batch_size; ++idx) {
      for (int pos = 0; pos < num_tokens; ++pos) {
        (reinterpret_cast<int64_t*>(positions_cpu_meta.data_ptr))[cpu_rotary_pos_idx++] = static_cast<int64_t>(pos);
      }
    }
    BufferMeta positions_meta = CopyToDevice<int64_t>(positions_cpu_meta);

    // [num_tokens, num_heads * head_size]
    BufferMeta query_meta = CreateBuffer<DataType>(
        MemoryType::MEMORY_GPU, {static_cast<size_t>(num_tokens), static_cast<size_t>(query_stride)}, true, 0, 1);
    // [num_tokens, num_kv_heads * head_size]
    BufferMeta key_meta;
    LoadNpy<DataType>("/tmp/tests/kernels/data/rotary_embedding/key_meta.npy", MemoryType::MEMORY_GPU, key_meta);
    // [max_position_embeddings, rotary_dim]
    BufferMeta cos_sin_cache_meta = CreateBuffer<DataType>(
        MemoryType::MEMORY_GPU, {static_cast<size_t>(max_position_embeddings), static_cast<size_t>(rotary_dim)});

    RotaryEmbeddingCuda<DataType> op;
    op.SetConfig(static_cast<DataType*>(cos_sin_cache_meta.data_ptr), rotary_dim, max_position_embeddings, base,
                 head_size, num_heads, num_kv_heads, query_stride, is_neox, stream);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    op.SetInput(static_cast<int64_t*>(positions_meta.data_ptr), static_cast<DataType*>(query_meta.data_ptr),
                static_cast<DataType*>(key_meta.data_ptr), num_tokens, stream);
    op.Forward();
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

    if (rotary_embedding_type == RotaryEmbeddingType::DEFAULT) {
      BufferMeta cos_sin_cache_ref_meta;
      LoadNpy<DataType>("/tmp/tests/kernels/data/rotary_embedding/cos_sin_cache_meta.npy", MemoryType::MEMORY_GPU,
                        cos_sin_cache_ref_meta);
      CheckResult<DataType>("rotary_embedding_cos_sin_cache_meta_half_check", cos_sin_cache_meta,
                            cos_sin_cache_ref_meta, 1e-5f, 1e-5f);

      BufferMeta query_ref_meta;
      LoadNpy<DataType>("/tmp/tests/kernels/data/rotary_embedding/query_meta.npy", MemoryType::MEMORY_GPU,
                        query_ref_meta);
      CheckResult<DataType>("rotary_embedding_query_meta_half_check", query_meta, query_ref_meta, 1e-5f, 1e-5f);
    } else if (rotary_embedding_type == RotaryEmbeddingType::DYNAMIC_NTK_SCALING) {
      BufferMeta cos_sin_cache_ref_meta;
      LoadNpy<DataType>("/tmp/tests/kernels/data/rotary_embedding/cos_sin_cache_dynamic_ntk_scaling_meta.npy",
                        MemoryType::MEMORY_GPU, cos_sin_cache_ref_meta);
      CheckResult<DataType>("rotary_embedding_cos_sin_cache_meta_half_check", cos_sin_cache_meta,
                            cos_sin_cache_ref_meta, 1e-5f, 1e-5f);

      BufferMeta query_ref_meta;
      LoadNpy<DataType>("/tmp/tests/kernels/data/rotary_embedding/query_dynamic_ntk_scaling_meta.npy",
                        MemoryType::MEMORY_GPU, query_ref_meta);
      CheckResult<DataType>("rotary_embedding_query_meta_half_check", query_meta, query_ref_meta, 1e-5f, 1e-5f);
    }
  }
};

TEST_F(LlamaNvidiaRotaryEmbeddingTestSuit, LlamaRotaryEmbeddingTest) { ForwardLlamaRotaryEmbedding<half>(); }

TEST_F(LlamaNvidiaRotaryEmbeddingTestSuit, LlamaDynamicNTKScalingRotaryEmbeddingTest) {
  rotary_embedding_type = RotaryEmbeddingType::DYNAMIC_NTK_SCALING;
  scaling_factor = 4.0f;
  ForwardLlamaRotaryEmbedding<half>();
}

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels
