/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include <gtest/gtest.h>

#include "csrc/kernels/nvidia/embedding/embedding.h"
#include "csrc/utils/nvidia/cuda_utils.h"
#include "tests/kernels/nvidia/utils/testsuit_base.h"

namespace llm_kernels {
namespace nvidia {
namespace test {

template <typename T>
__global__ void InitTablesKernel(T* emb_table_ptr, T* pos_table_ptr) {
  uint32_t glb_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  emb_table_ptr[glb_thread_idx] = (T)(glb_thread_idx / 10000.0f);
  pos_table_ptr[glb_thread_idx] = (T)(glb_thread_idx / 10000.0f);
}

template <typename T>
__global__ void RefLookupEmbedding(T* emb_table_ptr, T* pos_table_ptr, const int32_t idx, const int32_t token_idx,
                                   const int32_t start_step, T* result_ptr, const int64_t hidden_units) {
  uint32_t glb_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t emb_id = glb_thread_idx % hidden_units;

  int32_t real_idx = token_idx;
  int32_t step = start_step + real_idx;

  T embedding = emb_table_ptr[idx * hidden_units + emb_id];
  T pos_embed = pos_table_ptr == nullptr ? (T)0.f : pos_table_ptr[(step - 1) * hidden_units + emb_id];

  result_ptr[emb_id] = embedding + pos_embed;
}

class LlamaNvidiaEmbeddingTestSuit : public NvidiaTestSuitBase {
 public:
  void SetUp() override { NvidiaTestSuitBase::SetUp(); }

  void TearDown() override { NvidiaTestSuitBase::TearDown(); }

 protected:
  using NvidiaTestSuitBase::stream;

 protected:
  const size_t vocab_size = 4ul;
  const size_t vocab_id = 0ul;
  const size_t hidden_units = 4096ul;
  const int32_t max_length = 8;
  const int32_t batch_size = 2;
  const int32_t input_prompt_num = 2;
  const std::vector<std::vector<int32_t>> input_prompt_token_ids = {{1, 2}, {1024, 3, 0}};
  std::vector<size_t> ids_lens;

  template <typename T>
  void PrepareTables(BufferMeta& emb_table_meta, BufferMeta& pos_table_meta) {
    size_t total_nums = vocab_size * hidden_units;
    size_t block_size = 512ul;
    dim3 grid((total_nums + block_size - 1) / block_size);
    dim3 block(block_size);
    InitTablesKernel<T><<<grid, block, 0, stream>>>((T*)emb_table_meta.data_ptr, (T*)pos_table_meta.data_ptr);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
  }

  template <typename T>
  void ReferLookupEmbedding(T* emb_table_ptr, T* pos_table_ptr, const int32_t idx, const int32_t token_idx,
                            const int32_t start_step, T* result_ptr, const int64_t hidden_units) {
    size_t total_nums = hidden_units;
    size_t block_size = 512ul;
    dim3 grid((total_nums + block_size - 1) / block_size);
    dim3 block(block_size);
    // input ids data: [[1,2], [1024, 3, 0]]
    // for example token 1, idx = 1, token_idx = 0
    // for example token 0, idx = 0, token_idx = 2
    if (idx < 0 || idx >= vocab_size) {
      return;
    }
    RefLookupEmbedding<T><<<grid, block, 0, stream>>>(emb_table_ptr, pos_table_ptr, idx, token_idx, start_step,
                                                      result_ptr, hidden_units);
    CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
  }
};

TEST_F(LlamaNvidiaEmbeddingTestSuit, LlamaFusedEmbeddingWithCSRHalfTest) {
  // emb table shape: [4, 4096]
  // [[0.1, 0.2, ..., 409.6],
  //  [409.7, 409.8, ..., 819.2],
  //  ...
  //  [1228.9, 1229.0, ..., 1638.4]]
  BufferMeta emb_table_meta = CreateBuffer<half>(MemoryType::MEMORY_GPU, {vocab_size, hidden_units});
  // pos table shape: [4, 4096]
  // [[0.001, 0.002, ..., 4.096],
  //  [4.097, 4.098, ..., 8.192],
  //  ...
  //  [12.289, 12.290, ..., 16.384]]
  BufferMeta pos_table_meta = CreateBuffer<half>(MemoryType::MEMORY_GPU, {vocab_size, hidden_units});
  PrepareTables<half>(emb_table_meta, pos_table_meta);

  int32_t start_step = 1;
  std::vector<size_t> ids_offsets_host(batch_size + 1, 0ul);
  size_t total_ids_num = 0ul;
  for (size_t tokens_idx = 0; tokens_idx < input_prompt_token_ids.size(); ++tokens_idx) {
    total_ids_num += input_prompt_token_ids[tokens_idx].size();
    ids_offsets_host[tokens_idx + 1] = ids_offsets_host[tokens_idx] + input_prompt_token_ids[tokens_idx].size();
  }

  int32_t* input_ids;
  size_t* ids_offsets;
  half* output_hidden_units;
  CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&input_ids, sizeof(int32_t) * total_ids_num));
  CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&ids_offsets, sizeof(size_t) * (batch_size + 1)));
  CHECK_NVIDIA_CUDA_ERROR(cudaMalloc(&output_hidden_units, sizeof(float) * total_ids_num * hidden_units));
  for (size_t tokens_idx = 0; tokens_idx < input_prompt_token_ids.size(); ++tokens_idx) {
    CHECK_NVIDIA_CUDA_ERROR(
        cudaMemcpy(input_ids + ids_offsets_host[tokens_idx], input_prompt_token_ids[tokens_idx].data(),
                   sizeof(int32_t) * input_prompt_token_ids[tokens_idx].size(), cudaMemcpyHostToDevice));
  }
  CHECK_NVIDIA_CUDA_ERROR(cudaMemcpy(ids_offsets, ids_offsets_host.data(), sizeof(size_t) * ids_offsets_host.size(),
                                     cudaMemcpyHostToDevice));
  CHECK_NVIDIA_CUDA_ERROR(cudaMemset(output_hidden_units, 0x0, total_ids_num * hidden_units));
  CHECK_NVIDIA_CUDA_ERROR(cudaDeviceSynchronize());

  LookupFusedEmbeddingWithCSRInputs(
      reinterpret_cast<half*>(output_hidden_units), reinterpret_cast<const half*>(emb_table_meta.data_ptr),
      reinterpret_cast<const half*>(pos_table_meta.data_ptr), InvokeInputIdsEmbeddingLookupPosEncodingParam<half>{},
      input_ids, start_step, ids_offsets, batch_size, hidden_units, vocab_size, vocab_id, stream);
  CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

  for (int32_t prompt_id = 0; prompt_id < input_prompt_num; ++prompt_id) {
    half* host_result_ptr;
    CHECK_NVIDIA_CUDA_ERROR(
        cudaMallocHost(&host_result_ptr, sizeof(half) * input_prompt_token_ids[prompt_id].size() * hidden_units));
    CHECK_NVIDIA_CUDA_ERROR(
        cudaMemcpy(host_result_ptr, output_hidden_units + ids_offsets_host[prompt_id] * hidden_units,
                   sizeof(half) * (ids_offsets_host[prompt_id + 1] - ids_offsets_host[prompt_id]) * hidden_units,
                   cudaMemcpyDeviceToHost));
    CHECK_NVIDIA_CUDA_ERROR(cudaDeviceSynchronize());

    // compute refer
    std::vector<float> host_ref_result_vec(input_prompt_token_ids[prompt_id].size() * hidden_units, 0.0f);
    for (size_t prompt_token_id = 0; prompt_token_id < input_prompt_token_ids[prompt_id].size(); ++prompt_token_id) {
      int32_t token_id = input_prompt_token_ids[prompt_id][prompt_token_id];
      int32_t step = start_step + prompt_token_id;

      for (size_t emb_id = 0; emb_id < hidden_units; ++emb_id) {
        float ref_emb_value = (float)((half_float::half)((token_id * hidden_units + emb_id) / 10000.0f));
        float ref_pos_emb_value = (float)((half_float::half)(((step - 1) * hidden_units + emb_id) / 10000.0f));
        float ref_value = ref_emb_value + ref_pos_emb_value;
        if (static_cast<size_t>(token_id) > vocab_size) {
          ref_value = 0.0f;
        }
        float result_value = (float)(host_result_ptr[prompt_token_id * hidden_units + emb_id]);
        EXPECT_TRUE((ref_value - result_value) < 1e-3)
            << "Fail in token: " << token_id << " emb: " << emb_id << ", result_value: " << result_value
            << ", ref_value: " << ref_value;
      }
    }

    CHECK_NVIDIA_CUDA_ERROR(cudaFreeHost(host_result_ptr));
  }

  CHECK_NVIDIA_CUDA_ERROR(cudaFree(output_hidden_units));
  CHECK_NVIDIA_CUDA_ERROR(cudaFree(ids_offsets));
  CHECK_NVIDIA_CUDA_ERROR(cudaFree(input_ids));
}

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels