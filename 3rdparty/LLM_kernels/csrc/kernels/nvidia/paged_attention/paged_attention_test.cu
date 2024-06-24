// Copyright 2024 Tencent Inc.  All rights reserved.

#include <gtest/gtest.h>

#include "csrc/kernels/nvidia/paged_attention/cache_copy.h"
#include "csrc/kernels/nvidia/paged_attention/paged_attention.h"
#include "tests/kernels/nvidia/utils/testsuit_base.h"

namespace llm_kernels {
namespace nvidia {
namespace test {
class LlamaNvidiaPagedAttentionTestSuit : public NvidiaTestSuitBase {
 public:
  void SetUp() override { NvidiaTestSuitBase::SetUp(); }

  void TearDown() override { NvidiaTestSuitBase::TearDown(); }

 protected:
  using NvidiaTestSuitBase::stream;

 protected:
  // following config is loaded from FasterTransformers LLAMA bs 16, in_seq 512, out_seq 512
  int num_seqs{16};
  int seq_blocks{16};
  int block_size{32};
  int num_heads{40};
  int num_kv_heads{40};
  int head_size{128};
  int max_context_len{1024};
  int stride_size{5120};
};

template <typename T>
__global__ void reshape_kv_cache_kernel(
    T* __restrict__ ref_key,      // num_blocks, block_size, num_heads, head_size / x, x
    T* __restrict__ ref_value,    // num_blocks, block_size, num_heads, head_size
    const T* __restrict__ key,    // num_blocks, num_kv_heads, head_size / x, block_size, x
    const T* __restrict__ value,  // num_blocks, num_kv_heads, head_size / x, block_size
    const int num_blocks, const int num_kv_heads, const int num_heads, const int head_size, const int block_size,
    const int x) {
  int i_num_blocks = blockIdx.x;
  int i_block_size = blockIdx.y;
  int i_num_heads = blockIdx.z;
  int i_num_kv_heads = i_num_heads / (num_heads / num_kv_heads);
  int i_head_size = threadIdx.x;
  // num_blocks, block_size, num_heads, head_size
  ref_key[i_num_blocks * block_size * num_heads * head_size + i_block_size * num_heads * head_size +
          i_num_heads * head_size + i_head_size] =
      // num_blocks, num_kv_heads, head_size / x, block_size, x
      key[i_num_blocks * num_kv_heads * head_size * block_size + i_num_kv_heads * head_size * block_size +
          i_head_size / x * block_size * x + i_block_size * x + i_head_size % x];
  // num_blocks, block_size, num_heads, head_size
  ref_value[i_num_blocks * block_size * num_heads * head_size + i_block_size * num_heads * head_size +
            i_num_heads * head_size + i_head_size] =
      // num_blocks, num_kv_heads, head_size, block_size
      value[i_num_blocks * num_kv_heads * head_size * block_size + i_num_kv_heads * head_size * block_size +
            i_head_size * block_size + i_block_size];
}

template <typename T>
void reshape_kv_cache(T* ref_key,      // num_blocks, block_size, num_heads, head_size / x, x
                      T* ref_value,    // num_blocks, block_size, num_heads, head_size
                      const T* key,    // num_blocks, num_kv_heads, head_size / x, block_size, x
                      const T* value,  // num_blocks, num_kv_heads, head_size / x, block_size
                      const int num_blocks, const int num_kv_heads, const int num_heads, const int head_size,
                      const int block_size, const int x, cudaStream_t stream) {
  dim3 grid(num_blocks, block_size, num_heads);
  dim3 block(head_size);
  reshape_kv_cache_kernel<<<grid, block, 0, stream>>>(ref_key, ref_value, key, value, num_blocks, num_kv_heads,
                                                      num_heads, head_size, block_size, x);
}

TEST_F(LlamaNvidiaPagedAttentionTestSuit, LlamaPagedAttentionHalfTest) {
  using DataType = uint16_t;
  // create buffer
  BufferMeta out_meta = CreateBuffer<DataType>(
      MemoryType::MEMORY_GPU,
      {static_cast<size_t>(num_seqs), static_cast<size_t>(num_heads), static_cast<size_t>(head_size)});
  BufferMeta query_meta = CreateBuffer<DataType>(
      MemoryType::MEMORY_GPU,
      {static_cast<size_t>(num_seqs), static_cast<size_t>(num_heads), static_cast<size_t>(head_size)}, true, 0, 1);

  // need fill cache_offsets
  int seq_blocks = max_context_len / block_size;
  int x = 16 / sizeof(DataType);
  // 0, seq_blocks, 2xseq_blocks, 3xseq_blocks,...
  BufferMeta cache_offsets_meta = CreateBuffer<int>(MemoryType::MEMORY_GPU, {static_cast<size_t>(num_seqs)});
  InvokeRange(static_cast<int*>(cache_offsets_meta.data_ptr), 0, num_seqs, seq_blocks, stream);
  // max_context_len, max_context_len, max_context_len,...
  BufferMeta context_lens_meta = CreateBuffer<int>(MemoryType::MEMORY_GPU, {static_cast<size_t>(num_seqs)});
  InvokeRange(static_cast<int*>(context_lens_meta.data_ptr), max_context_len, num_seqs, 0, stream);

  BufferMeta key_caches = CreateBuffer<DataType>(
      MemoryType::MEMORY_GPU,
      {static_cast<size_t>(num_seqs), static_cast<size_t>(seq_blocks), static_cast<size_t>(num_kv_heads),
       static_cast<size_t>(head_size / x), static_cast<size_t>(block_size), static_cast<size_t>(x)},
      true, 0, 1);
  BufferMeta value_caches = CreateBuffer<DataType>(
      MemoryType::MEMORY_GPU,
      {static_cast<size_t>(num_seqs), static_cast<size_t>(seq_blocks), static_cast<size_t>(num_kv_heads),
       static_cast<size_t>(head_size), static_cast<size_t>(block_size)},
      true, 0, 1);

  BufferMeta key_cache_ptrs =
      CreateBuffer<uint64_t>(MemoryType::MEMORY_GPU, {static_cast<size_t>(num_seqs * seq_blocks)});
  BufferMeta value_cache_ptrs =
      CreateBuffer<uint64_t>(MemoryType::MEMORY_GPU, {static_cast<size_t>(num_seqs * seq_blocks)});

  // set block ptrs
  InvokeRange(reinterpret_cast<DataType**>(key_cache_ptrs.data_ptr), static_cast<DataType*>(key_caches.data_ptr),
              num_seqs * seq_blocks, num_kv_heads * head_size * block_size, stream);
  InvokeRange(reinterpret_cast<DataType**>(value_cache_ptrs.data_ptr), static_cast<DataType*>(value_caches.data_ptr),
              num_seqs * seq_blocks, num_kv_heads * head_size * block_size, stream);

  // run paged_attention(..., stream);
  PagedAttentionCuda<DataType, DataType, false> op;
  op.SetConfig(num_kv_heads, num_heads, head_size, block_size, stride_size);
  size_t work_size = op.GetWorkSpaceSize(num_seqs, max_context_len);
  BufferMeta workspace_meta = CreateBuffer<char>(MemoryType::MEMORY_GPU, {work_size});
  op.SetInput(reinterpret_cast<DataType*>(out_meta.data_ptr), reinterpret_cast<DataType*>(query_meta.data_ptr),
              reinterpret_cast<DataType**>(key_cache_ptrs.data_ptr),
              reinterpret_cast<DataType**>(value_cache_ptrs.data_ptr),
              reinterpret_cast<int*>(cache_offsets_meta.data_ptr), reinterpret_cast<int*>(context_lens_meta.data_ptr),
              max_context_len, num_seqs, stream, workspace_meta.data_ptr, work_size);
  op.Forward();
  CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

  // create ref buffer
  // reshape and transpose ref cache
  BufferMeta ref_key_cache = CreateBuffer<DataType>(
      MemoryType::MEMORY_GPU, {static_cast<size_t>(num_seqs), static_cast<size_t>(max_context_len),
                               static_cast<size_t>(num_heads), static_cast<size_t>(head_size)});
  BufferMeta ref_value_cache = CreateBuffer<DataType>(
      MemoryType::MEMORY_GPU, {static_cast<size_t>(num_seqs), static_cast<size_t>(max_context_len),
                               static_cast<size_t>(num_heads), static_cast<size_t>(head_size)});
  // run ref attention
  reshape_kv_cache(reinterpret_cast<DataType*>(ref_key_cache.data_ptr),
                   reinterpret_cast<DataType*>(ref_value_cache.data_ptr),
                   reinterpret_cast<DataType*>(key_caches.data_ptr), reinterpret_cast<DataType*>(value_caches.data_ptr),
                   num_seqs * seq_blocks, num_kv_heads, num_heads, head_size, block_size, x, stream);

  CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));
  // TODO Infer a result using FT's Multi-head Attention module, and compare the results of Paged Attention using the
  // same weight, kv-cache, and input.
}

TEST(CacheCopyTest, CacheCopyTest) {
  std::vector<size_t> h_input_offsets = {0};
  std::vector<size_t> h_block_offsets = {0};
  size_t block_size = 8;
  int bs = 2;
  int num_heads = 5;
  int head_size = 8;
  int x = 16 / sizeof(float);
  int token_data_size = num_heads * head_size;
  int stride_size = num_heads * head_size;
  std::vector<int> inputs = {17, 41};
  for (int i = 0; i < bs; i++) {
    h_input_offsets.push_back(inputs[i] + h_input_offsets.back());
    h_block_offsets.push_back((inputs[i] + block_size - 1) / block_size + h_block_offsets.back());
  }
  int total_len = h_input_offsets.back();
  std::vector<float> h_src;
  for (int i = 1; i <= bs; i++) {
    for (int j = 1; j <= inputs[i - 1]; j++) {
      for (int k = 1; k <= token_data_size; k++) {
        h_src.push_back((i * 100 + j) * 100 + k);
      }
    }
  }

  // 分配设备内存
  float* d_src;
  cudaMalloc(&d_src, h_src.size() * sizeof(float));
  void** d_k_list;
  cudaMalloc(&d_k_list, h_block_offsets.back() * sizeof(void*));
  void** d_v_list;
  cudaMalloc(&d_v_list, h_block_offsets.back() * sizeof(void*));
  size_t* d_input_offsets;
  cudaMalloc(&d_input_offsets, h_input_offsets.size() * sizeof(size_t));
  size_t* d_prefix_offsets;
  cudaMalloc(&d_prefix_offsets, h_input_offsets.size() * sizeof(size_t));
  int* d_block_offsets;
  cudaMalloc(&d_block_offsets, h_block_offsets.size() * sizeof(int));

  // 将主机数据复制到设备上
  cudaMemcpy(d_src, h_src.data(), h_src.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_input_offsets, h_input_offsets.data(), h_input_offsets.size() * sizeof(size_t), cudaMemcpyHostToDevice);
  cudaMemset(d_prefix_offsets, 0, h_input_offsets.size() * sizeof(size_t));
  cudaMemcpy(d_block_offsets, h_block_offsets.data(), h_block_offsets.size() * sizeof(int), cudaMemcpyHostToDevice);

  // 为 k_list 分配内存并初始化
  std::vector<float*> h_k_list_ptrs(h_block_offsets.back());
  for (size_t i = 0; i < h_block_offsets.back(); i++) {
    cudaMalloc(&h_k_list_ptrs[i], block_size * token_data_size * sizeof(float));
  }
  // 为 v_list 分配内存并初始化
  std::vector<float*> h_v_list_ptrs(h_block_offsets.back());
  for (size_t i = 0; i < h_block_offsets.back(); i++) {
    cudaMalloc(&h_v_list_ptrs[i], block_size * token_data_size * sizeof(float));
  }
  cudaMemcpy(d_k_list, h_k_list_ptrs.data(), h_k_list_ptrs.size() * sizeof(float*), cudaMemcpyHostToDevice);
  cudaMemcpy(d_v_list, h_v_list_ptrs.data(), h_v_list_ptrs.size() * sizeof(float*), cudaMemcpyHostToDevice);

  // 调用核函数
  CacheCopy<float, float, false>(d_src, d_src, d_k_list, d_v_list, d_input_offsets, d_prefix_offsets, d_block_offsets,
                                 block_size, bs, total_len, num_heads, head_size, stride_size, nullptr);
  cudaDeviceSynchronize();

  // 将结果从设备复制回主机并验证
  std::vector<float> h_dst(h_block_offsets.back() * token_data_size * block_size);
  for (size_t i = 0; i < h_block_offsets.back(); i++) {
    cudaMemcpy(h_dst.data() + i * block_size * token_data_size, h_k_list_ptrs[i],
               block_size * token_data_size * sizeof(float), cudaMemcpyDeviceToHost);
  }
  // for (int num_head_i = 0; num_head_i < num_heads; num_head_i++) {
  //     for (int head_size_i = 0; head_size_i < head_size / x; head_size_i++) {
  //             for (int i = 0; i < block_size; i++ ){
  //                 for (int j = 0; j < x; j++) {
  //                     int k_dst_index = num_head_i * (head_size * block_size) + head_size_i * (block_size * x) + i *
  //                     x + j; printf("%f ", *(h_dst.data() + k_dst_index));
  //                 }
  //                 printf("\n");
  //             }
  //             printf("\n");
  //     }
  //     printf("\n");
  // }

  // 验证结果
  int cache_offset_indx = 0;
  int src_index = 0;
  for (size_t i = 0; i < h_block_offsets.back() * block_size; i++) {
    if (h_input_offsets[cache_offset_indx] == i) {
      // printf("i %d -> %d\n", i, h_block_offsets[cache_offset_indx] * block_size);
      i = h_block_offsets[cache_offset_indx] * block_size;
      cache_offset_indx++;
      if (!(i < h_block_offsets.back() * block_size)) break;
    }
    int index = i % block_size;
    float* h_src_base = h_src.data() + src_index * token_data_size;
    float* h_dst_base = h_dst.data() + (i / block_size) * block_size * token_data_size;
    // printf("i = %d h_src_base = %d h_dst_base = %d\n", i, int(src_index * token_data_size), int((i / block_size) *
    // block_size * token_data_size));
    for (int num_head_i = 0; num_head_i < num_heads; num_head_i++) {
      for (int head_size_i = 0; head_size_i < head_size / x; head_size_i++) {
        for (int j = 0; j < x; j++) {
          int k_src_index = num_head_i * head_size + head_size_i * x + j;
          int k_dst_index = num_head_i * (head_size * block_size) + head_size_i * (block_size * x) + index * x + j;
          // printf("num_head_i = %d head_size_i = %d j = %d x = %d\n", num_head_i, head_size_i, j, x);
          // printf("k_src_index = %d k_dst_index = %d index = %d\n", k_src_index, k_dst_index, index);
          EXPECT_EQ(*(h_src_base + k_src_index), *(h_dst_base + k_dst_index));
        }
      }
    }
    //    printf("%f ", *(h_dst.data() + i * token_data_size + j));
    //    printf("%f ", *(h_src.data() + src_index * token_data_size + j));
    // printf("\n");
    src_index++;
    if (i == 1) break;
  }

  // 释放设备内存
  cudaFree(d_src);
  cudaFree(d_k_list);
  cudaFree(d_input_offsets);
  cudaFree(d_prefix_offsets);
  cudaFree(d_block_offsets);
  for (auto ptr : h_k_list_ptrs) {
    cudaFree(ptr);
  }
  for (auto ptr : h_v_list_ptrs) {
    cudaFree(ptr);
  }
}

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels
