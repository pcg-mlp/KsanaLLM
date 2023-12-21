/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "numerous_llm/kernels/nvidia/paged_attention.h"

#include <optional>

#include "3rdparty/LLM_kernels/csrc/kernels/nvidia/paged_attention/paged_attention.h"
#include "numerous_llm/utils/tensor.h"

namespace numerous_llm {

size_t get_cache_ptrs_numel(std::vector<Tensor>& cache) {
  size_t cache_ptrs_numel = 0;
  for (int i = 0; i < static_cast<int>(cache.size()); ++i) {
    cache_ptrs_numel += cache.at(i).blocks.size();
  }
  return cache_ptrs_numel;
}

void copy_cache_offsets(std::vector<Tensor>& cache, int* cache_offsets, cudaStream_t stream) {
  std::vector<int> cache_offsets_host;
  int offset = 0;
  for (int i = 0; i < static_cast<int>(cache.size()); ++i) {
    offset += static_cast<int>(cache.at(i).blocks.size());
    cache_offsets_host.push_back(offset);
  }
  cudaMemcpyAsync(cache_offsets, cache_offsets_host.data(), cache.size() * sizeof(int), cudaMemcpyHostToDevice, stream);
}

void copy_cache_ptrs(std::vector<Tensor>& cache, int* cache_offsets, void** cache_ptrs, cudaStream_t stream) {
  std::vector<void*> cache_ptrs_host;
  for (int i = 0; i < static_cast<int>(cache.size()); ++i) {
    std::vector<void*> data_ptrs = cache.at(i).GetPtrs<void>();
    cache_ptrs_host.insert(cache_ptrs_host.end(), data_ptrs.begin(), data_ptrs.end());
  }
  cudaMemcpyAsync(cache_ptrs, cache_ptrs_host.data(), get_cache_ptrs_numel(cache) * sizeof(void*),
                  cudaMemcpyHostToDevice, stream);
}

template <typename T>
void run_paged_attention(
    Tensor& out,                       // [num_seqs, num_heads, head_size]
    const Tensor& query,                     // [num_seqs, num_heads, head_size]
    std::vector<Tensor>& key_cache,    // num_seqs,[seq_blocks,num_kv_heads,head_size/x,block_size,x],x=16/sizeof(T)
    std::vector<Tensor>& value_cache,  // num_seqs,[seq_blocks, num_kv_heads, head_size, block_size]
    const Tensor& context_lens,              // [num_seqs]
    int max_context_len, cudaStream_t stream,
    void** key_cache_ptrs,    // num_seqs,[seq_blocks]
    void** value_cache_ptrs,  // num_seqs,[seq_blocks]
    int* cache_offsets_ptr,   // num_seqs
    void* workspace, size_t work_size, const std::optional<Tensor>& alibi_slopes) {
  int num_seqs = query.shape[0];
  int num_heads = query.shape[1];
  int head_size = query.shape[2];
  int num_kv_heads = value_cache.at(0).shape[1];
  int block_size = value_cache.at(0).shape[3];

  T* out_ptr = out.GetPtr<T>();
  const T* query_ptr = query.GetPtr<T>();
  const int* context_lens_ptr = context_lens.GetPtr<int>();
  const float* alibi_slopes_ptr =
      reinterpret_cast<const float*>(alibi_slopes.has_value() ? alibi_slopes.value().GetPtr<T>() : nullptr);

  llm_kernels::nvidia::PagedAttentionCuda<T> op;
  op.SetConfig(num_kv_heads, num_heads, head_size, block_size);

  op.SetInput(out_ptr, query_ptr, reinterpret_cast<T**>(key_cache_ptrs), reinterpret_cast<T**>(value_cache_ptrs),
              cache_offsets_ptr, context_lens_ptr, max_context_len, num_seqs, stream, workspace, work_size,
              alibi_slopes_ptr);
  op.Forward();
}

void paged_attention(
    Tensor& out,                       // [num_seqs, num_heads, head_size]
    const Tensor& query,                     // [num_seqs, num_heads, head_size]
    std::vector<Tensor>& key_cache,    // num_seqs,[seq_blocks,num_kv_heads,head_size/x,block_size,x],x=16/sizeof(T)
    std::vector<Tensor>& value_cache,  // num_seqs,[seq_blocks, num_kv_heads, head_size, block_size]
    const Tensor& context_lens,              // [num_seqs]
    int max_context_len, cudaStream_t stream, void* workspace, size_t work_size,
    const std::optional<Tensor>& alibi_slopes) {
  // get gpu buffers of key_cache_ptrs, value_cache_ptrs and cache_offsets_ptr from workspace
  size_t cache_size = get_cache_ptrs_numel(key_cache) * sizeof(void*);
  size_t offsets_size = query.shape[0] * sizeof(int);
  if (work_size < 2 * cache_size + offsets_size) {
    throw std::runtime_error("work_size < 2 * cache_size + offsets_size");
  }
  char* work_ptr = static_cast<char*>(workspace);
  size_t remaining_size = work_size;
  void** key_cache_ptrs = reinterpret_cast<void**>(work_ptr);  // num_seqs,[seq_blocks]
  work_ptr += cache_size;
  remaining_size -= cache_size;
  void** value_cache_ptrs = reinterpret_cast<void**>(work_ptr);  // num_seqs,[seq_blocks]
  work_ptr += cache_size;
  remaining_size -= cache_size;
  int* cache_offsets_ptr = reinterpret_cast<int*>(work_ptr);  // num_seqs
  work_ptr += offsets_size;
  remaining_size -= offsets_size;
  copy_cache_offsets(key_cache, cache_offsets_ptr, stream);
  copy_cache_ptrs(key_cache, cache_offsets_ptr, key_cache_ptrs, stream);
  copy_cache_ptrs(value_cache, cache_offsets_ptr, value_cache_ptrs, stream);

  DataType dtype = query.dtype;
  // specialize template according to dtype
  if (dtype == TYPE_FP32) {
    run_paged_attention<float>(out, query, key_cache, value_cache, context_lens, max_context_len, stream,
                               key_cache_ptrs, value_cache_ptrs, cache_offsets_ptr, work_ptr, remaining_size,
                               alibi_slopes);
  } else if (dtype == TYPE_FP16) {
    run_paged_attention<uint16_t>(out, query, key_cache, value_cache, context_lens, max_context_len, stream,
                                  key_cache_ptrs, value_cache_ptrs, cache_offsets_ptr, work_ptr, remaining_size,
                                  alibi_slopes);
  } else if (dtype == TYPE_BF16) {
    run_paged_attention<__nv_bfloat16>(out, query, key_cache, value_cache, context_lens, max_context_len, stream,
                                       key_cache_ptrs, value_cache_ptrs, cache_offsets_ptr, work_ptr, remaining_size,
                                       alibi_slopes);
  } else {
    throw std::runtime_error("Unsupported data type");
  }
}

}  // namespace numerous_llm
