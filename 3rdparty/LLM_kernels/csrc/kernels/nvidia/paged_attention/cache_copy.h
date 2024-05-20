#pragma once

#include <cuda_runtime.h>
namespace llm_kernels {
namespace nvidia {

template <typename T>
void CacheCopy(T* k_src, T* v_src, void** k_list, void** v_list, size_t* input_offsets, size_t* prefix_offsets,
               int* block_offsets, int block_size, int bs, int total_len, int num_heads, int head_size, int stride_size,
               cudaStream_t stream);

template <typename T>
void CachePosCopy(T* k_src, T* v_src, void** k_list, void** v_list, void* pos, size_t* input_offsets,
                  int* block_offsets, int block_size, int bs, int total_len, int num_heads, int head_size,
                  int stride_size, cudaStream_t stream);

template <typename T>
void ReverseCacheCopy(T* k_src, T* v_src, void** k_list, void** v_list, size_t* input_offsets, size_t* prefix_offsets,
                      int* block_offsets, int block_size, int bs, int total_len, int num_heads, int head_size,
                      int stride_size, cudaStream_t stream);

}  // namespace nvidia
}  // namespace llm_kernels
