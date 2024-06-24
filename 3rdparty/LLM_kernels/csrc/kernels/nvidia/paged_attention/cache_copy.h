#pragma once

#include <cuda_runtime.h>
namespace llm_kernels {
namespace nvidia {

template <typename SCALAR_T, typename CACHE_T, bool FP8_E5M2>
void CacheCopy(SCALAR_T* k_src, SCALAR_T* v_src, void** k_list, void** v_list, size_t* input_offsets,
               size_t* prefix_offsets, int* block_offsets, int block_size, int bs, int total_len, int num_heads,
               int head_size, int stride_size, cudaStream_t stream);

template <typename SCALAR_T, typename CACHE_T, bool FP8_E5M2>
void CachePosCopy(SCALAR_T* k_src, SCALAR_T* v_src, void** k_list, void** v_list, void* pos, size_t* input_offsets,
                  int* block_offsets, int block_size, int bs, int total_len, int num_heads, int head_size,
                  int stride_size, cudaStream_t stream);

template <typename SCALAR_T, typename CACHE_T, bool FP8_E5M2>
void ReverseCacheCopy(SCALAR_T* k_src, SCALAR_T* v_src, void** k_list, void** v_list, size_t* input_offsets,
                      size_t* prefix_offsets, int* block_offsets, int block_size, int bs, int total_len, int num_heads,
                      int head_size, int stride_size, cudaStream_t stream);

}  // namespace nvidia
}  // namespace llm_kernels
