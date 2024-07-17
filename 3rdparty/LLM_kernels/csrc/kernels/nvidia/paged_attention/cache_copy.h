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

/**
 * @brief Converts data to FP8 format and then back to its original format.
 *
 * This function is designed for testing the effects of computations in lower precision by
 * first converting the input data to FP8 format, performing operations, and then converting it back.
 *
 * @param data Pointer to the data array that will be converted.
 * @param dim0 The size of the first dimension of the data array.
 * @param dim1 The size of the second dimension of the data array.
 * @param stride_size The stride size, indicating the separation between consecutive elements.
 * @param stream The CUDA stream for asynchronous execution.
 * @tparam SCALAR_T The data type of the input data (e.g., __nv_bfloat16, half).
 * @tparam CACHE_T The data type used for intermediate storage during conversion.
 * @tparam FP8_E5M2 A boolean template parameter that determines the FP8 format to use.
 *                  True for E5M2 format.
 */
template <typename SCALAR_T, typename CACHE_T, bool FP8_E5M2>
void ConvertFP8AndBack(SCALAR_T* data, size_t dim0, size_t dim1, int stride_size, cudaStream_t stream);

}  // namespace nvidia
}  // namespace llm_kernels
