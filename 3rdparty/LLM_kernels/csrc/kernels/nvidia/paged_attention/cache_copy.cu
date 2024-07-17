
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include "csrc/kernels/nvidia/paged_attention/cache_copy.h"
#include "csrc/kernels/nvidia/paged_attention/quant_utils.cuh"
namespace llm_kernels {
namespace nvidia {

#define MAX_THREADS_PER_BLOCK 1024

__device__ int k_chunk_size = 16;
/*
  block_size:     Number of tokens stored in each block.
  block_offsets:  Records the number of blocks for each batch size   [bs + 1,]
  prefix_offsets: Records the prefix length for each batch size (bs)
                  (accumulated from 0 to the current batch).         [bs + 1,]
*/
template <typename SCALAR_T, typename CACHE_T, bool FP8_E5M2>
__global__ void CacheCopyKernel(SCALAR_T* k_src, SCALAR_T* v_src, void** k_list, void** v_list, size_t* input_offsets,
                                size_t* prefix_offsets, int* block_offsets, int block_size, int bs, int total_len,
                                int num_heads, int head_size, int stride_size) {
  /*
    x:           In PagedAttention storage, KV-Blocks are divided into chunks to store head_size.
                 The variable x represents the size of each chunk.
    head_size_i: Indicates which chunk the head_size to be processed belongs to.
    j:           Represents the offset of the head_size to be processed within a single chunk.
  */
  int x = k_chunk_size / sizeof(SCALAR_T);
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int hs_i = blockIdx.y;
  int head_size_i = hs_i / x;
  int j = hs_i % x;
  if (idx < total_len) {
    int block_idx = 0;
    for (block_idx = 0; block_idx < bs; block_idx++) {
      if (idx < input_offsets[block_idx + 1]) {
        break;
      }
    }
    size_t prefix_limit = prefix_offsets[block_idx + 1] - prefix_offsets[block_idx] + input_offsets[block_idx];
    if (idx < prefix_limit) {
      return;
    }
    int cur_block_offset = (idx - input_offsets[block_idx]) / block_size;
    int cur_batch_offset = (idx - input_offsets[block_idx]) % block_size;
    CACHE_T* k_dst_base = reinterpret_cast<CACHE_T*>(k_list[block_offsets[block_idx] + cur_block_offset]);
    CACHE_T* v_dst_base = reinterpret_cast<CACHE_T*>(v_list[block_offsets[block_idx] + cur_block_offset]);
    SCALAR_T* k_src_ptr = k_src + idx * stride_size;
    SCALAR_T* v_src_ptr = v_src + idx * stride_size;

    for (int num_head_i = threadIdx.y; num_head_i < num_heads; num_head_i += blockDim.y) {
      int k_src_index = num_head_i * head_size + head_size_i * x + j;
      int k_dst_index =
          num_head_i * (head_size * block_size) + head_size_i * (block_size * x) + cur_batch_offset * x + j;
      int i = head_size_i * x + j;
      int v_src_index = num_head_i * head_size + i;
      int v_dst_index = num_head_i * (head_size * block_size) + i * block_size + cur_batch_offset;
      // Assignment operation
      if constexpr (FP8_E5M2) {
        k_dst_base[k_dst_index] = fp8_e5m2_unscaled::vec_conversion<CACHE_T, SCALAR_T>(k_src_ptr[k_src_index]);
        v_dst_base[v_dst_index] = fp8_e5m2_unscaled::vec_conversion<CACHE_T, SCALAR_T>(v_src_ptr[v_src_index]);
      } else {
        k_dst_base[k_dst_index] = k_src_ptr[k_src_index];
        v_dst_base[v_dst_index] = v_src_ptr[v_src_index];
      }
    }
  }
}

/*
  block_size:     Number of tokens stored in each block.
  block_offsets:  Records the number of blocks for each batch size   [bs + 1,]
  prefix_offsets: Records the prefix length for each batch size (bs)
                  (accumulated from 0 to the current batch).         [bs + 1,]
*/
template <typename SCALAR_T, typename CACHE_T, bool FP8_E5M2>
__global__ void ReverseCacheCopyKernel(SCALAR_T* k_src, SCALAR_T* v_src, void** k_list, void** v_list,
                                       size_t* input_offsets, size_t* prefix_offsets, int* block_offsets,
                                       int block_size, int bs, int total_len, int num_heads, int head_size,
                                       int stride_size) {
  /*
    x:           In PagedAttention storage, KV-Blocks are divided into chunks to store head_size.
                 The variable x represents the size of each chunk.
    head_size_i: Indicates which chunk the head_size to be processed belongs to.
    j:           Represents the offset of the head_size to be processed within a single chunk.
  */
  int x = k_chunk_size / sizeof(SCALAR_T);
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int hs_i = blockIdx.y;
  int head_size_i = hs_i / x;
  int j = hs_i % x;
  if (idx < total_len) {
    int block_idx = 0;
    for (block_idx = 0; block_idx < bs; block_idx++) {
      if (idx < input_offsets[block_idx + 1]) {
        break;
      }
    }
    size_t prefix_limit = prefix_offsets[block_idx + 1] - prefix_offsets[block_idx] + input_offsets[block_idx];
    if (idx >= prefix_limit) {
      return;
    }
    int cur_block_offset = (idx - input_offsets[block_idx]) / block_size;
    int cur_batch_offset = (idx - input_offsets[block_idx]) % block_size;
    CACHE_T* k_dst_base = reinterpret_cast<CACHE_T*>(k_list[block_offsets[block_idx] + cur_block_offset]);
    CACHE_T* v_dst_base = reinterpret_cast<CACHE_T*>(v_list[block_offsets[block_idx] + cur_block_offset]);
    SCALAR_T* k_src_ptr = k_src + idx * stride_size;
    SCALAR_T* v_src_ptr = v_src + idx * stride_size;

    for (int num_head_i = threadIdx.y; num_head_i < num_heads; num_head_i += blockDim.y) {
      int k_src_index = num_head_i * head_size + head_size_i * x + j;
      int k_dst_index =
          num_head_i * (head_size * block_size) + head_size_i * (block_size * x) + cur_batch_offset * x + j;
      int i = head_size_i * x + j;
      int v_src_index = num_head_i * head_size + i;
      int v_dst_index = num_head_i * (head_size * block_size) + i * block_size + cur_batch_offset;
      // Reverse assignment operation
      if constexpr (FP8_E5M2) {
        k_src_ptr[k_src_index] = fp8_e5m2_unscaled::vec_conversion<SCALAR_T, CACHE_T>(k_dst_base[k_dst_index]);
        v_src_ptr[v_src_index] = fp8_e5m2_unscaled::vec_conversion<SCALAR_T, CACHE_T>(v_dst_base[v_dst_index]);
      } else {
        k_src_ptr[k_src_index] = k_dst_base[k_dst_index];
        v_src_ptr[v_src_index] = v_dst_base[v_dst_index];
      }
    }
  }
}

/*
  block_size:    Number of tokens stored in each block.
  block_offsets: Records the number of blocks for each batch size   [bs + 1,]
*/
template <typename SCALAR_T, typename CACHE_T, bool FP8_E5M2>
__global__ void CachePosCopyKernel(SCALAR_T* k_src, SCALAR_T* v_src, void** k_list, void** v_list, void* pos,
                                   size_t* input_offsets, int* block_offsets, int block_size, int bs, int total_len,
                                   int num_heads, int head_size, int stride_size) {
  /*
    x:           In PagedAttention storage, KV-Blocks are divided into chunks to store head_size.
                 The variable x represents the size of each chunk.
    head_size_i: Indicates which chunk the head_size to be processed belongs to.
    j:           Represents the offset of the head_size to be processed within a single chunk.
  */
  int x = k_chunk_size / sizeof(SCALAR_T);
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int hs_i = blockIdx.y;
  int head_size_i = hs_i / x;
  int j = hs_i % x;
  if (idx < total_len) {
    int input_len = reinterpret_cast<int64_t*>(pos)[idx];
    int cur_block_offset = input_len / block_size;
    int cur_batch_offset = input_len % block_size;
    CACHE_T* k_dst_base = reinterpret_cast<CACHE_T*>(k_list[block_offsets[idx] + cur_block_offset]);
    CACHE_T* v_dst_base = reinterpret_cast<CACHE_T*>(v_list[block_offsets[idx] + cur_block_offset]);
    SCALAR_T* k_src_ptr = k_src + idx * stride_size;
    SCALAR_T* v_src_ptr = v_src + idx * stride_size;

    for (int num_head_i = threadIdx.y; num_head_i < num_heads; num_head_i += blockDim.y) {
      int k_src_index = num_head_i * head_size + head_size_i * x + j;
      int k_dst_index =
          num_head_i * (head_size * block_size) + head_size_i * (block_size * x) + cur_batch_offset * x + j;
      int i = head_size_i * x + j;
      int v_src_index = num_head_i * head_size + i;
      int v_dst_index = num_head_i * (head_size * block_size) + i * block_size + cur_batch_offset;
      //  赋值操作
      if constexpr (FP8_E5M2) {
        k_dst_base[k_dst_index] = fp8_e5m2_unscaled::vec_conversion<CACHE_T, SCALAR_T>(k_src_ptr[k_src_index]);
        v_dst_base[v_dst_index] = fp8_e5m2_unscaled::vec_conversion<CACHE_T, SCALAR_T>(v_src_ptr[v_src_index]);
      } else {
        k_dst_base[k_dst_index] = k_src_ptr[k_src_index];
        v_dst_base[v_dst_index] = v_src_ptr[v_src_index];
      }
    }
  }
}

template <typename SCALAR_T, typename CACHE_T, bool FP8_E5M2>
void CacheCopy(SCALAR_T* k_src, SCALAR_T* v_src, void** k_list, void** v_list, size_t* input_offsets,
               size_t* prefix_offsets, int* block_offsets, int block_size, int bs, int total_len, int num_heads,
               int head_size, int stride_size, cudaStream_t stream) {
  int threadsPerBlock = 32;
  int min_num_heads = std::min(MAX_THREADS_PER_BLOCK / threadsPerBlock, num_heads);
  int blocks = (total_len + threadsPerBlock - 1) / threadsPerBlock;
  dim3 grid_shape(blocks, head_size);
  dim3 block_shape(threadsPerBlock, min_num_heads);
  CacheCopyKernel<SCALAR_T, CACHE_T, FP8_E5M2><<<grid_shape, block_shape, 0, stream>>>(
      k_src, v_src, k_list, v_list, input_offsets, prefix_offsets, block_offsets, block_size, bs, total_len, num_heads,
      head_size, stride_size);
}

template <typename SCALAR_T, typename CACHE_T, bool FP8_E5M2>
void ReverseCacheCopy(SCALAR_T* k_src, SCALAR_T* v_src, void** k_list, void** v_list, size_t* input_offsets,
                      size_t* prefix_offsets, int* block_offsets, int block_size, int bs, int total_len, int num_heads,
                      int head_size, int stride_size, cudaStream_t stream) {
  int threadsPerBlock = 32;
  int min_num_heads = std::min(MAX_THREADS_PER_BLOCK / threadsPerBlock, num_heads);
  int blocks = (total_len + threadsPerBlock - 1) / threadsPerBlock;
  dim3 grid_shape(blocks, head_size);
  dim3 block_shape(threadsPerBlock, min_num_heads);
  ReverseCacheCopyKernel<SCALAR_T, CACHE_T, FP8_E5M2><<<grid_shape, block_shape, 0, stream>>>(
      k_src, v_src, k_list, v_list, input_offsets, prefix_offsets, block_offsets, block_size, bs, total_len, num_heads,
      head_size, stride_size);
}

template <typename SCALAR_T, typename CACHE_T, bool FP8_E5M2>
void CachePosCopy(SCALAR_T* k_src, SCALAR_T* v_src, void** k_list, void** v_list, void* pos, size_t* input_offsets,
                  int* block_offsets, int block_size, int bs, int total_len, int num_heads, int head_size,
                  int stride_size, cudaStream_t stream) {
  int threadsPerBlock = 32;
  int min_num_heads = std::min(MAX_THREADS_PER_BLOCK / threadsPerBlock, num_heads);
  int blocks = (total_len + threadsPerBlock - 1) / threadsPerBlock;
  dim3 grid_shape(blocks, head_size);
  dim3 block_shape(threadsPerBlock, min_num_heads);
  CachePosCopyKernel<SCALAR_T, CACHE_T, FP8_E5M2>
      <<<grid_shape, block_shape, 0, stream>>>(k_src, v_src, k_list, v_list, pos, input_offsets, block_offsets,
                                               block_size, bs, total_len, num_heads, head_size, stride_size);
}

template <typename SCALAR_T, typename CACHE_T, bool FP8_E5M2>
__global__ void ConvertFP8AndBackKernel(SCALAR_T* data, size_t dim0, size_t dim1, int stride_size) {
  if constexpr (!FP8_E5M2) {
    return;
  }
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < dim0 * dim1) {
    // FP16 to FP8
    auto data_idx = idx / dim1 * stride_size + idx % dim1;
    CACHE_T temp = fp8_e5m2_unscaled::vec_conversion<CACHE_T, SCALAR_T>(data[data_idx]);
    // FP8 back to FP16
    data[data_idx] = fp8_e5m2_unscaled::vec_conversion<SCALAR_T, CACHE_T>(temp);
  }
}

template <typename SCALAR_T, typename CACHE_T, bool FP8_E5M2>
void ConvertFP8AndBack(SCALAR_T* data, size_t dim0, size_t dim1, int stride_size, cudaStream_t stream) {
  int threads_per_block = 256;
  int blocks_per_grid = (dim0 * dim1 + threads_per_block - 1) / threads_per_block;
  ConvertFP8AndBackKernel<SCALAR_T, CACHE_T, FP8_E5M2>
      <<<blocks_per_grid, threads_per_block, 0, stream>>>(data, dim0, dim1, stride_size);
}

#define CACHE_COPY_FUNCTION_DECLARATION(SCALAR_T, CACHE_T, FP8_E5M2)                                                   \
  template void CacheCopy<SCALAR_T, CACHE_T, FP8_E5M2>(                                                                \
      SCALAR_T * k_src, SCALAR_T * v_src, void** k_list, void** v_list, size_t* input_offsets, size_t* prefix_offsets, \
      int* block_offsets, int block_size, int bs, int total_len, int num_heads, int head_size, int stride_size,        \
      cudaStream_t stream);                                                                                            \
  template void ReverseCacheCopy<SCALAR_T, CACHE_T, FP8_E5M2>(                                                         \
      SCALAR_T * k_src, SCALAR_T * v_src, void** k_list, void** v_list, size_t* input_offsets, size_t* prefix_offsets, \
      int* block_offsets, int block_size, int bs, int total_len, int num_heads, int head_size, int stride_size,        \
      cudaStream_t stream);                                                                                            \
  template void CachePosCopy<SCALAR_T, CACHE_T, FP8_E5M2>(                                                             \
      SCALAR_T * k_src, SCALAR_T * v_src, void** k_list, void** v_list, void* pos, size_t* input_offsets,              \
      int* block_offsets, int block_size, int bs, int total_len, int num_heads, int head_size, int stride_size,        \
      cudaStream_t stream);                                                                                            \
  template void ConvertFP8AndBack<SCALAR_T, CACHE_T, FP8_E5M2>(SCALAR_T * data, size_t dim0, size_t dim1,              \
                                                               int stride_size, cudaStream_t stream);

CACHE_COPY_FUNCTION_DECLARATION(float, float, false);
CACHE_COPY_FUNCTION_DECLARATION(float, uint8_t, true);
CACHE_COPY_FUNCTION_DECLARATION(half, half, false);
CACHE_COPY_FUNCTION_DECLARATION(half, uint8_t, true);
CACHE_COPY_FUNCTION_DECLARATION(__nv_bfloat16, __nv_bfloat16, false);
CACHE_COPY_FUNCTION_DECLARATION(__nv_bfloat16, uint8_t, true);
#undef CACHE_COPY_FUNCTION_DECLARATION

}  // namespace nvidia
}  // namespace llm_kernels
