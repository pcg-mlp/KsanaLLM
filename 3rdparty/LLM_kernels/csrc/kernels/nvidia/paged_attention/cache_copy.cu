
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include "csrc/kernels/nvidia/paged_attention/cache_copy.h"
namespace llm_kernels {
namespace nvidia {

#define MAX_THREADS_PER_BLOCK 1024

__device__ int k_chunk_size = 16;
/*
block_size：一个block块存几个token
block_offset：记录每个bs的block个数情况
total_len： 总的长度
*/
template <typename T>
__global__ void CacheCopyKernel(T* k_src, T* v_src, void** k_list, void** v_list, size_t* input_offst,
                                int* block_offset, int block_size, int bs, int total_len, int num_heads, int head_size,
                                int stride_size) {
  int x = k_chunk_size / sizeof(T);
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int hs_i = blockIdx.y;
  int head_size_i = hs_i / x;
  int j = hs_i % x;
  if (idx < total_len) {
    int block_idx = 0;
    for (block_idx = 0; block_idx < bs; block_idx++) {
      if (idx < input_offst[block_idx + 1]) {
        break;
      }
    }
    int cur_block_offset = (idx - input_offst[block_idx]) / block_size;
    int cur_batch_offset = (idx - input_offst[block_idx]) % block_size;
    T* k_dst_base = reinterpret_cast<T*>(k_list[block_offset[block_idx] + cur_block_offset]);
    T* v_dst_base = reinterpret_cast<T*>(v_list[block_offset[block_idx] + cur_block_offset]);
    T* k_src_ptr = k_src + idx * stride_size;
    T* v_src_ptr = v_src + idx * stride_size;

    for (int num_head_i = threadIdx.y; num_head_i < num_heads; num_head_i += blockDim.y) {
      int k_src_index = num_head_i * head_size + head_size_i * x + j;
      int k_dst_index =
          num_head_i * (head_size * block_size) + head_size_i * (block_size * x) + cur_batch_offset * x + j;
      int i = head_size_i * x + j;
      int v_src_index = num_head_i * head_size + i;
      int v_dst_index = num_head_i * (head_size * block_size) + i * block_size + cur_batch_offset;
      //  赋值操作
      k_dst_base[k_dst_index] = k_src_ptr[k_src_index];
      v_dst_base[v_dst_index] = v_src_ptr[v_src_index];
    }
  }
}

/*
block_size：一个block块存几个token
block_offset：记录每个bs的block个数情况
total_len： 总的长度
*/
template <typename T>
__global__ void CachePosCopyKernel(T* k_src, T* v_src, void** k_list, void** v_list, void* pos, size_t* input_offst,
                                   int* block_offset, int block_size, int bs, int total_len, int num_heads,
                                   int head_size, int stride_size) {
  int x = k_chunk_size / sizeof(T);
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int hs_i = blockIdx.y;
  int head_size_i = hs_i / x;
  int j = hs_i % x;
  if (idx < total_len) {
    int input_len = reinterpret_cast<int64_t*>(pos)[idx];
    int cur_block_offset = input_len / block_size;
    int cur_batch_offset = input_len % block_size;
    T* k_dst_base = reinterpret_cast<T*>(k_list[block_offset[idx] + cur_block_offset]);
    T* v_dst_base = reinterpret_cast<T*>(v_list[block_offset[idx] + cur_block_offset]);
    T* k_src_ptr = k_src + idx * stride_size;
    T* v_src_ptr = v_src + idx * stride_size;

    for (int num_head_i = threadIdx.y; num_head_i < num_heads; num_head_i += blockDim.y) {
      int k_src_index = num_head_i * head_size + head_size_i * x + j;
      int k_dst_index =
          num_head_i * (head_size * block_size) + head_size_i * (block_size * x) + cur_batch_offset * x + j;
      int i = head_size_i * x + j;
      int v_src_index = num_head_i * head_size + i;
      int v_dst_index = num_head_i * (head_size * block_size) + i * block_size + cur_batch_offset;
      //  赋值操作
      k_dst_base[k_dst_index] = k_src_ptr[k_src_index];
      v_dst_base[v_dst_index] = v_src_ptr[v_src_index];
    }
  }
}

template <typename T>
void CacheCopy(T* k_src, T* v_src, void** k_list, void** v_list, size_t* input_offst, int* block_offset, int block_size,
               int bs, int total_len, int num_heads, int head_size, int stride_size, cudaStream_t stream) {
  int threadsPerBlock = 32;
  int min_num_heads = std::min(MAX_THREADS_PER_BLOCK / threadsPerBlock, num_heads);
  int blocks = (total_len + threadsPerBlock - 1) / threadsPerBlock;
  dim3 grid_shape(blocks, head_size);
  dim3 block_shape(threadsPerBlock, min_num_heads);
  CacheCopyKernel<<<grid_shape, block_shape, 0, stream>>>(k_src, v_src, k_list, v_list, input_offst, block_offset,
                                                          block_size, bs, total_len, num_heads, head_size, stride_size);
}

template <typename T>
void CachePosCopy(T* k_src, T* v_src, void** k_list, void** v_list, void* pos, size_t* input_offst, int* block_offset,
                  int block_size, int bs, int total_len, int num_heads, int head_size, int stride_size,
                  cudaStream_t stream) {
  int threadsPerBlock = 32;
  int min_num_heads = std::min(MAX_THREADS_PER_BLOCK / threadsPerBlock, num_heads);
  int blocks = (total_len + threadsPerBlock - 1) / threadsPerBlock;
  dim3 grid_shape(blocks, head_size);
  dim3 block_shape(threadsPerBlock, min_num_heads);
  CachePosCopyKernel<<<grid_shape, block_shape, 0, stream>>>(k_src, v_src, k_list, v_list, pos, input_offst,
                                                             block_offset, block_size, bs, total_len, num_heads,
                                                             head_size, stride_size);
}

template void CacheCopy<float>(float* k_src, float* v_src, void** k_list, void** v_list, size_t* input_offst,
                               int* block_offset, int block_size, int bs, int total_len, int num_heads, int head_size,
                               int stride_size, cudaStream_t stream);
template void CacheCopy<__nv_bfloat16>(__nv_bfloat16* k_src, __nv_bfloat16* v_src, void** k_list, void** v_list,
                                       size_t* input_offst, int* block_offset, int block_size, int bs, int total_len,
                                       int num_heads, int head_size, int stride_size, cudaStream_t stream);
template void CacheCopy<half>(half* k_src, half* v_src, void** k_list, void** v_list, size_t* input_offst,
                              int* block_offset, int block_size, int bs, int total_len, int num_heads, int head_size,
                              int stride_size, cudaStream_t stream);

template void CachePosCopy<float>(float* k_src, float* v_src, void** k_list, void** v_list, void* pos,
                                  size_t* input_offst, int* block_offset, int block_size, int bs, int total_len,
                                  int num_heads, int head_size, int stride_size, cudaStream_t stream);
template void CachePosCopy<__nv_bfloat16>(__nv_bfloat16* k_src, __nv_bfloat16* v_src, void** k_list, void** v_list,
                                          void* pos, size_t* input_offst, int* block_offset, int block_size, int bs,
                                          int total_len, int num_heads, int head_size, int stride_size,
                                          cudaStream_t stream);
template void CachePosCopy<half>(half* k_src, half* v_src, void** k_list, void** v_list, void* pos, size_t* input_offst,
                                 int* block_offset, int block_size, int bs, int total_len, int num_heads, int head_size,
                                 int stride_size, cudaStream_t stream);

}  // namespace nvidia
}  // namespace llm_kernels
