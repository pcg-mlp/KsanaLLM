
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include "cache_copy_flash_attn_layout.h"
#include "quant_utils.cuh"
namespace llm_kernels {
namespace nvidia {

#define MAX_THREADS_PER_BLOCK 1024
#define MAX_BLOCKS_PER_GRID_Y 65535

__device__ int k_chunk_size = 16;
/*
  block_size:     Number of tokens stored in each block.
  block_offsets:  Records the number of blocks for each batch size   [bs + 1,]
  prefix_offsets: Records the prefix length for each batch size (bs)
                  (accumulated from 0 to the current batch).         [bs + 1,]
*/
template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
__global__ void CacheCopyFlashAttnLayoutKernel(SCALAR_T* k_src, SCALAR_T* v_src, void** k_list, void** v_list,
                                               size_t* input_offsets, size_t* prefix_offsets,
                                               size_t* without_prefix_offsets, int* block_offsets, int block_size,
                                               int bs, int total_len, int num_heads, int head_size, int stride_size,
                                               float k_scale, float v_scale) {
  // copy from k,v(without_prefix_offsets) to cache list (input_offsets with prefix offsets)
  int idx = blockIdx.y + blockIdx.z * gridDim.y;
  if (idx < total_len) {
    int batch_idx = 0;
    for (batch_idx = 0; batch_idx < bs; batch_idx++) {
      if (idx < without_prefix_offsets[batch_idx + 1]) {
        break;
      }
    }
    // size_t prefix_limit = prefix_offsets[batch_idx + 1] - prefix_offsets[batch_idx] + input_offsets[batch_idx];
    int cur_batch_token_idx_with_prefix =
        (prefix_offsets[batch_idx + 1] - prefix_offsets[batch_idx]) + (idx - without_prefix_offsets[batch_idx]);
    int cur_block_offset = cur_batch_token_idx_with_prefix / block_size;
    int cur_batch_offset = cur_batch_token_idx_with_prefix % block_size;
    CACHE_T* k_dst_base = reinterpret_cast<CACHE_T*>(k_list[block_offsets[batch_idx] + cur_block_offset]);
    CACHE_T* v_dst_base = reinterpret_cast<CACHE_T*>(v_list[block_offsets[batch_idx] + cur_block_offset]);
    SCALAR_T* k_src_ptr = k_src + idx * stride_size;
    SCALAR_T* v_src_ptr = v_src + idx * stride_size;
    for (int head_size_i = threadIdx.x; head_size_i < head_size; head_size_i += blockDim.x) {
      for (int num_head_i = blockIdx.x; num_head_i < num_heads; num_head_i += gridDim.x) {
        int k_src_index = num_head_i * head_size + head_size_i;
        int k_dst_index = cur_batch_offset * num_heads * head_size + num_head_i * head_size + head_size_i;
        int v_src_index = num_head_i * head_size + head_size_i;
        int v_dst_index = cur_batch_offset * num_heads * head_size + num_head_i * head_size + head_size_i;
        // Assignment operation
        if constexpr (KV_DTYPE == llm_kernels::utils::KVCacheType::kAuto) {
          k_dst_base[k_dst_index] = k_src_ptr[k_src_index];
          v_dst_base[v_dst_index] = v_src_ptr[v_src_index];
        } else {
          k_dst_base[k_dst_index] = fp8::scaled_convert<CACHE_T, SCALAR_T, KV_DTYPE>(k_src_ptr[k_src_index], k_scale);
          v_dst_base[v_dst_index] = fp8::scaled_convert<CACHE_T, SCALAR_T, KV_DTYPE>(v_src_ptr[v_src_index], v_scale);
        }
      }
    }
  }
}

/*
  block_size:    Number of tokens stored in each block.
  block_offsets: Records the number of blocks for each batch size   [bs + 1,]
*/
template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
__global__ void CachePosCopyFlashAttnLayoutKernel(SCALAR_T* k_src, SCALAR_T* v_src, void** k_list, void** v_list,
                                                  void* pos, size_t* input_offsets, int* block_offsets, int block_size,
                                                  int bs, int total_len, int num_heads, int head_size, int stride_size,
                                                  float k_scale, float v_scale) {
  int idx = blockIdx.y;
  if (idx < total_len) {
    int input_len = reinterpret_cast<int64_t*>(pos)[idx];
    int cur_block_offset = input_len / block_size;
    int cur_batch_offset = input_len % block_size;
    CACHE_T* k_dst_base = reinterpret_cast<CACHE_T*>(k_list[block_offsets[idx] + cur_block_offset]);
    CACHE_T* v_dst_base = reinterpret_cast<CACHE_T*>(v_list[block_offsets[idx] + cur_block_offset]);
    SCALAR_T* k_src_ptr = k_src + idx * stride_size;
    SCALAR_T* v_src_ptr = v_src + idx * stride_size;

    for (int head_size_i = threadIdx.x; head_size_i < head_size; head_size_i += blockDim.x) {
      for (int num_head_i = blockIdx.x; num_head_i < num_heads; num_head_i += gridDim.x) {
        int k_src_index = num_head_i * head_size + head_size_i;
        int k_dst_index = cur_batch_offset * num_heads * head_size + num_head_i * head_size + head_size_i;
        int v_src_index = num_head_i * head_size + head_size_i;
        int v_dst_index = cur_batch_offset * num_heads * head_size + num_head_i * head_size + head_size_i;
        //  赋值操作
        if constexpr (KV_DTYPE == llm_kernels::utils::KVCacheType::kAuto) {
          k_dst_base[k_dst_index] = k_src_ptr[k_src_index];
          v_dst_base[v_dst_index] = v_src_ptr[v_src_index];
        } else {
          k_dst_base[k_dst_index] = fp8::scaled_convert<CACHE_T, SCALAR_T, KV_DTYPE>(k_src_ptr[k_src_index], k_scale);
          v_dst_base[v_dst_index] = fp8::scaled_convert<CACHE_T, SCALAR_T, KV_DTYPE>(v_src_ptr[v_src_index], v_scale);
        }
      }
    }
  }
}

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void CacheCopyFlashAttnLayout(SCALAR_T* k_src, SCALAR_T* v_src, void** k_list, void** v_list, size_t* input_offsets,
                              size_t* prefix_offsets, size_t* without_prefix_offsets, int* block_offsets,
                              int block_size, int bs, int total_len, int num_heads, int head_size, int stride_size,
                              float k_scale, float v_scale, cudaStream_t stream) {
  int grid_y = std::min(total_len, MAX_BLOCKS_PER_GRID_Y);
  int grid_z = (total_len + MAX_BLOCKS_PER_GRID_Y - 1) / MAX_BLOCKS_PER_GRID_Y;
  dim3 grid_shape(num_heads, grid_y, grid_z);

  dim3 block_shape(std::min(head_size, MAX_THREADS_PER_BLOCK));
  CacheCopyFlashAttnLayoutKernel<SCALAR_T, CACHE_T, KV_DTYPE><<<grid_shape, block_shape, 0, stream>>>(
      k_src, v_src, k_list, v_list, input_offsets, prefix_offsets, without_prefix_offsets, block_offsets, block_size,
      bs, total_len, num_heads, head_size, stride_size, k_scale, v_scale);
}

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void CachePosCopyFlashAttnLayout(SCALAR_T* k_src, SCALAR_T* v_src, void** k_list, void** v_list, void* pos,
                                 size_t* input_offsets, int* block_offsets, int block_size, int bs, int total_len,
                                 int num_heads, int head_size, int stride_size, float k_scale, float v_scale,
                                 cudaStream_t stream) {
  dim3 grid_shape(num_heads, total_len);
  dim3 block_shape(std::min(head_size, MAX_THREADS_PER_BLOCK));
  CachePosCopyFlashAttnLayoutKernel<SCALAR_T, CACHE_T, KV_DTYPE><<<grid_shape, block_shape, 0, stream>>>(
      k_src, v_src, k_list, v_list, pos, input_offsets, block_offsets, block_size, bs, total_len, num_heads, head_size,
      stride_size, k_scale, v_scale);
}

#define CACHE_COPY_FLASH_ATTN_LAYOUT_FUNCTION_DECLARATION(SCALAR_T, CACHE_T, KV_DTYPE)                                 \
  template void CacheCopyFlashAttnLayout<SCALAR_T, CACHE_T, KV_DTYPE>(                                                 \
      SCALAR_T * k_src, SCALAR_T * v_src, void** k_list, void** v_list, size_t* input_offsets, size_t* prefix_offsets, \
      size_t* without_prefix_offsets, int* block_offsets, int block_size, int bs, int total_len, int num_heads,        \
      int head_size, int stride_size, float k_scale, float v_scale, cudaStream_t stream);                              \
  template void CachePosCopyFlashAttnLayout<SCALAR_T, CACHE_T, KV_DTYPE>(                                              \
      SCALAR_T * k_src, SCALAR_T * v_src, void** k_list, void** v_list, void* pos, size_t* input_offsets,              \
      int* block_offsets, int block_size, int bs, int total_len, int num_heads, int head_size, int stride_size,        \
      float k_scale, float v_scale, cudaStream_t stream);

CACHE_COPY_FLASH_ATTN_LAYOUT_FUNCTION_DECLARATION(float, float, llm_kernels::utils::KVCacheType::kAuto);
CACHE_COPY_FLASH_ATTN_LAYOUT_FUNCTION_DECLARATION(float, uint8_t, llm_kernels::utils::KVCacheType::kFp8E4M3);
CACHE_COPY_FLASH_ATTN_LAYOUT_FUNCTION_DECLARATION(float, uint8_t, llm_kernels::utils::KVCacheType::kFp8E5M2);
CACHE_COPY_FLASH_ATTN_LAYOUT_FUNCTION_DECLARATION(half, half, llm_kernels::utils::KVCacheType::kAuto);
CACHE_COPY_FLASH_ATTN_LAYOUT_FUNCTION_DECLARATION(half, uint8_t, llm_kernels::utils::KVCacheType::kFp8E4M3);
CACHE_COPY_FLASH_ATTN_LAYOUT_FUNCTION_DECLARATION(half, uint8_t, llm_kernels::utils::KVCacheType::kFp8E5M2);
CACHE_COPY_FLASH_ATTN_LAYOUT_FUNCTION_DECLARATION(__nv_bfloat16, __nv_bfloat16, llm_kernels::utils::KVCacheType::kAuto);
CACHE_COPY_FLASH_ATTN_LAYOUT_FUNCTION_DECLARATION(__nv_bfloat16, uint8_t, llm_kernels::utils::KVCacheType::kFp8E4M3);
CACHE_COPY_FLASH_ATTN_LAYOUT_FUNCTION_DECLARATION(__nv_bfloat16, uint8_t, llm_kernels::utils::KVCacheType::kFp8E5M2);
#undef CACHE_COPY_FLASH_ATTN_LAYOUT_FUNCTION_DECLARATION

}  // namespace nvidia
}  // namespace llm_kernels
