/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <algorithm>
#include <cstdint>

#include <cuda_runtime.h>

namespace ksana_llm {

// Optimized batched device-to-device copy kernel.
//
// gridDim.x  : iterates over the (src, dst) pairs in the address lists (stride-loop).
// gridDim.y  : together with blockDim.x cooperatively copies the block_size payload of
//              one pair using the widest aligned vector path available (16B / 8B / 4B / 1B).
//
// Caller is required to pass block_size that is a multiple of 256, so each vector path
// (16B / 8B / 4B) divides block_size exactly and no tail handling is needed.
__global__ void BatchMemoryD2DKernel(const uint64_t* __restrict__ src_addr_list,
                                     const uint64_t* __restrict__ dst_addr_list, size_t src_offset, size_t dst_offset,
                                     size_t block_size, int N) {
  for (int i = blockIdx.x; i < N; i += gridDim.x) {
    const uint8_t* src_ptr = reinterpret_cast<const uint8_t*>(src_offset) + src_addr_list[i];
    uint8_t* dst_ptr = reinterpret_cast<uint8_t*>(dst_offset) + dst_addr_list[i];

    uintptr_t src_align = reinterpret_cast<uintptr_t>(src_ptr);
    uintptr_t dst_align = reinterpret_cast<uintptr_t>(dst_ptr);

    size_t stride = blockDim.x * gridDim.y;
    size_t offset = blockIdx.y * blockDim.x + threadIdx.x;

    if (((src_align | dst_align) & 15) == 0) {
      const int4* src_vec = reinterpret_cast<const int4*>(src_ptr);
      int4* dst_vec = reinterpret_cast<int4*>(dst_ptr);
      size_t num_elements = block_size >> 4;
#pragma unroll 4
      for (size_t idx = offset; idx < num_elements; idx += stride) {
        dst_vec[idx] = src_vec[idx];
      }
    } else if (((src_align | dst_align) & 7) == 0) {
      const int2* src_vec = reinterpret_cast<const int2*>(src_ptr);
      int2* dst_vec = reinterpret_cast<int2*>(dst_ptr);
      size_t num_elements = block_size >> 3;
#pragma unroll 4
      for (size_t idx = offset; idx < num_elements; idx += stride) {
        dst_vec[idx] = src_vec[idx];
      }
    } else if (((src_align | dst_align) & 3) == 0) {
      const int* src_vec = reinterpret_cast<const int*>(src_ptr);
      int* dst_vec = reinterpret_cast<int*>(dst_ptr);
      size_t num_elements = block_size >> 2;
#pragma unroll 4
      for (size_t idx = offset; idx < num_elements; idx += stride) {
        dst_vec[idx] = src_vec[idx];
      }
    } else {
#pragma unroll 4
      for (size_t idx = offset; idx < block_size; idx += stride) {
        dst_ptr[idx] = src_ptr[idx];
      }
    }
  }
}

void LaunchBatchMemoryD2DKernel(const uint64_t* src_addr_list, const uint64_t* dst_addr_list, size_t src_offset,
                                size_t dst_offset, size_t block_size, int N, cudaStream_t stream) {
  if (N <= 0 || block_size == 0) {
    return;
  }

  // Assume the 16-byte aligned path is the common case; size the launch geometry for it.
  size_t max_vec_elements = block_size >> 4;

  // Adaptive thread-block size: when block_size is tiny (e.g. 256B → 16 vec elements),
  // shrink to the smallest warp-multiple to avoid idle threads.
  int block_size_threads = 256;
  if (max_vec_elements < 256) {
    block_size_threads = static_cast<int>(((max_vec_elements + 31) / 32) * 32);
  }

  unsigned int grid_x = static_cast<unsigned int>(N);

  size_t chunks = (max_vec_elements + block_size_threads - 1) / block_size_threads;
  if (chunks == 0) {
    chunks = 1;
  }
  unsigned int grid_y = static_cast<unsigned int>(std::min(chunks, static_cast<size_t>(64)));

  dim3 grid(grid_x, grid_y);
  dim3 block(block_size_threads);

  BatchMemoryD2DKernel<<<grid, block, 0, stream>>>(src_addr_list, dst_addr_list, src_offset, dst_offset, block_size, N);
}

}  // namespace ksana_llm
