/*
 * Copyright 2025 Tencent Inc.  All rights reserved.
 */

#include <mutex>
#include <unordered_map>

#include "csrc/kernels/nvidia/assemble_tokens_hidden/assemble_tokens_hidden.h"
#include "csrc/kernels/nvidia/common/vec_dtypes.cuh"

using namespace llm_kernels::utils;

namespace llm_kernels {
namespace nvidia {

// AssembleTokensHiddenKernel optimized with shared memory
template <typename T, int BLOCK_SIZE = 256, int ELEMENTS_PER_THREAD = 4>
__global__ void AssembleTokensHiddenKernelSharedMem(const T* input, const size_t* accepted_tokens_idx,
                                                    const int32_t accepted_tokens_size, const int32_t hidden_units_num,
                                                    T* output) {
  // NOTE(karlluo): Declare shared memory to store hidden states for one batch
  // Dynamically allocate shared memory size
  extern __shared__ __align__(sizeof(T)) unsigned char shared_mem[];
  T* shared_data = reinterpret_cast<T*>(shared_mem);

  int batch_idx = blockIdx.x;
  if (batch_idx >= accepted_tokens_size) {
    return;
  }
  size_t input_offset = accepted_tokens_idx[batch_idx] * hidden_units_num;
  size_t output_offset = batch_idx * hidden_units_num;
  const int thread_idx = threadIdx.x;
  const int grid_size = blockDim.x;

  // NOTE(karlluo): Load data from global memory to shared memory
  // Use vectorized loading to improve memory bandwidth utilization
  if (hidden_units_num >= 4 && hidden_units_num % 4 == 0) {
    for (int i = thread_idx * 4; i < hidden_units_num; i += grid_size * 4) {
      if (i + 3 < hidden_units_num) {
        vec_t<T, 4> vec_data;
        vec_data.load(input + input_offset + i);
        vec_data.store(shared_data + i);
      }
    }
  } else {
    // Each thread processes multiple elements
#pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
      int idx = thread_idx + i * grid_size;
      if (idx < hidden_units_num) {
        shared_data[idx] = input[input_offset + idx];
      }
    }
    for (int idx = thread_idx + ELEMENTS_PER_THREAD * grid_size; idx < hidden_units_num; idx += grid_size) {
      shared_data[idx] = input[input_offset + idx];
    }
  }

  __syncthreads();

  // NOTE(karlluo): Write from shared memory to global memory output
  // Use vectorized storing to improve memory bandwidth utilization
  if (hidden_units_num >= 4 && hidden_units_num % 4 == 0) {
    for (int i = thread_idx * 4; i < hidden_units_num; i += grid_size * 4) {
      if (i + 3 < hidden_units_num) {
        vec_t<T, 4> vec_data;
        vec_data.load(shared_data + i);
        vec_data.store(output + output_offset + i);
      }
    }
  } else {
#pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
      int idx = thread_idx + i * grid_size;
      if (idx < hidden_units_num) {
        output[output_offset + idx] = shared_data[idx];
      }
    }
    for (int idx = thread_idx + ELEMENTS_PER_THREAD * grid_size; idx < hidden_units_num; idx += grid_size) {
      output[output_offset + idx] = shared_data[idx];
    }
  }
}

// AssembleTokensHiddenKernel optimized with shared memory and vectorized loading/storing
template <typename T, int VECTOR_SIZE = 4>
__global__ void AssembleTokensHiddenKernelVectorized(const T* input, const size_t* accepted_tokens_idx,
                                                     const int32_t accepted_tokens_size, const int32_t hidden_units_num,
                                                     T* output) {
  // NOTE(karlluo): Declare shared memory to store hidden states for one batch
  extern __shared__ __align__(sizeof(T)) unsigned char shared_mem[];
  T* shared_data = reinterpret_cast<T*>(shared_mem);

  int batch_idx = blockIdx.x;
  if (batch_idx >= accepted_tokens_size) return;
  size_t input_offset = accepted_tokens_idx[batch_idx] * hidden_units_num;
  size_t output_offset = batch_idx * hidden_units_num;

  // NOTE(karlluo): Use vectorized loading and storing to improve memory bandwidth utilization
  int vector_elements = (hidden_units_num / VECTOR_SIZE) * VECTOR_SIZE;

  // NOTE(karlluo):  Load data from global memory to shared memory
  for (int i = threadIdx.x * VECTOR_SIZE; i < vector_elements; i += blockDim.x * VECTOR_SIZE) {
    vec_t<T, VECTOR_SIZE> vec_data;
    vec_data.load(input + input_offset + i);
    vec_data.store(shared_data + i);
  }

  // NOTE(karlluo): Process remaining elements
  if (threadIdx.x < (hidden_units_num - vector_elements)) {
    int idx = vector_elements + threadIdx.x;
    shared_data[idx] = input[input_offset + idx];
  }

  __syncthreads();

  // NOTE(karlluo): Write from shared memory to global memory output
  for (int i = threadIdx.x * VECTOR_SIZE; i < vector_elements; i += blockDim.x * VECTOR_SIZE) {
    vec_t<T, VECTOR_SIZE> vec_data;
    vec_data.load(shared_data + i);
    vec_data.store(output + output_offset + i);
  }

  // NOTE(karlluo): Process remaining elements
  if (threadIdx.x < (hidden_units_num - vector_elements)) {
    int idx = vector_elements + threadIdx.x;
    output[output_offset + idx] = shared_data[idx];
  }
}

namespace {

// TODO(perf): The AssembleTokensHiddenKernel{SharedMem,Vectorized} kernels above declare
// dynamic shared memory of size hidden_units_num * sizeof(T) but use it only as a
// per-thread passthrough -- each thread loads its own slots into SMEM and immediately
// reads them back, with no inter-thread reuse, so the __syncthreads() is effectively a
// no-op and the SMEM provides no bandwidth or latency benefit. A cleaner long-term fix
// is to rewrite the kernels to copy directly from global to global, which would also
// remove the per-block dynamic SMEM ceiling that the helpers below have to work around.
// The opt-in helpers here are the minimal patch needed to unblock large hidden_units_num
// cases such as DeepSeek-V4-Pro MTP gather, where hidden_units_num = hc_mult * hidden_size
// = 28672 in bf16 needs 56 KB per block, above the 48 KB default per-block dynamic SMEM
// limit -- without opt-in, the kernel launch fails with cudaErrorInvalidValue.

// Returns the current device's per-block dynamic SMEM opt-in cap, cached per device id.
int GetMaxDynamicSmemPerBlockOptIn(int device_id) {
  static std::mutex cache_mu;
  static std::unordered_map<int, int> per_device_cap;
  {
    std::lock_guard<std::mutex> lock(cache_mu);
    auto it = per_device_cap.find(device_id);
    if (it != per_device_cap.end()) {
      return it->second;
    }
  }

  int max_smem = 0;
  if (cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device_id) != cudaSuccess) {
    return 0;
  }

  std::lock_guard<std::mutex> lock(cache_mu);
  per_device_cap[device_id] = max_smem;
  return max_smem;
}

// Opt-in the given kernel to use up to the device's max dynamic SMEM per block.
// Idempotent and cheap (CUDA driver tracks state per (kernel, device)); safe to call
// before every launch. On query / set failure this is a silent no-op -- the launch
// itself will then surface any actual error via the caller's existing CUDA error check,
// preserving the original error path.
template <typename KernelFn>
void EnsureMaxDynamicSmemOptIn(KernelFn kernel_fn) {
  int device_id = 0;
  if (cudaGetDevice(&device_id) != cudaSuccess) {
    return;
  }

  const int max_smem = GetMaxDynamicSmemPerBlockOptIn(device_id);
  if (max_smem <= 0) {
    return;
  }

  cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, max_smem);
}

}  // namespace

template <typename T>
void AssembleTokensHidden(const T* input, const size_t* accepted_tokens_idx, const int32_t accepted_tokens_size,
                       const int32_t hidden_units_num, T* output, cudaStream_t& stream) {
  // NOTE(karlluo): Choose appropriate thread block size and number of elements per thread based on hidden_units_num
  // size
  constexpr int BLOCK_SIZE = 256;

  // NOTE(karlluo): Calculate grid and block dimensions
  dim3 grid(min(static_cast<int32_t>(accepted_tokens_size), DEFAULT_CUDA_GPU_DEVICE_MAX_BLOCKS_NUM));
  dim3 block(BLOCK_SIZE);

  // NOTE(karlluo): Calculate shared memory size (in bytes)
  size_t shared_mem_size = hidden_units_num * sizeof(T);

  // NOTE(karlluo): Choose appropriate kernel and vectorization size based on hidden_units_num
  if (hidden_units_num >= 1024 && hidden_units_num % 4 == 0) {
    // For larger hidden_units_num divisible by 4, use vectorized kernel
    EnsureMaxDynamicSmemOptIn(AssembleTokensHiddenKernelVectorized<T, 4>);
    AssembleTokensHiddenKernelVectorized<T, 4><<<grid, block, shared_mem_size, stream>>>(
        input, accepted_tokens_idx, accepted_tokens_size, hidden_units_num, output);
  } else if (hidden_units_num <= 1024) {
    // For smaller hidden_units_num, each thread processes 4 elements
    EnsureMaxDynamicSmemOptIn(AssembleTokensHiddenKernelSharedMem<T, BLOCK_SIZE, 4>);
    AssembleTokensHiddenKernelSharedMem<T, BLOCK_SIZE, 4><<<grid, block, shared_mem_size, stream>>>(
        input, accepted_tokens_idx, accepted_tokens_size, hidden_units_num, output);
  } else if (hidden_units_num <= 4096) {
    // For medium-sized hidden_units_num, each thread processes 8 elements
    EnsureMaxDynamicSmemOptIn(AssembleTokensHiddenKernelSharedMem<T, BLOCK_SIZE, 8>);
    AssembleTokensHiddenKernelSharedMem<T, BLOCK_SIZE, 8><<<grid, block, shared_mem_size, stream>>>(
        input, accepted_tokens_idx, accepted_tokens_size, hidden_units_num, output);
  } else {
    // For larger hidden_units_num, each thread processes 16 elements
    EnsureMaxDynamicSmemOptIn(AssembleTokensHiddenKernelSharedMem<T, BLOCK_SIZE, 16>);
    AssembleTokensHiddenKernelSharedMem<T, BLOCK_SIZE, 16><<<grid, block, shared_mem_size, stream>>>(
        input, accepted_tokens_idx, accepted_tokens_size, hidden_units_num, output);
  }
}

#define INSTANTIATE_ASSEMBLE_TOKENS_HIDDEN(T)                                                                          \
  template void AssembleTokensHidden(const T* input, const size_t* accepted_tokens_idx,                             \
                                     const int32_t accepted_tokens_size, const int32_t hidden_units_num, T* output, \
                                     cudaStream_t& stream);

INSTANTIATE_ASSEMBLE_TOKENS_HIDDEN(float);
INSTANTIATE_ASSEMBLE_TOKENS_HIDDEN(half);
INSTANTIATE_ASSEMBLE_TOKENS_HIDDEN(__nv_bfloat16);

#undef INSTANTIATE_ASSEMBLE_TOKENS_HIDDEN

}  // namespace nvidia
}  // namespace llm_kernels
