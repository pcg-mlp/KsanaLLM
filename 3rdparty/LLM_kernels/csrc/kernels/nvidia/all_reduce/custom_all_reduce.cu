/*
 * Adapted from
 * https://github.com/vllm-project/vllm/blob/v0.3.1/csrc/cuda_utils_kernels.cuh
 * Copyright (c) 2024, Tencent Inc.
 * Copyright (c) 2024, The vLLM team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <iostream>
#include <limits>
#include <map>
#include <unordered_map>
#include <vector>

#include "csrc/kernels/nvidia/all_reduce/custom_all_reduce.h"
#include "csrc/utils/nvidia/cuda_utils.h"

using namespace llm_kernels::utils;

namespace llm_kernels {
namespace nvidia {

#define DINLINE __device__ __forceinline__

// scalar cast functions
DINLINE float upcast_s(half val) { return __half2float(val); }

template <typename T>
DINLINE T downcast_s(float val);
template <>
DINLINE half downcast_s(float val) {
  return __float2half(val);
}

// scalar add functions
// for some reason when compiling with Pytorch, the + operator for half and
// bfloat is disabled so we call the intrinsics directly
DINLINE half &assign_add(half &a, half b) {
  a = __hadd(a, b);
  return a;
}
DINLINE float &assign_add(float &a, float b) { return a += b; }

#if (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
DINLINE float upcast_s(nv_bfloat16 val) { return __bfloat162float(val); }
template <>
DINLINE nv_bfloat16 downcast_s(float val) {
  return __float2bfloat16(val);
}
DINLINE nv_bfloat16 &assign_add(nv_bfloat16 &a, nv_bfloat16 b) {
  a = __hadd(a, b);
  return a;
}
#endif

template <typename T, int N>
DINLINE array_t<T, N> &packed_assign_add(array_t<T, N> &a, array_t<T, N> b) {
#pragma unroll
  for (int i = 0; i < N; i++) {
    assign_add(a.data[i], b.data[i]);
  }
  return a;
}

template <typename T, int N>
DINLINE array_t<float, N> upcast(array_t<T, N> val) {
  if constexpr (std::is_same<T, float>::value) {
    return val;
  } else {
    array_t<float, N> out;
#pragma unroll
    for (int i = 0; i < N; i++) {
      out.data[i] = upcast_s(val.data[i]);
    }
    return out;
  }
}

template <typename O>
DINLINE O downcast(array_t<float, O::size> val) {
  if constexpr (std::is_same<typename O::type, float>::value) {
    return val;
  } else {
    O out;
#pragma unroll
    for (int i = 0; i < O::size; i++) {
      out.data[i] = downcast_s<typename O::type>(val.data[i]);
    }
    return out;
  }
}

// compute flag at compile time
__host__ __device__ constexpr uint64_t compute_flag(int ngpus) {
  auto m = std::numeric_limits<uint64_t>::max();
  return m >> ((8 - ngpus) * 8);
}

template <int ngpus>
DINLINE void start_sync(const RankSignals &sg, volatile Metadata *meta, int rank) {
  constexpr auto FLAG = compute_flag(ngpus);
  if (blockIdx.x == 0) {
    if (threadIdx.x < ngpus)
      // simultaneously write to the corresponding byte to all other ranks.
      // Latency = 1 p2p write
      sg.signals[threadIdx.x]->start.data[rank] = 255;
    else if (threadIdx.x == 32)
      // reset
      meta->sg.end.flag = 0;
  }
  if (threadIdx.x == 0) {
    while (meta->sg.start.flag != FLAG)
      ;
  }
  __syncthreads();
}

template <int ngpus, bool final_sync = false>
DINLINE void end_sync(const RankSignals &sg, volatile Metadata *meta, int rank) {
  constexpr auto FLAG = compute_flag(ngpus);
  __syncthreads();
  __shared__ int num;
  if (threadIdx.x == 0) num = atomicAdd((int *)&meta->counter, 1);
  __syncthreads();

  // Only the last completing block can perform the end synchronization
  // This can ensures when the final busy wait ends, all ranks must have
  // finished reading each other's buffer.
  if (num == gridDim.x - 1) {
    if (threadIdx.x == 32) {
      // reset in a different warp
      meta->counter = 0;
      meta->sg.start.flag = 0;
    } else if (threadIdx.x < ngpus) {
      // simultaneously write to the corresponding byte to all other ranks.
      // Latency = 1 p2p write
      sg.signals[threadIdx.x]->end.data[rank] = 255;
    }
    // if this is the final sync, only one block needs it
    // because kernel exit can serve as sync
    if constexpr (final_sync) {
      if (threadIdx.x == 0) {
        while (meta->sg.end.flag != FLAG)
          ;
      }
    }
  }
  if constexpr (!final_sync) {
    if (threadIdx.x == 0) {
      while (meta->sg.end.flag != FLAG)
        ;
    }
    __syncthreads();
  }
}

template <typename P, int ngpus, typename A>
DINLINE P packed_reduce(const P *ptrs[], int idx) {
  A tmp = upcast(ptrs[0][idx]);
#pragma unroll
  for (int i = 1; i < ngpus; i++) {
    packed_assign_add(tmp, upcast(ptrs[i][idx]));
  }
  return downcast<P>(tmp);
}

template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1)
    cross_device_reduce_1stage(RankData *_dp, RankSignals sg, volatile Metadata *meta, T *__restrict__ result, int rank,
                               int size) {
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  // note: we don't reorder the address so the accumulation order is the same
  // for all ranks, ensuring bitwise identical results
  auto dp = *_dp;
  start_sync<ngpus>(sg, meta, rank);

  // do the actual reduction
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += gridDim.x * blockDim.x) {
    ((P *)result)[idx] = packed_reduce<P, ngpus, A>((const P **)&dp.ptrs[0], idx);
  }
  end_sync<ngpus, true>(sg, meta, rank);
}

template <typename P>
DINLINE P *get_tmp_buf(volatile Signal *sg) {
  return (P *)(((Metadata *)sg));
}

template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1)
    cross_device_reduce_2stage(RankData *_dp, RankSignals sg, volatile Metadata *meta, T *__restrict__ result, int rank,
                               int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  int part = size / ngpus;
  int start = rank * part;
  int end = rank == ngpus - 1 ? size : start + part;
  const P *ptrs[ngpus];
  P *tmps[ngpus];
#pragma unroll
  for (int i = 0; i < ngpus; i++) {
    int target = (rank + i) % ngpus;
    ptrs[i] = (const P *)_dp->ptrs[target];
    tmps[i] = get_tmp_buf<P>(sg.buffers[target]);
  }
  auto tmp_out = tmps[0];
  start_sync<ngpus>(sg, meta, rank);
  // stage 1: reduce scatter
  for (int idx = start + tid; idx < end; idx += stride) {
    tmp_out[idx - start] = packed_reduce<P, ngpus, A>(ptrs, idx);
  }
  // Maybe TODO: replace this with per-block release-acquire
  // can save about 1-2us (not a lot though)
  end_sync<ngpus>(sg, meta, rank);

  // stage 2: allgather
  for (int idx = tid; idx < part; idx += stride) {
#pragma unroll
    for (int i = 0; i < ngpus; i++) {
      int dst_idx = ((rank + i) % ngpus) * part + idx;
      ((P *)result)[dst_idx] = tmps[i][idx];
    }
  }
  // process the last larger partition
  int remaining = size - part * ngpus;
  if (tid < remaining) {
    int dst_idx = tid + part * ngpus;
    ((P *)result)[dst_idx] = get_tmp_buf<P>(sg.buffers[ngpus - 1])[part + tid];
  }

  // faster than this
  // for (int idx = tid; idx < size; idx += stride) {
  //   int target_rank = idx / part;
  //   if (target_rank == ngpus) target_rank -= 1;
  //   ((P *)result)[idx] = tmps[target_rank][idx - target_rank * part];
  // }
}

template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1)
    cross_device_reduce_half_butterfly(RankData *_dp, RankSignals sg, volatile Metadata *meta, T *__restrict__ result,
                                       int rank, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  auto tmp_out = get_tmp_buf<P>(sg.buffers[rank]);
  constexpr int hg = ngpus / 2;
  // Actually not quite half butterfly.
  // This is an all-to-all within each group containing half of the ranks
  // followed by cross-group add. Equivalent to half butterfly when there
  // are 4 GPUs, a common case for PCIe cards like T4 and A10.
  const P *ptrs[hg];
  {
    int start = rank - rank % hg;
#pragma unroll
    for (int i = 0; i < hg; i++) {
      ptrs[i] = (const P *)_dp->ptrs[i + start];
    }
  }
  start_sync<ngpus>(sg, meta, rank);
  for (int idx = tid; idx < size; idx += stride) {
    tmp_out[idx] = packed_reduce<P, hg, A>(ptrs, idx);
  }
  end_sync<ngpus>(sg, meta, rank);

  auto src = get_tmp_buf<P>(sg.buffers[(ngpus - 1) - rank % ngpus]);
  // do the cross group reduction
  for (int idx = tid; idx < size; idx += stride) {
    auto tmp = tmp_out[idx];
    packed_assign_add(tmp, src[idx]);
    ((P *)result)[idx] = tmp;
  }
}

/**
 * meta is a pointer to device metadata and temporary buffer for allreduce.
 *
 * There's a total of sizeof(Metadata) of prefix before the actual data,
 * so meta + 1 points to actual temporary buffer.
 *
 * note: this class does not own any device memory. Any required buffers
 * are passed in from the constructor
 */
CustomAllreduce::CustomAllreduce(void **meta, void *rank_data, size_t rank_data_sz, void **handles,
                                 const std::vector<int64_t> &offsets, int rank, bool full_nvlink)
    : rank_(rank),
      world_size_(offsets.size()),
      full_nvlink_(full_nvlink),
      meta_(static_cast<Metadata *>(meta[rank])),
      d_rank_data_base_(reinterpret_cast<RankData *>(rank_data)),
      d_rank_data_end_(d_rank_data_base_ + rank_data_sz / sizeof(RankData)) {
  for (int i = 0; i < world_size_; i++) {
    sg_.signals[i] = &(static_cast<Metadata *>(meta[i])->sg);
    sg_.buffers[i] = reinterpret_cast<Signal *>((char *)(handles[i]) + offsets[i]);
  }
}

void CustomAllreduce::CheckRankDataCapacity(size_t num) {
  if (d_rank_data_base_ + num > d_rank_data_end_)
    throw std::runtime_error("Rank data buffer is overflowed by " +
                             std::to_string(d_rank_data_base_ + num - d_rank_data_end_));
}

void CustomAllreduce::RegisterBuffer(const std::vector<std::string> &handles, const std::vector<int64_t> &offsets,
                                     void *self, cudaStream_t &stream) {
  CheckRankDataCapacity();
  RankData data;
  for (int i = 0; i < world_size_; i++) {
    if (i != rank_) {
      char *handle = (char *)(*(void **)(handles[i].data()));
      handle += offsets[i];
      data.ptrs[i] = handle;
    } else {
      data.ptrs[i] = self;
    }
  }
  auto d_data = d_rank_data_base_;
  CHECK_NVIDIA_CUDA_ERROR(cudaMemcpyAsync(d_data, &data, sizeof(RankData), cudaMemcpyHostToDevice, stream));
  buffers_[self] = d_data;
}

/**
 * This is the result after careful grid search. Using 36 blocks give the best
 * or close to the best runtime on the devices I tried: A100, A10, A30, T4,
 * V100. You'll notice that NCCL kernels also only take a small amount of SMs.
 * Not quite sure the underlying reason, but my guess is that too many SMs
 * will cause contention on NVLink bus.
 */
template <typename T>
void CustomAllreduce::AllReduce(cudaStream_t stream, T *input, T *output, int size, int threads, int block_limit) {
  auto d = packed_t<T>::P::size;
  if (size % d != 0)
    throw std::runtime_error(
        "custom allreduce currently requires input length to be multiple "
        "of " +
        std::to_string(d));

  RankData *ptrs;
  cudaStreamCaptureStatus status;
  CHECK_NVIDIA_CUDA_ERROR(cudaStreamIsCapturing(stream, &status));
  auto it = buffers_.find(input);
  if (it == buffers_.end())
    throw std::runtime_error("buffer address " + std::to_string(reinterpret_cast<uint64_t>(input)) +
                             " is not registered!");
  ptrs = it->second;

  size /= d;
  auto bytes = size * sizeof(typename packed_t<T>::P);
  int blocks = std::min(block_limit, (size + threads - 1) / threads);
#define KL(ngpus, name) name<T, ngpus><<<blocks, threads, 0, stream>>>(ptrs, sg_, meta_, output, rank_, size);
#define REDUCE_CASE(ngpus)                                                                        \
  case ngpus: {                                                                                   \
    if (world_size_ == 2) {                                                                       \
      KL(ngpus, cross_device_reduce_1stage);                                                      \
    } else if (full_nvlink_) {                                                                    \
      if ((world_size_ <= 4 && bytes < 512 * 1024) || (world_size_ <= 8 && bytes < 256 * 1024)) { \
        KL(ngpus, cross_device_reduce_1stage);                                                    \
      } else {                                                                                    \
        KL(ngpus, cross_device_reduce_2stage);                                                    \
      }                                                                                           \
    } else {                                                                                      \
      KL(ngpus, cross_device_reduce_half_butterfly);                                              \
    }                                                                                             \
    break;                                                                                        \
  }

  switch (world_size_) {
    REDUCE_CASE(2)
    REDUCE_CASE(4)
    REDUCE_CASE(6)
    REDUCE_CASE(8)
    default:
      throw std::runtime_error(
          "custom allreduce only supports num gpus in (2,4,6,8). Actual num "
          "gpus = " +
          std::to_string(world_size_));
  }
#undef REDUCE_CASE
#undef KL
}

CustomAllreduce::~CustomAllreduce() {}
/**
 * To inspect PTX/SASS, copy paste this header file to compiler explorer and add
 * a template instantiation:
 */
template void CustomAllreduce::AllReduce<float>(cudaStream_t, float *, float *, int, int, int);
template void CustomAllreduce::AllReduce<half>(cudaStream_t, half *, half *, int, int, int);
template void CustomAllreduce::AllReduce<__nv_bfloat16>(cudaStream_t, __nv_bfloat16 *, __nv_bfloat16 *, int, int, int);

}  // namespace nvidia
}  // namespace llm_kernels
