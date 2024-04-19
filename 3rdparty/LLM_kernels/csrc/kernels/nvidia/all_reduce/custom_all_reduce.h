/*
 * Adapted from
 * https://github.com/vllm-project/vllm/blob/v0.3.1/csrc/custom_all_reduce.cuh
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

#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <iostream>
#include <limits>
#include <map>
#include <unordered_map>
#include <vector>

namespace llm_kernels {
namespace nvidia {

struct Signal {
  alignas(64) union {
    uint64_t flag;
    unsigned char data[8];
  } start;
  alignas(64) union {
    uint64_t flag;
    unsigned char data[8];
  } end;
};

struct Metadata {
  alignas(128) Signal sg;
  alignas(128) int counter;
};
static_assert(offsetof(Metadata, counter) == 128);
static_assert(sizeof(Metadata) == 256);

struct __align__(16) RankData { const void *__restrict__ ptrs[8]; };

struct Buffer {
  alignas(16) int data;
};

struct RankSignals {
  volatile Signal *signals[8];
  volatile Signal *buffers[8];
};

// like std::array, but aligned
template <typename T, int sz>
struct __align__(alignof(T) * sz) array_t {
  T data[sz];
  using type = T;
  static constexpr int size = sz;
};

// use packed type to maximize memory efficiency
// goal: generate ld.128 and st.128 instructions
template <typename T>
struct packed_t {
  // the (P)acked type for load/store
  using P = array_t<T, 16 / sizeof(T)>;
  // the (A)ccumulator type for reduction
  using A = array_t<float, 16 / sizeof(T)>;
};

class CustomAllreduce {
 public:
  int rank_;
  int world_size_;
  bool full_nvlink_;

  // below are device pointers
  RankSignals sg_;
  std::unordered_map<void *, RankData *> buffers_;
  Metadata *meta_;

  // stores the registered device pointers from all ranks
  RankData *d_rank_data_base_, *d_rank_data_end_;

  /**
   * meta is a pointer to device metadata and temporary buffer for allreduce.
   *
   * There's a total of sizeof(Metadata) of prefix before the actual data,
   * so meta + 1 points to actual temporary buffer.
   *
   * note: this class does not own any device memory. Any required buffers
   * are passed in from the constructor
   */
  CustomAllreduce(void **meta, void *rank_data, size_t rank_data_sz, void **handles,
                  const std::vector<int64_t> &offsets, int rank, bool full_nvlink = true);

  void CheckRankDataCapacity(size_t num = 1);

  void RegisterBuffer(const std::vector<std::string> &handles, const std::vector<int64_t> &offsets, void *self,
                      cudaStream_t &stream);

  /**
   * This is the result after careful grid search. Using 36 blocks give the best
   * or close to the best runtime on the devices I tried: A100, A10, A30, T4,
   * V100. You'll notice that NCCL kernels also only take a small amount of SMs.
   * Not quite sure the underlying reason, but my guess is that too many SMs
   * will cause contention on NVLink bus.
   */
  template <typename T>
  void AllReduce(cudaStream_t stream, T *input, T *output, int size, int threads = 512, int block_limit = 36);

  ~CustomAllreduce();
};

}  // namespace nvidia
}  // namespace llm_kernels
