/*
 * Adapted from
 * https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/decodingCommon.cu
 * https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/common/reduceKernelUtils.cuh
 * Copyright (c) 2024, Tencent Inc.
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.  All rights reserved.
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

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cub/cub.cuh>
#include "decodingCommon.h"

namespace tensorrt_llm {
namespace kernels {
static float constexpr HALF_FLT_MAX = 65504.F;
#define FINAL_MASK 0xffffffff

template <typename T>
inline __device__ T add(T a, T b) {
  return a + b;
}

template <typename T>
__inline__ __device__ T warpReduceSum(T val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    val = add<T>(val, __shfl_xor_sync(FINAL_MASK, val, mask, 32));  //__shfl_sync bf16 return float when sm < 80
  return val;
}

/* Calculate the sum of all elements in a block */
template <typename T>
__inline__ __device__ T blockReduceSum(T val) {
  static __shared__ T shared[32];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  val = warpReduceSum<T>(val);

  if (lane == 0) shared[wid] = val;

  __syncthreads();

  // Modify from blockDim.x << 5 to blockDim.x / 32. to prevent
  // blockDim.x is not divided by 32
  val = (threadIdx.x < (blockDim.x / 32.f)) ? shared[lane] : (T)(0.0f);
  val = warpReduceSum<T>(val);

  return val;
}

template <typename T>
__inline__ __device__ T warpReduceMax(T val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) val = max(val, __shfl_xor_sync(FINAL_MASK, val, mask, 32));
  return val;
}

/* Calculate the maximum of all elements in a block */
template <typename T>
__inline__ __device__ T blockReduceMax(T val) {
  static __shared__ T shared[32];
  int lane = threadIdx.x & 0x1f;  // in-warp idx
  int wid = threadIdx.x >> 5;     // warp idx

  val = warpReduceMax(val);  // get maxx in each warp

  if (lane == 0)  // record in-warp maxx by warp Idx
    shared[wid] = val;

  __syncthreads();

  // Modify from blockDim.x << 5 to blockDim.x / 32. to prevent
  // blockDim.x is not divided by 32
  val = (threadIdx.x < (blockDim.x / 32.f)) ? shared[lane] : -1e20f;
  val = warpReduceMax(val);

  return val;
}

__global__ void curandInitialize(curandState_t* state, const int* batchSlots, const int size,
                                 const uint64_t randomSeed) {
  int const idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < size) {
    auto const batchSlot = batchSlots != nullptr ? batchSlots[idx] : idx;
    curand_init(randomSeed, 0, 0, &state[batchSlot]);
  }
}

void invokeCurandInitialize(curandState_t* state, const int* batchSlots, const size_t batchSize,
                            const uint64_t randomSeed, cudaStream_t stream) {
  dim3 block(256);
  dim3 grid((int)(ceil(batchSize * 1.0 / 256)));
  curandInitialize<<<grid, block, 0, stream>>>(state, batchSlots, batchSize, randomSeed);
}

__global__ void curandBatchInitialize(curandState_t* states, const int* batchSlots, const int size,
                                      const uint64_t* randomSeeds) {
  int const idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < size) {
    auto const batchSlot = batchSlots != nullptr ? batchSlots[idx] : idx;
    curand_init(randomSeeds[batchSlot], 0, 0, &states[batchSlot]);
  }
}

void invokeCurandBatchInitialize(curandState_t* states, const int* batchSlots, const size_t batchSize,
                                 const uint64_t* randomSeeds, cudaStream_t stream) {
  dim3 block(256);
  dim3 grid((int)(ceil(batchSize * 1.0 / 256)));
  curandBatchInitialize<<<grid, block, 0, stream>>>(states, batchSlots, batchSize, randomSeeds);
}

template <typename T>
__global__ void addBiasSoftMax(T* logits, T** logitsPtrs, T* temperatures, T const* bias, int32_t const* endIds,
                               FinishedState const* finished, int32_t const* batchSlots, int32_t batchSize,
                               int32_t maxBatchSize, int32_t beamWidth, int32_t vocabSize, int32_t vocabSizePadded,
                               bool skipSoftMax, bool batchSlotsLogits) {
  auto const batchIdx = blockIdx.x;
  auto const beamIdx = blockIdx.y;
  auto const batchSlot = batchSlots != nullptr ? batchSlots[batchIdx] : batchIdx;
  auto const batchIdxLogits = batchSlotsLogits ? batchSlot : batchIdx;
  FinishedState const finishState =
      finished != nullptr ? finished[beamIdx * maxBatchSize + batchSlot] : FinishedState::empty();
  if (finishState.isSkipDecoding()) {
    return;
  }

  auto logitsPtr = logitsPtrs ? logitsPtrs[batchIdx] + beamIdx * vocabSizePadded
                              : logits + (batchIdxLogits * beamWidth + beamIdx) * vocabSizePadded;

  T temperature = temperatures ? temperatures[batchIdx] : T(1.0f);
  temperature = temperature == T(0.0f) ? T(1.0f) : temperature;
  bool finish = finishState.isFinished();
  int offset = (batchIdxLogits * beamWidth + beamIdx) * vocabSizePadded;

  float maxVal = -1 * FLT_MAX;
  bool const IS_FP16 = std::is_same<T, half>::value;
  T const MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;
  __shared__ float sMaxVal;
  __shared__ float sSumVal;

  for (int tid = threadIdx.x; tid < vocabSizePadded; tid += blockDim.x) {
    logitsPtr[tid] = logitsPtr[tid] / temperature;
    auto logit = logitsPtr[tid];
    if (tid < vocabSize) {
      if (finish && endIds != nullptr) {
        logit = (tid == endIds[batchSlot]) ? MAX_T_VAL : -MAX_T_VAL;
      } else {
        T bias_val = (bias != nullptr) ? bias[tid] : (T)0.0f;
        logit += bias_val;
      }
    } else {
      logit = -MAX_T_VAL;
    }
    maxVal = max(maxVal, (float)logit);
    logitsPtr[tid] = logit;
  }

  if (!skipSoftMax) {
    maxVal = blockReduceMax<float>((float)maxVal);
    if (threadIdx.x == 0) {
      sMaxVal = maxVal;
    }
    __syncthreads();

    float sumVal = 0.0f;
    for (int tid = threadIdx.x; tid < vocabSizePadded; tid += blockDim.x) {
      logitsPtr[tid] = __expf((float)logitsPtr[tid] - sMaxVal);
      sumVal += (float)logitsPtr[tid];
    }

    sumVal = blockReduceSum<float>(sumVal);
    if (threadIdx.x == 0) {
      sSumVal = sumVal;
    }
    __syncthreads();

    for (int tid = threadIdx.x; tid < vocabSizePadded; tid += blockDim.x) {
      logitsPtr[tid] = ((float)logitsPtr[tid] / (sSumVal + 1e-6f));
    }
  }
}

template <typename T>
void invokeAddBiasSoftMax(T* logits, T** logitsPtrs, T* temperatures, T const* bias, int32_t const* endIds,
                          FinishedState const* finished, int32_t const* batchSlots, int32_t batchSize,
                          int32_t maxBatchSize, int32_t beamWidth, int32_t vocabSize, int32_t vocabSizePadded,
                          bool skipSoftMax, bool batchSlotsLogits, cudaStream_t stream) {
  dim3 grid(batchSize, beamWidth);
  auto const vocabRoundedToWarp = (vocabSize + 31) & ~31;
  dim3 block(min(vocabRoundedToWarp, 1024));
  // vocabSize, e.g., 30000, 7000.... vocabSize is usually very big.
  addBiasSoftMax<<<grid, block, 0, stream>>>(logits, logitsPtrs, temperatures, bias, endIds, finished, batchSlots,
                                             batchSize, maxBatchSize, beamWidth, vocabSize, vocabSizePadded,
                                             skipSoftMax, batchSlotsLogits);
}

template void invokeAddBiasSoftMax(float* logits, float** logitsPtrs, float* temperatures, float const* bias,
                                   int32_t const* endIds, FinishedState const* finished, int32_t const* batchSlots,
                                   int32_t batchSize, int32_t maxBatchSize, int32_t beamWidth, int32_t vocabSize,
                                   int32_t vocabSizePadded, bool skipSoftMax, bool batchSlotsLogits,
                                   cudaStream_t stream);

template void invokeAddBiasSoftMax(half* logits, half** logitsPtrs, half* temperatures, half const* bias,
                                   int32_t const* endIds, FinishedState const* finished, int32_t const* batchSlots,
                                   int32_t batchSize, int32_t maxBatchSize, int32_t beamWidth, int32_t vocabSize,
                                   int32_t vocabSizePadded, bool skipSoftMax, bool batchSlotsLogits,
                                   cudaStream_t stream);

}  // namespace kernels
}  // namespace tensorrt_llm
