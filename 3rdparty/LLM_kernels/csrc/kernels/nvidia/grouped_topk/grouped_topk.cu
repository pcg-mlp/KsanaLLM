/*
 * Adapted from
 * https://github.com/flashinfer-ai/flashinfer/blob/main/csrc/fused_moe/noAuxTcKernels.cu
 *
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 * Copyright 2026 Tencent Inc.  All rights reserved.
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

#include "csrc/kernels/nvidia/grouped_topk/grouped_topk.h"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/torch.h>
#include <cmath>
#include <cub/cub.cuh>

#include "csrc/utils/nvidia/cuda_utils.h"

namespace llm_kernels {
namespace nvidia {

namespace cg = cooperative_groups;

static constexpr int WARP_SIZE = 32;
static constexpr int NumKimiK2Experts = 384;
static constexpr int NumDeepseekExperts = 256;
static constexpr int MaxNumExpertsUnit = 128;
static constexpr int NumTopGroupScores = 2;
static constexpr int MaxNumTopExperts = 8;
static constexpr int MaxNumTopGroups = 4;

template <typename T_>
struct TopKRedType {
  using T = T_;
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, half> || std::is_same_v<T, __nv_bfloat16> ||
                    std::is_same_v<T, int>,
                "Top K reduction only implemented for int, float, float16 and bfloat16");

  using TypeCmp = std::conditional_t<sizeof(T) == 4, uint64_t, uint32_t>;
  using IdxT = std::conditional_t<sizeof(T) == 4, int32_t, int16_t>;

  static constexpr int kMoveBits = (sizeof(T) == 4) ? 32 : 16;
  static constexpr int kMaxIdx = 65535;
  TypeCmp compValIdx;

  static __host__ __device__ inline TypeCmp makeCmpVal(T val, int32_t idx = 0) {
    auto valueBits = cub::Traits<T>::TwiddleIn(reinterpret_cast<typename cub::Traits<T>::UnsignedBits&>(val));
    TypeCmp compactTmp = valueBits;
    compactTmp = (compactTmp << kMoveBits) | (0xFFFF & (kMaxIdx - idx));
    // Use 65535 minus idx to give higher priority to elements with smaller indices.
    return compactTmp;
  }

  static __host__ __device__ void unpack(T& value, int32_t& index, TypeCmp cmp) {
    // Since “65535-idx” is always smaller than 65536 and positive, we can directly use it as the
    // lower 16 bits
    index = kMaxIdx - static_cast<int32_t>((cmp & 0xFFFF));

    auto compactTmp = cmp >> kMoveBits;
    auto valueBits = cub::Traits<T>::TwiddleOut(reinterpret_cast<typename cub::Traits<T>::UnsignedBits&>(compactTmp));
    value = reinterpret_cast<T&>(valueBits);
  }

  __host__ __device__ TopKRedType() = default;

  __host__ __device__ TopKRedType(T val, int32_t idx) : compValIdx(makeCmpVal(val, idx)) {}

  __host__ __device__ operator TypeCmp() const noexcept { return compValIdx; }

  __device__ inline TypeCmp reduce(cg::thread_block_tile<WARP_SIZE> const& warp) {
    return cg::reduce(warp, compValIdx, cg::greater<TypeCmp>{});
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int K_, bool Enable_>
struct TopKIdx {
  // by default, empty
};

template <int K_>
struct TopKIdx<K_, true> {
  static constexpr int K = K_;
  int32_t val[K];
};

////////////////////////////////////////////////////////////////////////////////////////////////////

#define TOPK_SWAP(I, J)                                         \
  {                                                             \
    auto pairMin = min(topK[I].compValIdx, topK[J].compValIdx); \
    auto pairMax = max(topK[I].compValIdx, topK[J].compValIdx); \
    topK[I].compValIdx = pairMax;                               \
    topK[J].compValIdx = pairMin;                               \
  }

template <int N, typename RedType>
struct Sort;

template <typename RedType>
struct Sort<1, RedType> {
  static __device__ void run(RedType* topK) {}
};

template <typename RedType>
struct Sort<2, RedType> {
  static __device__ void run(RedType* topK) { TOPK_SWAP(0, 1); }
};

template <typename RedType>
struct Sort<3, RedType> {
  static __device__ void run(RedType* topK) {
    TOPK_SWAP(0, 1);
    TOPK_SWAP(1, 2);
    TOPK_SWAP(0, 1);
  }
};

template <typename RedType>
struct Sort<4, RedType> {
  static __device__ void run(RedType* topK) {
    TOPK_SWAP(0, 2);
    TOPK_SWAP(1, 3);
    TOPK_SWAP(0, 1);
    TOPK_SWAP(2, 3);
    TOPK_SWAP(1, 2);
  }
};

template <int K, typename Type>
__forceinline__ __device__ void reduceTopK(cg::thread_block_tile<WARP_SIZE> const& warp, Type (&out)[K],
                                           int32_t (&outIdx)[K], Type value, int32_t idx, Type const minValue,
                                           int actualK = K) {
  static_assert(K > 0, "Top K must have K > 0");
  static_assert(K < WARP_SIZE, "Top K must have K < WARP_SIZE");
  using RedType = TopKRedType<Type>;
  RedType topK{value, idx};
  typename RedType::TypeCmp packedMax{};
#pragma unroll
  for (int kk = 0; kk < actualK; ++kk)  //@todo: check if actualK is correct
  {
    topK = kk > 0 && packedMax == topK.compValIdx ? RedType{minValue, idx} : topK;
    // get the next largest value
    packedMax = topK.reduce(warp);
    RedType::unpack(out[kk], outIdx[kk], packedMax);
  }
};

template <int K, typename Type, int N, bool IsSorted = false>
__device__ void reduceTopKFunc(cg::thread_block_tile<WARP_SIZE> const& warp, Type (&out)[K], int32_t (&outIdx)[K],
                               Type (&value)[N], int32_t (&idx)[N], Type minValue, int actualK = K) {
  static_assert(K > 0, "Top K must have K > 0");
  static_assert(K < WARP_SIZE, "Top K must have K < WARP_SIZE");
  static_assert(N > 0, "Top K must have N > 0");
  static_assert(N < 5, "Only support candidates number less than or equal to 128");
  using RedType = TopKRedType<Type>;
  RedType topK[N];
#pragma unroll
  for (int nn = 0; nn < N; ++nn) {
    topK[nn] = RedType{value[nn], idx[nn]};
  }

  if constexpr (!IsSorted) {
    Sort<N, RedType>::run(topK);
  }
  typename RedType::TypeCmp packedMax{};
#pragma unroll
  for (int kk = 0; kk < actualK; ++kk) {
    bool update = kk > 0 && packedMax == topK[0].compValIdx;
#pragma unroll
    for (int nn = 0; nn < N; ++nn) {
      topK[nn] = update && nn == N - 1 ? RedType{minValue, idx[nn]} : update ? topK[nn + 1] : topK[nn];
    }
    // get the next largest value
    packedMax = topK[0].reduce(warp);
    RedType::unpack(out[kk], outIdx[kk], packedMax);
  }
};

template <int K, typename Type, int N>
__forceinline__ __device__ void reduceTopK(cg::thread_block_tile<WARP_SIZE> const& warp, Type (&out)[K],
                                           int32_t (&outIdx)[K], Type (&value)[N], int32_t (&idx)[N],
                                           Type const minValue, int actualK = K) {
  static_assert(K > 0, "Top K must have K > 0");
  static_assert(K < WARP_SIZE, "Top K must have K < WARP_SIZE");
  static_assert(N > 0, "Top K must have N > 0");
  static_assert(N <= 16, "Only support candidates number less than or equal to 16*32=512");
  static_assert(N <= 4 || N % 4 == 0,
                "Only support candidates number is a multiple of 4*32=128 or less than or equal to 4");
  using RedType = TopKRedType<Type>;

  if constexpr (N <= 4) {
    reduceTopKFunc<K, Type, N>(warp, out, outIdx, value, idx, minValue, actualK);
  } else {
    constexpr int numLoops = N / 4;
    constexpr int numResults = (numLoops * K - 1) / WARP_SIZE + 1;

    Type topKBufferValue[numResults];
    int32_t topKBufferIdx[numResults];
    int32_t laneIdx = threadIdx.x % WARP_SIZE;

    for (int ii = 0; ii < numResults; ++ii) {
      topKBufferValue[ii] = minValue;
      topKBufferIdx[ii] = ii * WARP_SIZE - 1;  //@todo: check if this is correct
    }
    for (int loop = 0; loop < numLoops; ++loop) {
      int start = loop * 4;
      Type topKValue[K];
      int32_t topKIdx[K];
      Type inValue[4];
      int32_t inIdx[4];
      for (int i = 0; i < 4; ++i) {
        inValue[i] = value[start + i];
        inIdx[i] = idx[start + i];
      }
      reduceTopKFunc<K, Type, 4>(warp, topKValue, topKIdx, inValue, inIdx, minValue, actualK);
      int inOffset = laneIdx % K;
      if (laneIdx >= loop * K && laneIdx < (loop + 1) * K) {
        topKBufferValue[0] = topKValue[inOffset];
        topKBufferIdx[0] = topKIdx[inOffset];
      }
      if (loop == numLoops - 1 && (laneIdx < (numLoops * K - WARP_SIZE))) {
        topKBufferValue[1] = topKValue[inOffset];
        topKBufferIdx[1] = topKIdx[inOffset];
      }
    }

    reduceTopKFunc<K, Type, numResults>(warp, out, outIdx, topKBufferValue, topKBufferIdx, minValue, actualK);
  }
};

#undef TOPK_SWAP

static __device__ inline float sigmoid_accurate(float x) { return 0.5f * tanhf(0.5f * x) + 0.5f; }

template <typename InputT, typename BiasT, typename OutputT, typename IdxT, int MaxNumExperts, bool UseGroups>
__global__ void deepseek_v3_topk_kernel(InputT* scores, OutputT* topkValues, IdxT* topkIndices, BiasT* routingBias,
                                        int64_t const numTokens, int64_t const numGroup, int64_t const topkGroup,
                                        int64_t const topk, int64_t const numExperts, int64_t const numExpertsPerGroup,
                                        double const routedScalingFactor, float const scoreThreshold) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

  // declare shared memory structure
  // number of experts is bounded by number of threads
  __shared__ float __attribute((aligned(128))) smemScoreSigmoid[MaxNumExperts];
  __shared__ float __attribute((aligned(128))) smemScoreBias[MaxNumExperts];
  // number of expert groups is bounded by number of warps
  int constexpr NumWarps = MaxNumExperts / WARP_SIZE;
  __shared__ float __attribute((aligned(128))) smemGroupScores[NumWarps];

  // needed for warp reduce
  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<WARP_SIZE>(block);

  // for the final reduction of weight norm, only some lanes need to participate
  int32_t laneIdx = threadIdx.x % WARP_SIZE;
  int32_t warpIdx = __shfl_sync(0xffffffff, threadIdx.x / WARP_SIZE, 0);

  if constexpr (UseGroups) {
    if (warpIdx >= numGroup) {
      return;
    }
  }

  // note that for invalid scores, we simply use a negative value:
  // they work well even with the compacted format used in topK, and
  // sigmoid / bias activated scores cannot be negative
  static constexpr float invalidScoreFloat = float{-INFINITY};
  const OutputT invalidScore = OutputT{invalidScoreFloat};

  // load bias already; each warp represents one expert group
  auto threadExpert = threadIdx.x;
  bool expertSelected = threadExpert < numExperts;
  if constexpr (UseGroups) {
    threadExpert = warpIdx * numExpertsPerGroup + laneIdx;
    expertSelected = laneIdx < numExpertsPerGroup;
  }

  auto scoreIdx = int64_t{blockIdx.x} * int64_t{numExperts} + threadExpert;
  auto biasVal = expertSelected ? static_cast<float>(routingBias[threadExpert]) : invalidScoreFloat;
  topkValues += blockIdx.x * topk;
  topkIndices += blockIdx.x * topk;

  // get our assigned thread score; each warp represents one expert group
  float score = expertSelected ? static_cast<float>(scores[scoreIdx]) : invalidScoreFloat;
  auto scoreSigmoid = sigmoid_accurate(score);
  // write the sigmoid score to shared for later use
  if (expertSelected) {
    smemScoreSigmoid[threadExpert] = scoreSigmoid;
  }

  // get the score with bias
  // note that with invalid values, because sigmoid is < 1 and bias is -1,
  // we must get a negative value, which is smaller than any valid value
  auto scoreBias = float{scoreSigmoid + float{biasVal}};

  if (expertSelected) {
    smemScoreBias[threadExpert] = scoreBias;
  }

  // registers for top group score reduction
  float topExpGroupScores[NumTopGroupScores];
  [[maybe_unused]] int32_t topExpGroupIdx[NumTopGroupScores];
  float topGroups[MaxNumTopGroups];  // bound of numGroup
  int32_t topGroupIdx[MaxNumTopGroups];
  float expertScoreGroup[MaxNumTopGroups];
  int32_t expertIdxGroup[MaxNumTopGroups];
  float topScores[MaxNumTopExperts];  // bound of topk
  int32_t topExperts[MaxNumTopExperts];

  if constexpr (UseGroups) {
    reduceTopK(warp, topExpGroupScores, topExpGroupIdx, scoreBias, threadExpert,
               /* minValue */ invalidScoreFloat);

    // get the final group score and write it to shared
    if (laneIdx == 0) {
      auto groupScore = topExpGroupScores[0] + topExpGroupScores[1];
      smemGroupScores[warpIdx] = groupScore;
    }
  }

  // make group scores available to all warps
  __syncthreads();

  if constexpr (UseGroups) {
    if (warpIdx == 0) {
      // a single warp performs the selection of top groups, and goes on to select the final experts
      float groupScore = laneIdx < numGroup ? smemGroupScores[laneIdx] : invalidScoreFloat;

      reduceTopK(warp, topGroups, topGroupIdx, groupScore, laneIdx,
                 /* minValue */ invalidScoreFloat);

      // final expert selection: get relevant indexes and scores from shared

#pragma unroll
      for (int ii = 0; ii < MaxNumTopGroups; ++ii) {  // bound of numGroup
        // auto groupIdx = topGroupIdx[ii];
        auto groupIdx = (ii < topkGroup) ? topGroupIdx[ii] : 0;
        expertIdxGroup[ii] = groupIdx * numExpertsPerGroup + laneIdx;

        expertScoreGroup[ii] =
            (ii < topkGroup) && expertSelected ? smemScoreBias[expertIdxGroup[ii]] : invalidScoreFloat;
      }

      reduceTopK(warp, topScores, topExperts, expertScoreGroup, expertIdxGroup,
                 /* minValue */ invalidScoreFloat, topk);
    }
  } else if constexpr (MaxNumExperts > MaxNumExpertsUnit) {
    // without groups, and the expert number is larger than MaxNumExpertsUnit,
    // we need to use multiple warps to calculate the intermediate topk results

    int constexpr NumExpertWarps = (MaxNumExperts - 1) / MaxNumExpertsUnit + 1;
    int constexpr NumInterTopK = NumExpertWarps * MaxNumTopExperts;
    __shared__ float __attribute((aligned(128))) smemInterTopScores[NumInterTopK];
    __shared__ int32_t __attribute((aligned(128))) smemInterTopExperts[NumInterTopK];
    if (warpIdx < NumExpertWarps) {
      int offset = warpIdx * WARP_SIZE * MaxNumTopGroups;
#pragma unroll
      for (int ii = 0; ii < MaxNumTopGroups; ++ii) {
        auto expertIdx = ii * WARP_SIZE + laneIdx;
        expertIdxGroup[ii] = offset + expertIdx;
        expertScoreGroup[ii] = offset + expertIdx < numExperts ? smemScoreBias[offset + expertIdx] : invalidScoreFloat;
      }
      reduceTopK(warp, topScores, topExperts, expertScoreGroup, expertIdxGroup,
                 /* minValue */ invalidScoreFloat, topk);

      if (laneIdx < topk) {
        smemInterTopScores[warpIdx * MaxNumTopExperts + laneIdx] = topScores[laneIdx];
        smemInterTopExperts[warpIdx * MaxNumTopExperts + laneIdx] = topExperts[laneIdx];
      }
    }
    __syncthreads();
    if (warpIdx == 0) {
      int constexpr NumInterTopKPerThread = (NumInterTopK * NumExpertWarps - 1) / WARP_SIZE + 1;
      float intermidiateScore[NumInterTopKPerThread];
      int32_t intermidiateExpert[NumInterTopKPerThread];
      for (int i = laneIdx; i < NumInterTopKPerThread * WARP_SIZE; i += WARP_SIZE) {
        int ii = i / WARP_SIZE;
        if (i < NumInterTopK) {
          intermidiateScore[ii] = smemInterTopScores[i];
          intermidiateExpert[ii] = smemInterTopExperts[i];
        } else {
          intermidiateScore[ii] = invalidScoreFloat;
          intermidiateExpert[ii] = MaxNumExperts - 1;
        }
      }
      reduceTopK(warp, topScores, topExperts, intermidiateScore, intermidiateExpert,
                 /* minValue */ invalidScoreFloat, topk);
    }
  } else {
    // without groups, and the expert number is smaller than MaxNumExpertsUnit
    // each thread just takes `MaxNumTopGroups` experts
    if (warpIdx == 0) {
#pragma unroll
      for (int ii = 0; ii < MaxNumTopGroups; ++ii) {
        auto expertIdx = ii * WARP_SIZE + laneIdx;
        expertIdxGroup[ii] = expertIdx;
        expertScoreGroup[ii] = expertIdx < numExperts ? smemScoreBias[expertIdx] : invalidScoreFloat;
      }
      reduceTopK(warp, topScores, topExperts, expertScoreGroup, expertIdxGroup,
                 /* minValue */ invalidScoreFloat, topk);
    }
  }

  if (warpIdx == 0) {
    // determine our lane's expert index and write to output
    int32_t expertIdx = laneIdx < topk ? topExperts[laneIdx] : MaxNumExperts - 1;
    // norm the value
    float scoreNorm = laneIdx < topk ? smemScoreSigmoid[expertIdx] : 0.F;
    auto redNorm = cg::reduce(warp, scoreNorm, cg::plus<float>{});
    auto finalScore = scoreNorm * routedScalingFactor / (redNorm + 1e-20);

    // Score threshold sparsification: zero out experts below threshold and re-normalize
    if (scoreThreshold > 0.F) {
      // First normalize to get per-expert ratio
      float normalizedScore = finalScore / (routedScalingFactor + 1e-20f);
      // Mark experts below threshold as cut
      bool isCut = (laneIdx < topk) && (normalizedScore < scoreThreshold);
      float cutScore = isCut ? finalScore : 0.F;
      float totalCutScore = cg::reduce(warp, cutScore, cg::plus<float>{});
      float remainingSum = routedScalingFactor - totalCutScore;

      if (isCut) {
        finalScore = 0.F;
        // NOTE: Experts with normalized scores below the threshold are marked with index -1,
        // which the MoE kernel must handle accordingly
        expertIdx = -1;
      } else if (laneIdx < topk && remainingSum > 1e-20f) {
        // Re-normalize remaining experts to preserve total scaling factor
        finalScore = finalScore * routedScalingFactor / remainingSum;
      }
    }

    // store the topk scores and experts to output
    if (laneIdx < topk) {
      topkValues[laneIdx] = static_cast<OutputT>(finalScore);
      topkIndices[laneIdx] = expertIdx;
    }
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

// Wrapper function for external calls
template <typename T>
void InvokeDeepSeekV3GroupedTopk(void* gating_output, void* e_bias, float routed_scaling_factor, void* topk_weights,
                                 void* topk_ids, int tokens_num, int num_experts, int topk, int num_expert_group,
                                 int topk_group, float score_threshold, cudaStream_t stream) {
  // Check if we can use the optimized deepseek_v3_topk_kernel
  bool const is_single_group = (num_expert_group == 1) && (num_experts <= NumKimiK2Experts);

  KLLM_KERNEL_CHECK_WITH_INFO(num_experts % num_expert_group == 0,
                              "num_experts (%d) must be divisible by num_expert_group (%d)", num_experts,
                              num_expert_group);
  KLLM_KERNEL_CHECK_WITH_INFO(topk <= MaxNumTopExperts, "topk (%d) exceeds MaxNumTopExperts (%d)", topk,
                              MaxNumTopExperts);
  KLLM_KERNEL_CHECK_WITH_INFO(topk_group <= MaxNumTopGroups, "topk_group (%d) exceeds MaxNumTopGroups (%d)", topk_group,
                              MaxNumTopGroups);
  int64_t const experts_per_group = num_experts / num_expert_group;
  bool const is_multi_group = (num_expert_group != 1) && (num_experts <= NumDeepseekExperts) &&
                              (experts_per_group <= WARP_SIZE) && (experts_per_group * topk_group <= MaxNumExpertsUnit);

  KLLM_KERNEL_CHECK_WITH_INFO(is_single_group || is_multi_group, "Unsupported input configuration for grouped topk");
  cudaLaunchConfig_t config;
  auto* kernel_instance = &deepseek_v3_topk_kernel<T, float, float, int, NumDeepseekExperts, true>;
  int num_threads = NumDeepseekExperts;
  if (is_single_group) {
    if (num_experts > MaxNumExpertsUnit) {
      kernel_instance = &deepseek_v3_topk_kernel<T, float, float, int, NumKimiK2Experts, false>;
      num_threads = NumKimiK2Experts;
    } else {
      kernel_instance = &deepseek_v3_topk_kernel<T, float, float, int, MaxNumExpertsUnit, false>;
      num_threads = MaxNumExpertsUnit;
    }
  }

  config.gridDim = tokens_num;
  config.blockDim = num_threads;
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = llm_kernels::utils::GetEnablePDL();
  config.numAttrs = 1;
  config.attrs = attrs;

  cudaLaunchKernelEx(&config, kernel_instance, reinterpret_cast<T*>(gating_output),
                     reinterpret_cast<float*>(topk_weights), reinterpret_cast<int*>(topk_ids),
                     reinterpret_cast<float*>(e_bias), tokens_num, num_expert_group, topk_group, topk, num_experts,
                     experts_per_group, routed_scaling_factor, score_threshold);
}
#define INVOKE_DEEPSEEK_V3_GROUPED_TOPK(T)                                                                      \
  template void InvokeDeepSeekV3GroupedTopk<T>(void* gating_output, void* e_bias, float routed_scaling_factor,  \
                                               void* topk_weights_ptr, void* topk_ids_ptr, int tokens_num,      \
                                               int num_experts, int topk, int num_expert_group, int topk_group, \
                                               float score_threshold, cudaStream_t stream)
INVOKE_DEEPSEEK_V3_GROUPED_TOPK(float);
INVOKE_DEEPSEEK_V3_GROUPED_TOPK(half);
INVOKE_DEEPSEEK_V3_GROUPED_TOPK(__nv_bfloat16);
#undef INVOKE_DEEPSEEK_V3_GROUPED_TOPK

template <int MAX_K>
__device__ void GetTopKIndices(const float* data, int* indices, int n, int k, int index_offset) {
  // Local arrays used as heap storage
  float heap_values[MAX_K];
  int heap_indices[MAX_K];

  // Initialize the heap
  for (int i = 0; i < k; i++) {
    heap_values[i] = data[i];
    heap_indices[i] = i + index_offset;
  }

  // Build min heap
  for (int i = k / 2 - 1; i >= 0; i--) {
    int parent = i;
    while (true) {
      int leftChild = 2 * parent + 1;
      int rightChild = 2 * parent + 2;
      int smallest = parent;

      if (leftChild < k && heap_values[leftChild] < heap_values[smallest]) {
        smallest = leftChild;
      }
      if (rightChild < k && heap_values[rightChild] < heap_values[smallest]) {
        smallest = rightChild;
      }

      if (smallest == parent) break;

      // Swap elements
      float temp_val = heap_values[parent];
      heap_values[parent] = heap_values[smallest];
      heap_values[smallest] = temp_val;

      int temp_idx = heap_indices[parent];
      heap_indices[parent] = heap_indices[smallest];
      heap_indices[smallest] = temp_idx;

      parent = smallest;
    }
  }

  // Process remaining elements
  for (int i = k; i < n; i++) {
    if (data[i] > heap_values[0]) {
      heap_values[0] = data[i];
      heap_indices[0] = i + index_offset;

      // Sift down
      int parent = 0;
      while (true) {
        int leftChild = 2 * parent + 1;
        int rightChild = 2 * parent + 2;
        int smallest = parent;

        if (leftChild < k && heap_values[leftChild] < heap_values[smallest]) {
          smallest = leftChild;
        }
        if (rightChild < k && heap_values[rightChild] < heap_values[smallest]) {
          smallest = rightChild;
        }

        if (smallest == parent) break;

        float temp_val = heap_values[parent];
        heap_values[parent] = heap_values[smallest];
        heap_values[smallest] = temp_val;

        int temp_idx = heap_indices[parent];
        heap_indices[parent] = heap_indices[smallest];
        heap_indices[smallest] = temp_idx;

        parent = smallest;
      }
    }
  }

  // Extract results
  for (int i = k - 1; i > 0; i--) {
    indices[k - 1 - i] = heap_indices[0];

    heap_values[0] = heap_values[i];
    heap_indices[0] = heap_indices[i];

    int parent = 0;
    int heapSize = i;

    while (true) {
      int leftChild = 2 * parent + 1;
      int rightChild = 2 * parent + 2;
      int smallest = parent;

      if (leftChild < heapSize && heap_values[leftChild] < heap_values[smallest]) {
        smallest = leftChild;
      }
      if (rightChild < heapSize && heap_values[rightChild] < heap_values[smallest]) {
        smallest = rightChild;
      }

      if (smallest == parent) break;

      float temp_val = heap_values[parent];
      heap_values[parent] = heap_values[smallest];
      heap_values[smallest] = temp_val;

      int temp_idx = heap_indices[parent];
      heap_indices[parent] = heap_indices[smallest];
      heap_indices[smallest] = temp_idx;

      parent = smallest;
    }
  }

  indices[k - 1] = heap_indices[0];
}

// Select the top k from data of size n specified by data_indices.
template <int MAX_K>
__device__ void GetTopKIndices(const float* data, int* indices, int n, int k, int32_t* data_indices) {
  // Local arrays used as heap storage
  float heap_values[MAX_K];
  int heap_indices[MAX_K];

  // Initialize the heap
  for (int i = 0; i < k; i++) {
    heap_values[i] = data[data_indices[i]];
    heap_indices[i] = data_indices[i];
  }

  // Build min heap
  for (int i = k / 2 - 1; i >= 0; i--) {
    int parent = i;
    while (true) {
      int leftChild = 2 * parent + 1;
      int rightChild = 2 * parent + 2;
      int smallest = parent;

      if (leftChild < k && heap_values[leftChild] < heap_values[smallest]) {
        smallest = leftChild;
      }
      if (rightChild < k && heap_values[rightChild] < heap_values[smallest]) {
        smallest = rightChild;
      }

      if (smallest == parent) break;

      // Swap elements
      float temp_val = heap_values[parent];
      heap_values[parent] = heap_values[smallest];
      heap_values[smallest] = temp_val;

      int temp_idx = heap_indices[parent];
      heap_indices[parent] = heap_indices[smallest];
      heap_indices[smallest] = temp_idx;

      parent = smallest;
    }
  }

  // Process remaining elements
  for (int i = k; i < n; i++) {
    if (data[data_indices[i]] > heap_values[0]) {
      heap_values[0] = data[data_indices[i]];
      heap_indices[0] = data_indices[i];

      // Sift down
      int parent = 0;
      while (true) {
        int leftChild = 2 * parent + 1;
        int rightChild = 2 * parent + 2;
        int smallest = parent;

        if (leftChild < k && heap_values[leftChild] < heap_values[smallest]) {
          smallest = leftChild;
        }
        if (rightChild < k && heap_values[rightChild] < heap_values[smallest]) {
          smallest = rightChild;
        }

        if (smallest == parent) break;

        float temp_val = heap_values[parent];
        heap_values[parent] = heap_values[smallest];
        heap_values[smallest] = temp_val;

        int temp_idx = heap_indices[parent];
        heap_indices[parent] = heap_indices[smallest];
        heap_indices[smallest] = temp_idx;

        parent = smallest;
      }
    }
  }

  // Extract results
  for (int i = k - 1; i > 0; i--) {
    indices[k - 1 - i] = heap_indices[0];

    heap_values[0] = heap_values[i];
    heap_indices[0] = heap_indices[i];

    int parent = 0;
    int heapSize = i;

    while (true) {
      int leftChild = 2 * parent + 1;
      int rightChild = 2 * parent + 2;
      int smallest = parent;

      if (leftChild < heapSize && heap_values[leftChild] < heap_values[smallest]) {
        smallest = leftChild;
      }
      if (rightChild < heapSize && heap_values[rightChild] < heap_values[smallest]) {
        smallest = rightChild;
      }

      if (smallest == parent) break;

      float temp_val = heap_values[parent];
      heap_values[parent] = heap_values[smallest];
      heap_values[smallest] = temp_val;

      int temp_idx = heap_indices[parent];
      heap_indices[parent] = heap_indices[smallest];
      heap_indices[smallest] = temp_idx;

      parent = smallest;
    }
  }

  indices[k - 1] = heap_indices[0];
}

// Basic softmax + topk kernel for simple cases without grouping
template <typename T>
__global__ void BasicSoftmaxTopkKernel(T* gating_output, float* topk_weights, int32_t* topk_ids, int num_rows,
                                       int num_experts, int topk, float routed_scaling_factor) {
  int row_idx = blockIdx.x;
  if (row_idx >= num_rows) return;

  extern __shared__ float shared_mem[];
  float* scores = shared_mem;  // 每个block处理一行，shared memory存储该行的softmax结果

  T* output_row = gating_output + row_idx * num_experts;

  // Step 1: 在专家维度上计算softmax
  // 每个线程负责一个专家，并行加载数据到shared memory
  if (threadIdx.x < num_experts) {
    scores[threadIdx.x] = static_cast<float>(output_row[threadIdx.x]);
  }
  __syncthreads();

  // 找到最大值以保证数值稳定性 - 让thread 0来做串行计算确保正确性
  if (threadIdx.x == 0) {
    float max_val = scores[0];
    for (int i = 1; i < num_experts; i++) {
      max_val = fmaxf(max_val, scores[i]);
    }

    // 计算 exp(x - max) 并求和
    float sum = 0.0f;
    for (int i = 0; i < num_experts; i++) {
      scores[i] = expf(scores[i] - max_val);
      sum += scores[i];
    }

    // 归一化得到softmax概率
    for (int i = 0; i < num_experts; i++) {
      scores[i] = scores[i] / sum;
    }
  }
  __syncthreads();

  // 将softmax结果写回到gating_output，保留注释方便debug
  // if (threadIdx.x < num_experts) {
  //   output_row[threadIdx.x] = static_cast<T>(scores[threadIdx.x]);
  // }

  // Step 2: 使用现有的GetTopKIndices函数进行top-k选择
  if (threadIdx.x == 0) {
    int* temp_indices = reinterpret_cast<int*>(scores + num_experts);  // 复用shared memory
    GetTopKIndices<32>(scores, temp_indices, num_experts, topk, 0);

    // 将结果写入输出数组，并应用scaling factor
    for (int i = 0; i < topk; i++) {
      topk_ids[row_idx * topk + i] = temp_indices[i];
      topk_weights[row_idx * topk + i] = scores[temp_indices[i]] * routed_scaling_factor;
    }
  }
}

// Wrapper function for basic softmax + topk
template <typename T>
void InvokeBasicSoftmaxTopk(void* gating_output, void* topk_weights_ptr, void* topk_ids_ptr, int num_rows,
                            int num_experts, int topk, float routed_scaling_factor, cudaStream_t stream) {
  // 验证参数
  if (num_experts >= 1024 || topk > num_experts || topk > 32) {
    // 参数不合法，直接返回
    return;
  }

  dim3 grid_size(num_rows);
  // 由于 num_experts < 1024，可以直接使用 num_experts 作为 block_size
  // 但为了更好的性能，使用 warp 对齐的大小
  int block_size = ((num_experts + 31) / 32) * 32;  // 向上对齐到 32 的倍数（warp size）

  // Shared memory: num_experts * sizeof(float) for scores + 32 * sizeof(int) for temp indices
  size_t shared_mem_size = num_experts * sizeof(float) + 32 * sizeof(int);

  BasicSoftmaxTopkKernel<T><<<grid_size, block_size, shared_mem_size, stream>>>(
      static_cast<T*>(gating_output), static_cast<float*>(topk_weights_ptr), static_cast<int32_t*>(topk_ids_ptr),
      num_rows, num_experts, topk, routed_scaling_factor);
}

#define INVOKE_BASIC_SOFTMAX_TOPK(T)                                                                            \
  template void InvokeBasicSoftmaxTopk<T>(void* gating_output, void* topk_weights_ptr, void* topk_ids_ptr,      \
                                          int num_rows, int num_experts, int topk, float routed_scaling_factor, \
                                          cudaStream_t stream)

INVOKE_BASIC_SOFTMAX_TOPK(float);
INVOKE_BASIC_SOFTMAX_TOPK(half);
INVOKE_BASIC_SOFTMAX_TOPK(__nv_bfloat16);
#undef INVOKE_BASIC_SOFTMAX_TOPK

// fill GPU memory with random integers using PyTorch
void FillRandomInts(int* data_ptr, int size, int start_int, int end_int, int rank, cudaStream_t stream) {
  // Set manual seed for reproducibility
  // NOTE(rockcao): Using torch to facilitate the same random number generation method across different frameworks
  torch::manual_seed(42);

  auto options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, rank).requires_grad(false);

  auto tensor = torch::from_blob(data_ptr, {size}, options);
  tensor.random_(start_int, end_int);
}

}  // namespace nvidia
}  // namespace llm_kernels
