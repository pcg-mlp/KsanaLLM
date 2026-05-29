/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <algorithm>
#include <functional>
#include <memory>
#include <queue>
#include <type_traits>
#include <vector>

#include "device/device_types.h"
#include "device/device_utils.h"
#include "utils/status.h"
#include "utils/utils.h"

namespace ksana_llm {

// Forward declaration — Tensor lives in utils/tensor.h which itself depends on memory_utils.h,
// so we keep the relationship one-directional by exposing only an opaque class name here.
class Tensor;

struct WorkspaceMeta {
  void* space_ptr = nullptr;
  size_t space_size = 0ul;
};

template <typename T>
__attribute__((unused)) static T DivRoundUp(T dividend, T divisor) {
  static_assert(std::is_integral<T>::value, "DivRoundUp requires integral types");
  return (dividend + divisor - 1) / divisor;
}

template <typename T>
__attribute__((unused)) static T DivRoundDown(T dividend, T divisor) {
  static_assert(std::is_integral<T>::value, "DivRoundDown requires integral types");
  return dividend / divisor;
}

template <typename T>
__attribute__((unused)) static T RoundUp(T value, T multiple) {
  static_assert(std::is_integral<T>::value, "RoundUp requires integral types");
  return ((value + multiple - 1) / multiple) * multiple;
}

// 保持向后兼容性
__attribute__((unused)) static int64_t DivRoundUp(int64_t dividend, int64_t divisor) {
  return DivRoundUp<int64_t>(dividend, divisor);
}

__attribute__((unused)) static int64_t DivRoundDown(int64_t dividend, int64_t divisor) {
  return DivRoundDown<int64_t>(dividend, divisor);
}

__attribute__((unused)) static int64_t RoundUp(int64_t value, int64_t multiple) {
  return RoundUp<int64_t>(value, multiple);
}

/**
 * The AlignedMemoryQueue class is designed to facilitate the allocation and alignment of memory blocks
 * in a queued manner, ensuring that each block is aligned to a specified boundary. This is particularly
 * useful for optimizing memory access patterns in systems where alignment matters, such as on GPUs or for SIMD
 * operations.
 *
 * Usage Example:
 * --------------
 * // Initialize the AlignedMemoryQueue with the desired memory alignment size and the custom allocator.
 * AlignedMemoryQueue aligned_memory_queue(kCudaMemAlignmentSize, allocator);
 *
 * // Add memory allocation requests to the queue. Each request specifies a pointer to store the allocated
 * // memory address and the count of the memory block needed.
 * int* ptr1 = nullptr;
 * double* ptr2 = nullptr;
 * double* ptr3 = nullptr;
 * queue.Add(ptr1, 10);
 * queue.Add(ptr2, 5);
 * queue.Add(ptr3, 7);
 *
 * // Once all requests are added, call AllocateAndAlign to process the queue, allocate, and align all requested memory
 * blocks. aligned_memory_queue.AllocateAndAlign();
 *
 * Notes:
 * ------
 * - The allocator function is a critical component that must be provided to handle the actual memory allocation.
 *   It can be customized to integrate with various memory management systems.
 * - The AlignedMemoryQueue ensures that each allocated block is aligned to the specified alignment boundary,
 *   which can help improve memory access efficiency in certain applications.
 */
class AlignedMemoryQueue {
 public:
  // Define an Allocator type, a function pointer for memory allocation
  using Allocator = std::function<void*(size_t)>;

 public:
  // Constructor, requires the number of bytes for alignment and the allocation function
  AlignedMemoryQueue(size_t alignment, Allocator allocator);

  // Destructor
  ~AlignedMemoryQueue() {}

  // Add a memory request to the queue
  template <typename T>
  void Add(T*& ptr, size_t count) {
    queue_.push_back({reinterpret_cast<void**>(&ptr), sizeof(T) * count});
  }

  // Allocate and align all requested memory
  void AllocateAndAlign();

 private:
  // Calculate the size after alignment
  size_t AlignSize(size_t size);

  // Check if a number is a power of two
  static bool IsPowerOfTwo(size_t x);

 private:
  // Queue to store pointers and their requested sizes
  std::vector<std::pair<void**, size_t>> queue_;
  // Number of bytes for memory alignment
  size_t alignment_;
  // Allocation function
  Allocator allocator_;
};

// Get free & total memory in bytes of current selected device.
Status GetDeviceMemoryInfo(MemoryDevice device, size_t* free, size_t* total);

// Get free & total host memory in bytes.
Status GetHostMemoryInfo(size_t* free, size_t* total);

// Get workspace of size.
// It maintain a global memory block, and reallocated if size is not enough.
void GetWorkSpaceImpl(size_t size, void** ws_addr);

// Define a function to create kernel workspace.
typedef void (*WorkSpaceFunc)(size_t, void**);

// Get the workspace function.
WorkSpaceFunc GetWorkSpaceFunc();

// Compute the maximum Q-token chunk size that fits within the 4GB logits buffer constraint.
// Respects env var KSANA_INDEXER_MAX_Q_CHUNK_SIZE override if set.
//   max_seq_len: maximum KV sequence length (after compression if applicable).
//   max_step_token_num: upper bound on total tokens per step.
inline size_t ComputeIndexerMaxQChunkSize(size_t max_seq_len, size_t max_step_token_num) {
  constexpr size_t kSeqLenAlignment = 4;
  if (const int env_val = GetEnvAsPositiveInt("KLLM_INDEXER_MAX_CHUNKED_Q_SIZE", 0); env_val > 0) {
    return RoundUp(static_cast<size_t>(env_val), kSeqLenAlignment);
  }
  constexpr size_t kMaxLogitsBytes = 4ul << 30;  // 4 GB
  constexpr size_t kBlockKV = 256;
  const size_t aligned_seq_len_kv = RoundUp(max_seq_len, kBlockKV);
  const size_t computed_q_chunk = kMaxLogitsBytes / (aligned_seq_len_kv * sizeof(float));
  return RoundUp(std::min(max_step_token_num, computed_q_chunk), kSeqLenAlignment);
}

// Batched device-to-device copy with per-call layer offsets. For each i in [0, N), copies
// `block_size` bytes from
//     src_i = reinterpret_cast<uint8_t*>(src_offset) + src_addr_list[i]
//     dst_i = reinterpret_cast<uint8_t*>(dst_offset) + dst_addr_list[i]
// where N == src_addr_list.shape[0] == dst_addr_list.shape[0]. All copies share a single
// kernel launch on `stream`, eliminating the per-block launch overhead of looped MemcpyAsync.
//
// src_addr_list / dst_addr_list are 1-D device tensors of element type uint64_t holding
// per-pair (layer-invariant) byte addresses or offsets. They are intended to be built once
// per forward step; subsequent per-layer invocations reuse the same tensors and only vary
// (src_offset, dst_offset) to fold in the layer-specific base pointer. Empty lists
// (N == 0) are a no-op.
void BatchMemoryD2D(const Tensor& src_addr_list, const Tensor& dst_addr_list, size_t src_offset, size_t dst_offset,
                    size_t block_size, Stream& stream);

}  // namespace ksana_llm
