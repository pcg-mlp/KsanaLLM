/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <algorithm>
#include <memory>
#include <queue>
#include <vector>

#include "ksana_llm/block_manager/block_manager_interface.h"

namespace ksana_llm {

static int64_t DivRoundUp(int64_t dividend, int64_t divisor) { return (dividend + divisor - 1) / divisor; }

static int64_t DivRoundDown(int64_t dividend, int64_t divisor) { return dividend / divisor; }

/**
 * The AlignedMemoryQueue class is designed to facilitate the allocation and alignment of memory blocks
 * in a queued manner, ensuring that each block is aligned to a specified boundary. This is particularly
 * useful for optimizing memory access patterns in systems where alignment matters, such as on GPUs or for SIMD operations.
 *
 * Usage Example:
 * --------------
 * // Define a custom allocator function that integrates with your memory management system.
 * // In this example, the allocator configures a device buffer based on a specific device ID and allocates
 * // contiguous memory blocks from a block manager.
 * auto allocator = [this](size_t size) -> void* {
 *   GetBlockManager()->SetDeviceId(rank_);
 *   GetBlockManager()->AllocateContiguous(size, device_buffer_block_id_);
 *   GetBlockManager()->GetContiguousPtr(device_buffer_block_id_, device_buffer_);
 *   return device_buffer_;
 * };
 *
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
 * // Once all requests are added, call AllocateAndAlign to process the queue, allocate, and align all requested memory blocks.
 * aligned_memory_queue.AllocateAndAlign();
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

// Set a global block manager
void SetBlockManager(BlockManagerInterface* block_manager);

// Get the global block manager
BlockManagerInterface* GetBlockManager();

// Get block pointer.
template <typename T>
std::vector<T*> GetBlockPtrs(const std::vector<int>& blocks) {
  std::vector<void*> addrs;
  GetBlockManager()->GetBlockPtrs(blocks, addrs);
  std::vector<T*> results(addrs.size());
  std::transform(addrs.begin(), addrs.end(), results.begin(), [](void* p) { return reinterpret_cast<T*>(p); });
  return results;
}

// Get block pointer.
template <typename T>
T* GetContiguousPtr(int block_id) {
  void* addr;
  GetBlockManager()->GetContiguousPtr(block_id, addr);
  return reinterpret_cast<T*>(addr);
}

// Get host block pointer.
template <typename T>
T* GetHostContiguousPtr(int block_id) {
  void* addr;
  GetBlockManager()->GetHostContiguousPtr(block_id, addr);
  return reinterpret_cast<T*>(addr);
}

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

}  // namespace ksana_llm
