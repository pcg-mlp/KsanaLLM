/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cuda_runtime.h>

#include <cstring>
#include <stdexcept>
#include <vector>

#include "utils/blocking_queue.h"
#include "utils/logger.h"

namespace ksana_llm {

struct StagingBuffer {
  void* ptr = nullptr;
  size_t capacity = 0;
};

/**
 * @brief 自包含的 CUDA pinned memory buffer pool，用于 RDMA staging。
 *
 * 构造时通过 cudaHostAlloc 预分配固定数量的 pinned memory block，
 * 提供线程安全的 Get/Put 接口供并发传输使用。
 * 所有 block 可通过 AllBlocks() 遍历以注册 RDMA memory region。
 */
class StagingBufferPool {
 public:
  StagingBufferPool(size_t block_count, size_t block_size)
      : block_size_(block_size), total_bytes_(block_count * block_size) {
    int cur_dev = -1;
    cudaGetDevice(&cur_dev);

    cudaError_t err = cudaHostAlloc(&base_ptr_, total_bytes_, cudaHostAllocDefault);
    if (err != cudaSuccess || base_ptr_ == nullptr) {
      const cudaError_t peek_after = cudaPeekAtLastError();
      KLLM_LOG_ERROR << "cudaHostAlloc failed, block_count=" << block_count << " block_size=" << block_size
                     << " total_bytes=" << total_bytes_ << " cur_dev=" << cur_dev << " err=" << static_cast<int>(err)
                     << "(" << cudaGetErrorString(err) << ")"
                     << " peek_after=" << cudaGetErrorString(peek_after) << " base_ptr=" << base_ptr_
                     << " flag=cudaHostAllocDefault";
      throw std::runtime_error("StagingBufferPool: cudaHostAlloc failed");
    }
    memset(base_ptr_, 0, total_bytes_);

    blocks_.resize(block_count);
    for (size_t i = 0; i < block_count; ++i) {
      blocks_[i].ptr = static_cast<char*>(base_ptr_) + i * block_size;
      blocks_[i].capacity = block_size;
      free_queue_.Put(&blocks_[i]);
    }
  }

  ~StagingBufferPool() {
    free_queue_.Stop();
    if (base_ptr_) {
      cudaFreeHost(base_ptr_);
    }
  }

  StagingBufferPool(const StagingBufferPool&) = delete;
  StagingBufferPool& operator=(const StagingBufferPool&) = delete;

  /**
   * @brief 阻塞等待直到有可用 buffer 或 pool 被 Stop()。
   * @return 可用 buffer 指针；pool 已关闭时返回 nullptr。
   * @note 批量获取场景应使用 TryGet() 避免死锁。
   */
  StagingBuffer* Get() { return free_queue_.Get(); }

  /**
   * @brief 非阻塞获取：立即返回可用 buffer 或 nullptr（pool 为空/已关闭）。
   */
  StagingBuffer* TryGet() { return free_queue_.NonBlockingGet(); }

  /**
   * @brief 归还 staging buffer。
   */
  void Put(StagingBuffer* buf) {
    if (!buf) {
      return;
    }
    free_queue_.Put(buf);
  }

  void* BasePtr() const { return base_ptr_; }
  size_t TotalBytes() const { return total_bytes_; }
  size_t BlockCount() const { return blocks_.size(); }

 private:
  size_t block_size_;
  size_t total_bytes_;
  void* base_ptr_ = nullptr;
  std::vector<StagingBuffer> blocks_;
  BlockingQueue<StagingBuffer*> free_queue_;
};

}  // namespace ksana_llm
