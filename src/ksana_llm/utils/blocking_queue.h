/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <limits.h>
#include <stdint.h>

#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include "ksana_llm/utils/logger.h"

namespace ksana_llm {

template <typename T, typename Queue = std::queue<T>>
class BlockingQueue {
 public:
  explicit BlockingQueue(uint32_t max_size = UINT_MAX) : max_size_(max_size), is_closed_(false) {}

  ~BlockingQueue() = default;

  template <typename V>
  bool Put(V&& new_value) {
    std::unique_lock<std::mutex> lock(mutex_);
    cond_.wait(lock, [this] { return queue_.size() < max_size_ || is_closed_; });
    if (is_closed_) {
      return false;
    }

    queue_.push(std::forward<V>(new_value));
    cond_.notify_all();
    return true;
  }

  T Get() {
    std::unique_lock<std::mutex> lock(mutex_);
    cond_.wait(lock, [this] { return !queue_.empty() || is_closed_; });
    if (is_closed_) {
      return T();
    }

    T value = std::move(queue_.front());
    queue_.pop();
    cond_.notify_all();
    return value;
  }

  T Get(size_t timeout_ms) {
    std::unique_lock<std::mutex> lock(mutex_);
    cond_.wait(lock, std::chrono::milliseconds(timeout_ms), [this] { return !queue_.empty() || is_closed_; });
    if (is_closed_) {
      return T();
    }

    T value = std::move(queue_.front());
    queue_.pop();
    cond_.notify_all();
    return value;
  }

  size_t Size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
  }

  bool Empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.empty();
  }

  void Stop() {
    std::lock_guard<std::mutex> lk(mutex_);
    is_closed_ = true;
    cond_.notify_all();
  }

 private:
  Queue queue_;
  mutable std::mutex mutex_;
  std::condition_variable cond_;
  uint32_t max_size_;
  bool is_closed_;
};

}  // namespace ksana_llm
