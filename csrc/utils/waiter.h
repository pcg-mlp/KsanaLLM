/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>

namespace ksana_llm {

class AtomicCounter {
 public:
  explicit AtomicCounter(int num = 1) : num_(num) {}

  AtomicCounter() = delete;

  void Reset(int num) { num_ = num; }

  bool IsZero() { return num_ == 0; }

  bool DecAndIsZero() { return (--num_ == 0) ? true : false; }

 private:
  std::atomic<int> num_;
};

class Waiter {
 public:
  explicit Waiter(int num_wait = 1) : num_wait_(num_wait), stop_(false) {}
  explicit Waiter(int num_wait, std::function<void()> done) : num_wait_(num_wait), stop_(false), done_(done) {}

  Waiter() = delete;

  // Wait until zero.
  void Wait();

  void Notify();

  void Reset(int num_wait);

  void Inc();

  int Cnt();

  void Stop();

 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  int num_wait_;
  bool stop_;
  // done call back
  std::function<void()> done_;
};

}  // namespace ksana_llm
