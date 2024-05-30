/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <math.h>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <queue>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <vector>

namespace ksana_llm {

// / The thread pool
class ThreadPool {
  public:
    using Task = std::function<void()>;

    ThreadPool(size_t size) : size_(size) {}

    ~ThreadPool() {}

    size_t Idle() { return idle_; }

    void Stop() {
      {
        std::lock_guard<std::mutex> lock(this->mutex_);
        stopped_.store(true);
      }
      cv_.notify_all();
      for (std::thread& thread : pool_) {
        if (thread.joinable()) {
          thread.join();
        }
      }
    }

    void Start() {
      idle_ = size_ < 1 ? 1 : size_;
      stopped_.store(false);
      for (size_ = 0; size_ < idle_; ++size_) {
        pool_.emplace_back([this] {
          while (!this->stopped_) {
            std::function<void()> task;
            {
              std::unique_lock<std::mutex> lock{this->mutex_};
              this->cv_.wait(lock, [this] { return this->stopped_.load() || !this->tasks_.empty(); });
              if (this->stopped_ && this->tasks_.empty()) {
                return;
              }
              task = std::move(this->tasks_.front());
              this->tasks_.pop();
            }
            idle_--;
            task();
            idle_++;
          }
        });
      }
    }

    template <class Fun, class... Args>
    auto Submit(Fun&& f, Args&&... args) -> std::future<decltype(f(args...))> {
      if (stopped_.load()) {
        throw std::runtime_error("Submit on stopped threadpool.");
      }

      using RetType = decltype(f(args...));
      auto task =
          std::make_shared<std::packaged_task<RetType()>>(std::bind(std::forward<Fun>(f), std::forward<Args>(args)...));
      std::future<RetType> future = task->get_future();
      {
        std::lock_guard<std::mutex> lock{mutex_};
        tasks_.emplace([task]() { (*task)(); });
      }
      cv_.notify_one();
      return future;
    }

    size_t Size() { return size_; }

  private:
    std::condition_variable cv_;
    std::mutex mutex_;

    std::atomic<size_t> idle_;
    std::atomic<bool> stopped_;

    std::vector<std::thread> pool_;
    std::queue<Task> tasks_;

    size_t size_;
};

}  // namespace ksana_llm
