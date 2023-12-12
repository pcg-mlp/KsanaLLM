/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/

#pragma once

#include <memory>
#include <mutex>

namespace numerous_llm {

// The singleton instance implement
template <typename T>
class Singleton {
 public:
  // Get singleton instance, with constructor arguments
  template <typename... Args>
  static T* GetInstance(Args&&... args) {
    if (!singleton_instance_) {
      std::lock_guard<std::mutex> lock(singleton_mutex_);
      if (!singleton_instance_) {
        singleton_instance_ = new T(std::forward<Args>(args)...);
      }
    }
    return singleton_instance_;
  }

 private:
  Singleton();
  Singleton(const Singleton&) = delete;
  Singleton& operator=(const Singleton&) = delete;
  ~Singleton();

 private:
  static T* singleton_instance_;
  static std::mutex singleton_mutex_;
};

template <typename T>
T* Singleton<T>::singleton_instance_ = nullptr;

template <typename T>
std::mutex Singleton<T>::singleton_mutex_;

}  // namespace numerous_llm
