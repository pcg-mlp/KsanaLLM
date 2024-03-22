/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/utils/common_context.h"

#include <iostream>
#include <stdexcept>
#include "fmt/core.h"
#include "ksana_llm/utils/device_helper.h"
#include "ksana_llm/utils/device_types.h"

namespace ksana_llm {

template <int T>
ContextT<T>::ContextT(const int tensor_parallel_size, const int pipeline_parallel_size, const MemoryDevice device_type)
    : tensor_parallel_size_(tensor_parallel_size),
      pipeline_parallel_size_(pipeline_parallel_size),
      device_type_(device_type) {
  if (pipeline_parallel_size_ != 1) {
    throw std::runtime_error("Only support pipeline_parallel_size == 1");
  }

  GetDeviceCount(&device_num_);
  if (device_num_ < tensor_parallel_size_ * pipeline_parallel_size_) {
    throw std::runtime_error(fmt::format("{} tensor_parallel_size should not bigger than devices num: {}",
                                         tensor_parallel_size_, device_num_));
  }

  memory_manage_streams_.reserve(tensor_parallel_size_);
  compute_streams_.reserve(tensor_parallel_size_);
  h2d_streams_.reserve(tensor_parallel_size_);
  d2h_streams_.reserve(tensor_parallel_size_);
  d2d_streams_.reserve(tensor_parallel_size_);
  nccl_streams_.reserve(tensor_parallel_size_);
  for (int worker_id = 0; worker_id < tensor_parallel_size_; ++worker_id) {
    InitStreams(worker_id);
  }

  // Initialize the device extension.
  InitializeExtension();
}

template <int T>
ContextT<T>::~ContextT() {
  DestroyExtension();

  for (int worker_id = 0; worker_id < tensor_parallel_size_; ++worker_id) {
    memory_manage_streams_[worker_id].Destroy();
    compute_streams_[worker_id].Destroy();
    h2d_streams_[worker_id].Destroy();
    d2h_streams_[worker_id].Destroy();
    d2d_streams_[worker_id].Destroy();
    nccl_streams_[worker_id].Destroy();
  }

  memory_manage_streams_.clear();
  compute_streams_.clear();
  h2d_streams_.clear();
  d2h_streams_.clear();
  d2d_streams_.clear();
  nccl_streams_.clear();
}

template <int T>
void ContextT<T>::InitStreams(const int worker_id) {
  memory_manage_streams_.emplace_back(worker_id);
  compute_streams_.emplace_back(worker_id);
  h2d_streams_.emplace_back(worker_id);
  d2h_streams_.emplace_back(worker_id);
  d2d_streams_.emplace_back(worker_id);
  nccl_streams_.emplace_back(worker_id);
}

template class ContextT<ACTIVE_DEVICE_TYPE>;

}  // namespace ksana_llm
