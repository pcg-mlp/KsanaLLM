/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/models/base/base_model.h"

namespace ksana_llm {

BaseModel::~BaseModel() { ReleaseBufferTensors(); }

Status BaseModel::CreateBufferTensor(Tensor& buf_tensor, const std::vector<size_t> shape, const DataType dtype,
                                     const MemoryDevice memory_device) {
  STATUS_CHECK_FAILURE(CreateTensor(buf_tensor, shape, dtype, rank_, memory_device));
  buffer_tensor_heap_.push_back(buf_tensor);
  total_buffer_size_ += buf_tensor.GetTotalBytes();
  return Status();
}

Status BaseModel::ReleaseBufferTensors() {
  for (Tensor& buf_tensor_ptr : buffer_tensor_heap_) {
    if (buf_tensor_ptr.GetElementNumber() > 0) {
      STATUS_CHECK_FAILURE(DestroyTensor(buf_tensor_ptr, rank_));
    }
  }
  buffer_tensor_heap_.clear();
  total_buffer_size_ = 0ul;
  return Status();
}

const size_t BaseModel::GetBufferTensorsMemoryUsed() { return total_buffer_size_; }

}  // namespace ksana_llm
