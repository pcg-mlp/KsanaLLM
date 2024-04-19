// Copyright 2024 Tencent Inc.  All rights reserved.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stdexcept>

typedef enum NmDataType { NM_TYPE_INVALID = -1, NM_TYPE_FP32 = 0, NM_TYPE_FP16 = 1, NM_TYPE_BF16 = 3 } DataType;

size_t get_nm_dtype_size(NmDataType dtype) {
  if (dtype == NM_TYPE_FP32) {
    return sizeof(float);
  } else if (dtype == NM_TYPE_FP16) {
    return sizeof(uint16_t);
  } else if (dtype == NM_TYPE_BF16) {
    return sizeof(__nv_bfloat16);
  } else {
    throw std::runtime_error("Unsupported data type");
    return 0;
  }
}

struct NmTensor {
  int stride(int index) { return stride_[index]; }
  int size(int index) { return size_[index]; }
  void* data_ptr() const { return data_; }
  std::vector<void*> data_ptrs() const { return datas_; }
  NmDataType dtype() { return dtype_; }
  int numel() { return size_[0] * stride_[0]; }
  void* data_;
  std::vector<void*> datas_;
  std::vector<int> size_;
  std::vector<int> stride_;
  NmDataType dtype_;
};
