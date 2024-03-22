/* Copyright 2023 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cstdint>
#include <type_traits>

#ifdef ENABLE_CUDA
#  include <cuda_fp16.h>
#endif

#ifdef ENABLE_ACL
#  include <acl/acl.h>
#endif

namespace ksana_llm {

// All the available data types.
enum DataType {
  TYPE_INVALID,
  TYPE_BOOL,
  TYPE_UINT8,
  TYPE_UINT16,
  TYPE_UINT32,
  TYPE_UINT64,
  TYPE_INT8,
  TYPE_INT16,
  TYPE_INT32,
  TYPE_INT64,
  TYPE_FP16,
  TYPE_FP32,
  TYPE_FP64,
  TYPE_BYTES,
  TYPE_BF16,
  TYPE_FP8_E4M3,
  TYPE_STR,
  TYPE_POINTER,
  TYPE_VOID,
};

template <typename T>
DataType GetTensorType() {
  if (std::is_same<T, float>::value || std::is_same<T, const float>::value) {
    return TYPE_FP32;
  }
#ifdef ENABLE_CUDA
  else if (std::is_same<T, half>::value || std::is_same<T, const half>::value) {
    return TYPE_FP16;
  }
#  ifdef ENABLE_BF16
  else if (std::is_same<T, __nv_bfloat16>::value || std::is_same<T, const __nv_bfloat16>::value) {
    return TYPE_BF16;
  }
#  endif
#  ifdef ENABLE_FP8
  else if (std::is_same<T, __nv_fp8_e4m3>::value || std::is_same<T, const __nv_fp8_e4m3>::value) {
    return TYPE_FP8_E4M3;
  }
#  endif
#endif
  else if (std::is_same<T, int>::value || std::is_same<T, const int>::value) {
    return TYPE_INT32;
  } else if (std::is_same<T, int8_t>::value || std::is_same<T, const int8_t>::value) {
    return TYPE_INT8;
  } else if (std::is_same<T, uint8_t>::value || std::is_same<T, const uint8_t>::value) {
    return TYPE_UINT8;
  } else if (std::is_same<T, unsigned int>::value || std::is_same<T, const unsigned int>::value) {
    return TYPE_UINT32;
  } else if (std::is_same<T, unsigned long long int>::value || std::is_same<T, const unsigned long long int>::value) {
    return TYPE_UINT64;
  } else if (std::is_same<T, bool>::value || std::is_same<T, const bool>::value) {
    return TYPE_BOOL;
  } else if (std::is_same<T, char>::value || std::is_same<T, const char>::value) {
    return TYPE_BYTES;
  } else if (std::is_pointer<T>::value) {
    return TYPE_POINTER;
  } else {
    return TYPE_INVALID;
  }
}

}  // namespace ksana_llm
