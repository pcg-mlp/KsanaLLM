/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <unordered_map>

// All supported device type.
#define DEVICE_TYPE_NVIDIA 0
#define DEVICE_TYPE_ASCEND 1

// Unknown device type.
#define DEVICE_TYPE_UNKNOWN -1

// Select active device type.
#ifdef ENABLE_CUDA
#  define ACTIVE_DEVICE_TYPE DEVICE_TYPE_NVIDIA
#elif defined(ENABLE_ACL)
#  define ACTIVE_DEVICE_TYPE DEVICE_TYPE_ASCEND
#else
#  define ACTIVE_DEVICE_TYPE DEVICE_TYPE_UNKNOWN
#endif

// Include necessary header files.
#ifdef ENABLE_CUDA
#  include <cublasLt.h>
#  include <cublas_v2.h>
#  include <cuda_runtime.h>
#endif

#ifdef ENABLE_ACL
#  include <acl/acl.h>
#  include <acl/acl_base.h>
#  include <acl/acl_rt.h>
#endif

namespace ksana_llm {

// The type define.
#if defined(ENABLE_CUDA)
typedef half float16;
#  ifdef ENABLE_BFLOAT16
typedef __nv_bfloat16 bfloat16;
#  endif
#elif defined(ENABLE_ACL)
typedef aclFloat16 float16;
#endif

// All the available data format, for ascend aclTensor.
enum DataFormat {
#if defined(ENABLE_CUDA)
  FORMAT_DEFAULT
#elif defined(ENABLE_ACL)
  FORMAT_DEFAULT = aclFormat::ACL_FORMAT_ND
#endif
};

// All the available data types.
enum DataType {
#if defined(ENABLE_CUDA)
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
  TYPE_BF16,
  TYPE_FP16,
  TYPE_FP32,
  TYPE_FP64,
  TYPE_BYTES,
  TYPE_FP8_E4M3,
  TYPE_VOID,
  TYPE_POINTER
#elif defined(ENABLE_ACL)
  TYPE_INVALID = aclDataType::ACL_DT_UNDEFINED,
  TYPE_BOOL = aclDataType::ACL_BOOL,
  TYPE_UINT8 = aclDataType::ACL_INT8,
  TYPE_UINT16 = aclDataType::ACL_UINT16,
  TYPE_UINT32 = aclDataType::ACL_UINT32,
  TYPE_UINT64 = aclDataType::ACL_INT64,
  TYPE_INT8 = aclDataType::ACL_INT8,
  TYPE_INT16 = aclDataType::ACL_INT16,
  TYPE_INT32 = aclDataType::ACL_INT32,
  TYPE_INT64 = aclDataType::ACL_INT64,
  TYPE_BF16 = aclDataType::ACL_BF16,
  TYPE_FP16 = aclDataType::ACL_FLOAT16,
  TYPE_FP32 = aclDataType::ACL_FLOAT,
  TYPE_FP64 = aclDataType::ACL_DOUBLE,
  TYPE_BYTES = aclDataType::ACL_STRING,
  TYPE_FP8_E4M3 = aclDataType::ACL_DT_UNDEFINED,
  TYPE_VOID = aclDataType::ACL_DT_UNDEFINED,
  TYPE_POINTER = aclDataType::ACL_DT_UNDEFINED
#endif
};

size_t GetTypeSize(DataType dtype);

// The memory device.
enum MemoryDevice { MEMORY_HOST, MEMORY_DEVICE };

// A dummy class used as a real defined class.
struct DummyClass {};

}  // namespace ksana_llm
