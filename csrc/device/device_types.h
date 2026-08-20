/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <torch/torch.h>

#include <string>
#include <unordered_map>

// All supported device type.
#define DEVICE_TYPE_NVIDIA 0
#define DEVICE_TYPE_ASCEND 1
#define DEVICE_TYPE_ZIXIAO 2
#define DEVICE_TYPE_KUNLUN 3

// Unknown device type.
#define DEVICE_TYPE_UNKNOWN -1

// Select active device type.
#ifdef ENABLE_CUDA
#  define ACTIVE_DEVICE_TYPE DEVICE_TYPE_NVIDIA
#elif defined(ENABLE_ACL)
#  define ACTIVE_DEVICE_TYPE DEVICE_TYPE_ASCEND
#elif defined(ENABLE_TOPS)
#  define ACTIVE_DEVICE_TYPE DEVICE_TYPE_ZIXIAO
#elif defined(ENABLE_KUNLUN)
#  define ACTIVE_DEVICE_TYPE DEVICE_TYPE_KUNLUN
#else
#  define ACTIVE_DEVICE_TYPE DEVICE_TYPE_UNKNOWN
#endif

// Include necessary header files.
#ifdef ENABLE_CUDA
#  include <cublasLt.h>
#  include <cublas_v2.h>
#  include <cuda_runtime.h>
// Include cuda fp8 header
#  ifdef ENABLE_FP8
#    include <cuda_fp8.h>
#  endif
#endif

#ifdef ENABLE_ACL
#  include <acl/acl.h>
#  include <acl/acl_base.h>
#  include <acl/acl_rt.h>
#endif

// Kunlun XPU: only include the runtime header here (XPUStream / XPUEvent typedefs
// are required by kunlun_device.h via StreamTypeTraits/EventTypeTraits specializations).
// BKCL header lives next to runtime header but is included in bkcl_utils.h, NOT here,
// to avoid pulling BKCL types into every translation unit.
//
// Note: <xpu/runtime.h> defines `typedef void* XPUStream` and `typedef void* XPUEvent`
// (both opaque handles, NOT aliases of cudaStream_t — cudart compatibility uses a
// separate path through libcudart.so).
#ifdef ENABLE_KUNLUN
#  include <xpu/runtime.h>
#endif

namespace ksana_llm {

// The type define.
#if defined(ENABLE_CUDA)
typedef half float16;
typedef __nv_bfloat16 bfloat16;
#  ifdef ENABLE_FP8
typedef __nv_fp8_e4m3 fp8e4m3;
typedef __nv_fp8_e5m2 fp8e5m2;
#  endif
#elif defined(ENABLE_ACL)
typedef aclFloat16 float16;
typedef int16_t bfloat16;
#elif defined(ENABLE_TOPS)
typedef int16_t float16;
typedef uint16_t bfloat16;
#elif defined(ENABLE_KUNLUN)
// Kunlun XPU: use int16_t/uint16_t as float16/bfloat16 storage types.
// Note: xdnn's libxpu_dnn.so uses its own struct float16/bfloat16 internally,
// but they are memory-layout compatible (both 2 bytes). kernel_wrapper.cpp
// uses reinterpret_cast to bridge the type mismatch at link time.
typedef int16_t float16;
typedef uint16_t bfloat16;
#endif

// All the available data format, for ascend aclTensor.
enum DataFormat {
#if defined(ENABLE_CUDA)
  FORMAT_DEFAULT
#elif defined(ENABLE_ACL)
  FORMAT_DEFAULT = aclFormat::ACL_FORMAT_ND,
  FORMAT_ND = aclFormat::ACL_FORMAT_ND,
  FORMAT_NZ = aclFormat::ACL_FORMAT_FRACTAL_NZ
#elif defined(ENABLE_TOPS)
  FORMAT_DEFAULT
#elif defined(ENABLE_KUNLUN)
  FORMAT_DEFAULT
#endif
};

#ifdef ENABLE_ACL
inline aclFormat GetACLFormat(DataFormat data_format) {
  switch (data_format) {
    case FORMAT_ND:
      return aclFormat::ACL_FORMAT_ND;
    case FORMAT_NZ:
      return aclFormat::ACL_FORMAT_FRACTAL_NZ;
    default:
      return aclFormat::ACL_FORMAT_ND;
  }
}

inline DataFormat GetTensorFormat(aclFormat data_format) {
  switch (data_format) {
    case aclFormat::ACL_FORMAT_ND:
      return FORMAT_DEFAULT;
    case aclFormat::ACL_FORMAT_FRACTAL_NZ:
      return FORMAT_NZ;
    default:
      return FORMAT_DEFAULT;
  }
}
#endif

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
  TYPE_I4_GROUP,
  TYPE_BLOCK_FP8_E4M3,
  TYPE_FP8_E5M2,
  TYPE_VOID,
  TYPE_POINTER,
  TYPE_UINT4x2,
  TYPE_INT4x2,
  TYPE_FP8_DS_MLA,
  TYPE_FP8_E8M0,
  TYPE_FP4_E2M1FN_x2,
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
  TYPE_FP8_E4M3 = aclDataType::ACL_DT_UNDEFINED - 1,
  TYPE_FP8_E5M2 = aclDataType::ACL_DT_UNDEFINED - 2,
  TYPE_I4_GROUP = aclDataType::ACL_DT_UNDEFINED - 3,
  TYPE_BLOCK_FP8_E4M3 = aclDataType::ACL_DT_UNDEFINED - 4,
  TYPE_VOID = aclDataType::ACL_DT_UNDEFINED - 5,
  TYPE_POINTER = aclDataType::ACL_DT_UNDEFINED - 6,
  TYPE_UINT4x2 = aclDataType::ACL_DT_UNDEFINED - 7,
  TYPE_INT4x2 = aclDataType::ACL_DT_UNDEFINED - 8,
  TYPE_FP8_DS_MLA = aclDataType::ACL_DT_UNDEFINED - 9,
  TYPE_FP8_E8M0 = aclDataType::ACL_DT_UNDEFINED - 10,
  TYPE_FP4_E2M1FN_x2 = aclDataType::ACL_DT_UNDEFINED - 11,
#elif defined(ENABLE_TOPS)
  TYPE_INVALID,
  TYPE_FP32,
  TYPE_FP16,
  TYPE_BF16,
  TYPE_TF24,
  TYPE_TF32,
  TYPE_INT8,
  TYPE_UINT8,
  TYPE_INT32,
  TYPE_INT64,
  TYPE_UINT32,
  TYPE_UINT64,
  TYPE_BOOL,
  TYPE_BIT,
  TYPE_BIT2,
  TYPE_BIT4,
  TYPE_INT16,
  TYPE_FP8_E4M3,
  TYPE_FP8_E5M2,
  TYPE_FP64,
  TYPE_UINT16,
  TYPE_BYTES,
  TYPE_VOID,
  TYPE_POINTER,
  TYPE_I4_GROUP,
  TYPE_BLOCK_FP8_E4M3,
  TYPE_UINT4x2,
  TYPE_INT4x2,
  TYPE_FP8_DS_MLA,
  TYPE_FP8_E8M0,
  TYPE_FP4_E2M1FN_x2,
#elif defined(ENABLE_KUNLUN)
  // Kunlun XPU Stage 0: enumerate ALL possible DataType values (mirroring CUDA/TOPS)
  // so downstream code that references e.g. TYPE_FP8_E4M3 still compiles.
  // Runtime support (in kernels/layers) will KLLM_THROW for unsupported dtypes.
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
  TYPE_I4_GROUP,
  TYPE_BLOCK_FP8_E4M3,
  TYPE_FP8_E5M2,
  TYPE_VOID,
  TYPE_POINTER,
  TYPE_UINT4x2,
  TYPE_INT4x2,
  TYPE_FP8_DS_MLA,
  TYPE_FP8_E8M0,
  TYPE_FP4_E2M1FN_x2,
#endif
};

__attribute__((unused)) static std::string GetTypeString(DataType dtype) {
  static const std::unordered_map<DataType, std::string> type_map{{TYPE_INVALID, "invalid"},
                                                                  {TYPE_BOOL, "bool"},
                                                                  {TYPE_BYTES, "bytes"},
                                                                  {TYPE_UINT8, "uint8"},
                                                                  {TYPE_UINT16, "uint16"},
                                                                  {TYPE_UINT32, "uint32"},
                                                                  {TYPE_UINT64, "uint64"},
                                                                  {TYPE_INT8, "int8"},
                                                                  {TYPE_INT16, "int16"},
                                                                  {TYPE_INT32, "int32"},
                                                                  {TYPE_INT64, "int64"},
                                                                  {TYPE_FP16, "float16"},
                                                                  {TYPE_BF16, "bfloat16"},
                                                                  {TYPE_FP32, "float32"},
                                                                  {TYPE_FP64, "float64"},
                                                                  {TYPE_BYTES, "bytes"},
                                                                  {TYPE_FP8_E4M3, "fp8_e4m3"},
                                                                  {TYPE_I4_GROUP, "int4_group"},
                                                                  {TYPE_BLOCK_FP8_E4M3, "block_fp8_e4m3"},
                                                                  {TYPE_FP8_E5M2, "fp8_e5m2"},
                                                                  {TYPE_VOID, "void"},
                                                                  {TYPE_POINTER, "pointer"},
                                                                  {TYPE_FP8_DS_MLA, "fp8_ds_mla"},
                                                                  {TYPE_FP8_E8M0, "fp8_e8m0"},
                                                                  {TYPE_FP4_E2M1FN_x2, "fp4_e2m1fn_x2"}};
  return type_map.count(dtype) ? type_map.at(dtype) : "invalid";
}

size_t GetTypeSize(DataType dtype);

// The memory device.
enum MemoryDevice { MEMORY_HOST, MEMORY_DEVICE };

// The memory location.
enum MemoryLocation {
  LOCATION_UNKNOWN,
  LOCATION_HOST,
  LOCATION_DEVICE,
  LOCATION_DEVICE_IPC,
  LOCATION_NVSHMEM,
  LOCATION_MULTICAST
};

// A dummy class used as a real defined class.
struct DummyClass {};

c10::ScalarType GetTorchTypeFromDataType(const DataType& data_type);

DataType GetDataTypeFromTorchType(const c10::ScalarType& torch_type);

}  // namespace ksana_llm
