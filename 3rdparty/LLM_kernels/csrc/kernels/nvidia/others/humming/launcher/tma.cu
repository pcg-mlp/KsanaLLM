/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 * TMA descriptor builder implementation.
 */

#include "csrc/kernels/nvidia/others/humming/launcher/tma.h"
#include "csrc/utils/nvidia/assert.h"

#include <algorithm>
#include <cstring>

namespace llm_kernels {
namespace nvidia {
namespace humming {

CUtensorMapSwizzle GetSwizzleEnum(uint32_t swizzle_bytes) {
  switch (swizzle_bytes) {
    case 0:   return CU_TENSOR_MAP_SWIZZLE_NONE;
    case 32:  return CU_TENSOR_MAP_SWIZZLE_32B;
    case 64:  return CU_TENSOR_MAP_SWIZZLE_64B;
    case 128: return CU_TENSOR_MAP_SWIZZLE_128B;
    default:
      KLLM_KERNEL_CHECK_WITH_INFO(false,
          "Unsupported swizzle_bytes=%u. Must be 0, 32, 64, or 128.", swizzle_bytes);
      return CU_TENSOR_MAP_SWIZZLE_NONE;  // unreachable
  }
}

CUtensorMapDataType GetTmaDataType(uint32_t element_bytes) {
  switch (element_bytes) {
    case 1: return CU_TENSOR_MAP_DATA_TYPE_UINT8;
    case 2: return CU_TENSOR_MAP_DATA_TYPE_UINT16;
    case 4: return CU_TENSOR_MAP_DATA_TYPE_UINT32;
    case 8: return CU_TENSOR_MAP_DATA_TYPE_UINT64;
    default:
      KLLM_KERNEL_CHECK_WITH_INFO(false,
          "Unsupported element_bytes=%u for TMA dtype.", element_bytes);
      return CU_TENSOR_MAP_DATA_TYPE_UINT8;  // unreachable
  }
}

void BuildTmaDescriptor(CUtensorMap* tensor_map,
                        const void* data_ptr,
                        CUtensorMapDataType dtype,
                        uint32_t ndim,
                        const uint64_t* gmem_dims,
                        const uint64_t* gmem_strides,
                        const uint32_t* smem_box_dims,
                        uint32_t swizzle_bytes) {
  KLLM_KERNEL_CHECK_WITH_INFO(ndim >= 1 && ndim <= 5,
      "TMA ndim must be 1-5, got %u", ndim);
  KLLM_KERNEL_CHECK_WITH_INFO(tensor_map != nullptr, "tensor_map must not be null");
  KLLM_KERNEL_CHECK_WITH_INFO(data_ptr != nullptr, "data_ptr must not be null");

  // cuTensorMapEncodeTiled expects dimensions in innermost-first order.
  // Our inputs are in logical (outermost-first) order, so reverse them.
  uint64_t reversed_gmem_dims[5];
  uint64_t reversed_gmem_strides[4];  // ndim-1 entries
  uint32_t reversed_smem_box_dims[5];
  uint32_t element_strides[5];

  for (uint32_t i = 0; i < ndim; ++i) {
    reversed_gmem_dims[i] = gmem_dims[ndim - 1 - i];
    reversed_smem_box_dims[i] = smem_box_dims[ndim - 1 - i];
    element_strides[i] = 1;
  }

  // Strides: gmem_strides has ndim-1 entries (outermost first).
  // After reversal: strides[i] corresponds to dimension i+1 (skipping innermost).
  for (uint32_t i = 0; i < ndim - 1; ++i) {
    reversed_gmem_strides[i] = gmem_strides[ndim - 2 - i];
  }

  CUresult result = cuTensorMapEncodeTiled(
      tensor_map,
      dtype,
      static_cast<cuuint32_t>(ndim),
      const_cast<void*>(data_ptr),
      reversed_gmem_dims,
      reversed_gmem_strides,
      reversed_smem_box_dims,
      element_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE,
      GetSwizzleEnum(swizzle_bytes),
      CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

  if (result != CUDA_SUCCESS) {
    const char* err_str = nullptr;
    cuGetErrorString(result, &err_str);
    KLLM_KERNEL_CHECK_WITH_INFO(false,
        "cuTensorMapEncodeTiled failed: %s", err_str ? err_str : "unknown error");
  }
}

}  // namespace humming
}  // namespace nvidia
}  // namespace llm_kernels
