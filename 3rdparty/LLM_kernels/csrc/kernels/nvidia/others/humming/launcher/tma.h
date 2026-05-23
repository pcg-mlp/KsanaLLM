/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 * TMA descriptor builder utilities for humming kernels.
 */
#pragma once
#include <cuda.h>
#include <cstdint>

namespace llm_kernels {
namespace nvidia {
namespace humming {

/// Convert swizzle byte count to CU enum.
CUtensorMapSwizzle GetSwizzleEnum(uint32_t swizzle_bytes);

/// Map element byte size to CUtensorMapDataType.
CUtensorMapDataType GetTmaDataType(uint32_t element_bytes);

/// Build a TMA descriptor via cuTensorMapEncodeTiled.
/// All array parameters are in LOGICAL order (outermost dimension first).
/// Internally they are reversed to match TMA's innermost-first convention.
///
/// @param tensor_map     Output TMA descriptor.
/// @param data_ptr       Device pointer to tensor data.
/// @param dtype          TMA data type enum.
/// @param ndim           Number of dimensions (1-5).
/// @param gmem_dims      [ndim] global memory dimensions, logical order.
/// @param gmem_strides   [ndim-1] strides in bytes, logical order (outermost first).
///                        Innermost stride is implicit (= element_size * element_stride).
/// @param smem_box_dims  [ndim] shared memory box dimensions, logical order.
/// @param swizzle_bytes  Swizzle granularity: 0, 32, 64, or 128.
void BuildTmaDescriptor(CUtensorMap* tensor_map,
                        const void* data_ptr,
                        CUtensorMapDataType dtype,
                        uint32_t ndim,
                        const uint64_t* gmem_dims,
                        const uint64_t* gmem_strides,
                        const uint32_t* smem_box_dims,
                        uint32_t swizzle_bytes);

}  // namespace humming
}  // namespace nvidia
}  // namespace llm_kernels
