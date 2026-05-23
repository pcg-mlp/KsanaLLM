/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 *
 * Helpers shared by the sglang gate kernels: input-dtype enum (host-side) and
 * a __device__ load_as_float<T>() that converts fp32/bf16 → float at the load
 * site, so the kernel body can stay in float.
 */
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <type_traits>

#include "sgl_kernel/utils.cuh"

namespace llm_kernels::nvidia::sglang_gate {

enum class GateInputDtype { kFloat32, kBFloat16 };

template <typename T>
SGL_DEVICE float load_as_float(const T* p, size_t idx) {
  if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    return __bfloat162float(p[idx]);
  } else {
    return static_cast<float>(p[idx]);
  }
}

}  // namespace llm_kernels::nvidia::sglang_gate
