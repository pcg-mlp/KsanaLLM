/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 */

// PDL (Programmatic Dependent Launch) compatibility shims.
//
// Provides:
//   PDLWaitPrimary<bool>()   — wait for grid dependency (CUDA 12, sm_90+)
//   PDLTriggerSecondary<bool>() — signal completion to dependent grid
//
// When the template argument is `false`, the calls are trivially optimised out.
// On architectures below sm_90 the intrinsics are not available, so the bodies
// are empty even when `true`.

#pragma once

#include <cuda_runtime.h>

template <bool kUsePDL>
__forceinline__ __device__ void PDLWaitPrimary() {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  if constexpr (kUsePDL) {
    cudaGridDependencySynchronize();
  }
#endif
}

template <bool kUsePDL>
__forceinline__ __device__ void PDLTriggerSecondary() {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  if constexpr (kUsePDL) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
#endif
}
