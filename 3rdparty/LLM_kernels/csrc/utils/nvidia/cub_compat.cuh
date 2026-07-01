/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 *
 * Compatibility shim for CUDA 13.0+ where the CCCL/CUB library removed the
 * `cub::Sum` and `cub::Max` binary functor types. On CUDA 12.x these are still
 * provided by `<cub/thread/thread_operators.cuh>`, so the definitions below
 * are only emitted when building against CUDA 13.0 or newer.
 */
#pragma once

#include <cub/cub.cuh>

// Note: use CUDART_VERSION (from cuda_runtime.h, pulled in by cub) rather than
// CUDA_VERSION (from cuda.h) so this works even when cuda.h is not included.
#if CUDART_VERSION >= 13000
#  include <cuda/functional>      // cuda::maximum
#  include <cuda/std/functional>  // cuda::std::plus

CUB_NAMESPACE_BEGIN

struct Sum
{
  template <typename T>
  __host__ __device__ __forceinline__ T operator()(const T& a, const T& b) const
  {
    return a + b;
  }
};

struct Max
{
  template <typename T>
  __host__ __device__ __forceinline__ T operator()(const T& a, const T& b) const
  {
    return a < b ? b : a;
  }
};

CUB_NAMESPACE_END

#endif // CUDART_VERSION >= 13000
