// Copyright 2024 Tencent Inc.  All rights reserved.

#pragma once

#include <cassert>

namespace llm_kernels {
namespace nvidia {

// Class for auto compute extent each dimention's neighbor element stride.
// for example: [2, 3, 4], dim1, dim2, dim3, its stride [12, 4, 1]
template <typename T, int N>
class NdIndexOffsetHelper {
 public:
  __forceinline__ NdIndexOffsetHelper() = default;

  template <class... Ts>
  __device__ __host__ __forceinline__ explicit NdIndexOffsetHelper(T d0, Ts... dims) {
    constexpr int n = 1 + sizeof...(dims);
    static_assert(n <= N, "");
    T dims_arr[n] = {d0, static_cast<T>(dims)...};
    InitStrides(dims_arr, n);
  }

  __device__ __host__ __forceinline__ explicit NdIndexOffsetHelper(const T* dims) { InitStrides(dims, N); }

  template <typename U>
  __device__ __host__ __forceinline__ explicit NdIndexOffsetHelper(const U* dims) {
    T dims_arr[N];
    for (int i = 0; i < N; ++i) {
      dims_arr[i] = dims[i];
    }
    InitStrides(dims_arr, N);
  }

  __device__ __host__ __forceinline__ explicit NdIndexOffsetHelper(const T* dims, int n) { InitStrides(dims, n); }

  template <typename U>
  __device__ __host__ __forceinline__ explicit NdIndexOffsetHelper(const U* dims, int n) {
    T dims_arr[N];
    for (int i = 0; i < N; ++i) {
      if (i < n) {
        dims_arr[i] = dims[i];
      }
    }
    InitStrides(dims_arr, n);
  }

  virtual ~NdIndexOffsetHelper() = default;

  __device__ __host__ __forceinline__ T NdIndexToOffset(const T* index) const {
    T offset = 0;
#ifdef __CUDA_ARCH__
#  pragma unroll
#endif
    for (int i = 0; i < N; ++i) {
      offset += index[i] * stride_[i];
    }
    return offset;
  }

  __device__ __host__ __forceinline__ T NdIndexToOffset(const T* index, int n) const {
    assert(n <= N);
    T offset = 0;
#ifdef __CUDA_ARCH__
#  pragma unroll
#endif
    for (int i = 0; i < N; ++i) {
      if (i < n) {
        offset += index[i] * stride_[i];
      }
    }
    return offset;
  }

  template <class... Ts>
  __device__ __host__ __forceinline__ T NdIndexToOffset(T d0, Ts... others) const {
    constexpr int n = 1 + sizeof...(others);
    static_assert(n <= N, "");
    T index[n] = {d0, others...};
    T offset = 0;
#ifdef __CUDA_ARCH__
#  pragma unroll
#endif
    for (int i = 0; i < n - 1; ++i) {
      offset += index[i] * stride_[i];
    }
    if (n == N) {
      offset += index[n - 1];
    } else {
      offset += index[n - 1] * stride_[n - 1];
    }
    return offset;
  }

  __device__ __host__ __forceinline__ void OffsetToNdIndex(T offset, T* index) const {
    T remaining = offset;
#ifdef __CUDA_ARCH__
#  pragma unroll
#endif
    for (int i = 0; i < N - 1; ++i) {
      const T idx = remaining / stride_[i];
      index[i] = idx;
      remaining = remaining - idx * stride_[i];
    }
    index[N - 1] = remaining;
  }

  __device__ __host__ __forceinline__ void OffsetToNdIndex(T offset, T* index, int n) const {
    assert(n <= N);
    T remaining = offset;
#ifdef __CUDA_ARCH__
#  pragma unroll
#endif
    for (int i = 0; i < N; ++i) {
      if (i < n) {
        const T idx = remaining / stride_[i];
        index[i] = idx;
        remaining = remaining - idx * stride_[i];
      }
    }
  }

  template <class... Ts>
  __device__ __host__ __forceinline__ void OffsetToNdIndex(T offset, T& d0, Ts&... others) const {
    constexpr int n = 1 + sizeof...(others);
    static_assert(n <= N, "");
    T* index[n] = {&d0, &others...};
    T remaining = offset;
#ifdef __CUDA_ARCH__
#  pragma unroll
#endif
    for (int i = 0; i < n - 1; ++i) {
      const T idx = remaining / stride_[i];
      *index[i] = idx;
      remaining = remaining - idx * stride_[i];
    }
    if (n == N) {
      *index[n - 1] = remaining;
    } else {
      *index[n - 1] = remaining / stride_[n - 1];
    }
  }

  __device__ __host__ __forceinline__ constexpr int Size() const { return N; }

 protected:
  __device__ __host__ __forceinline__ void InitStrides(const T* dims, const int n) {
    for (int i = n - 1; i < N; ++i) {
      stride_[i] = 1;
    }
    for (int i = n - 2; i >= 0; --i) {
      stride_[i] = dims[i + 1] * stride_[i + 1];
    }
  }

  T stride_[N];
};

template <typename T, int N>
class NdIndexStrideOffsetHelper : public NdIndexOffsetHelper<T, N> {
 public:
  __forceinline__ NdIndexStrideOffsetHelper() = default;
  __device__ __host__ __forceinline__ explicit NdIndexStrideOffsetHelper(const T* strides) {
    for (int i = 0; i < N; ++i) {
      stride_[i] = strides[i];
    }
  }

  template <typename U>
  __device__ __host__ __forceinline__ explicit NdIndexStrideOffsetHelper(const U* strides) {
    for (int i = 0; i < N; ++i) {
      stride_[i] = static_cast<T>(strides[i]);
    }
  }

  __device__ __host__ __forceinline__ explicit NdIndexStrideOffsetHelper(const T* strides, int n) {
    for (int i = 0; i < N; ++i) {
      if (i < n) {
        stride_[i] = strides[i];
      }
    }
  }

  template <typename U>
  __device__ __host__ __forceinline__ explicit NdIndexStrideOffsetHelper(const U* strides, int n) {
    for (int i = 0; i < N; ++i) {
      if (i < n) {
        stride_[i] = static_cast<T>(strides[i]);
      }
    }
  }

 private:
  using NdIndexOffsetHelper<T, N>::stride_;
};

}  // namespace nvidia
}  // namespace llm_kernels
