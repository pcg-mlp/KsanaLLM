/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#pragma once

#include <cuda_runtime.h>

namespace llm_kernels {
namespace nvidia {

template <typename T>
void InvokeCopyElements(T** src_ptrs, T* dest, size_t num_elements, cudaStream_t& stream);

}  // namespace nvidia
}  // namespace llm_kernels