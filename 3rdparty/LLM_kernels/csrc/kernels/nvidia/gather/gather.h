/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 */
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace llm_kernels {
namespace nvidia {

/**
 * @brief Gather hidden units along the first dimension according to the given indices.
 *
 * Treats `input` as an array of `len_input` elements along the first dimension, where
 * each element has size `data_size` in bytes. For each index in `indices[0..len-1]`,
 * the corresponding slice is copied from `input` to `output` in order.
 *
 * @param input      Pointer to the input data [_, data_size]
 * @param indices    Pointer to an array of indices of the first dimension indices of input
 * @param len        Size of indices
 * @param data_size  Size in bytes of a single hidden units.
 * @param output     Pointer to the output data where gathered hidden units are written [len, data_size]
 * @param stream     CUDA stream used to launch the gather operation
 */
void GatherFirstDim(const void* input, const int* indices, const int64_t len, const int64_t data_size, void* output,
                    cudaStream_t stream);

}  // namespace nvidia
}  // namespace llm_kernels
