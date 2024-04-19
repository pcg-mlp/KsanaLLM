/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */
#pragma once

#include <cstdint>

#include <cuda_runtime.h>

namespace llm_kernels {
namespace nvidia {

constexpr int64_t MAX_LORA_HIDDEN_UNITS = 65536;

template <typename Y_T, typename X_T, typename W_T>
void InvokeBGMV(Y_T* y, const X_T* x, const W_T* w, const int64_t* indices, const int64_t layer_idx, const float scale,
                const int64_t batch_size, const int64_t num_layers, const int64_t feat_in, const int64_t feat_out,
                const int64_t y_offset, const int64_t full_y_size, cudaStream_t& stream);

}  // namespace nvidia
}  // namespace llm_kernels