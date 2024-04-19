/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

// This file is adopt
// from
// https://github.com/vllm-project/vllm/blob/891070257c145b506a20666a3cb70afcf674d4ca/csrc/punica/bgmv/bgmv_config.h

#pragma once

#include <cstdint>

namespace llm_kernels {
namespace nvidia {

template <int32_t feat_in, int32_t feat_out, typename in_T, typename out_T, typename W_T>
void InvokeBGMVWrapper(out_T *__restrict__ Y, const in_T *__restrict__ X, const W_T *__restrict__ W,
                       const int64_t *__restrict__ indices, const int64_t y_offset, const int64_t full_y_size,
                       const int64_t batch_size, const int64_t num_layers, const int64_t layer_idx, const float scale,
                       cudaStream_t &stream);

// clang-format off

#define FOR_BGMV_WIDE(f, in_T, out_T, W_T, narrow) \
    f(in_T, out_T, W_T, narrow, 128) \
    f(in_T, out_T, W_T, narrow, 256) \
    f(in_T, out_T, W_T, narrow, 512) \
    f(in_T, out_T, W_T, narrow, 1024) \
    f(in_T, out_T, W_T, narrow, 1280) \
    f(in_T, out_T, W_T, narrow, 1728) \
    f(in_T, out_T, W_T, narrow, 1792) \
    f(in_T, out_T, W_T, narrow, 2048) \
    f(in_T, out_T, W_T, narrow, 2560) \
    f(in_T, out_T, W_T, narrow, 2752) \
    f(in_T, out_T, W_T, narrow, 3072) \
    f(in_T, out_T, W_T, narrow, 3456) \
    f(in_T, out_T, W_T, narrow, 3584) \
    f(in_T, out_T, W_T, narrow, 4096) \
    f(in_T, out_T, W_T, narrow, 5120) \
    f(in_T, out_T, W_T, narrow, 5504) \
    f(in_T, out_T, W_T, narrow, 5632) \
    f(in_T, out_T, W_T, narrow, 6912) \
    f(in_T, out_T, W_T, narrow, 7168) \
    f(in_T, out_T, W_T, narrow, 8192) \
    f(in_T, out_T, W_T, narrow, 9216) \
    f(in_T, out_T, W_T, narrow, 10240) \
    f(in_T, out_T, W_T, narrow, 11008) \
    f(in_T, out_T, W_T, narrow, 12288) \
    f(in_T, out_T, W_T, narrow, 13824) \
    f(in_T, out_T, W_T, narrow, 14336) \
    f(in_T, out_T, W_T, narrow, 16384) \
    f(in_T, out_T, W_T, narrow, 20480) \
    f(in_T, out_T, W_T, narrow, 28672) \
    f(in_T, out_T, W_T, narrow, 32000) \
    f(in_T, out_T, W_T, narrow, 32256) \
    f(in_T, out_T, W_T, narrow, 32512) \
    f(in_T, out_T, W_T, narrow, 32768) \
    f(in_T, out_T, W_T, narrow, 33024) \
    f(in_T, out_T, W_T, narrow, 36864) \
    f(in_T, out_T, W_T, narrow, 49152) \

#define FOR_BGMV_WIDE_NARROW(f, in_T, out_T, W_T) \
    FOR_BGMV_WIDE(f, in_T, out_T, W_T, 8)  \
    FOR_BGMV_WIDE(f, in_T, out_T, W_T, 16) \
    FOR_BGMV_WIDE(f, in_T, out_T, W_T, 32) \
    FOR_BGMV_WIDE(f, in_T, out_T, W_T, 64)

// clang-format on

}  // namespace nvidia
}  // namespace llm_kernels