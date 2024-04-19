/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

// This file is adopt
// from
// https://github.com/vllm-project/vllm/blob/891070257c145b506a20666a3cb70afcf674d4ca/csrc/punica/bgmv/bgmv_all.cu

#include "bgmv.h"
#include "bgmv_config.h"
#include "bgmv_impl.cuh"

namespace llm_kernels {
namespace nvidia {

FOR_BGMV_WIDE_NARROW(INST_BGMV_TWOSIDE, nv_half, nv_half, nv_half)
FOR_BGMV_WIDE_NARROW(INST_BGMV_TWOSIDE, nv_half, nv_half, nv_bfloat16)
FOR_BGMV_WIDE_NARROW(INST_BGMV_TWOSIDE, nv_bfloat16, nv_bfloat16, nv_bfloat16)
FOR_BGMV_WIDE_NARROW(INST_BGMV_TWOSIDE, nv_bfloat16, nv_bfloat16, nv_half)
FOR_BGMV_WIDE_NARROW(INST_BGMV_TWOSIDE, nv_half, nv_bfloat16, nv_half)
FOR_BGMV_WIDE_NARROW(INST_BGMV_TWOSIDE, nv_bfloat16, nv_half, nv_half)
FOR_BGMV_WIDE_NARROW(INST_BGMV_TWOSIDE, nv_half, nv_bfloat16, nv_bfloat16)
FOR_BGMV_WIDE_NARROW(INST_BGMV_TWOSIDE, nv_bfloat16, nv_half, nv_bfloat16)
FOR_BGMV_WIDE_NARROW(INST_BGMV_TWOSIDE, float, nv_half, nv_half)
FOR_BGMV_WIDE_NARROW(INST_BGMV_TWOSIDE, float, nv_half, nv_bfloat16)
FOR_BGMV_WIDE_NARROW(INST_BGMV_TWOSIDE, float, nv_bfloat16, nv_bfloat16)
FOR_BGMV_WIDE_NARROW(INST_BGMV_TWOSIDE, float, nv_bfloat16, nv_half)
FOR_BGMV_WIDE_NARROW(INST_BGMV_TWOSIDE, nv_half, float, nv_half)
FOR_BGMV_WIDE_NARROW(INST_BGMV_TWOSIDE, nv_half, float, nv_bfloat16)
FOR_BGMV_WIDE_NARROW(INST_BGMV_TWOSIDE, nv_bfloat16, float, nv_bfloat16)
FOR_BGMV_WIDE_NARROW(INST_BGMV_TWOSIDE, nv_bfloat16, float, nv_half)
FOR_BGMV_WIDE_NARROW(INST_BGMV_TWOSIDE, float, float, nv_half)
FOR_BGMV_WIDE_NARROW(INST_BGMV_TWOSIDE, float, float, nv_bfloat16)

inline constexpr uint32_t PackU16(uint16_t a, uint16_t b) { return (uint32_t(a) << 16) | uint32_t(b); }

template <typename Y_T, typename X_T, typename W_T>
void InvokeBGMV(Y_T* y, const X_T* x, const W_T* w, const int64_t* indices, const int64_t layer_idx, const float scale,
                const int64_t batch_size, const int64_t num_layers, const int64_t feat_in, const int64_t feat_out,
                const int64_t y_offset, const int64_t full_y_size, cudaStream_t& stream) {
  if (feat_in < MAX_LORA_HIDDEN_UNITS && feat_out < MAX_LORA_HIDDEN_UNITS) {
    switch (PackU16(feat_in, feat_out)) {
#define CASE_ONESIDE(_in_T, _out_T, _W_T, feat_in, feat_out)                                                         \
  case PackU16(feat_in, feat_out):                                                                                   \
    InvokeBGMVWrapper<feat_in, feat_out>(y, x, w, indices, y_offset, full_y_size, batch_size, num_layers, layer_idx, \
                                         scale, stream);                                                             \
    break;
#define CASE(_in_T, _out_T, _W_T, narrow, wide) \
  CASE_ONESIDE(X_T, Y_T, W_T, narrow, wide)     \
  CASE_ONESIDE(X_T, Y_T, W_T, wide, narrow)
      FOR_BGMV_WIDE_NARROW(CASE, _, _, _)
#undef CASE
#undef CASE_ONESIDE
      default:
        return;
    }
    return;

  } else {
    throw std::invalid_argument("Lora hidden units is larger than 65536");
  }
}

#define INSTANTIATE_INVOKE_BGMV(Y_T, X_T, W_T)                                                                  \
  template void InvokeBGMV(Y_T* y, const X_T* x, const W_T* w, const int64_t* indices, const int64_t layer_idx, \
                           const float scale, const int64_t batch_size, const int64_t num_layers,               \
                           const int64_t feat_in, const int64_t feat_out, const int64_t y_offset,               \
                           const int64_t full_y_size, cudaStream_t& stream);

INSTANTIATE_INVOKE_BGMV(nv_half, nv_half, nv_half)
INSTANTIATE_INVOKE_BGMV(nv_half, nv_half, nv_bfloat16)
INSTANTIATE_INVOKE_BGMV(nv_bfloat16, nv_bfloat16, nv_bfloat16)
INSTANTIATE_INVOKE_BGMV(nv_bfloat16, nv_bfloat16, nv_half)
INSTANTIATE_INVOKE_BGMV(nv_half, nv_bfloat16, nv_half)
INSTANTIATE_INVOKE_BGMV(nv_bfloat16, nv_half, nv_half)
INSTANTIATE_INVOKE_BGMV(nv_half, nv_bfloat16, nv_bfloat16)
INSTANTIATE_INVOKE_BGMV(nv_bfloat16, nv_half, nv_bfloat16)
INSTANTIATE_INVOKE_BGMV(float, nv_half, nv_half)
INSTANTIATE_INVOKE_BGMV(float, nv_half, nv_bfloat16)
INSTANTIATE_INVOKE_BGMV(float, nv_bfloat16, nv_bfloat16)
INSTANTIATE_INVOKE_BGMV(float, nv_bfloat16, nv_half)
INSTANTIATE_INVOKE_BGMV(nv_half, float, nv_half)
INSTANTIATE_INVOKE_BGMV(nv_half, float, nv_bfloat16)
INSTANTIATE_INVOKE_BGMV(nv_bfloat16, float, nv_bfloat16)
INSTANTIATE_INVOKE_BGMV(nv_bfloat16, float, nv_half)
INSTANTIATE_INVOKE_BGMV(float, float, nv_half)
INSTANTIATE_INVOKE_BGMV(float, float, nv_bfloat16)

#undef INSTANTIATE_INVOKE_BGMV

}  // namespace nvidia
}  // namespace llm_kernels