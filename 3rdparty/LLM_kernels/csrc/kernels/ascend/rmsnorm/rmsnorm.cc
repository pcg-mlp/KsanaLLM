/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include "csrc/kernels/ascend/cast/cast.h"

#include "aclrtlaunch_InvokeRmsNormFloatKernel.h"
#include "aclrtlaunch_InvokeRmsNormHalfKernel.h"
#include "csrc/kernels/ascend/rmsnorm/rmsnorm_tiling.h"
#include "csrc/utils/ascend/common.h"

namespace llm_kernels {
namespace ascend {

template <typename DTYPE>
void InvokeRmsLayerNorm(DTYPE* out, DTYPE* input, DTYPE* gamma, DTYPE* beta, const float layernorm_eps, const int32_t m,
                        const int32_t n, aclrtStream& stream, void (*ws_func)(size_t, void**)) {
  // NOTE(karlluo): m is seq length, n is hidden units number
  RmsNormTilingConfig tiling;
  // for continue batching, we dont need batch size
  tiling.bLength = 1;
  tiling.sLength = m;
  tiling.hLength = n;
  tiling.originalHLength = n;
  tiling.reciprocalOfHLength = float(1.0f) / float(n);
  // TODO(karlluo): how many elements for each reduce sum handle, if n is too large to load to UB, we need split it to
  // multiple loop round.
  tiling.loopRound = 1;
  // NOTE(karlluo): relate to x² fp32 buffer size
  tiling.mainBshLength = n;
  // NOTE(karlluo): relate to reduce sum buffer size
  tiling.mainBsLengthAlign = n;
  tiling.eps = layernorm_eps;
  tiling.mainBsLength = 1;

  RmsNormTilingConfig* buf = &tiling;
  size_t tiling_size = sizeof(RmsNormTilingConfig);
  void* tiling_device;
  ws_func(tiling_size, &tiling_device);
  ACL_CHECK_RET(
      aclrtMemcpyAsync(tiling_device, tiling_size, (void*)buf, tiling_size, ACL_MEMCPY_HOST_TO_DEVICE, stream));
  ACL_CHECK_RET(aclrtSynchronizeStream(stream));

  if (std::is_same<DTYPE, aclFloat16>::value) {
    ACL_CHECK_RET(ACLRT_LAUNCH_KERNEL(InvokeRmsNormHalfKernel)(m, stream, (uint8_t*)input, (uint8_t*)gamma,
                                                               (uint8_t*)out, (uint8_t*)tiling_device));
  } else if (std::is_same<DTYPE, float>::value) {
    ACL_CHECK_RET(ACLRT_LAUNCH_KERNEL(InvokeRmsNormFloatKernel)(m, stream, (uint8_t*)input, (uint8_t*)gamma,
                                                                (uint8_t*)out, (uint8_t*)tiling_device));
  } else {
    throw std::invalid_argument("Invalid rms norm compute type, only support float16 or float32.");
  }
}

template void InvokeRmsLayerNorm(aclFloat16* out, aclFloat16* input, aclFloat16* gamma, aclFloat16* beta,
                                 const float layernorm_eps, const int32_t m, const int32_t n, aclrtStream& stream,
                                 void (*ws_func)(size_t, void**));
template void InvokeRmsLayerNorm(float* out, float* input, float* gamma, float* beta, const float layernorm_eps,
                                 const int32_t m, const int32_t n, aclrtStream& stream,
                                 void (*ws_func)(size_t, void**));

}  // namespace ascend
}  // namespace llm_kernels