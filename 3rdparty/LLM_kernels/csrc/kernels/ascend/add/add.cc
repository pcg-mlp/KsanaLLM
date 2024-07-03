/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */
#include <vector>

#include "3rdparty/half/include/half.hpp"
#include "acl/acl.h"
#include "acl/acl_op_compiler.h"
#include "aclnn/acl_meta.h"

#include "aclrtlaunch_InvokeAddFloatKernel.h"
#include "aclrtlaunch_InvokeAddHalfKernel.h"
#include "csrc/kernels/ascend/add/add_tiling.h"
#include "csrc/utils/ascend/common.h"

namespace llm_kernels {
namespace ascend {

template <typename DTYPE>
void InvokeAdd(DTYPE alpha, DTYPE* input_a, DTYPE* input_b, DTYPE* bias, DTYPE* output, uint32_t hidden_units_num,
               uint32_t total_seq_len, aclrtStream& stream, void (*ws_func)(size_t, void**)) {
  AddTilingConfig tiling_config;
  tiling_config.total_elem_num = total_seq_len * hidden_units_num;
  tiling_config.block_elem_num = hidden_units_num;
  if (std::is_same<DTYPE, float>::value) {
    tiling_config.alpha = alpha;
  } else if (std::is_same<DTYPE, aclFloat16>::value) {
    aclFloat16* alpha_buf = (aclFloat16*)(&tiling_config.alpha);
    alpha_buf[0] = aclFloatToFloat16(alpha);
  } else if (std::is_same<DTYPE, half_float::half>::value) {
    half_float::half* alpha_buf = (half_float::half*)(&tiling_config.alpha);
    alpha_buf[0] = half_float::half(alpha);
  } else {
    throw std::invalid_argument("Invalid add type type, only support float16 or float32.");
  }
  AddTilingConfig* buf = &tiling_config;
  void* tiling_device = nullptr;
  ws_func(sizeof(AddTilingConfig), &tiling_device);
  ACL_CHECK_RET(aclrtMemcpyAsync(tiling_device, sizeof(AddTilingConfig), (void*)buf, sizeof(AddTilingConfig),
                                 ACL_MEMCPY_HOST_TO_DEVICE, stream));
  ACL_CHECK_RET(aclrtSynchronizeStream(stream));

  if (std::is_same<DTYPE, float>::value) {
    ACL_CHECK_RET(ACLRT_LAUNCH_KERNEL(InvokeAddFloatKernel)(
        total_seq_len, stream, reinterpret_cast<uint8_t*>(input_a), reinterpret_cast<uint8_t*>(input_b),
        reinterpret_cast<uint8_t*>(bias), reinterpret_cast<uint8_t*>(output), tiling_device));
  } else if (std::is_same<DTYPE, aclFloat16>::value) {
    ACL_CHECK_RET(ACLRT_LAUNCH_KERNEL(InvokeAddHalfKernel)(
        total_seq_len, stream, reinterpret_cast<uint8_t*>(input_a), reinterpret_cast<uint8_t*>(input_b),
        reinterpret_cast<uint8_t*>(bias), reinterpret_cast<uint8_t*>(output), tiling_device));
  } else {
    throw std::invalid_argument("Invalid add compute type in InvokeAdd, only support float16 or float32.");
  }
}

template void InvokeAdd(float alpha, float* input_a, float* input_b, float* bias, float* output,
                        uint32_t hidden_units_num, uint32_t total_seq_len, aclrtStream& stream,
                        void (*ws_func)(size_t, void**));

template void InvokeAdd(aclFloat16 alpha, aclFloat16* input_a, aclFloat16* input_b, aclFloat16* bias,
                        aclFloat16* output, uint32_t hidden_units_num, uint32_t total_seq_len, aclrtStream& stream,
                        void (*ws_func)(size_t, void**));

}  // namespace ascend
}  // namespace llm_kernels