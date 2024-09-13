/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include "argmax.h"

#include "aclnnop/aclnn_argmax.h"
#include "aclrtlaunch_InvokeArgmaxFloatKernel.h"
#include "aclrtlaunch_InvokeArgmaxHalfKernel.h"
#include "atb/infer_op_params.h"
#include "csrc/kernels/ascend/argmax/argmax_kernel.h"
#include "csrc/utils/ascend/common.h"

namespace llm_kernels {
namespace ascend {

template <typename T>
void InvokeArgmax(const T* input, const uint32_t* ids_offset, const int32_t batch_size, const int32_t vocab_size,
                  uint32_t* result, aclrtStream& stream, void (*ws_func)(size_t, void**)) {
  ArgmaxConfigTiling argmax_config_tiling;
  argmax_config_tiling.batch_size = batch_size;
  argmax_config_tiling.vocab_size = vocab_size;
  // TODO(karlluo): load ub size from config and config tile num
  argmax_config_tiling.tile_num = 4;
  argmax_config_tiling.block_handle_num =
      (argmax_config_tiling.batch_size + ARGMAX_SINGLE_BLOCK_CAPACITY - 1) / ARGMAX_SINGLE_BLOCK_CAPACITY;
  ArgmaxConfigTiling* buf = &argmax_config_tiling;
  size_t tiling_size = sizeof(ArgmaxConfigTiling);
  void* tiling_device = nullptr;
  ws_func(tiling_size, &tiling_device);

  ACL_CHECK_RET(
      aclrtMemcpyAsync(tiling_device, tiling_size, (void*)buf, tiling_size, ACL_MEMCPY_HOST_TO_DEVICE, stream));
  ACL_CHECK_RET(aclrtSynchronizeStream(stream));

  if (std::is_same<T, aclFloat16>::value) {
    ACL_CHECK_RET(ACLRT_LAUNCH_KERNEL(InvokeArgmaxHalfKernel)(argmax_config_tiling.block_handle_num, stream,
                                                              (uint8_t*)(input), result,
                                                              reinterpret_cast<uint8_t*>(tiling_device)));
  } else if (std::is_same<T, float>::value) {
    ACL_CHECK_RET(ACLRT_LAUNCH_KERNEL(InvokeArgmaxFloatKernel)(argmax_config_tiling.block_handle_num, stream,
                                                               (uint8_t*)(input), result,
                                                               reinterpret_cast<uint8_t*>(tiling_device)));
  } else {
    throw std::invalid_argument("Invalid argmax data type, only support float16 or float32.");
  }
}

template void InvokeArgmax(const float* input, const uint32_t* ids_offset, const int32_t batch_size,
                           const int32_t vocab_size, uint32_t* result, aclrtStream& stream,
                           void (*ws_func)(size_t, void**));

template void InvokeArgmax(const aclFloat16* input, const uint32_t* ids_offset, const int32_t batch_size,
                           const int32_t vocab_size, uint32_t* result, aclrtStream& stream,
                           void (*ws_func)(size_t, void**));

}  // namespace ascend
}  // namespace llm_kernels
