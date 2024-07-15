/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include "csrc/kernels/ascend/assemble_last_token/assemble_last_token.h"
#include <type_traits>

#include "aclrtlaunch_InvokeAssembleLastTokenFloatKernel.h"
#include "aclrtlaunch_InvokeAssembleLastTokenHalfKernel.h"
#include "csrc/kernels/ascend/assemble_last_token/assemble_last_token_tiling.h"
#include "csrc/utils/ascend/common.h"
#include "csrc/utils/ascend/tiling_data_types.h"

namespace llm_kernels {
namespace ascend {

template <typename DTYPE>
void InvokeAssembleLastToken(DTYPE* input, size_t* ids_offsets, size_t* prefix_offsets, const int32_t batch_size,
                             const int32_t hidden_units_num, DTYPE* output, aclrtStream& stream,
                             void (*ws_func)(size_t, void**)) {
  AssembleLastTokenTiling assemble_last_token_tiling;
  assemble_last_token_tiling.batch_size = batch_size;
  assemble_last_token_tiling.hidden_units_num = hidden_units_num;
  // TODO(karlluo): will optimize according input size
  assemble_last_token_tiling.tile_num = 1;
  AssembleLastTokenTiling* buf = &assemble_last_token_tiling;
  void* tiling_device = nullptr;
  ws_func(sizeof(AssembleLastTokenTiling), &tiling_device);
  ACL_CHECK_RET(aclrtMemcpyAsync(tiling_device, sizeof(AssembleLastTokenTiling), (void*)buf,
                                 sizeof(AssembleLastTokenTiling), ACL_MEMCPY_HOST_TO_DEVICE, stream));
  ACL_CHECK_RET(aclrtSynchronizeStream(stream));
  if (std::is_same<DTYPE, aclFloat16>::value) {
    ACL_CHECK_RET(ACLRT_LAUNCH_KERNEL(InvokeAssembleLastTokenHalfKernel)(
        batch_size, stream, (uint8_t*)input, (uint8_t*)ids_offsets, (uint8_t*)output, tiling_device));
  } else if (std::is_same<DTYPE, float>::value) {
    ACL_CHECK_RET(ACLRT_LAUNCH_KERNEL(InvokeAssembleLastTokenFloatKernel)(
        batch_size, stream, (uint8_t*)input, (uint8_t*)ids_offsets, (uint8_t*)output, tiling_device));
  } else {
    throw std::invalid_argument(
        "Invalid assemble last token compute type in InvokeAssembleLastToken, only support float16 or float32.");
  }
}

template void InvokeAssembleLastToken(float* input, size_t* ids_offsets, size_t* prefix_offsets,
                                      const int32_t batch_size, const int32_t hidden_units_num, float* output,
                                      aclrtStream& stream, void (*ws_func)(size_t, void**));

template void InvokeAssembleLastToken(aclFloat16* input, size_t* ids_offsets, size_t* prefix_offsets,
                                      const int32_t batch_size, const int32_t hidden_units_num, aclFloat16* output,
                                      aclrtStream& stream, void (*ws_func)(size_t, void**));

}  // namespace ascend
}  // namespace llm_kernels