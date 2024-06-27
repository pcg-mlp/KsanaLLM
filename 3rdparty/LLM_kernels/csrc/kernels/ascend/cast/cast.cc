/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include "csrc/kernels/ascend/cast/cast.h"

#include "aclrtlaunch_InvokeCastFloatToHalfKernel.h"
#include "aclrtlaunch_InvokeCastHalfToFloatKernel.h"
#include "csrc/kernels/ascend/cast/cast_tiling.h"
#include "csrc/utils/ascend/common.h"

namespace llm_kernels {
namespace ascend {

template <typename SRC_DTYPE, typename DST_DTYPE>
void InvokeCast(SRC_DTYPE* input, DST_DTYPE* output, uint32_t seq_len, uint32_t hidden_units_num, aclrtStream& stream,
                void (*ws_func)(size_t, void**)) {
  if (std::is_same<SRC_DTYPE, DST_DTYPE>::value) {
    return;
  }
  CastTilingConfig tiling;
  tiling.total_elem_num = seq_len * hidden_units_num;
  tiling.block_elem_num = hidden_units_num;
  tiling.tile_num = CAST_TILE_NUM;
  CastTilingConfig* buf = &tiling;
  void* tiling_device = nullptr;
  ws_func(sizeof(CastTilingConfig), &tiling_device);
  if (std::is_same<SRC_DTYPE, DST_DTYPE>::value) {
    ACL_CHECK_RET(aclrtMemcpyAsync(output, tiling.total_elem_num * sizeof(DST_DTYPE), input,
                                   tiling.total_elem_num * sizeof(SRC_DTYPE), ACL_MEMCPY_DEVICE_TO_DEVICE, stream));
    return;
  }
  ACL_CHECK_RET(aclrtMemcpyAsync(tiling_device, sizeof(CastTilingConfig), (void*)buf, sizeof(CastTilingConfig),
                                 ACL_MEMCPY_HOST_TO_DEVICE, stream));
  ACL_CHECK_RET(aclrtSynchronizeStream(stream));
  if (std::is_same<SRC_DTYPE, aclFloat16>::value && std::is_same<DST_DTYPE, float>::value) {
    ACL_CHECK_RET(ACLRT_LAUNCH_KERNEL(InvokeCastHalfToFloatKernel)(seq_len, stream, (uint8_t*)input, (uint8_t*)output,
                                                                   tiling_device));
  } else if (std::is_same<SRC_DTYPE, float>::value && std::is_same<DST_DTYPE, aclFloat16>::value) {
    ACL_CHECK_RET(ACLRT_LAUNCH_KERNEL(InvokeCastFloatToHalfKernel)(seq_len, stream, (uint8_t*)input, (uint8_t*)output,
                                                                   tiling_device));
  } else {
    throw std::invalid_argument(
        "Invalid cast compute type in InvokeCast, only support float16 to float32 or float32 to float16.");
  }
}

template void InvokeCast(aclFloat16* input, float* output, uint32_t seq_len, uint32_t hidden_units_num,
                         aclrtStream& stream, void (*ws_func)(size_t, void**));
template void InvokeCast(float* input, aclFloat16* output, uint32_t seq_len, uint32_t hidden_units_num,
                         aclrtStream& stream, void (*ws_func)(size_t, void**));
template void InvokeCast(aclFloat16* input, aclFloat16* output, uint32_t seq_len, uint32_t hidden_units_num,
                         aclrtStream& stream, void (*ws_func)(size_t, void**));
template void InvokeCast(float* input, float* output, uint32_t seq_len, uint32_t hidden_units_num, aclrtStream& stream,
                         void (*ws_func)(size_t, void**));

}  // namespace ascend
}  // namespace llm_kernels
