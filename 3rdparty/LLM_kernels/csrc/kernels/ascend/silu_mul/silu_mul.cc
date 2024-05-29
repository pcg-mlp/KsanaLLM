/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include "csrc/kernels/ascend/silu_mul/silu_mul.h"
#include "csrc/kernels/ascend/silu_mul/silu_mul_kernel.h"

#include "aclrtlaunch_InvokeSiluMulFloatKernel.h"
#include "aclrtlaunch_InvokeSiluMulHalfKernel.h"

namespace llm_kernels {
namespace ascend {

template <typename T>
void InvokeSiluMul(T* input, T* weight, const size_t m, const size_t n, T* output, aclrtStream stream,
                   llm_kernels::utils::WorkSpaceFunc ws_func) {
  uint32_t total_elem_num = m * n;
  uint32_t block_elem_num = n;
  uint32_t dim_num = total_elem_num / block_elem_num;
  constexpr uint32_t tile_num = 1;

  SiluMulTilingConfig tiling;
  tiling.total_elem_num = total_elem_num;
  tiling.block_elem_num = block_elem_num;
  tiling.tile_num = tile_num;
  SiluMulTilingConfig* buf = &tiling;
  size_t tiling_size = sizeof(SiluMulTilingConfig);
  void* workspace_ptr = nullptr;
  ws_func(tiling_size, &workspace_ptr);
  uint8_t* tiling_device = (uint8_t*)workspace_ptr;
  ACL_CHECK_RET(aclrtMemcpy(tiling_device, tiling_size, (void*)buf, tiling_size, ACL_MEMCPY_HOST_TO_DEVICE));
  if (std::is_same<T, aclFloat16>::value) {
    ACL_CHECK_RET(ACLRT_LAUNCH_KERNEL(InvokeSiluMulHalfKernel)(dim_num, stream, input, weight, output, tiling_device));
  } else if (std::is_same<T, float>::value) {
    ACL_CHECK_RET(ACLRT_LAUNCH_KERNEL(InvokeSiluMulFloatKernel)(dim_num, stream, input, weight, output, tiling_device));
  } else {
    throw std::invalid_argument("Unsupported inference type in SiluMul.");
  }
  ACL_CHECK_RET(aclrtSynchronizeStream(stream));
}

template void InvokeSiluMul(aclFloat16* input, aclFloat16* weight, const size_t m, const size_t n, aclFloat16* output,
                            aclrtStream stream, llm_kernels::utils::WorkSpaceFunc ws_func);

template void InvokeSiluMul(float* input, float* weight, const size_t m, const size_t n, float* output,
                            aclrtStream stream, llm_kernels::utils::WorkSpaceFunc ws_func);

}  // namespace ascend
}  // namespace llm_kernels