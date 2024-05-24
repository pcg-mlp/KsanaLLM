/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include "slice.h"

#include "aclnnop/aclnn_slice.h"
#include "csrc/utils/ascend/common.h"

#include "aclrtlaunch_InvokeSliceKernel.h"

#include "csrc/kernels/ascend/slice/slice_tiling.h"

namespace llm_kernels {
namespace ascend {

// The max block size of slice.
constexpr uint32_t MAX_SLICE_BLOCK_SIZE = 16 * 16;

// The max used ai core number.
constexpr uint32_t MAX_USED_CORE_NUM = 24;

void Slice(const aclTensor* input, const int sliceDim, const int sliceStart, const int sliceEnd, const int sliceStep,
           aclTensor** output, aclrtStream& stream, void (*ws_func)(size_t, void**)) {
  uint64_t ws_size = 0ull;
  void* workspace = nullptr;
  aclOpExecutor* executor = nullptr;
  ACL_CHECK_RET(
      aclnnSliceGetWorkspaceSize(input, sliceDim, sliceStart, sliceEnd, sliceStep, *output, &ws_size, &executor));
  ws_func(ws_size, &workspace);
  ACL_CHECK_RET(aclnnSlice(workspace, ws_size, executor, stream));
}

Slice2::Slice2() {
  tiling_size_ = sizeof(SliceTilingData);
  ACL_CHECK_RET(aclrtMalloc(&tiling_buffer_gm_, tiling_size_, ACL_MEM_MALLOC_HUGE_FIRST));

  // TODO: Get block num from device info.
  tiling_data_.used_core_num = 24;

  size_t usr_workspace_size = 4 * 1024;
  size_t sys_workspace_size = 16 * 1024 * 1024;
  ACL_CHECK_RET(aclrtMalloc(&workspace_gm_, usr_workspace_size + sys_workspace_size, ACL_MEM_MALLOC_HUGE_FIRST));
}

Slice2::~Slice2() {
  ACL_CHECK_RET(aclrtFree(tiling_buffer_gm_));
  ACL_CHECK_RET(aclrtFree(workspace_gm_));
}

void Slice2::CopyTilingToDevice(aclrtStream stream) {
  ACL_CHECK_RET(aclrtMemcpyAsync(tiling_buffer_gm_, tiling_size_, &tiling_data_, tiling_size_,
                                 aclrtMemcpyKind::ACL_MEMCPY_HOST_TO_DEVICE, stream));
  ACL_CHECK_RET(aclrtSynchronizeStream(stream));
}

void Slice2::Forward(void* output, void* input, int start, int length, int step, int times, aclrtStream stream) {
  tiling_data_.start = start;
  tiling_data_.length = length;
  tiling_data_.step = step;
  tiling_data_.times = times;

  if (length < MAX_SLICE_BLOCK_SIZE) {
    tiling_data_.block_size = length;
    tiling_data_.tail_block_size = length;
    tiling_data_.step_block_num = 1;
  } else {
    tiling_data_.block_size = MAX_SLICE_BLOCK_SIZE;
    tiling_data_.tail_block_size = length % MAX_SLICE_BLOCK_SIZE;
    tiling_data_.step_block_num = (length + MAX_SLICE_BLOCK_SIZE - 1) / MAX_SLICE_BLOCK_SIZE;
  }

  if (tiling_data_.times * tiling_data_.step_block_num < MAX_USED_CORE_NUM) {
    tiling_data_.used_core_num = tiling_data_.times * tiling_data_.step_block_num;
  } else {
    tiling_data_.used_core_num = MAX_USED_CORE_NUM;
  }

  CopyTilingToDevice(stream);

  ACLRT_LAUNCH_KERNEL(InvokeSliceKernel)
  (tiling_data_.used_core_num, stream, input, output, workspace_gm_, tiling_buffer_gm_);
}

}  // namespace ascend
}  // namespace llm_kernels
