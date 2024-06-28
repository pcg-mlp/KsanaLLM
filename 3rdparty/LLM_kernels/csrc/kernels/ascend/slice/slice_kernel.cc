/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"

#include "csrc/kernels/ascend/slice/slice_tiling.h"

using namespace AscendC;
using namespace llm_kernels::ascend;

using matmul::Matmul;
using matmul::MatmulType;

constexpr int32_t BUFFER_NUM = 1;  // tensor num for each queue

__aicore__ inline void CopyTiling(SliceTilingData* tiling, GM_ADDR tiling_gm) {
  uint32_t* tiling_ptr = reinterpret_cast<uint32_t*>(tiling);
  __gm__ uint32_t* tiling_gm_ptr = reinterpret_cast<__gm__ uint32_t*>(tiling_gm);

  for (int i = 0; i < sizeof(SliceTilingData) / sizeof(uint32_t); i++, tiling_ptr++) {
    *tiling_ptr = *(tiling_gm_ptr + i);
  }
}

// A implement of paged attention.
class SliceKernel {
 public:
  __aicore__ inline SliceKernel() {}

  __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, SliceTilingData* tiling, TPipe* pipe_ptr);

  __aicore__ inline void Process();

 private:
  // The three stage process.
  __aicore__ inline void CopyIn(int32_t loop_idx, uint32_t src_offset, uint16_t src_size);
  __aicore__ inline void CopyOut(int32_t loop_idx, uint32_t dst_offset, uint16_t dst_size);

  // The memory pipe.
  TPipe* pipe_;

  // The tiling config.
  SliceTilingData* tiling_;

  GlobalTensor<int32_t> input_gm_;
  GlobalTensor<int32_t> output_gm_;

  // The input and output queue.
  TQue<QuePosition::VECIN, BUFFER_NUM> input_queue_;

  int32_t block_idx_ = 0;
  int32_t block_dim_ = 0;

  uint32_t padded_block_size_ = 0;
};

__aicore__ void SliceKernel::Init(GM_ADDR input, GM_ADDR output, SliceTilingData* tiling, TPipe* pipe_ptr) {
  pipe_ = pipe_ptr;
  tiling_ = tiling;

  block_idx_ = GetBlockIdx();
  block_dim_ = tiling_->used_core_num;

  input_gm_.SetGlobalBuffer((__gm__ int32_t*)input);
  output_gm_.SetGlobalBuffer((__gm__ int32_t*)output);

  // Pad block size to 32 bytes.
  padded_block_size_ = ((tiling_->block_size + (32) - 1) & ~((32) - 1));

  pipe_->InitBuffer(input_queue_, BUFFER_NUM, padded_block_size_);
}

__aicore__ void SliceKernel::Process() {
  int32_t loop_num = 0;
  if (block_idx_ < (tiling_->times * tiling_->step_block_num) % block_dim_) {
    loop_num = (tiling_->times * tiling_->step_block_num) / block_dim_ + 1;
  } else {
    loop_num = (tiling_->times * tiling_->step_block_num) / block_dim_;
  }

  for (int32_t loop_idx = 0; loop_idx < loop_num; ++loop_idx) {
    uint32_t total_block_idx = loop_idx * block_dim_ + block_idx_;
    uint32_t step_block_idx = total_block_idx % tiling_->step_block_num;

    uint32_t step_idx = total_block_idx / tiling_->step_block_num;

    uint16_t src_size = 0;
    uint32_t src_offset = 0;

    uint16_t dst_size = 0;
    uint32_t dst_offset = 0;

    if (step_block_idx == tiling_->step_block_num - 1) {
      src_offset = tiling_->start + (step_idx * tiling_->step) + (step_block_idx * tiling_->block_size);
      src_size = tiling_->tail_block_size;

      dst_offset = (step_idx * tiling_->length) + (step_block_idx * tiling_->block_size);
      dst_size = tiling_->tail_block_size;
    } else {
      src_offset = tiling_->start + (step_idx * tiling_->step) + (step_block_idx * tiling_->block_size);
      src_size = tiling_->block_size;

      dst_offset = (step_idx * tiling_->length) + (step_block_idx * tiling_->block_size);
      dst_size = tiling_->block_size;
    }

    CopyIn(loop_idx, src_offset, src_size);
    CopyOut(loop_idx, dst_offset, dst_size);
  }
}

__aicore__ void SliceKernel::CopyIn(int32_t loop_idx, uint32_t src_offset, uint16_t src_size) {
  LocalTensor<int32_t> local_tensor = input_queue_.AllocTensor<int32_t>();

  src_offset /= sizeof(int32_t);

  DataCopyPadParams pad_params;
  DataCopyParams copy_params{1, src_size, 0, 0};
  DataCopyPad(local_tensor, input_gm_[src_offset], copy_params, pad_params);
  pipe_barrier(PIPE_ALL);

  input_queue_.EnQue(local_tensor);
}

__aicore__ void SliceKernel::CopyOut(int32_t loop_idx, uint32_t dst_offset, uint16_t dst_size) {
  dst_offset /= sizeof(int32_t);

  DataCopyParams copy_params{1, dst_size, 0, 0};
  LocalTensor<int32_t> local_tensor = input_queue_.DeQue<int32_t>();
  DataCopyPad(output_gm_[dst_offset], local_tensor, copy_params);
  pipe_barrier(PIPE_ALL);

  input_queue_.FreeTensor(local_tensor);
}

extern "C" __global__ __aicore__ void InvokeSliceKernel(GM_ADDR input, GM_ADDR output, 
                                                        GM_ADDR __restrict tiling_gm) {
  TPipe pipe;

  SliceTilingData tiling;
  CopyTiling(&tiling, tiling_gm);
  if (GetBlockIdx() >= tiling.used_core_num) {
    return;
  }

  SliceKernel op;
  op.Init(input, output, &tiling, &pipe);
  op.Process();
}
