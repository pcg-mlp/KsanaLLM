/**
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include "argmax_kernel.h"

#include "interface/kernel_common.h"
#include "interface/kernel_type.h"
#include "kernel_operator.h"
using namespace AscendC;
using namespace llm_kernels::ascend;

constexpr int ARGMAX_BUFFER_NUM = 1;

constexpr float FLOAT_MIN = 1.175494351e-38;

__aicore__ inline void CopyTiling(llm_kernels::ascend::ArgmaxConfigTiling* tiling, GM_ADDR tiling_gm) {
  uint32_t* tiling_ptr = reinterpret_cast<uint32_t*>(tiling);
  __gm__ uint32_t* tiling_gm_ptr = reinterpret_cast<__gm__ uint32_t*>(tiling_gm);

  for (int i = 0; i < sizeof(llm_kernels::ascend::ArgmaxConfigTiling) / sizeof(uint32_t); i++, tiling_ptr++) {
    *tiling_ptr = *(tiling_gm_ptr + i);
  }
}

template <typename DTYPE>
class ArgmaxKernel {
 public:
  __aicore__ inline ArgmaxKernel() {}
  __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, GM_ADDR tiling_gm);
  __aicore__ inline void Process();

 private:
  __aicore__ inline void CopyIn(uint64_t current_batch_offset, uint32_t loop_idx);
  __aicore__ inline void ArgmaxCompute(float& max_value, uint32_t& max_index, uint32_t loop_idx);

  llm_kernels::ascend::ArgmaxConfigTiling tiling_;
  GlobalTensor<DTYPE> input_gm_;
  GlobalTensor<uint32_t> output_gm_;
  TPipe pipe_;

  // create queues for input, in this case depth is equal to buffer num
  TQue<QuePosition::VECIN, ARGMAX_BUFFER_NUM> input_queue_;
  // create queue for output, in this case depth is equal to buffer num
  TQue<QuePosition::VECOUT, ARGMAX_BUFFER_NUM> output_queue_;

  TBuf<QuePosition::LCM> workspace_buf_;
  TBuf<QuePosition::LCM> output_buf_;

  // for input tensor with shape [batch_size, vocab]
  uint32_t tile_elem_num_;
  uint32_t vocab_elems_data_size_;
  // handle_capacity for output
  uint32_t current_block_handle_capacity_;
};

template <typename DTYPE>
__aicore__ inline void ArgmaxKernel<DTYPE>::Init(GM_ADDR input, GM_ADDR output, GM_ADDR tiling_gm) {
  CopyTiling(&tiling_, tiling_gm);

  current_block_handle_capacity_ = ARGMAX_SINGLE_BLOCK_CAPACITY;
  // compute the tail value in last block
  if (GetBlockIdx() == GetBlockNum() - 1) {
    current_block_handle_capacity_ = tiling_.batch_size - GetBlockIdx() * ARGMAX_SINGLE_BLOCK_CAPACITY;
  }

  // TODO(karlluo): supported can not divide up
  tile_elem_num_ = tiling_.vocab_size / ARGMAX_BUFFER_NUM / tiling_.tile_num;
  vocab_elems_data_size_ = tiling_.vocab_size * sizeof(DTYPE);

  input_gm_.SetGlobalBuffer((__gm__ DTYPE*)input);
  output_gm_.SetGlobalBuffer((__gm__ uint32_t*)output);

  pipe_.InitBuffer(input_queue_, ARGMAX_BUFFER_NUM, tile_elem_num_ * sizeof(DTYPE));
  // only one output is enought
  pipe_.InitBuffer(output_queue_, ARGMAX_BUFFER_NUM, ARGMAX_SINGLE_BLOCK_CAPACITY * sizeof(uint32_t));
  // NOTE(karlluo): for ascend not support half operator in aicore, we have to cast all input into float, and
  // workspace_buf is for this buffer
  pipe_.InitBuffer(workspace_buf_, tile_elem_num_ * sizeof(float));
}

template <typename DTYPE>
__aicore__ inline void ArgmaxKernel<DTYPE>::CopyIn(uint64_t current_batch_offset, uint32_t loop_idx) {
  // alloc tensor from queue memory
  LocalTensor<DTYPE> input_local = input_queue_.AllocTensor<DTYPE>();
  // copy progress_th tile from global tensor to local tensor
  DataCopy(input_local, input_gm_[current_batch_offset + loop_idx * tile_elem_num_], tile_elem_num_);
  // enque input tensors to VECIN queue
  input_queue_.EnQue(input_local);
}

template <typename DTYPE>
__aicore__ inline void ArgmaxKernel<DTYPE>::ArgmaxCompute(float& max_value, uint32_t& max_index, uint32_t loop_idx) {
  // deque input tensors from VECIN queue
  LocalTensor<DTYPE> input_local = input_queue_.DeQue<DTYPE>();
  LocalTensor<float> workspace_local = workspace_buf_.Get<float>();
  __ubuf__ float* candidates_ptr = nullptr;

  // NOTE(karlluo): for aicore can not handle fp16 single operation. we have to cast it to float.
  if (std::is_same<half, DTYPE>::value) {
    Cast<float, DTYPE>(workspace_local, input_local, RoundMode::CAST_NONE, tile_elem_num_);
    pipe_barrier(PIPE_V);
  }

  for (uint32_t idx = 0; idx < tile_elem_num_; ++idx) {
    // float val = candidates_ptr[idx];
    float val = FLOAT_MIN;
    if (std::is_same<half, DTYPE>::value) {
      val = workspace_local.GetValue(idx);
    } else if (std::is_same<float, DTYPE>::value) {
      val = input_local.GetValue(idx);
    }
    if (val > max_value) {
      max_index = loop_idx * tile_elem_num_ + idx;
      max_value = val;
    }
  }
  // free input tensors for reuse
  input_queue_.FreeTensor(input_local);
}

template <typename DTYPE>
__aicore__ inline void ArgmaxKernel<DTYPE>::Process() {
  // other block skip running
  if (GetBlockIdx() >= tiling_.batch_size) {
    return;
  }
  // output_local shape: [ARGMAX_SINGLE_BLOCK_CAPACITY] of uint32_t but we only use current_block_handle_capacity_
  LocalTensor<uint32_t> output_local = output_queue_.AllocTensor<uint32_t>();
  // loop count need to be doubled, due to double buffer
  int32_t loop_count = tiling_.tile_num * ARGMAX_BUFFER_NUM;
  for (uint32_t batch_idx = 0; batch_idx < current_block_handle_capacity_; ++batch_idx) {
    uint64_t current_batch_offset =
        GetBlockIdx() * ARGMAX_SINGLE_BLOCK_CAPACITY * tiling_.vocab_size + batch_idx * tiling_.vocab_size;
    float max_value = FLOAT_MIN;
    uint32_t max_index = 0;

    // tiling strategy, pipeline parallel
    for (int32_t loop_idx = 0; loop_idx < loop_count; loop_idx++) {
      CopyIn(current_batch_offset, loop_idx);
      ArgmaxCompute(max_value, max_index, loop_idx);
    }
    output_local.SetValue(batch_idx, max_index);
  }
  output_queue_.EnQue(output_local);
  output_local = output_queue_.DeQue<uint32_t>();
  DataCopy(output_gm_[GetBlockIdx() * ARGMAX_SINGLE_BLOCK_CAPACITY], output_local, ARGMAX_SINGLE_BLOCK_CAPACITY);
  output_queue_.FreeTensor(output_local);
}

extern "C" __global__ __aicore__ void InvokeArgmaxHalfKernel(GM_ADDR input, GM_ADDR output, GM_ADDR tiling_gm) {
  ArgmaxKernel<half> argmax_kernel;
  argmax_kernel.Init(input, output, tiling_gm);
  argmax_kernel.Process();
}

extern "C" __global__ __aicore__ void InvokeArgmaxFloatKernel(GM_ADDR input, GM_ADDR output, GM_ADDR tiling_gm) {
  ArgmaxKernel<float> argmax_kernel;
  argmax_kernel.Init(input, output, tiling_gm);
  argmax_kernel.Process();
}