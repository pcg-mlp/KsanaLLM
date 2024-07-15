/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

#include "csrc/kernels/ascend/assemble_last_token/assemble_last_token_tiling.h"

using namespace AscendC;
using namespace llm_kernels::ascend;

constexpr int32_t ASSEMBLE_LAST_TOKEN_BUFFER_NUM = 1;  // tensor num for each queue

__aicore__ inline void CopyTiling(AssembleLastTokenTiling* tiling, GM_ADDR tiling_gm) {
  uint32_t* tiling_ptr = reinterpret_cast<uint32_t*>(tiling);
  __gm__ uint32_t* tiling_gm_ptr = reinterpret_cast<__gm__ uint32_t*>(tiling_gm);

  for (int i = 0; i < sizeof(AssembleLastTokenTiling) / sizeof(uint32_t); i++, tiling_ptr++) {
    *tiling_ptr = *(tiling_gm_ptr + i);
  }
}

template <typename T>
class AssembleLastTokenKernel {
 public:
  __aicore__ inline AssembleLastTokenKernel() {}

  __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, GM_ADDR ids_offsets, GM_ADDR tiling);

  __aicore__ inline void Process();

 private:
  // The in/out stage process.
  __aicore__ inline void CopyIn(int32_t src_idx);
  __aicore__ inline void CopyOut(int32_t dst_idx);

  // The tiling config.
  AssembleLastTokenTiling* tiling_;

  TPipe pipe_;
  GlobalTensor<T> input_gm_;
  GlobalTensor<T> output_gm_;
  GlobalTensor<uint64_t> ids_offsets_gm_;

  // The input and output queue.
  TQue<QuePosition::VECIN, ASSEMBLE_LAST_TOKEN_BUFFER_NUM> input_queue_;

  int32_t block_idx_ = 0;
  int32_t block_dim_ = 0;
  uint32_t tile_elem_num_ = 0;
  uint32_t input_offset_ = 0;
};

template <typename T>
__aicore__ void AssembleLastTokenKernel<T>::Init(GM_ADDR input, GM_ADDR output, GM_ADDR ids_offsets, GM_ADDR tiling) {
  CopyTiling(tiling_, tiling);
  tile_elem_num_ = tiling_->hidden_units_num / ASSEMBLE_LAST_TOKEN_BUFFER_NUM / tiling_->tile_num;
  block_idx_ = GetBlockIdx();
  block_dim_ = GetBlockNum();
  uint64_t block_data_size = tiling_->hidden_units_num * sizeof(T);
  ids_offsets_gm_.SetGlobalBuffer((__gm__ uint64_t*)ids_offsets);
  uint64_t ids_offsets_val = ids_offsets_gm_.GetValue(block_idx_);
  uint64_t ids_offsets_val_next = ids_offsets_gm_.GetValue(block_idx_ + 1);
  uint64_t batch_offset = ids_offsets_val * tiling_->hidden_units_num;
  uint64_t cur_seq_len = ids_offsets_val_next - ids_offsets_val;
  input_offset_ = batch_offset + (cur_seq_len - 1) * tiling_->hidden_units_num;
  input_gm_.SetGlobalBuffer((__gm__ T*)input);
  output_gm_.SetGlobalBuffer((__gm__ T*)output);
  pipe_.InitBuffer(input_queue_, ASSEMBLE_LAST_TOKEN_BUFFER_NUM, tile_elem_num_ * sizeof(T));
}

template <typename T>
__aicore__ inline void AssembleLastTokenKernel<T>::CopyIn(int32_t src_idx) {
  LocalTensor<T> local_tensor = input_queue_.AllocTensor<T>();
  DataCopy(local_tensor, input_gm_[input_offset_ + src_idx], tile_elem_num_);
  input_queue_.EnQue(local_tensor);
}

template <typename T>
__aicore__ inline void AssembleLastTokenKernel<T>::CopyOut(int32_t dst_idx) {
  LocalTensor<T> local_tensor = input_queue_.DeQue<T>();
  DataCopy(output_gm_[block_idx_ * tiling_->hidden_units_num + dst_idx], local_tensor, tile_elem_num_);
  input_queue_.FreeTensor(local_tensor);
}

template <typename T>
__aicore__ void AssembleLastTokenKernel<T>::Process() {
  // loop count need to be doubled, due to double buffer
  int32_t loop_count = tiling_->tile_num * ASSEMBLE_LAST_TOKEN_BUFFER_NUM;
  // tiling strategy, pipeline parallel
  for (int32_t loop_idx = 0; loop_idx < loop_count; loop_idx++) {
    CopyIn(loop_idx);
    CopyOut(loop_idx);
  }
}

extern "C" __global__ __aicore__ void InvokeAssembleLastTokenHalfKernel(GM_ADDR input, GM_ADDR ids_offsets,
                                                                        GM_ADDR output, GM_ADDR tiling_gm) {
  AssembleLastTokenKernel<half> kernel;
  kernel.Init(input, ids_offsets, output, tiling_gm);
  kernel.Process();
}

extern "C" __global__ __aicore__ void InvokeAssembleLastTokenFloatKernel(GM_ADDR input, GM_ADDR ids_offsets,
                                                                         GM_ADDR output, GM_ADDR tiling_gm) {
  AssembleLastTokenKernel<float> kernel;
  kernel.Init(input, ids_offsets, output, tiling_gm);
  kernel.Process();
}