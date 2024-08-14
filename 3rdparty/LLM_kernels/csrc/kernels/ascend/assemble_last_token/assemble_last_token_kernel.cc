/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

#include "csrc/kernels/ascend/assemble_last_token/assemble_last_token_tiling.h"

using namespace AscendC;
using namespace llm_kernels::ascend;

constexpr int32_t ASSEMBLE_LAST_TOKEN_BUFFER_NUM = 1;  // tensor num for each queue

template <typename T>
class AssembleLastTokenKernel {
 public:
  __aicore__ inline AssembleLastTokenKernel() {}

  __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, GM_ADDR ids_offsets, GM_ADDR tiling_gm);

  __aicore__ inline void Process();

 private:
  // The in/out stage process.
  __aicore__ inline void CopyIn(int32_t src_idx);
  __aicore__ inline void CopyOut(int32_t dst_idx);
  // The tiling config.
  AssembleLastTokenTiling tiling;
  TPipe pipe_;
  GlobalTensor<T> input_gm_;
  GlobalTensor<T> output_gm_;
  GlobalTensor<uint64_t> ids_offsets_gm_;
  // The input and output queue.
  TQue<QuePosition::VECIN, ASSEMBLE_LAST_TOKEN_BUFFER_NUM> input_queue_;
  uint32_t tile_elem_num_ = 0;
};

template <typename T>
__aicore__ void AssembleLastTokenKernel<T>::Init(GM_ADDR input, GM_ADDR output, GM_ADDR ids_offsets, GM_ADDR tiling_gm) {
  auto tmp_tiling_gm = (__gm__ uint32_t*)tiling_gm;
  auto tmp_tiling = (uint32_t*)&tiling;
  for (int32_t i = 0; i < sizeof(AssembleLastTokenTiling) / sizeof(uint32_t);
       ++i, ++tmp_tiling_gm, ++tmp_tiling) {
    *tmp_tiling = *tmp_tiling_gm;
  }

  tile_elem_num_ = tiling.hidden_units_num / ASSEMBLE_LAST_TOKEN_BUFFER_NUM / tiling.tile_num;

  ids_offsets_gm_.SetGlobalBuffer((__gm__ uint64_t*)ids_offsets);
  uint64_t ids_offsets_val = ids_offsets_gm_.GetValue(GetBlockIdx());
  uint64_t ids_offsets_val_next = ids_offsets_gm_.GetValue(GetBlockIdx() + 1);
  size_t cur_seq_len = ids_offsets_val_next - ids_offsets_val;
  size_t offset = (ids_offsets_val + cur_seq_len - 1) * tiling.hidden_units_num;
  size_t block_data_size = tiling.hidden_units_num * sizeof(T);

  input_gm_.SetGlobalBuffer((__gm__ T*)input + offset, block_data_size);
  output_gm_.SetGlobalBuffer((__gm__ T*)output + GetBlockIdx() * tiling.hidden_units_num, block_data_size);

  pipe_.InitBuffer(input_queue_, ASSEMBLE_LAST_TOKEN_BUFFER_NUM, tile_elem_num_ * sizeof(T));
}

template <typename T>
__aicore__ inline void AssembleLastTokenKernel<T>::CopyIn(int32_t src_idx) {
  LocalTensor<T> local_tensor = input_queue_.AllocTensor<T>();
  DataCopy(local_tensor, input_gm_[src_idx], tile_elem_num_);
  pipe_barrier(PIPE_ALL);
  input_queue_.EnQue(local_tensor);
}

template <typename T>
__aicore__ inline void AssembleLastTokenKernel<T>::CopyOut(int32_t dst_idx) {
  LocalTensor<T> local_tensor = input_queue_.DeQue<T>();
  DataCopy(output_gm_[dst_idx], local_tensor, tile_elem_num_);
  pipe_barrier(PIPE_ALL);
  input_queue_.FreeTensor(local_tensor);
}

template <typename T>
__aicore__ void AssembleLastTokenKernel<T>::Process() {
  // loop count need to be doubled, due to double buffer
  int32_t loop_count = tiling.tile_num * ASSEMBLE_LAST_TOKEN_BUFFER_NUM;
  // tiling strategy, pipeline parallel
  for (int32_t loop_idx = 0; loop_idx < loop_count; loop_idx++) {
    CopyIn(loop_idx);
    CopyOut(loop_idx);
  }
}

extern "C" __global__ __aicore__ void InvokeAssembleLastTokenHalfKernel(GM_ADDR input, GM_ADDR ids_offsets,
                                                                        GM_ADDR output, GM_ADDR tiling_gm) {
  AssembleLastTokenKernel<half> kernel;
  kernel.Init(input, output, ids_offsets, tiling_gm);
  kernel.Process();
}

extern "C" __global__ __aicore__ void InvokeAssembleLastTokenFloatKernel(GM_ADDR input, GM_ADDR ids_offsets,
                                                                         GM_ADDR output, GM_ADDR tiling_gm) {
  AssembleLastTokenKernel<float> kernel;
  kernel.Init(input, output, ids_offsets, tiling_gm);
  kernel.Process();
}