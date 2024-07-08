/**
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include "add_tiling.h"
#include "interface/kernel_operator_vec_vconv_intf.h"
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t ADD_BUFFER_NUM = 1;  // tensor num for each queue

template <typename DTYPE>
class AddKernel {
 public:
  __aicore__ inline AddKernel() {}
  __aicore__ inline void Init(GM_ADDR input_a, GM_ADDR input_b, GM_ADDR bias, GM_ADDR output, GM_ADDR tiling_gm);
  __aicore__ inline void Process();

 private:
  __aicore__ inline void CopyIn(uint32_t loop_idx);
  __aicore__ inline void AddCompute(uint32_t loop_idx);
  __aicore__ inline void CopyOut(uint32_t loop_idx);

  GlobalTensor<DTYPE> input_a_global;
  GlobalTensor<DTYPE> input_b_global;
  GlobalTensor<DTYPE> bias_global;
  GlobalTensor<DTYPE> output_global;
  TPipe pipe;
  llm_kernels::ascend::AddTilingConfig tiling;

  uint32_t tile_elem_num;
  uint32_t block_data_size;
  DTYPE alpha;
  bool is_bias;

  // create queues for input, in this case depth is equal to buffer num
  TQue<QuePosition::VECIN, ADD_BUFFER_NUM> input_a_queue;
  TQue<QuePosition::VECIN, ADD_BUFFER_NUM> input_b_queue;
  TQue<QuePosition::VECIN, ADD_BUFFER_NUM> bias_queue;
  // create queue for output, in this case depth is equal to buffer num
  TQue<QuePosition::VECOUT, ADD_BUFFER_NUM> output_queue;
};

template <typename DTYPE>
__aicore__ inline void AddKernel<DTYPE>::Init(GM_ADDR input_a, GM_ADDR input_b, GM_ADDR bias, GM_ADDR output,
                                              GM_ADDR tiling_gm) {
  ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
  is_bias = false;
  auto tmp_tiling_gm = (__gm__ uint32_t*)tiling_gm;
  auto tmp_tiling = (uint32_t*)&tiling;
  for (int32_t i = 0; i < sizeof(llm_kernels::ascend::AddTilingConfig) / sizeof(uint32_t);
       ++i, ++tmp_tiling_gm, ++tmp_tiling) {
    *tmp_tiling = *tmp_tiling_gm;
  }
  ASSERT(tiling.tile_num != 0 && "tile num can not be zero!");
  tile_elem_num = tiling.block_elem_num / ADD_BUFFER_NUM / tiling.tile_num;
  block_data_size = tiling.block_elem_num * sizeof(DTYPE);

  // get start index for current core, core parallel
  input_a_global.SetGlobalBuffer((__gm__ DTYPE*)input_a + GetBlockIdx() * tiling.block_elem_num, block_data_size);
  input_b_global.SetGlobalBuffer((__gm__ DTYPE*)input_b + GetBlockIdx() * tiling.block_elem_num, block_data_size);
  if (bias != nullptr) {
    bias_global.SetGlobalBuffer((__gm__ DTYPE*)bias, block_data_size);
  }
  output_global.SetGlobalBuffer((__gm__ DTYPE*)output + GetBlockIdx() * tiling.block_elem_num, block_data_size);

  pipe.InitBuffer(input_a_queue, ADD_BUFFER_NUM, tile_elem_num * sizeof(DTYPE));
  pipe.InitBuffer(input_b_queue, ADD_BUFFER_NUM, tile_elem_num * sizeof(DTYPE));
  if (bias != nullptr) {
    pipe.InitBuffer(bias_queue, ADD_BUFFER_NUM, tile_elem_num * sizeof(DTYPE));
    is_bias = true;
  }
  pipe.InitBuffer(output_queue, ADD_BUFFER_NUM, tile_elem_num * sizeof(DTYPE));

  if (std::is_same<DTYPE, half>::value) {
    alpha = ((DTYPE*)(&(tiling.alpha)))[0];
  } else {
    alpha = tiling.alpha;
  }
}

template <typename DTYPE>
__aicore__ inline void AddKernel<DTYPE>::Process() {
  // loop count need to be doubled, due to double buffer
  int32_t loop_count = tiling.tile_num * ADD_BUFFER_NUM;
  // tiling strategy, pipeline parallel
  for (int32_t loop_idx = 0; loop_idx < loop_count; loop_idx++) {
    CopyIn(loop_idx);
    AddCompute(loop_idx);
    CopyOut(loop_idx);
  }
}

template <typename DTYPE>
__aicore__ inline void AddKernel<DTYPE>::CopyIn(uint32_t loop_idx) {
  // alloc tensor from queue memory
  LocalTensor<DTYPE> input_a_local = input_a_queue.AllocTensor<DTYPE>();
  // copy progress_th tile from global tensor to local tensor
  DataCopy(input_a_local, input_a_global[loop_idx * tile_elem_num], tile_elem_num);
  // enque input tensors to VECIN queue
  input_a_queue.EnQue(input_a_local);
  // alloc tensor from queue memory
  LocalTensor<DTYPE> input_b_local = input_b_queue.AllocTensor<DTYPE>();
  // copy progress_th tile from global tensor to local tensor
  DataCopy(input_b_local, input_b_global[loop_idx * tile_elem_num], tile_elem_num);
  // enque input tensors to VECIN queue
  input_b_queue.EnQue(input_b_local);

  if (is_bias) {
    LocalTensor<DTYPE> bias_local = bias_queue.AllocTensor<DTYPE>();
    DataCopy(bias_local, bias_global[loop_idx * tile_elem_num], tile_elem_num);
    bias_queue.EnQue(bias_local);
  }
}

template <typename DTYPE>
__aicore__ inline void AddKernel<DTYPE>::AddCompute(uint32_t loop_idx) {
  // deque input tensors from VECIN queue
  LocalTensor<DTYPE> input_a_local = input_a_queue.DeQue<DTYPE>();
  LocalTensor<DTYPE> input_b_local = input_b_queue.DeQue<DTYPE>();
  LocalTensor<DTYPE> output_local = output_queue.AllocTensor<DTYPE>();

  // compute (a + b)
  Add(output_local, input_a_local, input_b_local, tile_elem_num);
  pipe_barrier(PIPE_V);
  // compute (a + b) + bias
  if (is_bias) {
    LocalTensor<DTYPE> bias_local = bias_queue.AllocTensor<DTYPE>();
    Add(output_local, output_local, bias_local, tile_elem_num);
    pipe_barrier(PIPE_V);
    bias_queue.FreeTensor(bias_local);
  }

  // enque the output tensor to VECOUT queue
  output_queue.EnQue<DTYPE>(output_local);
  // free input tensors for reuse
  input_a_queue.FreeTensor(input_a_local);
  input_b_queue.FreeTensor(input_b_local);
}

template <typename DTYPE>
__aicore__ inline void AddKernel<DTYPE>::CopyOut(uint32_t loop_idx) {
  // deque output tensor from VECOUT queue
  LocalTensor<DTYPE> output_local = output_queue.DeQue<DTYPE>();
  // copy progress_th tile from local tensor to global tensor
  DataCopy(output_global[loop_idx * tile_elem_num], output_local, tile_elem_num);
  // free output tensor for reuse
  output_queue.FreeTensor(output_local);
}

extern "C" __global__ __aicore__ void InvokeAddFloatKernel(GM_ADDR input_a, GM_ADDR input_b, GM_ADDR bias,
                                                           GM_ADDR output, GM_ADDR tiling_gm) {
  AddKernel<float> add_kernel;
  add_kernel.Init(input_a, input_b, bias, output, tiling_gm);
  add_kernel.Process();
}

extern "C" __global__ __aicore__ void InvokeAddHalfKernel(GM_ADDR input_a, GM_ADDR input_b, GM_ADDR bias,
                                                          GM_ADDR output, GM_ADDR tiling_gm) {
  AddKernel<half> add_kernel;
  add_kernel.Init(input_a, input_b, bias, output, tiling_gm);
  add_kernel.Process();
}