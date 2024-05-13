/**
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include "interface/kernel_type.h"
#include "kernel_operator.h"
#include "silu_mul_kernel.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;  // tensor num for each queue

template <typename DTYPE>
class SiluMulKernel {
 public:
  __aicore__ inline SiluMulKernel() {}
  __aicore__ inline void Init(GM_ADDR input, GM_ADDR weight, GM_ADDR output, GM_ADDR tiling_gm);
  __aicore__ inline void Process();

 private:
  __aicore__ inline void CopyIn(uint32_t loop_idx);
  __aicore__ inline void CastCompute(uint32_t loop_idx);
  __aicore__ inline void CopyOut(uint32_t loop_idx);

  GlobalTensor<DTYPE> input_global;
  GlobalTensor<DTYPE> weight_global;
  GlobalTensor<DTYPE> output_global;
  TPipe pipe;
  llm_kernels::ascend::SiluMulTilingConfig tiling;

  uint32_t src_block_data_size = 0;
  uint32_t dst_block_data_size = 0;
  uint32_t weight_block_data_size = 0;
  uint32_t tile_elem_num = 0;

  // create queues for input, in this case depth is equal to buffer num
  TQue<QuePosition::VECIN, BUFFER_NUM> input_queue;
  TQue<QuePosition::VECIN, BUFFER_NUM> weight_queue;
  // create queue for output, in this case depth is equal to buffer num
  TQue<QuePosition::VECOUT, BUFFER_NUM> output_queue;
};

template <typename DTYPE>
__aicore__ inline void SiluMulKernel<DTYPE>::Init(GM_ADDR input, GM_ADDR weight, GM_ADDR output, GM_ADDR tiling_gm) {
  ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
  auto tmp_tiling_gm = (__gm__ uint32_t*)tiling_gm;
  auto tmp_tiling = (uint32_t*)&tiling;
  for (int32_t i = 0; i < sizeof(llm_kernels::ascend::SiluMulTilingConfig) / sizeof(uint32_t);
       ++i, ++tmp_tiling_gm, ++tmp_tiling) {
    *tmp_tiling = *tmp_tiling_gm;
  }
  ASSERT(tiling.tile_num != 0 && "tile num can not be zero!");
  tile_elem_num = tiling.block_elem_num / BUFFER_NUM / tiling.tile_num;
  src_block_data_size = tiling.block_elem_num * sizeof(DTYPE);
  dst_block_data_size = tiling.block_elem_num * sizeof(DTYPE);
  weight_block_data_size = tiling.block_elem_num * sizeof(DTYPE);

  // get start index for current core, core parallel
  input_global.SetGlobalBuffer((__gm__ DTYPE*)input + GetBlockIdx() * tiling.block_elem_num, src_block_data_size);
  weight_global.SetGlobalBuffer((__gm__ DTYPE*)weight + GetBlockIdx() * tiling.block_elem_num, weight_block_data_size);
  output_global.SetGlobalBuffer((__gm__ DTYPE*)output + GetBlockIdx() * tiling.block_elem_num, dst_block_data_size);

  pipe.InitBuffer(input_queue, BUFFER_NUM, tile_elem_num * sizeof(DTYPE));
  pipe.InitBuffer(weight_queue, BUFFER_NUM, tile_elem_num * sizeof(DTYPE));
  pipe.InitBuffer(output_queue, BUFFER_NUM, tile_elem_num * sizeof(DTYPE));
}

template <typename DTYPE>
__aicore__ inline void SiluMulKernel<DTYPE>::Process() {
  // loop count need to be doubled, due to double buffer
  int32_t loop_count = tiling.tile_num * BUFFER_NUM;
  // tiling strategy, pipeline parallel
  for (int32_t loop_idx = 0; loop_idx < loop_count; loop_idx++) {
    CopyIn(loop_idx);
    CastCompute(loop_idx);
    CopyOut(loop_idx);
  }
}

template <typename DTYPE>
__aicore__ inline void SiluMulKernel<DTYPE>::CopyIn(uint32_t loop_idx) {
  // alloc tensor from queue memory
  LocalTensor<DTYPE> input_local = input_queue.AllocTensor<DTYPE>();
  LocalTensor<DTYPE> weight_local = weight_queue.AllocTensor<DTYPE>();
  // copy progress_th tile from global tensor to local tensor
  DataCopy(input_local, input_global[loop_idx * tile_elem_num], tile_elem_num);
  DataCopy(weight_local, weight_global[loop_idx * tile_elem_num], tile_elem_num);
  // enque input tensors to VECIN queue
  input_queue.EnQue(input_local);
  weight_queue.EnQue(weight_local);
}

template <typename DTYPE>
__aicore__ inline void SiluMulKernel<DTYPE>::CastCompute(uint32_t loop_idx) {
  // deque input tensors from VECIN queue
  LocalTensor<DTYPE> input_local = input_queue.DeQue<DTYPE>();
  LocalTensor<DTYPE> weight_local = weight_queue.DeQue<DTYPE>();
  LocalTensor<DTYPE> output_local = output_queue.AllocTensor<DTYPE>();
  // call cast
  AscendC::Silu<DTYPE>(output_local, input_local, src_block_data_size);
  AscendC::Mul<DTYPE>(output_local, output_local, weight_local, tile_elem_num);
  // enque the output tensor to VECOUT queue
  output_queue.EnQue<DTYPE>(output_local);
  // free input tensors for reuse
  input_queue.FreeTensor(input_local);
  weight_queue.FreeTensor(weight_local);
}

template <typename DTYPE>
__aicore__ inline void SiluMulKernel<DTYPE>::CopyOut(uint32_t loop_idx) {
  // deque output tensor from VECOUT queue
  LocalTensor<DTYPE> output_local = output_queue.DeQue<DTYPE>();
  // copy progress_th tile from local tensor to global tensor
  DataCopy(output_global[loop_idx * tile_elem_num], output_local, tile_elem_num);
  // free output tensor for reuse
  output_queue.FreeTensor(output_local);
}

extern "C" __global__ __aicore__ void InvokeSiluMulHalfKernel(GM_ADDR input, GM_ADDR weight, GM_ADDR output,
                                                              GM_ADDR tiling_gm) {
  SiluMulKernel<half> silu_mul_kernel;
  silu_mul_kernel.Init(input, weight, output, tiling_gm);
  silu_mul_kernel.Process();
}
