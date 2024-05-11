/**
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include "cast.h"
#include "interface/kernel_operator_vec_vconv_intf.h"
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;  // tensor num for each queue

template <typename SRC_DTYPE, typename DST_DTYPE>
class CastKernel {
 public:
  __aicore__ inline CastKernel() {}
  __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, GM_ADDR tiling_gm);
  __aicore__ inline void Process();

 private:
  __aicore__ inline void CopyIn(uint32_t loop_idx);
  __aicore__ inline void CastCompute(uint32_t loop_idx);
  __aicore__ inline void CopyOut(uint32_t loop_idx);

  GlobalTensor<SRC_DTYPE> input_global;
  GlobalTensor<DST_DTYPE> output_global;
  TPipe pipe;
  llm_kernels::ascend::CastTilingConfig tiling;

  uint32_t src_block_data_size = 0;
  uint32_t dst_block_data_size = 0;
  uint32_t tile_elem_num = 0;

  // create queues for input, in this case depth is equal to buffer num
  TQue<QuePosition::VECIN, BUFFER_NUM> input_queue;
  // create queue for output, in this case depth is equal to buffer num
  TQue<QuePosition::VECOUT, BUFFER_NUM> output_queue;
};

template <typename SRC_DTYPE, typename DST_DTYPE>
__aicore__ inline void CastKernel<SRC_DTYPE, DST_DTYPE>::Init(GM_ADDR input, GM_ADDR output, GM_ADDR tiling_gm) {
  ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
  auto tmp_tiling_gm = (__gm__ uint32_t*)tiling_gm;
  auto tmp_tiling = (uint32_t*)&tiling;
  for (int32_t i = 0; i < sizeof(llm_kernels::ascend::CastTilingConfig) / sizeof(uint32_t);
       ++i, ++tmp_tiling_gm, ++tmp_tiling) {
    *tmp_tiling = *tmp_tiling_gm;
  }
  ASSERT(tiling.tile_num != 0 && "tile num can not be zero!");
  tile_elem_num = tiling.block_elem_num / BUFFER_NUM / tiling.tile_num;
  src_block_data_size = tiling.block_elem_num * sizeof(SRC_DTYPE);
  dst_block_data_size = tiling.block_elem_num * sizeof(DST_DTYPE);

  // get start index for current core, core parallel
  input_global.SetGlobalBuffer((__gm__ SRC_DTYPE*)input + GetBlockIdx() * tiling.block_elem_num, src_block_data_size);
  output_global.SetGlobalBuffer((__gm__ DST_DTYPE*)output + GetBlockIdx() * tiling.block_elem_num, src_block_data_size);

  pipe.InitBuffer(input_queue, BUFFER_NUM, tile_elem_num * sizeof(SRC_DTYPE));
  pipe.InitBuffer(output_queue, BUFFER_NUM, tile_elem_num * sizeof(DST_DTYPE));
}

template <typename SRC_DTYPE, typename DST_DTYPE>
__aicore__ inline void CastKernel<SRC_DTYPE, DST_DTYPE>::Process() {
  // loop count need to be doubled, due to double buffer
  int32_t loop_count = tiling.tile_num * BUFFER_NUM;
  // tiling strategy, pipeline parallel
  for (int32_t loop_idx = 0; loop_idx < loop_count; loop_idx++) {
    CopyIn(loop_idx);
    CastCompute(loop_idx);
    CopyOut(loop_idx);
  }
}

template <typename SRC_DTYPE, typename DST_DTYPE>
__aicore__ inline void CastKernel<SRC_DTYPE, DST_DTYPE>::CopyIn(uint32_t loop_idx) {
  // alloc tensor from queue memory
  LocalTensor<SRC_DTYPE> input_local = input_queue.AllocTensor<SRC_DTYPE>();
  // copy progress_th tile from global tensor to local tensor
  DataCopy(input_local, input_global[loop_idx * tile_elem_num], tile_elem_num);
  // enque input tensors to VECIN queue
  input_queue.EnQue(input_local);
}

template <typename SRC_DTYPE, typename DST_DTYPE>
__aicore__ inline void CastKernel<SRC_DTYPE, DST_DTYPE>::CastCompute(uint32_t loop_idx) {
  // deque input tensors from VECIN queue
  LocalTensor<SRC_DTYPE> input_local = input_queue.DeQue<SRC_DTYPE>();
  LocalTensor<DST_DTYPE> output_local = output_queue.AllocTensor<DST_DTYPE>();
  // call cast
  AscendC::Cast<DST_DTYPE, SRC_DTYPE>(output_local, input_local, RoundMode::CAST_NONE, tile_elem_num);
  // enque the output tensor to VECOUT queue
  output_queue.EnQue<DST_DTYPE>(output_local);
  // free input tensors for reuse
  input_queue.FreeTensor(input_local);
}

template <typename SRC_DTYPE, typename DST_DTYPE>
__aicore__ inline void CastKernel<SRC_DTYPE, DST_DTYPE>::CopyOut(uint32_t loop_idx) {
  // deque output tensor from VECOUT queue
  LocalTensor<DST_DTYPE> output_local = output_queue.DeQue<DST_DTYPE>();
  // copy progress_th tile from local tensor to global tensor
  DataCopy(output_global[loop_idx * tile_elem_num], output_local, tile_elem_num);
  // free output tensor for reuse
  output_queue.FreeTensor(output_local);
}

extern "C" __global__ __aicore__ void InvokeCastFloatToHalfKernel(GM_ADDR input, GM_ADDR output, GM_ADDR tiling_gm) {
  CastKernel<float, half> cast_kernel;
  cast_kernel.Init(input, output, tiling_gm);
  cast_kernel.Process();
}
