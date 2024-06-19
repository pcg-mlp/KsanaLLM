/**
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include "embedding_kernel.h"

#include "interface/kernel_type.h"
#include "kernel_operator.h"
using namespace AscendC;

constexpr int EMB_BUFFER_NUM = 1;

template <typename DTYPE>
class EmbeddingKernel {
 public:
  __aicore__ inline EmbeddingKernel() {}
  __aicore__ inline void Init(GM_ADDR input_ids, GM_ADDR embedding_table, GM_ADDR output, GM_ADDR tiling_gm);
  __aicore__ inline void Process();

 private:
  __aicore__ inline void CopyIn(uint32_t loop_idx);
  __aicore__ inline void EmbCompute(uint32_t loop_idx);
  __aicore__ inline void CopyOut(uint32_t loop_idx);

  GlobalTensor<DTYPE> embedding_table_gm_;
  GlobalTensor<DTYPE> output_gm_;
  TPipe pipe_;
  llm_kernels::ascend::EmbeddingConfigTiling tiling_;

  // create queues for input, in this case depth is equal to buffer num
  TQue<QuePosition::VECIN, EMB_BUFFER_NUM> embedding_table_queue_;
  // create queue for output, in this case depth is equal to buffer num
  TQue<QuePosition::VECOUT, EMB_BUFFER_NUM> output_queue_;

  uint32_t tile_elem_num_;
  uint32_t hidden_units_data_size_;
  int32_t token_id_;
};

template <typename DTYPE>
__aicore__ inline void EmbeddingKernel<DTYPE>::Init(GM_ADDR input_ids, GM_ADDR embedding_table, GM_ADDR output,
                                                    GM_ADDR tiling_gm) {
  ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
  ASSERT(tiling_.hidden_units == 0 && "hidden_units can not be zero!");
  auto tmp_tiling_gm = (__gm__ uint32_t*)tiling_gm;
  auto tmp_tiling = (uint32_t*)&tiling_;
  for (int32_t i = 0; i < sizeof(llm_kernels::ascend::EmbeddingConfigTiling) / sizeof(uint32_t);
       ++i, ++tmp_tiling_gm, ++tmp_tiling) {
    *tmp_tiling = *tmp_tiling_gm;
  }
  ASSERT(tiling_.hidden_units != 0 && "hidden_units can not be zero!");

  tile_elem_num_ = tiling_.hidden_units / EMB_BUFFER_NUM / tiling_.tile_num;
  hidden_units_data_size_ = tiling_.hidden_units * sizeof(DTYPE);

  // get start index for current core, core parallel
  int32_t real_token_id = ((__gm__ int32_t*)input_ids)[GetBlockIdx()];
  // on each NPU, emb range is [vocab_id * vocab_size, (vocab_id + 1) * vocab_size)
  // read_id bigger than the vocabulary size, handle next id
  if (real_token_id >= ((tiling_.vocab_id + 1) * tiling_.vocab_size) ||
      real_token_id < (tiling_.vocab_id * tiling_.vocab_size)) {
    return;
  }
  embedding_table_gm_.SetGlobalBuffer(
      (__gm__ DTYPE*)embedding_table + (real_token_id - tiling_.vocab_id * tiling_.vocab_size) * tiling_.hidden_units,
      hidden_units_data_size_);
  output_gm_.SetGlobalBuffer((__gm__ DTYPE*)output + GetBlockIdx() * tiling_.hidden_units, hidden_units_data_size_);

  pipe_.InitBuffer(embedding_table_queue_, EMB_BUFFER_NUM, tile_elem_num_ * sizeof(DTYPE));
  pipe_.InitBuffer(output_queue_, EMB_BUFFER_NUM, tile_elem_num_ * sizeof(DTYPE));
}

template <typename DTYPE>
__aicore__ inline void EmbeddingKernel<DTYPE>::Process() {
  // loop count need to be doubled, due to double buffer
  int32_t loop_count = tiling_.tile_num * EMB_BUFFER_NUM;
  // tiling strategy, pipeline parallel
  for (int32_t loop_idx = 0; loop_idx < loop_count; loop_idx++) {
    CopyIn(loop_idx);
    EmbCompute(loop_idx);
    CopyOut(loop_idx);
  }
}

template <typename DTYPE>
__aicore__ inline void EmbeddingKernel<DTYPE>::CopyIn(uint32_t loop_idx) {
  // alloc tensor from queue memory
  LocalTensor<DTYPE> emb_table_local = embedding_table_queue_.AllocTensor<DTYPE>();
  // copy progress_th tile from global tensor to local tensor
  DataCopy(emb_table_local, embedding_table_gm_[loop_idx * tile_elem_num_], tile_elem_num_);
  // enque input tensors to VECIN queue
  embedding_table_queue_.EnQue(emb_table_local);
}

template <typename DTYPE>
__aicore__ inline void EmbeddingKernel<DTYPE>::EmbCompute(uint32_t loop_idx) {
  // deque input tensors from VECIN queue
  LocalTensor<DTYPE> emb_table_local = embedding_table_queue_.DeQue<DTYPE>();
  LocalTensor<DTYPE> output_local = output_queue_.AllocTensor<DTYPE>();

  DataCopy(output_local, emb_table_local, tile_elem_num_);
  // enque the output tensor to VECOUT queue
  output_queue_.EnQue<DTYPE>(output_local);
  // free input tensors for reuse
  embedding_table_queue_.FreeTensor(emb_table_local);
}

template <typename DTYPE>
__aicore__ inline void EmbeddingKernel<DTYPE>::CopyOut(uint32_t loop_idx) {
  // deque output tensor from VECOUT queue
  LocalTensor<DTYPE> output_local = output_queue_.DeQue<DTYPE>();
  // copy progress_th tile from local tensor to global tensor
  DataCopy(output_gm_[loop_idx * tile_elem_num_], output_local, tile_elem_num_);
  // free output tensor for reuse
  output_queue_.FreeTensor(output_local);
}

extern "C" __global__ __aicore__ void InvokeLookupEmbeddingHalfKernel(GM_ADDR input_ids, GM_ADDR embedding_table,
                                                                      GM_ADDR output, GM_ADDR tiling_gm) {
  EmbeddingKernel<half> emb_kernel;
  emb_kernel.Init(input_ids, embedding_table, output, tiling_gm);
  emb_kernel.Process();
}

extern "C" __global__ __aicore__ void InvokeLookupEmbeddingFloatKernel(GM_ADDR input_ids, GM_ADDR embedding_table,
                                                                      GM_ADDR output, GM_ADDR tiling_gm) {
  EmbeddingKernel<float> emb_kernel;
  emb_kernel.Init(input_ids, embedding_table, output, tiling_gm);
  emb_kernel.Process();
}