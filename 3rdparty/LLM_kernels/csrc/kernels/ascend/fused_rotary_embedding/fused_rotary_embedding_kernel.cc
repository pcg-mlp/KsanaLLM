/**
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include "fused_rotary_embedding_kernel.h"
#include "interface/kernel_operator_vec_vconv_intf.h"
#include "kernel_operator.h"

constexpr int32_t ROPE_EMBEDDING_BUFFER_NUM = 1;  // tensor num for each queue

template <typename T>
class RotaryEmbeddingKernel {
 public:
  __aicore__ inline RotaryEmbeddingKernel() {}
  __aicore__ inline void Init(GM_ADDR pos_ptr, GM_ADDR input_ptr, GM_ADDR cos_sin_cache_ptr, GM_ADDR tiling_gm,
                              GM_ADDR output_ptr, GM_ADDR workspace_ptr);
  __aicore__ inline void Process();

 private:
  __aicore__ inline void CopyIn();
  __aicore__ inline void Compute();
  __aicore__ inline void CopyOut();

  AscendC::GlobalTensor<T> input_global;
  AscendC::GlobalTensor<T> cos_sin_cache_global;
  AscendC::GlobalTensor<T> workspace_global;
  AscendC::GlobalTensor<T> output_global;
  AscendC::TPipe pipe;
  llm_kernels::ascend::RotaryEmbeddingTilingConfig tiling;

  int64_t pos = 0;
  int embed_dim = 0;
  int elem_num = 0;
  int loop_round = 0;
  int workspace_elem_num = 0;

  // create queues for input, in this case depth is equal to buffer num
  AscendC::TQue<AscendC::QuePosition::VECIN, ROPE_EMBEDDING_BUFFER_NUM> input_queue;
  AscendC::TQue<AscendC::QuePosition::VECIN, ROPE_EMBEDDING_BUFFER_NUM> cos_sin_cache_queue;
  AscendC::TQue<AscendC::QuePosition::VECIN, ROPE_EMBEDDING_BUFFER_NUM> workspace_queue;
  // create queue for output, in this case depth is equal to buffer num
  AscendC::TQue<AscendC::QuePosition::VECOUT, ROPE_EMBEDDING_BUFFER_NUM> output_queue;
};

template <typename T>
__aicore__ inline void RotaryEmbeddingKernel<T>::Init(GM_ADDR pos_ptr, GM_ADDR input_ptr, GM_ADDR cos_sin_cache_ptr,
                                                      GM_ADDR tiling_gm, GM_ADDR output_ptr, GM_ADDR workspace_ptr) {
  ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
  auto tmp_tiling_gm = (__gm__ uint32_t*)tiling_gm;
  auto tmp_tiling = (uint32_t*)&tiling;
  for (int32_t i = 0; i < sizeof(llm_kernels::ascend::RotaryEmbeddingTilingConfig) / sizeof(uint32_t);
       ++i, ++tmp_tiling_gm, ++tmp_tiling) {
    *tmp_tiling = *tmp_tiling_gm;
  }
  if (AscendC::GetBlockIdx() >= tiling.seq_len) {
    return;
  }

  pos = ((__gm__ int64_t*)pos_ptr)[AscendC::GetBlockIdx()];
  embed_dim = tiling.rotary_dim / 2;
  // 0: for query, 1:for key
  if (tiling.mode == 0) {
    elem_num = tiling.num_heads * tiling.head_size;
  } else {
    elem_num = tiling.num_kv_heads * tiling.head_size;
  }
  loop_round = elem_num / tiling.rotary_dim;

  __gm__ T* cache_ptr = ((__gm__ T*)cos_sin_cache_ptr) + pos * tiling.rotary_dim;
  input_global.SetGlobalBuffer((__gm__ T*)input_ptr + elem_num * AscendC::GetBlockIdx(), elem_num);
  pipe.InitBuffer(input_queue, ROPE_EMBEDDING_BUFFER_NUM, elem_num * sizeof(T));

  cos_sin_cache_global.SetGlobalBuffer(cache_ptr, tiling.rotary_dim);
  pipe.InitBuffer(cos_sin_cache_queue, ROPE_EMBEDDING_BUFFER_NUM, tiling.rotary_dim * sizeof(T));

  output_global.SetGlobalBuffer((__gm__ T*)output_ptr + elem_num * AscendC::GetBlockIdx(), elem_num);
  pipe.InitBuffer(output_queue, ROPE_EMBEDDING_BUFFER_NUM, elem_num * sizeof(T));

  workspace_elem_num = tiling.rotary_dim * 2;
  workspace_global.SetGlobalBuffer(((__gm__ T*)workspace_ptr) + AscendC::GetBlockIdx() * workspace_elem_num,
                                   workspace_elem_num);
  pipe.InitBuffer(workspace_queue, ROPE_EMBEDDING_BUFFER_NUM, workspace_elem_num * sizeof(T));
}

template <typename T>
__aicore__ inline void RotaryEmbeddingKernel<T>::Process() {
  CopyIn();
  Compute();
  CopyOut();
}

template <typename T>
__aicore__ inline void RotaryEmbeddingKernel<T>::CopyIn() {
  // alloc tensor from queue memory
  AscendC::LocalTensor<T> input_local = input_queue.AllocTensor<T>();
  AscendC::LocalTensor<T> cos_sin_cache_local = cos_sin_cache_queue.AllocTensor<T>();
  AscendC::LocalTensor<T> workspace_local = workspace_queue.AllocTensor<T>();
  // copy progress_th tile from global tensor to local tensor
  AscendC::DataCopy(input_local, input_global[0], elem_num);
  AscendC::DataCopy(cos_sin_cache_local, cos_sin_cache_global[0], tiling.rotary_dim);
  AscendC::DataCopy(workspace_local, workspace_global[0], workspace_elem_num);
  // enque input tensors to VECIN queue
  input_queue.EnQue(input_local);
  cos_sin_cache_queue.EnQue(cos_sin_cache_local);
  workspace_queue.EnQue(workspace_local);
}

template <typename T>
__aicore__ inline void RotaryEmbeddingKernel<T>::Compute() {
  // deque input tensors from VECIN queue
  // input_local shape: [1, num_heads * embed_dim] or [1, num_kv_heads * embed_dim]
  AscendC::LocalTensor<T> input_local = input_queue.DeQue<T>();
  // cos_sin_cache_local shape: [1, rotary_dim]
  AscendC::LocalTensor<T> cos_sin_cache_local = cos_sin_cache_queue.DeQue<T>();
  // workspace shape: [1, rotary_dim]
  AscendC::LocalTensor<T> workspace_local = workspace_queue.DeQue<T>();
  // output_local shape: [1, num_heads * embed_dim] or [1, num_kv_heads * embed_dim]
  AscendC::LocalTensor<T> output_local = output_queue.AllocTensor<T>();

  for (int head_idx = 0; head_idx < loop_round; ++head_idx) {
    set_mask_count();
    // equal to: arr[x_index] = x * cos - y * sin
    // equal to: workspace[0] = x * cos
    Mul(workspace_local[0], input_local[head_idx * tiling.rotary_dim], cos_sin_cache_local[0], embed_dim);
    pipe_barrier(PIPE_V);
    // equal to: workspace[1] = y * sin
    Mul(workspace_local[embed_dim], input_local[head_idx * tiling.rotary_dim + embed_dim],
        cos_sin_cache_local[embed_dim], embed_dim);
    pipe_barrier(PIPE_V);
    // equal to: arr[y_index] = y * cos + x * sin
    // equal to: workspace[2] = y * cos
    Mul(workspace_local[2 * embed_dim], input_local[head_idx * tiling.rotary_dim + embed_dim], cos_sin_cache_local[0],
        embed_dim);
    pipe_barrier(PIPE_V);
    // equal to: workspace[3] = x * sin
    Mul(workspace_local[3 * embed_dim], input_local[head_idx * tiling.rotary_dim], cos_sin_cache_local[embed_dim],
        embed_dim);
    pipe_barrier(PIPE_V);
    // equal to: arr[x_index] = workspace[0] - workspace[1]
    Sub(output_local[head_idx * tiling.rotary_dim], workspace_local[0], workspace_local[embed_dim], embed_dim);
    pipe_barrier(PIPE_V);
    // equal to: arr[y_index] = workspace[2] + workspace[3]
    Add(output_local[head_idx * tiling.rotary_dim + embed_dim], workspace_local[2 * embed_dim],
        workspace_local[3 * embed_dim], embed_dim);
    pipe_barrier(PIPE_V);
    set_mask_norm();
    AscendC::AscendCUtils::ResetMask();
  }

  // enque the output tensor to VECOUT queue
  output_queue.EnQue<T>(output_local);
  // free input tensors for reuse
  input_queue.FreeTensor(input_local);
  cos_sin_cache_queue.FreeTensor(cos_sin_cache_local);
  workspace_queue.FreeTensor(workspace_local);
}

template <typename T>
__aicore__ inline void RotaryEmbeddingKernel<T>::CopyOut() {
  // deque output tensor from VECOUT queue
  AscendC::LocalTensor<T> output_local = output_queue.DeQue<T>();
  // copy progress_th tile from local tensor to global tensor
  AscendC::DataCopy(output_global[0], output_local, elem_num);
  // free output tensor for reuse
  output_queue.FreeTensor(output_local);
}

extern "C" __global__ __aicore__ void InvokeRotaryEmbeddingHalfKernel(GM_ADDR pos_ptr, GM_ADDR input_ptr,
                                                                      GM_ADDR cos_sin_cache_ptr, GM_ADDR tiling_gm,
                                                                      GM_ADDR output_ptr, GM_ADDR workspace_ptr) {
  RotaryEmbeddingKernel<half> rope_emb_kernel;
  rope_emb_kernel.Init(pos_ptr, input_ptr, cos_sin_cache_ptr, tiling_gm, output_ptr, workspace_ptr);
  rope_emb_kernel.Process();
}

extern "C" __global__ __aicore__ void InvokeRotaryEmbeddingFloatKernel(GM_ADDR pos_ptr, GM_ADDR input_ptr,
                                                                       GM_ADDR cos_sin_cache_ptr, GM_ADDR tiling_gm,
                                                                       GM_ADDR output_ptr, GM_ADDR workspace_ptr) {
  RotaryEmbeddingKernel<float> rope_emb_kernel;
  rope_emb_kernel.Init(pos_ptr, input_ptr, cos_sin_cache_ptr, tiling_gm, output_ptr, workspace_ptr);
  rope_emb_kernel.Process();
}
