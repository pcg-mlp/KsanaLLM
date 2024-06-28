/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <linux/limits.h>
#include "csrc/kernels/ascend/paged_attention/paged_attention_tiling.h"

#include "acl/acl_base.h"

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
using namespace AscendC;
using namespace matmul;
using namespace llm_kernels::ascend;

constexpr uint32_t BUFFER_NUM = 1;  // tensor num for each queue

constexpr uint32_t cube_tiling_size = sizeof(TCubeTiling) / sizeof(uint32_t);
constexpr uint32_t softmax_tiling_size = sizeof(SoftMaxTiling) / sizeof(uint32_t);

__aicore__ inline void CopyTiling(PagedAttentionTilingData* tiling, GM_ADDR tiling_gm) {
  uint32_t* tiling_ptr = reinterpret_cast<uint32_t*>(tiling);
  __gm__ uint32_t* tiling_gm_ptr = reinterpret_cast<__gm__ uint32_t*>(tiling_gm);

  for (int i = 0; i < sizeof(PagedAttentionTilingData) / sizeof(uint32_t); i++, tiling_ptr++) {
    *tiling_ptr = *(tiling_gm_ptr + i);
  }
}

// A implement of paged attention.
template <typename T>
class PagedAttentionKernel {
 public:
  __aicore__ inline PagedAttentionKernel() {}

  __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR attn_mask, GM_ADDR k_list, GM_ADDR v_list,
                              GM_ADDR output, GM_ADDR workspace, PagedAttentionTilingData* tiling, TPipe* pipe_ptr);

  __aicore__ inline void Process();

  // mm
  typedef MatmulType<TPosition::GM, CubeFormat::ND, T> a1Type;
  typedef MatmulType<TPosition::GM, CubeFormat::ND, T> b1Type;
  typedef MatmulType<TPosition::GM, CubeFormat::ND, T> bias1Type;
  typedef MatmulType<TPosition::GM, CubeFormat::ND, T> c1Type;
  Matmul<a1Type, b1Type, c1Type, bias1Type> mm_;

  // mm2
  typedef MatmulType<TPosition::GM, CubeFormat::ND, T> a2Type;
  typedef MatmulType<TPosition::GM, CubeFormat::ND, T> b2Type;
  typedef MatmulType<TPosition::GM, CubeFormat::ND, T> bias2Type;
  typedef MatmulType<TPosition::GM, CubeFormat::ND, T> c2Type;
  Matmul<a2Type, b2Type, c2Type, bias2Type> mm2_;

 private:
  __aicore__ inline void ContextCopyIn(uint32_t head_idx, uint32_t tok_idx);
  __aicore__ inline void ContextCompute(uint32_t head_idx, uint32_t tok_idx);
  __aicore__ inline void ContextCopyOut(uint32_t head_idx, uint32_t tok_idx);

  __aicore__ inline void DecodeCopyIn(uint32_t head_idx, uint32_t tok_idx);
  __aicore__ inline void DecodeCompute(uint32_t head_idx, uint32_t tok_idx);
  __aicore__ inline void DecodeCopyOut(uint32_t head_idx, uint32_t tok_idx);

  __aicore__ inline void ProcessPrefill();
  __aicore__ inline void ProcessDecode();

  TPipe* pipe_;

  PagedAttentionTilingData* tiling_;

  TCubeTiling qk_tiling_;
  TCubeTiling wv_tiling_;

  SoftMaxTiling softmax_tiling_;

  T scale_;

  uint32_t block_idx_;

  GlobalTensor<T> q_gm_;
  GlobalTensor<T> k_gm_;
  GlobalTensor<T> v_gm_;

  GlobalTensor<T> attn_mask_gm_;

  GlobalTensor<T> output_gm_;
  GlobalTensor<T> workspace_gm_;

  // The k & v list.
  GlobalTensor<uint32_t> k_list_gm_;
  GlobalTensor<uint32_t> v_list_gm_;

  GlobalTensor<T> k_cache_gm_;
  GlobalTensor<T> v_cache_gm_;

  // The input and output queue.
  TQue<QuePosition::VECIN, BUFFER_NUM> input_queue_;
  TQue<QuePosition::VECOUT, BUFFER_NUM> output_queue_;

  TQue<QuePosition::VECIN, BUFFER_NUM> mask_input_queue_;

  // Used to save result of attn_w * v
  TQue<QuePosition::VECIN, BUFFER_NUM> attn_wv_result_queue_;
  TQue<QuePosition::VECIN, BUFFER_NUM> attn_wv_buffer_queue_;

  // For softmax, the minimum compute uinit is [1, seq_len].
  TQue<QuePosition::VECIN, BUFFER_NUM> softmax_max_queue_;
  TQue<QuePosition::VECIN, BUFFER_NUM> softmax_sum_queue_;

  // Pad seq_len up to 32 bytes.
  uint32_t padded_seq_len_ = 0;
};

template <typename T>
__aicore__ void PagedAttentionKernel<T>::Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR attn_mask, GM_ADDR k_list,
                                              GM_ADDR v_list, GM_ADDR output, GM_ADDR workspace,
                                              PagedAttentionTilingData* tiling, TPipe* pipe_ptr) {
  pipe_ = pipe_ptr;
  tiling_ = tiling;

  uint16_t scale = tiling->scale_fp16;
  scale_ = *reinterpret_cast<half*>(&scale);

  padded_seq_len_ = ((tiling_->seq_len + (32 / sizeof(T)) - 1) & ~((32 / sizeof(T)) - 1));

  block_idx_ = GetBlockIdx();

  q_gm_.SetGlobalBuffer((__gm__ T*)q);
  k_gm_.SetGlobalBuffer((__gm__ T*)k);
  v_gm_.SetGlobalBuffer((__gm__ T*)v);

  if (tiling_->context_stage == 0) {
    k_list_gm_.SetGlobalBuffer((__gm__ uint32_t*)k_list);
    v_list_gm_.SetGlobalBuffer((__gm__ uint32_t*)v_list);
  }

  attn_mask_gm_.SetGlobalBuffer((__gm__ T*)attn_mask);

  output_gm_.SetGlobalBuffer((__gm__ T*)output);
  workspace_gm_.SetGlobalBuffer((__gm__ T*)workspace);

  pipe_->InitBuffer(input_queue_, BUFFER_NUM, padded_seq_len_ * sizeof(T));
  pipe_->InitBuffer(output_queue_, BUFFER_NUM, padded_seq_len_ * sizeof(T));

  pipe_->InitBuffer(mask_input_queue_, BUFFER_NUM, padded_seq_len_ * sizeof(T));

  if (tiling_->context_stage == 0) {
    pipe_->InitBuffer(attn_wv_result_queue_, BUFFER_NUM, tiling_->head_dim * sizeof(T));
    pipe_->InitBuffer(attn_wv_buffer_queue_, BUFFER_NUM, tiling_->head_dim * sizeof(T));
  }
  pipe_->InitBuffer(softmax_max_queue_, BUFFER_NUM, padded_seq_len_ * sizeof(T));
  pipe_->InitBuffer(softmax_sum_queue_, BUFFER_NUM, padded_seq_len_ * sizeof(T));

  // Copy tiling to TCubeTiling & SoftMaxTiling
  uint32_t* assign_ptr = nullptr;
  uint32_t* tiling_ptr = reinterpret_cast<uint32_t*>(tiling_);
  for (int i = 0; (i < 2 * cube_tiling_size + softmax_tiling_size); ++i) {
    if (i == 0) {
      assign_ptr = reinterpret_cast<uint32_t*>(&qk_tiling_);
    } else if (i == cube_tiling_size) {
      assign_ptr = reinterpret_cast<uint32_t*>(&wv_tiling_);
    } else if (i == 2 * cube_tiling_size) {
      assign_ptr = reinterpret_cast<uint32_t*>(&softmax_tiling_);
    }

    *assign_ptr = *(tiling_ptr + i);
    assign_ptr++;
  }
}

template <typename T>
__aicore__ void PagedAttentionKernel<T>::ProcessPrefill() {
  REGIST_MATMUL_OBJ(pipe_, GetSysWorkSpacePtr(), mm_, &qk_tiling_, mm2_, &wv_tiling_);

  for (uint32_t head_idx = 0; head_idx < tiling_->head_size; ++head_idx) {
    if (head_idx % tiling_->head_size == block_idx_) {
      uint32_t head_offset = head_idx * tiling_->seq_len * tiling_->head_dim;

      uint32_t qk_offset = head_idx * tiling_->seq_len * tiling_->seq_len;

      mm_.SetTensorA(q_gm_[head_offset]);
      mm_.SetTensorB(k_gm_[head_offset]);
      mm_.IterateAll(workspace_gm_[qk_offset]);
      mm_.End();
      pipe_barrier(PIPE_ALL);

      // scaling & attn_mask & softmax.
      for (uint32_t tok_idx = 0; tok_idx < tiling_->seq_len; ++tok_idx) {
        // Loop seq_len * [1, seq_len]
        ContextCopyIn(head_idx, tok_idx);
        ContextCompute(head_idx, tok_idx);
        ContextCopyOut(head_idx, tok_idx);
      }
      pipe_barrier(PIPE_ALL);

      // attn_weight * v
      mm2_.SetTensorA(workspace_gm_[qk_offset]);
      mm2_.SetTensorB(v_gm_[head_offset]);
      mm2_.IterateAll(output_gm_[head_offset]);
      mm2_.End();
      pipe_barrier(PIPE_ALL);
    }
  }
}

template <typename T>
__aicore__ void PagedAttentionKernel<T>::ProcessDecode() {
  REGIST_MATMUL_OBJ(pipe_, GetSysWorkSpacePtr(), mm_, &qk_tiling_, mm2_, &wv_tiling_);

  uint32_t head_idx = block_idx_;
  uint32_t head_offset = head_idx * tiling_->head_dim;
  uint32_t qk_offset = head_idx * 1 * tiling_->seq_len;

  // q * k_T
  for (uint32_t tok_idx = 0; tok_idx <= tiling_->token_pos; ++tok_idx) {
    uint32_t k_block_idx = tok_idx / tiling_->block_token_num;
    uint32_t k_token_offset = (tok_idx % tiling_->block_token_num) * tiling_->head_size * tiling_->head_dim;
    uint32_t k_head_offset = head_idx * tiling_->head_dim;

    uint32_t k_cache_ptr0 = k_list_gm_.GetValue(k_block_idx * 2);
    uint32_t k_cache_ptr1 = k_list_gm_.GetValue(k_block_idx * 2 + 1);

    uint64_t k_cache_ptr = ((uint64_t)k_cache_ptr1 << 32) | k_cache_ptr0;
    k_cache_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(k_cache_ptr) + k_token_offset + k_head_offset);

    mm_.SetTensorA(q_gm_[head_offset]);
    mm_.SetTensorB(k_cache_gm_);
    mm_.IterateAll(workspace_gm_[qk_offset + tok_idx]);
    mm_.End();
  }

  // scaling & attn_mask & softmax.
  DecodeCopyIn(head_idx, tiling_->token_pos);
  DecodeCompute(head_idx, tiling_->token_pos);
  DecodeCopyOut(head_idx, tiling_->token_pos);

  LocalTensor<T> local_tensor_result = attn_wv_result_queue_.AllocTensor<T>();
  LocalTensor<T> local_tensor_buffer = attn_wv_buffer_queue_.AllocTensor<T>();

  // attn_weight * v
  for (uint32_t tok_idx = 0; tok_idx <= tiling_->token_pos; ++tok_idx) {
    uint32_t v_block_idx = tok_idx / tiling_->block_token_num;
    uint32_t v_token_offset = (tok_idx % tiling_->block_token_num) * tiling_->head_size * tiling_->head_dim;
    uint32_t v_head_offset = head_idx * tiling_->head_dim;

    uint32_t v_cache_ptr0 = v_list_gm_.GetValue(v_block_idx * 2);
    uint32_t v_cache_ptr1 = v_list_gm_.GetValue(v_block_idx * 2 + 1);

    uint64_t v_cache_ptr = ((uint64_t)v_cache_ptr1 << 32) | v_cache_ptr0;
    v_cache_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(v_cache_ptr) + v_token_offset + v_head_offset);

    mm2_.SetTensorA(workspace_gm_[qk_offset + tok_idx]);
    mm2_.SetTensorB(v_cache_gm_);
    mm2_.IterateAll(output_gm_[head_offset]);
    mm2_.End();

    // Add all output_gm_[head_offset] to get the result.
    if (tok_idx == 0) {
      DataCopy(local_tensor_buffer, output_gm_[head_offset], tiling_->head_dim);
      pipe_barrier(PIPE_ALL);
      DataCopy(local_tensor_result, output_gm_[head_offset], tiling_->head_dim);
      pipe_barrier(PIPE_ALL);
    } else {
      DataCopy(local_tensor_buffer, output_gm_[head_offset], tiling_->head_dim);
      pipe_barrier(PIPE_ALL);
      Add(local_tensor_result, local_tensor_result, local_tensor_buffer, tiling_->head_dim);
      pipe_barrier(PIPE_ALL);
    }
  }

  DataCopy(output_gm_[head_offset], local_tensor_result, tiling_->head_dim);

  attn_wv_result_queue_.FreeTensor(local_tensor_result);
  attn_wv_buffer_queue_.FreeTensor(local_tensor_buffer);
}

template <typename T>
__aicore__ void PagedAttentionKernel<T>::Process() {
  if (tiling_->context_stage == 1) {
    ProcessPrefill();
  } else if (tiling_->context_stage == 0) {
    ProcessDecode();
  }
}

template <typename T>
__aicore__ void PagedAttentionKernel<T>::ContextCopyIn(uint32_t head_idx, uint32_t tok_idx) {
  LocalTensor<T> local_tensor = input_queue_.AllocTensor<T>();

  uint32_t head_offset = head_idx * tiling_->seq_len * tiling_->seq_len;
  uint32_t tok_offset = tok_idx * tiling_->seq_len;

  DataCopyPadParams pad_params;
  DataCopyParams copy_params{1, static_cast<uint16_t>(tiling_->seq_len * sizeof(T)), 0, 0};
  DataCopyPad(local_tensor, workspace_gm_[head_offset + tok_offset], copy_params, pad_params);

  input_queue_.EnQue(local_tensor);

  LocalTensor<T> mask_tensor = mask_input_queue_.AllocTensor<T>();

  DataCopyPadParams pad_params2;
  DataCopyParams copy_params2{1, static_cast<uint16_t>(padded_seq_len_ * sizeof(T)), 0, 0};
  DataCopyPad(mask_tensor, attn_mask_gm_[tok_idx * tiling_->max_seq_len], copy_params2, pad_params2);

  mask_input_queue_.EnQue(mask_tensor);
}

template <typename T>
__aicore__ void PagedAttentionKernel<T>::ContextCompute(uint32_t head_idx, uint32_t tok_idx) {
  LocalTensor<T> local_tensor = input_queue_.DeQue<T>();

  LocalTensor<T> softmax_max_tensor = softmax_max_queue_.AllocTensor<T>();
  LocalTensor<T> softmax_sum_tensor = softmax_sum_queue_.AllocTensor<T>();

  LocalTensor<T> output_tensor = output_queue_.AllocTensor<T>();
  Muls(output_tensor, local_tensor, scale_, tiling_->seq_len);

  LocalTensor<T> mask_tensor = mask_input_queue_.DeQue<T>();
  Add(output_tensor, output_tensor, mask_tensor, padded_seq_len_);

  SoftMaxShapeInfo shape = {1, padded_seq_len_, 1, padded_seq_len_};
  SoftMax<T>(output_tensor, softmax_sum_tensor, softmax_max_tensor, output_tensor,
             *reinterpret_cast<SoftMaxTiling*>(&softmax_tiling_), shape);

  output_queue_.EnQue(output_tensor);
  input_queue_.FreeTensor(local_tensor);
  mask_input_queue_.FreeTensor(mask_tensor);

  softmax_max_queue_.FreeTensor(softmax_max_tensor);
  softmax_sum_queue_.FreeTensor(softmax_sum_tensor);
}

template <typename T>
__aicore__ void PagedAttentionKernel<T>::ContextCopyOut(uint32_t head_idx, uint32_t tok_idx) {
  LocalTensor<T> local_tensor = output_queue_.DeQue<T>();

  uint32_t head_offset = head_idx * tiling_->seq_len * tiling_->seq_len;
  uint32_t tok_offset = tok_idx * tiling_->seq_len;

  DataCopyParams copy_params{1, static_cast<uint16_t>(tiling_->seq_len * sizeof(T)), 0, 0};
  DataCopyPad(workspace_gm_[head_offset + tok_offset], local_tensor, copy_params);

  output_queue_.FreeTensor(local_tensor);
}

template <typename T>
__aicore__ void PagedAttentionKernel<T>::DecodeCopyIn(uint32_t head_idx, uint32_t tok_idx) {
  LocalTensor<T> local_tensor = input_queue_.AllocTensor<T>();

  uint32_t head_offset = head_idx * 1 * tiling_->seq_len;

  DataCopyPadParams pad_params;
  DataCopyParams copy_params{1, static_cast<uint16_t>(tiling_->seq_len * sizeof(T)), 0, 0};
  DataCopyPad(local_tensor, workspace_gm_[head_offset], copy_params, pad_params);

  input_queue_.EnQue(local_tensor);

  LocalTensor<T> mask_tensor = mask_input_queue_.AllocTensor<T>();

  // Use tok_idx, but padded to 32 bytes.
  DataCopyPadParams pad_params2;
  DataCopyParams copy_params2{1, static_cast<uint16_t>(padded_seq_len_ * sizeof(T)), 0, 0};
  DataCopyPad(mask_tensor, attn_mask_gm_[tok_idx * tiling_->max_seq_len], copy_params2, pad_params2);

  mask_input_queue_.EnQue(mask_tensor);
}

template <typename T>
__aicore__ void PagedAttentionKernel<T>::DecodeCompute(uint32_t head_idx, uint32_t tok_idx) {
  LocalTensor<T> local_tensor = input_queue_.DeQue<T>();

  LocalTensor<T> softmax_max_tensor = softmax_max_queue_.AllocTensor<T>();
  LocalTensor<T> softmax_sum_tensor = softmax_sum_queue_.AllocTensor<T>();

  LocalTensor<T> output_tensor = output_queue_.AllocTensor<T>();
  Muls(output_tensor, local_tensor, scale_, tiling_->seq_len);

  // Padded to 32 bytes.
  LocalTensor<T> mask_tensor = mask_input_queue_.DeQue<T>();
  Add(output_tensor, output_tensor, mask_tensor, padded_seq_len_);

  SoftMaxShapeInfo shape = {1, padded_seq_len_, 1, padded_seq_len_};
  SoftMax<T>(output_tensor, softmax_sum_tensor, softmax_max_tensor, output_tensor,
             *reinterpret_cast<SoftMaxTiling*>(&softmax_tiling_), shape);

  output_queue_.EnQue(output_tensor);
  input_queue_.FreeTensor(local_tensor);
  mask_input_queue_.FreeTensor(mask_tensor);

  softmax_max_queue_.FreeTensor(softmax_max_tensor);
  softmax_sum_queue_.FreeTensor(softmax_sum_tensor);
}

template <typename T>
__aicore__ void PagedAttentionKernel<T>::DecodeCopyOut(uint32_t head_idx, uint32_t tok_idx) {
  LocalTensor<T> local_tensor = output_queue_.DeQue<T>();

  uint32_t head_offset = head_idx * 1 * tiling_->seq_len;

  DataCopyParams copy_params{1, static_cast<uint16_t>(tiling_->seq_len * sizeof(T)), 0, 0};
  DataCopyPad(workspace_gm_[head_offset], local_tensor, copy_params);

  output_queue_.FreeTensor(local_tensor);
}

extern "C" __global__ __aicore__ void InvokePagedAttentionKernel(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR attn_mask,
                                                                 GM_ADDR k_list, GM_ADDR v_list, GM_ADDR output,
                                                                 GM_ADDR workspace, GM_ADDR tiling_gm) {
  PagedAttentionTilingData tiling;
  CopyTiling(&tiling, tiling_gm);

  uint32_t max_core_num = (g_coreType == AIC) ? tiling.used_core_num : tiling.used_core_num * 2;
  if (GetBlockIdx() >= max_core_num) {
    return;
  }

  TPipe pipe;
  if (tiling.data_type == 0) {
    PagedAttentionKernel<half> op;
    op.Init(q, k, v, attn_mask, k_list, v_list, output, workspace, &tiling, &pipe);
    op.Process();
  } else if (tiling.data_type == 1) {
    PagedAttentionKernel<float> op;
    op.Init(q, k, v, attn_mask, k_list, v_list, output, workspace, &tiling, &pipe);
    op.Process();
  }
}
