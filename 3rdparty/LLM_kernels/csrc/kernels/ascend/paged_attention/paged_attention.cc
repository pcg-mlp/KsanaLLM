/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "csrc/kernels/ascend/paged_attention/paged_attention.h"
#include "csrc/utils/ascend/common.h"
#include "csrc/utils/ascend/tiling_data_types.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <type_traits>

#include "aclrtlaunch_InvokePagedAttentionKernel.h"

#include "acl/acl.h"
#include "acl/acl_op_compiler.h"
#include "aclnn/acl_meta.h"

#include "tiling/softmax/softmax_tiling_intf.h"
#include "tiling/tiling_api.h"

// The max possible batch size.
#define MAX_BATCH_SIZE 256

// The max possible block num for one batch.
#define MAX_SEQ_BLOCK_NUM 2048

// The max total seq len of all prompts in one batch.
#define MAX_TOTAL_SEQ_LEN 4096

// The max seq_len
#define MAX_SEQ_LEN 4096

// The max used ai core number.
constexpr uint32_t MAX_USED_CORE_NUM = 24;

namespace llm_kernels {
namespace ascend {

template <typename T>
PagedAttention<T>::~PagedAttention() {
  free(prefill_token_offset_);
  free(decode_tokens_len_);

  free(kv_list_);
  free(kv_cache_offset_);

  ACL_CHECK_RET(aclrtFree(tiling_buffer_gm_));
  ACL_CHECK_RET(aclrtFree(workspace_gm_));
}

template <typename T>
void PagedAttention<T>::Initialize(uint32_t head_size, uint32_t kv_head_size, uint32_t head_dim, uint32_t layer_num,
                                   uint32_t layer_idx, uint32_t block_token_num, aclrtStream stream,
                                   const RotaryEmbeddingType scaling_type, const float scaling_factor) {
  head_size_ = head_size;
  kv_head_size_ = kv_head_size;

  head_dim_ = head_dim;
  layer_num_ = layer_num;
  block_token_num_ = block_token_num;

  tiling_size_ = sizeof(PagedAttentionTilingData);

  ACL_CHECK_RET(aclrtMalloc(&tiling_buffer_gm_, tiling_size_, ACL_MEM_TYPE_HIGH_BAND_WIDTH));

  prefill_token_offset_ = (uint64_t*)malloc(MAX_BATCH_SIZE * sizeof(uint64_t));
  decode_tokens_len_ = (int32_t*)malloc(MAX_BATCH_SIZE * sizeof(int32_t));

  kv_list_ = (void**)malloc(layer_num_ * MAX_SEQ_BLOCK_NUM * 2 * sizeof(void*));
  kv_cache_offset_ = (int32_t*)malloc(MAX_BATCH_SIZE * sizeof(int32_t));

  stride_size_ = head_size_ * head_dim_;

  scaling_type_ = scaling_type;
  scaling_factor_ = scaling_factor;

  ACL_CHECK_RET(
      aclrtMalloc(&cos_sin_cache_, max_position_embeddings_ * rotary_dim_ * sizeof(T), ACL_MEM_TYPE_HIGH_BAND_WIDTH));
  rope_.SetConfig(reinterpret_cast<T*>(cos_sin_cache_), rotary_dim_, max_position_embeddings_, rope_base_, head_dim_,
                  head_size_, kv_head_size_, stride_size_, is_neox_, stream, scaling_type_, scaling_factor_);

  size_t usr_workspace_size = MAX_SEQ_LEN * MAX_SEQ_LEN * sizeof(T);
  size_t sys_workspace_size = 1024 * 1024 * 1024;
  ACL_CHECK_RET(aclrtMalloc(&workspace_gm_, usr_workspace_size + sys_workspace_size, ACL_MEM_MALLOC_HUGE_FIRST));

  IniAttnMask();
}

template <typename T>
void PagedAttention<T>::IniAttnMask() {
  uint16_t min_value = 0xFBFF;
  std::vector<uint16_t> mask(MAX_SEQ_LEN * MAX_SEQ_LEN, 0);
  for (size_t i = 0; i < MAX_SEQ_LEN; ++i) {
    for (size_t j = 0; j < MAX_SEQ_LEN; ++j) {
      if (j > i) {
        mask[i * MAX_SEQ_LEN + j] = min_value;
      }
    }
  }

  ACL_CHECK_RET(aclrtMalloc(&attn_mask_gm_, MAX_SEQ_LEN * MAX_SEQ_LEN * sizeof(T), ACL_MEM_TYPE_HIGH_BAND_WIDTH));
  ACL_CHECK_RET(aclrtMemcpy(attn_mask_gm_, MAX_SEQ_LEN * MAX_SEQ_LEN * sizeof(T), mask.data(),
                            MAX_SEQ_LEN * MAX_SEQ_LEN * sizeof(T), aclrtMemcpyKind::ACL_MEMCPY_HOST_TO_DEVICE));
}

template <typename T>
void PagedAttention<T>::GenerateTilingData(bool is_context_stage, uint32_t seq_len, uint32_t seq_block_num,
                                           int32_t token_pos) {
  if (std::is_same<T, aclFloat16>::value) {
    tiling_data_.data_type = static_cast<uint32_t>(TilingDataType::FLOAT16);
  } else if (std::is_same<T, float>::value) {
    tiling_data_.data_type = static_cast<uint32_t>(TilingDataType::FLOAT32);
  }

  tiling_data_.context_stage = static_cast<uint32_t>(is_context_stage);
  tiling_data_.seq_len = seq_len;
  tiling_data_.seq_block_num = seq_block_num;
  tiling_data_.block_token_num = block_token_num_;
  tiling_data_.token_pos = token_pos;
  tiling_data_.head_size = head_size_;
  tiling_data_.head_dim = head_dim_;
  tiling_data_.used_core_num = MAX_USED_CORE_NUM;
  tiling_data_.max_seq_len = MAX_SEQ_LEN;

  float scale = 1.0 / std::sqrt(head_dim_);
  tiling_data_.scale = *reinterpret_cast<uint32_t*>(&scale);

  aclFloat16 scale_fp16 = aclFloatToFloat16(scale);
  tiling_data_.scale_fp16 = *reinterpret_cast<uint16_t*>(&scale_fp16);

  // q * k_T
  {
    matmul_tiling::DataType computing_dtype;

    if (std::is_same<T, aclFloat16>::value) {
      computing_dtype = matmul_tiling::DataType::DT_FLOAT16;
    }

    // Q
    matmul_tiling::TPosition left_pos = matmul_tiling::TPosition::GM;
    matmul_tiling::CubeFormat left_format = matmul_tiling::CubeFormat::ND;
    matmul_tiling::DataType left_dtype = computing_dtype;
    bool transpose_a = false;

    // K_T
    matmul_tiling::TPosition right_pos = matmul_tiling::TPosition::GM;
    matmul_tiling::CubeFormat right_format = matmul_tiling::CubeFormat::ND;
    matmul_tiling::DataType right_dtype = computing_dtype;
    bool transpose_b = is_context_stage ? false : true;

    matmul_tiling::TPosition res_pos = matmul_tiling::TPosition::GM;
    matmul_tiling::CubeFormat res_format = matmul_tiling::CubeFormat::ND;
    matmul_tiling::DataType res_dtype = computing_dtype;
    bool is_bias = false;

    matmul_tiling::TPosition bias_pos = matmul_tiling::TPosition::GM;
    matmul_tiling::CubeFormat bias_format = matmul_tiling::CubeFormat::ND;
    matmul_tiling::DataType bias_dtype = computing_dtype;

    size_t m = is_context_stage ? seq_len : 1;
    size_t n = is_context_stage ? seq_len : 1;
    size_t k = head_dim_;

    matmul_tiling::MatmulApiTiling tiling_api;
    tiling_api.SetAType(left_pos, left_format, left_dtype, transpose_a);
    tiling_api.SetBType(right_pos, right_format, right_dtype, transpose_b);
    tiling_api.SetCType(res_pos, res_format, res_dtype);
    tiling_api.SetBiasType(bias_pos, bias_format, bias_dtype);
    tiling_api.SetOrgShape(m, n, k);
    tiling_api.SetShape(m, n, k);
    tiling_api.SetBias(is_bias);
    tiling_api.SetBufferSpace(-1, -1, -1);

    optiling::TCubeTiling tiling_data;
    if (tiling_api.GetTiling(tiling_data) == -1) {
      std::cerr << "Get TCubeTiling error." << std::endl;
      assert(false);
    }

    matmul_tiling::SysTilingTempBufSize buf_size;
    if (MatmulGetTmpBufSize(tiling_data, buf_size) == -1) {
      std::cerr << "Get TCubeTiling temp buf size error." << std::endl;
      assert(false);
    }

    tiling_data_.qk_ub_size = buf_size.ubSize;

    uint32_t tiling_size = tiling_data.GetDataSize();
    tiling_data.SaveToBuffer(&tiling_data_.cube_tiling_qk, tiling_size);
  }

  // w * v
  {
    matmul_tiling::DataType computing_dtype;

    if (std::is_same<T, aclFloat16>::value) {
      computing_dtype = matmul_tiling::DataType::DT_FLOAT16;
    }

    // W
    matmul_tiling::TPosition left_pos = matmul_tiling::TPosition::GM;
    matmul_tiling::CubeFormat left_format = matmul_tiling::CubeFormat::ND;
    matmul_tiling::DataType left_dtype = computing_dtype;
    bool transpose_a = false;

    // V
    matmul_tiling::TPosition right_pos = matmul_tiling::TPosition::GM;
    matmul_tiling::CubeFormat right_format = matmul_tiling::CubeFormat::ND;
    matmul_tiling::DataType right_dtype = computing_dtype;
    bool transpose_b = false;

    matmul_tiling::TPosition res_pos = matmul_tiling::TPosition::GM;
    matmul_tiling::CubeFormat res_format = matmul_tiling::CubeFormat::ND;
    matmul_tiling::DataType res_dtype = computing_dtype;
    bool is_bias = false;

    matmul_tiling::TPosition bias_pos = matmul_tiling::TPosition::GM;
    matmul_tiling::CubeFormat bias_format = matmul_tiling::CubeFormat::ND;
    matmul_tiling::DataType bias_dtype = computing_dtype;

    size_t m = is_context_stage ? seq_len : 1;
    size_t n = head_dim_;
    size_t k = is_context_stage ? seq_len : 1;

    matmul_tiling::MatmulApiTiling tiling_api;
    tiling_api.SetAType(left_pos, left_format, left_dtype, transpose_a);
    tiling_api.SetBType(right_pos, right_format, right_dtype, transpose_b);
    tiling_api.SetCType(res_pos, res_format, res_dtype);
    tiling_api.SetBiasType(bias_pos, bias_format, bias_dtype);
    tiling_api.SetOrgShape(m, n, k);
    tiling_api.SetShape(m, n, k);
    tiling_api.SetBias(is_bias);
    tiling_api.SetBufferSpace(-1, -1, -1);

    optiling::TCubeTiling tiling_data;
    if (tiling_api.GetTiling(tiling_data) == -1) {
      assert(false);
    }

    matmul_tiling::SysTilingTempBufSize buf_size;
    if (MatmulGetTmpBufSize(tiling_data, buf_size) == -1) {
      assert(false);
    }

    tiling_data_.wv_ub_size = buf_size.ubSize;

    uint32_t tiling_size = tiling_data.GetDataSize();
    tiling_data.SaveToBuffer(&tiling_data_.cube_tiling_wv, tiling_size);
  }

  // Softmax
  std::vector<int64_t> shape_vec = {seq_len, seq_len};
  ge::Shape src_shape(shape_vec);

  optiling::SoftMaxTiling softmax_tiling;
  const uint32_t local_workspace_size = AscendC::GetSoftMaxMinTmpSize(src_shape, sizeof(aclFloat16), false);
  AscendC::SoftMaxTilingFunc(src_shape, sizeof(aclFloat16), local_workspace_size, softmax_tiling);

  uint32_t tiling_size = softmax_tiling.GetDataSize();
  softmax_tiling.SaveToBuffer(&tiling_data_.softmax_tiling, tiling_size);
}

template <typename T>
void PagedAttention<T>::CopyTilingToDevice(aclrtStream stream) {
  ACL_CHECK_RET(aclrtMemcpyAsync(tiling_buffer_gm_, tiling_size_, &tiling_data_, tiling_size_,
                                 aclrtMemcpyKind::ACL_MEMCPY_HOST_TO_DEVICE, stream));
  ACL_CHECK_RET(aclrtSynchronizeStream(stream));
}

template <typename T>
void PagedAttention<T>::Forward(void* output, void* qkv_tensor, void* seq_offset, void** kv_list, void* block_offset,
                                void* rope_pos, int batch_size, int total_token_num, int total_block_num,
                                int layer_index, bool is_context_stage, aclrtStream stream) {
  ACL_CHECK_RET(aclrtMemcpyAsync(kv_list_, layer_num_ * total_block_num * 2 * sizeof(void*), kv_list,
                                 layer_num_ * total_block_num * 2 * sizeof(void*),
                                 aclrtMemcpyKind::ACL_MEMCPY_DEVICE_TO_HOST, stream));

  ACL_CHECK_RET(aclrtMemcpyAsync(kv_cache_offset_, (batch_size + 1) * sizeof(int32_t), block_offset,
                                 (batch_size + 1) * sizeof(int32_t), aclrtMemcpyKind::ACL_MEMCPY_DEVICE_TO_HOST,
                                 stream));

  ACL_CHECK_RET(aclrtSynchronizeStream(stream));

  if (is_context_stage) {
    ACL_CHECK_RET(aclrtMemcpy(prefill_token_offset_, (batch_size + 1) * sizeof(uint64_t), seq_offset,
                              (batch_size + 1) * sizeof(uint64_t), aclrtMemcpyKind::ACL_MEMCPY_DEVICE_TO_HOST));

    // Loop every sequence.
    size_t total_seq_len_idx = 0;
    size_t total_block_num_idx = 0;
    for (size_t b_idx = 0; b_idx < batch_size; ++b_idx) {
      uint32_t cur_seq_len = prefill_token_offset_[b_idx + 1] - prefill_token_offset_[b_idx];
      uint32_t cur_block_num = kv_cache_offset_[b_idx + 1] - kv_cache_offset_[b_idx];

      void* q_buffer;
      void* k_buffer;
      void* v_buffer;
      void* o_buffer;
      size_t buffer_size = cur_seq_len * head_size_ * head_dim_ * sizeof(T);
      ACL_CHECK_RET(aclrtMalloc(&q_buffer, buffer_size, ACL_MEM_TYPE_HIGH_BAND_WIDTH));
      ACL_CHECK_RET(aclrtMalloc(&k_buffer, buffer_size, ACL_MEM_TYPE_HIGH_BAND_WIDTH));
      ACL_CHECK_RET(aclrtMalloc(&v_buffer, buffer_size, ACL_MEM_TYPE_HIGH_BAND_WIDTH));
      ACL_CHECK_RET(aclrtMalloc(&o_buffer, buffer_size, ACL_MEM_TYPE_HIGH_BAND_WIDTH));

      size_t b_offset = total_seq_len_idx * head_size_ * head_dim_ * sizeof(T) * 3;

      slice_.Forward(q_buffer, qkv_tensor + b_offset, 0, head_size_ * head_dim_, head_size_ * head_dim_ * 3,
                     cur_seq_len, stream);
      slice_.Forward(k_buffer, qkv_tensor + b_offset, head_size_ * head_dim_, head_size_ * head_dim_,
                     head_size_ * head_dim_ * 3, cur_seq_len, stream);
      slice_.Forward(v_buffer, qkv_tensor + b_offset, head_size_ * head_dim_ * 2, head_size_ * head_dim_,
                     head_size_ * head_dim_ * 3, cur_seq_len, stream);

      // ROPE
      size_t rope_offset = total_seq_len_idx * sizeof(int64_t);
      rope_.SetInput((int64_t*)(rope_pos + rope_offset), (T*)q_buffer, (T*)k_buffer, cur_seq_len, stream);
      rope_.Forward();

      // Cache KV
      void** k_list = kv_list_ + layer_index * total_block_num * 2 + total_block_num_idx;
      void** v_list = k_list + total_block_num;
      for (size_t token_idx = 0; token_idx < cur_seq_len; ++token_idx) {
        int b_block_index = token_idx / block_token_num_;
        int b_block_offset = token_idx % block_token_num_;

        void* k_dst = k_list[b_block_index] + b_block_offset * head_dim_ * kv_head_size_ * sizeof(T);
        void* v_dst = v_list[b_block_index] + b_block_offset * head_dim_ * kv_head_size_ * sizeof(T);

        void* k_src = k_buffer + token_idx * kv_head_size_ * head_dim_ * sizeof(T);
        void* v_src = v_buffer + token_idx * kv_head_size_ * head_dim_ * sizeof(T);

        ACL_CHECK_RET(aclrtMemcpyAsync(k_dst, kv_head_size_ * head_dim_ * sizeof(T), k_src,
                                       kv_head_size_ * head_dim_ * sizeof(T),
                                       aclrtMemcpyKind::ACL_MEMCPY_DEVICE_TO_DEVICE, stream));
        ACL_CHECK_RET(aclrtMemcpyAsync(v_dst, kv_head_size_ * head_dim_ * sizeof(T), v_src,
                                       kv_head_size_ * head_dim_ * sizeof(T),
                                       aclrtMemcpyKind::ACL_MEMCPY_DEVICE_TO_DEVICE, stream));
      }

      void* q_buffer_2;
      void* k_buffer_2;
      void* v_buffer_2;
      ACL_CHECK_RET(aclrtMalloc(&q_buffer_2, buffer_size, ACL_MEM_TYPE_HIGH_BAND_WIDTH));
      ACL_CHECK_RET(aclrtMalloc(&k_buffer_2, buffer_size, ACL_MEM_TYPE_HIGH_BAND_WIDTH));
      ACL_CHECK_RET(aclrtMalloc(&v_buffer_2, buffer_size, ACL_MEM_TYPE_HIGH_BAND_WIDTH));

      // [seq_len, head_size, head_dim] => [head_size, seq_len, head_dim]
      permute_.Forward(q_buffer_2, q_buffer, {cur_seq_len, head_size_, head_dim_}, {1, 0, 2}, stream);
      permute_.Forward(k_buffer_2, k_buffer, {cur_seq_len, head_size_, head_dim_}, {1, 2, 0}, stream);
      permute_.Forward(v_buffer_2, v_buffer, {cur_seq_len, head_size_, head_dim_}, {1, 0, 2}, stream);

      GenerateTilingData(true, cur_seq_len, cur_block_num, cur_seq_len - 1);
      CopyTilingToDevice(stream);

      ACLRT_LAUNCH_KERNEL(InvokePagedAttentionKernel)
      (tiling_data_.used_core_num, stream, q_buffer_2, k_buffer_2, v_buffer_2, attn_mask_gm_, k_list, v_list, o_buffer,
       workspace_gm_, tiling_buffer_gm_);

      size_t output_offset = total_seq_len_idx * head_size_ * head_dim_ * sizeof(T);
      permute_.Forward(output + output_offset, o_buffer, {head_size_, cur_seq_len, head_dim_}, {1, 0, 2}, stream);

      total_seq_len_idx += cur_seq_len;
      total_block_num_idx += cur_block_num;

      ACL_CHECK_RET(aclrtFree(q_buffer));
      ACL_CHECK_RET(aclrtFree(k_buffer));
      ACL_CHECK_RET(aclrtFree(v_buffer));
      ACL_CHECK_RET(aclrtFree(o_buffer));

      ACL_CHECK_RET(aclrtFree(q_buffer_2));
      ACL_CHECK_RET(aclrtFree(k_buffer_2));
      ACL_CHECK_RET(aclrtFree(v_buffer_2));
    }
  } else {
    // Decode.
    ACL_CHECK_RET(aclrtMemcpy(decode_tokens_len_, batch_size * sizeof(int32_t), seq_offset,
                              batch_size * sizeof(int32_t), aclrtMemcpyKind::ACL_MEMCPY_DEVICE_TO_HOST));

    // Loop every sequence.
    size_t total_block_num_idx = 0;
    for (size_t b_idx = 0; b_idx < batch_size; ++b_idx) {
      uint32_t cur_seq_len = decode_tokens_len_[b_idx];
      uint32_t cur_block_num = kv_cache_offset_[b_idx + 1] - kv_cache_offset_[b_idx];

      void* q_buffer;
      void* k_buffer;
      void* v_buffer;
      void* o_buffer;
      size_t buffer_size = head_size_ * head_dim_ * sizeof(T);
      ACL_CHECK_RET(aclrtMalloc(&q_buffer, buffer_size, ACL_MEM_TYPE_HIGH_BAND_WIDTH));
      ACL_CHECK_RET(aclrtMalloc(&k_buffer, buffer_size, ACL_MEM_TYPE_HIGH_BAND_WIDTH));
      ACL_CHECK_RET(aclrtMalloc(&v_buffer, buffer_size, ACL_MEM_TYPE_HIGH_BAND_WIDTH));
      ACL_CHECK_RET(aclrtMalloc(&o_buffer, buffer_size, ACL_MEM_TYPE_HIGH_BAND_WIDTH));

      size_t b_offset = b_idx * head_size_ * head_dim_ * sizeof(T) * 3;
      slice_.Forward(q_buffer, qkv_tensor + b_offset, 0, head_size_ * head_dim_, head_size_ * head_dim_ * 3, 1, stream);
      slice_.Forward(k_buffer, qkv_tensor + b_offset, head_size_ * head_dim_, head_size_ * head_dim_,
                     head_size_ * head_dim_ * 3, 1, stream);
      slice_.Forward(v_buffer, qkv_tensor + b_offset, head_size_ * head_dim_ * 2, head_size_ * head_dim_,
                     head_size_ * head_dim_ * 3, 1, stream);

      // ROPE
      size_t rope_offset = b_idx * sizeof(int64_t);
      rope_.SetInput((int64_t*)(rope_pos + rope_offset), (T*)q_buffer, (T*)k_buffer, 1, stream);
      rope_.Forward();

      // Cache KV
      void** k_list = kv_list_ + layer_index * total_block_num * 2 + total_block_num_idx;
      void** v_list = k_list + total_block_num;

      void** k_list_dev = kv_list + layer_index * total_block_num * 2 + total_block_num_idx;
      void** v_list_dev = k_list_dev + total_block_num;
      {
        size_t token_idx = cur_seq_len - 1;
        int b_block_index = token_idx / block_token_num_;
        int b_block_offset = token_idx % block_token_num_;

        void* k_dst = k_list[b_block_index] + b_block_offset * head_dim_ * kv_head_size_ * sizeof(T);
        void* v_dst = v_list[b_block_index] + b_block_offset * head_dim_ * kv_head_size_ * sizeof(T);

        void* k_src = k_buffer;
        void* v_src = v_buffer;

        ACL_CHECK_RET(aclrtMemcpyAsync(k_dst, kv_head_size_ * head_dim_ * sizeof(T), k_src,
                                       kv_head_size_ * head_dim_ * sizeof(T),
                                       aclrtMemcpyKind::ACL_MEMCPY_DEVICE_TO_DEVICE, stream));
        ACL_CHECK_RET(aclrtMemcpyAsync(v_dst, kv_head_size_ * head_dim_ * sizeof(T), v_src,
                                       kv_head_size_ * head_dim_ * sizeof(T),
                                       aclrtMemcpyKind::ACL_MEMCPY_DEVICE_TO_DEVICE, stream));
      }

      void* q_buffer_2;
      void* k_buffer_2;
      void* v_buffer_2;
      ACL_CHECK_RET(aclrtMalloc(&q_buffer_2, buffer_size, ACL_MEM_TYPE_HIGH_BAND_WIDTH));
      ACL_CHECK_RET(aclrtMalloc(&k_buffer_2, buffer_size, ACL_MEM_TYPE_HIGH_BAND_WIDTH));
      ACL_CHECK_RET(aclrtMalloc(&v_buffer_2, buffer_size, ACL_MEM_TYPE_HIGH_BAND_WIDTH));

      // [1, head_size, head_dim] => [head_size, 1, head_dim]
      permute_.Forward(q_buffer_2, q_buffer, {1, head_size_, head_dim_}, {1, 0, 2}, stream);
      permute_.Forward(k_buffer_2, k_buffer, {1, head_size_, head_dim_}, {1, 0, 2}, stream);
      permute_.Forward(v_buffer_2, v_buffer, {1, head_size_, head_dim_}, {1, 0, 2}, stream);

      GenerateTilingData(false, cur_seq_len, cur_block_num, cur_seq_len - 1);
      CopyTilingToDevice(stream);


      ACLRT_LAUNCH_KERNEL(InvokePagedAttentionKernel)
      (tiling_data_.used_core_num, stream, q_buffer_2, k_buffer_2, v_buffer_2, attn_mask_gm_, k_list_dev, v_list_dev,
       o_buffer, workspace_gm_, tiling_buffer_gm_);

      size_t output_offset = b_idx * head_size_ * head_dim_ * sizeof(T);
      permute_.Forward(output + output_offset, o_buffer, {head_size_, 1, head_dim_}, {1, 0, 2}, stream);

      total_block_num_idx += cur_block_num;

      ACL_CHECK_RET(aclrtFree(q_buffer));
      ACL_CHECK_RET(aclrtFree(k_buffer));
      ACL_CHECK_RET(aclrtFree(v_buffer));
      ACL_CHECK_RET(aclrtFree(o_buffer));

      ACL_CHECK_RET(aclrtFree(q_buffer_2));
      ACL_CHECK_RET(aclrtFree(k_buffer_2));
      ACL_CHECK_RET(aclrtFree(v_buffer_2));
    }
  }
}

template class PagedAttention<aclFloat16>;
template class PagedAttention<float>;

}  // namespace ascend
}  // namespace llm_kernels
