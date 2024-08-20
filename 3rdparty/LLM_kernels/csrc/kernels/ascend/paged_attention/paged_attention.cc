/* Copyright 2024 Tencent Inc.  All rights reserved.
Partialy modify from
https://github.com/PaddlePaddle/PaddleCustomDevice/blob/develop/backends/npu/custom_op/llama_infer/atb_ops

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

#include "tiling/softmax/softmax_tiling.h"
#include "tiling/tiling_api.h"

#include "csrc/kernels/ascend/paged_attention/paged_attention_tiling.h"

#ifdef ENABLE_ACL_ATB
#  include "atb/atb_infer.h"
#endif

// The max possible batch size.
#define MAX_BATCH_SIZE 256

// The max possible block num for one batch.
#define MAX_SEQ_BLOCK_NUM 2048

// The max total seq len of all prompts in one batch.
#define MAX_TOTAL_SEQ_LEN 4096

// The max seq_len
#define MAX_SEQ_LEN 4096

// The identify of permute mask.
#define PERMUTE_PREFILL_PRE 0x10000000
#define PERMUTE_PREFILL_POST 0x20000000

// The identify of slice mask.
#define SLICE_TILING_0 0x30000000
#define SLICE_TILING_1 0x40000000
#define SLICE_TILING_2 0x50000000

namespace llm_kernels {
namespace ascend {

// The tiling data for current request.
PagedAttentionTilingData prefill_tiling_data_;
PagedAttentionTilingData decode_tiling_data_;

template <typename T>
PagedAttention<T>::~PagedAttention() {
  free(prefill_token_offset_);
  free(decode_tokens_len_);

  free(kv_list_);
  free(kv_cache_offset_);

  ACL_CHECK_RET(aclrtFree(tiling_buffer_gm_));
  ACL_CHECK_RET(aclrtFree(workspace_gm_));

  ACL_CHECK_RET(aclrtFree(permute_tiling_gm_));
  ACL_CHECK_RET(aclrtFree(slice_tiling_gm_));

  ACL_CHECK_RET(aclrtFree(q_buffer_));
  ACL_CHECK_RET(aclrtFree(k_buffer_));
  ACL_CHECK_RET(aclrtFree(v_buffer_));
  ACL_CHECK_RET(aclrtFree(o_buffer_));

  ACL_CHECK_RET(aclrtFree(q_buffer_2_));
  ACL_CHECK_RET(aclrtFree(k_buffer_2_));
  ACL_CHECK_RET(aclrtFree(v_buffer_2_));
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

  // prefill: 2 * seq_len, decode: 2
  size_t permute_tiling_size = permute_.GetTilingSize() * MAX_SEQ_LEN * 2;
  ACL_CHECK_RET(aclrtMalloc(&permute_tiling_gm_, permute_tiling_size, ACL_MEM_TYPE_HIGH_BAND_WIDTH));

  size_t slice_tiling_size = slice_.GetTilingSize() * MAX_SEQ_LEN * 3;
  ACL_CHECK_RET(aclrtMalloc(&slice_tiling_gm_, slice_tiling_size, ACL_MEM_TYPE_HIGH_BAND_WIDTH));

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

  size_t buffer_size = MAX_SEQ_LEN * head_size_ * head_dim_ * sizeof(T);
  ACL_CHECK_RET(aclrtMalloc(&q_buffer_, buffer_size, ACL_MEM_TYPE_HIGH_BAND_WIDTH));
  ACL_CHECK_RET(aclrtMalloc(&k_buffer_, buffer_size, ACL_MEM_TYPE_HIGH_BAND_WIDTH));
  ACL_CHECK_RET(aclrtMalloc(&v_buffer_, buffer_size, ACL_MEM_TYPE_HIGH_BAND_WIDTH));
  ACL_CHECK_RET(aclrtMalloc(&o_buffer_, buffer_size, ACL_MEM_TYPE_HIGH_BAND_WIDTH));

  ACL_CHECK_RET(aclrtMalloc(&q_buffer_2_, buffer_size, ACL_MEM_TYPE_HIGH_BAND_WIDTH));
  ACL_CHECK_RET(aclrtMalloc(&k_buffer_2_, buffer_size, ACL_MEM_TYPE_HIGH_BAND_WIDTH));
  ACL_CHECK_RET(aclrtMalloc(&v_buffer_2_, buffer_size, ACL_MEM_TYPE_HIGH_BAND_WIDTH));

  InitAttnMask();
  InitPermuteTiling(stream);
  InitSliceTiling(stream);

  InitTilingData(true);
  InitTilingData(false);
}

template <typename T>
void PagedAttention<T>::InitAttnMask() {
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
void PagedAttention<T>::InitPermuteTiling(aclrtStream stream) {
  uint32_t offset;
  for (size_t i = 1; i <= MAX_SEQ_LEN; ++i) {
    offset = i * permute_.GetTilingSize() * 2;
    permute_.CacheTiling(permute_tiling_gm_ + offset, (PERMUTE_PREFILL_PRE | i), {i, head_size_, head_dim_}, {1, 0, 2},
                         stream);
    permute_.CacheTiling(permute_tiling_gm_ + offset + permute_.GetTilingSize(), (PERMUTE_PREFILL_POST | i),
                         {head_size_, i, head_dim_}, {1, 0, 2}, stream);
  }
}

template <typename T>
void PagedAttention<T>::InitSliceTiling(aclrtStream stream) {
  uint32_t offset;
  for (size_t i = 1; i <= MAX_SEQ_LEN; ++i) {
    offset = i * slice_.GetTilingSize() * 3;

    slice_.CacheTiling(slice_tiling_gm_ + offset, (SLICE_TILING_0 | i), 0, head_size_ * head_dim_,
                       head_size_ * head_dim_ * 3, i, stream);

    slice_.CacheTiling(slice_tiling_gm_ + offset + slice_.GetTilingSize(), (SLICE_TILING_1 | i), head_size_ * head_dim_,
                       head_size_ * head_dim_, head_size_ * head_dim_ * 3, i, stream);

    slice_.CacheTiling(slice_tiling_gm_ + offset + (slice_.GetTilingSize() * 2), (SLICE_TILING_2 | i),
                       head_size_ * head_dim_ * 2, head_size_ * head_dim_, head_size_ * head_dim_ * 3, i, stream);
  }
}

template <typename T>
void PagedAttention<T>::InitTilingData(bool is_context_stage) {
  PagedAttentionTilingData* tiling_data = is_context_stage ? &prefill_tiling_data_ : &decode_tiling_data_;

  if (std::is_same<T, aclFloat16>::value) {
    tiling_data->data_type = static_cast<uint32_t>(TilingDataType::FLOAT16);
  } else if (std::is_same<T, float>::value) {
    tiling_data->data_type = static_cast<uint32_t>(TilingDataType::FLOAT32);
  }

  tiling_data->context_stage = static_cast<uint32_t>(is_context_stage);
  tiling_data->block_token_num = block_token_num_;
  tiling_data->head_size = head_size_;
  tiling_data->head_dim = head_dim_;
  tiling_data->used_core_num = head_size_ / 2;
  tiling_data->max_seq_len = MAX_SEQ_LEN;

  float scale = 1.0 / std::sqrt(head_dim_);
  tiling_data->scale = *reinterpret_cast<uint32_t*>(&scale);

  aclFloat16 scale_fp16 = aclFloatToFloat16(scale);
  tiling_data->scale_fp16 = *reinterpret_cast<uint16_t*>(&scale_fp16);
}

template <typename T>
void PagedAttention<T>::GenerateTilingData(bool is_context_stage, uint32_t seq_len, uint32_t seq_block_num,
                                           int32_t token_pos) {
  PagedAttentionTilingData* tiling_data = is_context_stage ? &prefill_tiling_data_ : &decode_tiling_data_;

  tiling_data->seq_len = seq_len;
  tiling_data->seq_block_num = seq_block_num;
  tiling_data->token_pos = token_pos;

  matmul_tiling::DataType computing_dtype;
  if (std::is_same<T, aclFloat16>::value) {
    computing_dtype = matmul_tiling::DataType::DT_FLOAT16;
  }

  // q * k_T
  {
    // Reinitialize prefill tilling every time.
    // As a work around for matmul_tiling::MatmulApiTiling.GetTiling()'s occasional crash.
    matmul_tiling::MatmulApiTiling qk_tiling;

    qk_tiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, computing_dtype, false);
    qk_tiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, computing_dtype, true);
    qk_tiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, computing_dtype);
    qk_tiling.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, computing_dtype);
    qk_tiling.SetBias(false);
    qk_tiling.SetBufferSpace(-1, -1, -1);

    size_t m = is_context_stage ? seq_len : 1;
    size_t n = is_context_stage ? seq_len : 1;
    size_t k = head_dim_;

    qk_tiling.SetOrgShape(m, n, k);
    qk_tiling.SetShape(m, n, k);

    optiling::TCubeTiling cube_tiling_data;
    if (qk_tiling.GetTiling(cube_tiling_data) == -1) {
      std::cerr << "Get " << (is_context_stage ? "prefil" : "decode") << " qk TCubeTiling error, m:" << m << ", n:" << n
                << ", k:" << k << std::endl;
      assert(false);
    }

    uint32_t tiling_size = cube_tiling_data.GetDataSize();
    cube_tiling_data.SaveToBuffer(&tiling_data->cube_tiling_qk, tiling_size);
  }

  // w * v
  {
    matmul_tiling::MatmulApiTiling vw_tiling;

    vw_tiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, computing_dtype, false);
    vw_tiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, computing_dtype, false);
    vw_tiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, computing_dtype);
    vw_tiling.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, computing_dtype);
    vw_tiling.SetBias(false);
    vw_tiling.SetBufferSpace(-1, -1, -1);

    size_t m = is_context_stage ? seq_len : 1;
    size_t n = head_dim_;
    size_t k = is_context_stage ? seq_len : 1;

    vw_tiling.SetOrgShape(m, n, k);
    vw_tiling.SetShape(m, n, k);

    optiling::TCubeTiling cube_tiling_data;
    if (vw_tiling.GetTiling(cube_tiling_data) == -1) {
      std::cerr << "Get " << (is_context_stage ? "prefil" : "decode") << " vw TCubeTiling error, m:" << m << ", n:" << n
                << ", k:" << k << std::endl;
      assert(false);
    }

    uint32_t tiling_size = cube_tiling_data.GetDataSize();
    cube_tiling_data.SaveToBuffer(&tiling_data->cube_tiling_wv, tiling_size);
  }

  // Softmax
  std::vector<int64_t> shape_vec = {seq_len, seq_len};
  ge::Shape src_shape(shape_vec);

  optiling::SoftMaxTiling softmax_tiling;
  const uint32_t local_workspace_size = AscendC::GetSoftMaxMinTmpSize(src_shape, sizeof(aclFloat16), false);
  AscendC::SoftMaxTilingFunc(src_shape, sizeof(aclFloat16), local_workspace_size, softmax_tiling);

  uint32_t tiling_size = softmax_tiling.GetDataSize();
  softmax_tiling.SaveToBuffer(&tiling_data->softmax_tiling, tiling_size);
}

template <typename T>
void PagedAttention<T>::CopyTilingToDevice(bool is_context_stage, aclrtStream stream) {
  PagedAttentionTilingData* tiling_data = is_context_stage ? &prefill_tiling_data_ : &decode_tiling_data_;

  ACL_CHECK_RET(aclrtMemcpyAsync(tiling_buffer_gm_, tiling_size_, tiling_data, tiling_size_,
                                 aclrtMemcpyKind::ACL_MEMCPY_HOST_TO_DEVICE, stream));
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

      size_t b_offset = total_seq_len_idx * head_size_ * head_dim_ * sizeof(T) * 3;

      int slice0_block_dim;
      void* slice0_tiling_data =
          slice_.GetTilingData(SLICE_TILING_0 | static_cast<size_t>(cur_seq_len), slice0_block_dim);
      slice_.Forward(q_buffer_, qkv_tensor + b_offset, slice0_tiling_data, slice0_block_dim, stream);

      int slice1_block_dim;
      void* slice1_tiling_data =
          slice_.GetTilingData(SLICE_TILING_1 | static_cast<size_t>(cur_seq_len), slice1_block_dim);
      slice_.Forward(k_buffer_, qkv_tensor + b_offset, slice1_tiling_data, slice1_block_dim, stream);

      int slice2_block_dim;
      void* slice2_tiling_data =
          slice_.GetTilingData(SLICE_TILING_2 | static_cast<size_t>(cur_seq_len), slice2_block_dim);
      slice_.Forward(v_buffer_, qkv_tensor + b_offset, slice2_tiling_data, slice2_block_dim, stream);

      // stream different batch idx.
      if (b_idx > 0) {
        aclrtSynchronizeStream(stream);
      }

      // ROPE
      size_t rope_offset = total_seq_len_idx * sizeof(int64_t);
      rope_.SetInput((int64_t*)(rope_pos + rope_offset), (T*)q_buffer_, (T*)k_buffer_, cur_seq_len, stream);
      rope_.Forward();

      // Cache KV
      void** k_list = kv_list_ + layer_index * total_block_num * 2 + total_block_num_idx;
      void** v_list = k_list + total_block_num;
      for (size_t token_idx = 0; token_idx < cur_seq_len;) {
        int b_block_index = token_idx / block_token_num_;
        int b_block_offset = token_idx % block_token_num_;

        void* k_dst = k_list[b_block_index] + b_block_offset * head_dim_ * kv_head_size_ * sizeof(T);
        void* v_dst = v_list[b_block_index] + b_block_offset * head_dim_ * kv_head_size_ * sizeof(T);

        void* k_src = k_buffer_ + token_idx * kv_head_size_ * head_dim_ * sizeof(T);
        void* v_src = v_buffer_ + token_idx * kv_head_size_ * head_dim_ * sizeof(T);

        if (token_idx + block_token_num_ <= cur_seq_len) {
          ACL_CHECK_RET(aclrtMemcpyAsync(k_dst, block_token_num_ * kv_head_size_ * head_dim_ * sizeof(T), k_src,
                                         block_token_num_ * kv_head_size_ * head_dim_ * sizeof(T),
                                         aclrtMemcpyKind::ACL_MEMCPY_DEVICE_TO_DEVICE, stream));
          ACL_CHECK_RET(aclrtMemcpyAsync(v_dst, block_token_num_ * kv_head_size_ * head_dim_ * sizeof(T), v_src,
                                         block_token_num_ * kv_head_size_ * head_dim_ * sizeof(T),
                                         aclrtMemcpyKind::ACL_MEMCPY_DEVICE_TO_DEVICE, stream));
          token_idx += block_token_num_;
        } else {
          int tail = cur_seq_len % block_token_num_;
          ACL_CHECK_RET(aclrtMemcpyAsync(k_dst, tail * kv_head_size_ * head_dim_ * sizeof(T), k_src,
                                         tail * kv_head_size_ * head_dim_ * sizeof(T),
                                         aclrtMemcpyKind::ACL_MEMCPY_DEVICE_TO_DEVICE, stream));
          ACL_CHECK_RET(aclrtMemcpyAsync(v_dst, tail * kv_head_size_ * head_dim_ * sizeof(T), v_src,
                                         tail * kv_head_size_ * head_dim_ * sizeof(T),
                                         aclrtMemcpyKind::ACL_MEMCPY_DEVICE_TO_DEVICE, stream));
          token_idx += tail;
        }
      }

      int pre_block_dim;
      void* pre_tiling_data = permute_.GetTilingData(PERMUTE_PREFILL_PRE | cur_seq_len, pre_block_dim);
      permute_.Forward(q_buffer_2_, q_buffer_, pre_tiling_data, pre_block_dim, stream);
      permute_.Forward(k_buffer_2_, k_buffer_, pre_tiling_data, pre_block_dim, stream);
      permute_.Forward(v_buffer_2_, v_buffer_, pre_tiling_data, pre_block_dim, stream);

      GenerateTilingData(true, cur_seq_len, cur_block_num, cur_seq_len - 1);
      CopyTilingToDevice(true, stream);

      ACLRT_LAUNCH_KERNEL(InvokePagedAttentionKernel)
      (prefill_tiling_data_.used_core_num, stream, q_buffer_2_, k_buffer_2_, v_buffer_2_, attn_mask_gm_, k_list, v_list,
       o_buffer_, workspace_gm_, tiling_buffer_gm_);

      int post_block_dim;
      size_t output_offset = total_seq_len_idx * head_size_ * head_dim_ * sizeof(T);
      void* post_tiling_data = permute_.GetTilingData(PERMUTE_PREFILL_POST | cur_seq_len, post_block_dim);
      permute_.Forward(output + output_offset, o_buffer_, post_tiling_data, post_block_dim, stream);

      total_seq_len_idx += cur_seq_len;
      total_block_num_idx += cur_block_num;
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

      size_t b_offset = b_idx * head_size_ * head_dim_ * sizeof(T) * 3;

      static int slice0_block_dim = 0;
      static int slice1_block_dim = 0;
      static int slice2_block_dim = 0;
      static void* slice0_tiling_data = nullptr;
      static void* slice1_tiling_data = nullptr;
      static void* slice2_tiling_data = nullptr;
      if (slice0_tiling_data == nullptr) {
        slice0_tiling_data = slice_.GetTilingData(SLICE_TILING_0 | static_cast<size_t>(1), slice0_block_dim);
        slice1_tiling_data = slice_.GetTilingData(SLICE_TILING_1 | static_cast<size_t>(1), slice1_block_dim);
        slice2_tiling_data = slice_.GetTilingData(SLICE_TILING_2 | static_cast<size_t>(1), slice2_block_dim);
      }

      slice_.Forward(q_buffer_, qkv_tensor + b_offset, slice0_tiling_data, slice0_block_dim, stream);
      slice_.Forward(k_buffer_, qkv_tensor + b_offset, slice1_tiling_data, slice1_block_dim, stream);
      slice_.Forward(v_buffer_, qkv_tensor + b_offset, slice2_tiling_data, slice2_block_dim, stream);

      // stream different batch idx.
      if (b_idx > 0) {
        aclrtSynchronizeStream(stream);
      }

      // ROPE
      size_t rope_offset = b_idx * sizeof(int64_t);
      rope_.SetInput((int64_t*)(rope_pos + rope_offset), (T*)q_buffer_, (T*)k_buffer_, 1, stream);
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

        ACL_CHECK_RET(aclrtMemcpyAsync(k_dst, kv_head_size_ * head_dim_ * sizeof(T), k_buffer_,
                                       kv_head_size_ * head_dim_ * sizeof(T),
                                       aclrtMemcpyKind::ACL_MEMCPY_DEVICE_TO_DEVICE, stream));
        ACL_CHECK_RET(aclrtMemcpyAsync(v_dst, kv_head_size_ * head_dim_ * sizeof(T), v_buffer_,
                                       kv_head_size_ * head_dim_ * sizeof(T),
                                       aclrtMemcpyKind::ACL_MEMCPY_DEVICE_TO_DEVICE, stream));
      }

      GenerateTilingData(false, cur_seq_len, cur_block_num, cur_seq_len - 1);
      CopyTilingToDevice(false, stream);

      size_t output_offset = b_idx * head_size_ * head_dim_ * sizeof(T);
      ACLRT_LAUNCH_KERNEL(InvokePagedAttentionKernel)
      (decode_tiling_data_.used_core_num, stream, q_buffer_, k_buffer_, v_buffer_, attn_mask_gm_, k_list_dev,
       v_list_dev, output + output_offset, workspace_gm_, tiling_buffer_gm_);

      total_block_num_idx += cur_block_num;
    }
  }
}

template class PagedAttention<aclFloat16>;
template class PagedAttention<float>;

#ifdef ENABLE_ACL_ATB
template <typename DTYPE>
ATBPagedAttention<DTYPE>::~ATBPagedAttention() {}

template <typename DTYPE>
void ATBPagedAttention<DTYPE>::Forward(void* output, void* qkv_tensor, void* slot_mapping, void* k_cache, void* v_cache,
                                       void* block_tables, const uint32_t max_num_blocks_per_query,
                                       const uint32_t batch_size, const uint32_t total_token_num,
                                       const uint32_t total_block_num, const uint32_t block_token_num,
                                       const uint32_t layer_index, void* seq_len, const bool is_context_stage,
                                       atb::Context* atb_context, void (*ws_func)(size_t, void**)) {
  aclDataType acl_dtype;
  if (std::is_same<DTYPE, aclFloat16>::value) {
    acl_dtype = aclDataType::ACL_FLOAT16;
  } else if (std::is_same<DTYPE, float>::value) {
    acl_dtype = aclDataType::ACL_FLOAT;
  } else {
    throw std::invalid_argument("Invalid matmul type type, only support float16 or float32.");
  }

  atb_op_executor_.ResetVariantPack();
  // qkv_input_tensor_id
  atb_op_executor_.SetInputTensor(qkv_tensor, {total_token_num, (head_size_ + 2 * kv_head_size_) * head_dim_},
                                  acl_dtype);
  // rope_cos_input_tensor_id
  atb_op_executor_.SetInputTensor(rope_cos_workspace_ptr_, {total_token_num, head_dim_}, acl_dtype);
  // rope_sin_input_tensor_id
  atb_op_executor_.SetInputTensor(rope_sin_workspace_ptr_, {total_token_num, head_dim_}, acl_dtype);
  if (is_context_stage) {
    // mask_input_tensor_id
    atb_op_executor_.SetInputTensor(attn_mask_ptr_, {MAX_SEQ_LEN, MAX_SEQ_LEN}, acl_dtype);
  }
  // k_cache_input_tensor_id
  atb_op_executor_.SetInputTensor(k_cache, {total_block_num, block_token_num, kv_head_size_, head_dim_}, acl_dtype);
  // v_cache_input_tensor_id
  atb_op_executor_.SetInputTensor(v_cache, {total_block_num, block_token_num, kv_head_size_, head_dim_}, acl_dtype);
  // slots_input_tensor_id
  atb_op_executor_.SetInputTensor(slot_mapping, {total_token_num}, aclDataType::ACL_INT32);
  if (!is_context_stage) {
    // block_tables_input_tensor_id
    atb_op_executor_.SetInputTensor(block_tables, {total_token_num, max_num_blocks_per_query}, aclDataType::ACL_INT32);
  }
  // seqlen_input_tensor_id
  atb_op_executor_.SetInputTensor(seq_len, {batch_size}, aclDataType::ACL_INT32);
  if (!is_context_stage) {
    // batch_status_input_tensor_id
    // TODO(karlluo): tag which query should be handle
    atb_op_executor_.SetInputTensor(batch_status_.data(), {batch_size}, aclDataType::ACL_INT32);
  }
  atb_op_executor_.SetOutputTensor(output, {total_token_num, head_size_ * head_dim_}, acl_dtype);

  atb_op_executor_.Run(atb_context, ws_func);
}

template <typename DTYPE>
void ATBPagedAttention<DTYPE>::Initialize(uint32_t max_batch_size, uint32_t head_size, uint32_t kv_head_size,
                                          uint32_t head_dim, uint32_t layer_num, uint32_t layer_idx,
                                          uint32_t block_token_num, aclrtStream& stream, const int rank,
                                          const bool is_context_stage, const size_t max_position_embeddings,
                                          const float rope_base, const RotaryEmbeddingType scaling_type,
                                          const float scaling_factor) {
  max_batch_size_ = max_batch_size;
  head_size_ = head_size;
  kv_head_size_ = kv_head_size;
  head_dim_ = head_dim;
  layer_num_ = layer_num;
  block_token_num_ = block_token_num;
  rank_ = rank;
  is_prefill_ = is_context_stage;

  // TODO(karlluo): tag which query should be handle
  batch_status_.resize(max_batch_size_, 1);

  // init rope cos and sin
  InitRopeCosSinWorkspace(max_position_embeddings, rope_base, head_dim, scaling_factor, scaling_type, stream);

  if (is_context_stage) {
    InitAttnMask();
  }

  uint32_t tensor_id = 0;
  // input
  // shape: (ntokens, (head_size + 2 * kv_head_size) * head_dim)
  uint32_t qkv_input_tensor_id = tensor_id++;
  // shape: (ntokens, head_dim)
  uint32_t rope_cos_input_tensor_id = tensor_id++;
  // shape: (ntokens, head_dim)
  uint32_t rope_sin_input_tensor_id = tensor_id++;
  // is_context_stage == true:
  //   shape: (MAX_SEQ_LEN, MAX_SEQ_LEN)
  uint32_t mask_input_tensor_id = is_context_stage ? tensor_id++ : 0;
  // shape: (num_blocks, block_size, k_head_num, head_size)
  uint32_t k_cache_input_tensor_id = tensor_id++;
  // shape: (num_blocks, block_size, k_head_num, head_size)
  uint32_t v_cache_input_tensor_id = tensor_id++;
  // shape: (ntokens)
  uint32_t slots_input_tensor_id = tensor_id++;
  // is_context_stage == false:
  //   shape: (num_tokens, max_num_blocks_per_query)
  uint32_t block_tables_input_tensor_id = !is_context_stage ? tensor_id++ : 0;
  // shape: (batch)
  uint32_t seqlen_input_tensor_id = tensor_id++;
  // is_context_stage == false:
  //   shape: (batch, num_heads, 1, max_seqlen)
  uint32_t batch_status_input_tensor_id = !is_context_stage ? tensor_id++ : seqlen_input_tensor_id;
  // output
  // shape: (ntokens, head_size * head_dim)
  uint32_t attn_output_tensor_id = tensor_id++;

  // intermedia
  uint32_t q_inner_tensor_id = tensor_id++;
  uint32_t k_inner_tensor_id = tensor_id++;
  uint32_t v_inner_tensor_id = tensor_id++;
  uint32_t emb_q_inner_tensor_id = tensor_id++;
  uint32_t emb_k_inner_tensor_id = tensor_id++;

  atb::GraphParam op_graph;
  op_graph.name = "ATBPagedAttentionOp";
  op_graph.inTensorNum = batch_status_input_tensor_id - qkv_input_tensor_id + 1;
  op_graph.outTensorNum = 1;
  op_graph.internalTensorNum = emb_k_inner_tensor_id - q_inner_tensor_id + 1;
  op_graph.nodes.resize(4);  // split qkv + rope + write kv + flash attn/page attn
  uint32_t node_idx = 0;

  // split q,k,v
  {
    atb::Node& op_node = op_graph.nodes.at(node_idx++);
    CreateSplitQKVOperation(head_size, kv_head_size, head_dim, &op_node.operation);
    op_node.inTensorIds = {qkv_input_tensor_id};
    op_node.outTensorIds = {q_inner_tensor_id, k_inner_tensor_id, v_inner_tensor_id};
  }

  // rope
  {
    atb::Node& op_node = op_graph.nodes.at(node_idx++);
    atb::infer::RopeParam op_param;
    op_param.rotaryCoeff = 2;
    atb::CreateOperation(op_param, &op_node.operation);
    op_node.inTensorIds = {q_inner_tensor_id, k_inner_tensor_id, rope_cos_input_tensor_id, rope_sin_input_tensor_id,
                           seqlen_input_tensor_id};
    op_node.outTensorIds = {emb_q_inner_tensor_id, emb_k_inner_tensor_id};
  }

  // write kv
  {
    atb::Node& op_node = op_graph.nodes.at(node_idx++);
    atb::infer::ReshapeAndCacheParam op_param;
    atb::CreateOperation(op_param, &op_node.operation);
    op_node.inTensorIds = {emb_k_inner_tensor_id, v_inner_tensor_id, k_cache_input_tensor_id, v_cache_input_tensor_id,
                           slots_input_tensor_id};
    op_node.outTensorIds = {k_cache_input_tensor_id, v_cache_input_tensor_id};  // write in place
    op_node.inTensorReshapeFuncs.resize(op_node.inTensorIds.size());
    op_node.inTensorReshapeFuncs[0] = [=](const atb::Dims& old_shape, atb::Dims& new_shape) {
      new_shape.dimNum = 3;
      new_shape.dims[0] = old_shape.dims[0];
      new_shape.dims[1] = kv_head_size;
      new_shape.dims[2] = head_dim;
    };
    op_node.inTensorReshapeFuncs[1] = [=](const atb::Dims& old_shape, atb::Dims& new_shape) {
      new_shape.dimNum = 3;
      new_shape.dims[0] = old_shape.dims[0];
      new_shape.dims[1] = kv_head_size;
      new_shape.dims[2] = head_dim;
    };
  }

  // flash attn or paged attn
  if (is_context_stage) {
    atb::Node& op_node = op_graph.nodes.at(node_idx++);
    atb::infer::SelfAttentionParam op_param;
    op_param.headNum = head_size;
    op_param.kvHeadNum = kv_head_size;
    op_param.qkScale = 1.0f / sqrt(head_dim);
    op_param.calcType = atb::infer::SelfAttentionParam::CalcType::PA_ENCODER;
    op_param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM;
    op_param.isTriuMask = 1;
    atb::CreateOperation(op_param, &op_node.operation);
    op_node.inTensorIds = {emb_q_inner_tensor_id, emb_k_inner_tensor_id, v_inner_tensor_id, mask_input_tensor_id,
                           seqlen_input_tensor_id};
    op_node.outTensorIds = {attn_output_tensor_id};
    op_node.inTensorReshapeFuncs.resize(op_node.inTensorIds.size());
  } else {
    atb::Node& op_node = op_graph.nodes.at(node_idx++);
    atb::infer::PagedAttentionParam op_param;
    op_param.headNum = head_size;
    op_param.qkScale = 1.0f / sqrt(head_dim);
    op_param.kvHeadNum = kv_head_size;
    op_param.maskType = atb::infer::PagedAttentionParam::UNDEFINED;
    op_param.batchRunStatusEnable = true;

    atb::CreateOperation(op_param, &op_node.operation);

    op_node.inTensorIds = {emb_q_inner_tensor_id,        k_cache_input_tensor_id, v_cache_input_tensor_id,
                           block_tables_input_tensor_id, seqlen_input_tensor_id,  batch_status_input_tensor_id};
    op_node.outTensorIds = {attn_output_tensor_id};
    op_node.inTensorReshapeFuncs.resize(op_node.inTensorIds.size());
    op_node.inTensorReshapeFuncs[0] = [=](const atb::Dims& old_shape, atb::Dims& new_shape) {
      new_shape.dimNum = 3;
      new_shape.dims[0] = old_shape.dims[0];
      new_shape.dims[1] = kv_head_size;
      new_shape.dims[2] = head_dim;
    };
  }

  op_graph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc>& in_tensor_descs,
                                atb::SVector<atb::TensorDesc>& out_tensor_descs) {
    out_tensor_descs.resize(1);
    out_tensor_descs.at(0) = in_tensor_descs.at(0);
    out_tensor_descs.at(0).shape.dims[0] = in_tensor_descs.at(0).shape.dims[0];
    out_tensor_descs.at(0).shape.dims[1] = head_size * head_dim;
    return atb::NO_ERROR;
  };
  atb_op_executor_.Init(rank, op_graph);
  atb_op_executor_.ResetVariantPack();
}

template <typename DTYPE>
void ATBPagedAttention<DTYPE>::InitRopeCosSinWorkspace(const size_t max_position_embeddings, const float rope_base,
                                                       const uint32_t head_dim, const float scaling_factor,
                                                       const RotaryEmbeddingType scaling_type, aclrtStream& stream) {
  std::vector<DTYPE> cos_workspace_host(max_position_embeddings * head_dim, 0u);
  std::vector<DTYPE> sin_workspace_host(max_position_embeddings * head_dim, 0u);
  ACL_CHECK_RET(aclrtMalloc(&rope_cos_workspace_ptr_, max_position_embeddings * head_dim * sizeof(DTYPE),
                            ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK_RET(aclrtMalloc(&rope_sin_workspace_ptr_, max_position_embeddings * head_dim * sizeof(DTYPE),
                            ACL_MEM_MALLOC_HUGE_FIRST));
  size_t extend_max_len = max_position_embeddings;
  float new_base = rope_base;
  float scaling = 1.0f;
  // https://github.com/vllm-project/vllm/blob/523e30ea0c5abcb447763dcd9a77b54d5c5f3239/vllm/model_executor/layers/rotary_embedding.py#L219
  if (scaling_type == RotaryEmbeddingType::DYNAMIC_NTK_SCALING) {
    extend_max_len = max_position_embeddings * scaling_factor;
    new_base =
        std::pow(rope_base * ((scaling_factor * extend_max_len / max_position_embeddings) - (scaling_factor - 1)),
                 (head_dim / (head_dim - 2)));
  }
  if (scaling_type == RotaryEmbeddingType::LINEAR_SCALING) {
    extend_max_len = max_position_embeddings * scaling_factor;
    scaling = scaling_factor;
  }
  for (size_t token_idx = 0; token_idx < extend_max_len; ++token_idx) {
    int pos = token_idx;
    for (size_t rid = 0; rid < head_dim / 2; ++rid) {
      float inv_freq = 1.0 / std::pow(new_base, rid * 2 / float(head_dim));
      float freq = pos * inv_freq / scaling;
      if (std::is_same<DTYPE, aclFloat16>::value) {
        cos_workspace_host[pos * head_dim + rid] = aclFloatToFloat16(std::cos(freq));
        cos_workspace_host[pos * head_dim + head_dim / 2 + rid] = cos_workspace_host[pos * head_dim + rid];
        sin_workspace_host[pos * head_dim + rid] = aclFloatToFloat16(std::sin(freq));
        sin_workspace_host[pos * head_dim + head_dim / 2 + rid] = sin_workspace_host[pos * head_dim + rid];
      } else if (std::is_same<DTYPE, float>::value) {
        cos_workspace_host[pos * head_dim + rid] = DTYPE(std::cos(freq));
        cos_workspace_host[pos * head_dim + head_dim / 2 + rid] = cos_workspace_host[pos * head_dim + rid];
        sin_workspace_host[pos * head_dim + rid] = DTYPE(std::sin(freq));
        sin_workspace_host[pos * head_dim + head_dim / 2 + rid] = sin_workspace_host[pos * head_dim + rid];
      } else {
        throw std::invalid_argument("Invalid rope compute type, only support float16 or float32.");
      }
    }
  }
  ACL_CHECK_RET(aclrtMemcpyAsync(rope_cos_workspace_ptr_, cos_workspace_host.size() * sizeof(DTYPE),
                                 cos_workspace_host.data(), cos_workspace_host.size() * sizeof(DTYPE),
                                 ACL_MEMCPY_HOST_TO_DEVICE, stream));
  ACL_CHECK_RET(aclrtMemcpyAsync(rope_sin_workspace_ptr_, sin_workspace_host.size() * sizeof(DTYPE),
                                 sin_workspace_host.data(), sin_workspace_host.size() * sizeof(DTYPE),
                                 ACL_MEMCPY_HOST_TO_DEVICE, stream));
  ACL_CHECK_RET(aclrtSynchronizeStream(stream));
}

template <typename DTYPE>
void ATBPagedAttention<DTYPE>::InitAttnMask() {
  uint16_t min_value = 0xFBFF;
  std::vector<uint16_t> mask(MAX_SEQ_LEN * MAX_SEQ_LEN, 0);
  for (size_t i = 0; i < MAX_SEQ_LEN; ++i) {
    for (size_t j = 0; j < MAX_SEQ_LEN; ++j) {
      if (j > i) {
        mask[i * MAX_SEQ_LEN + j] = min_value;
      }
    }
  }

  ACL_CHECK_RET(aclrtMalloc(&attn_mask_ptr_, MAX_SEQ_LEN * MAX_SEQ_LEN * sizeof(DTYPE), ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK_RET(aclrtMemcpy(attn_mask_ptr_, MAX_SEQ_LEN * MAX_SEQ_LEN * sizeof(DTYPE), mask.data(),
                            MAX_SEQ_LEN * MAX_SEQ_LEN * sizeof(DTYPE), aclrtMemcpyKind::ACL_MEMCPY_HOST_TO_DEVICE));
}

template <typename DTYPE>
void ATBPagedAttention<DTYPE>::CreateSplitQKVOperation(uint32_t head_size, uint32_t kv_head_size, uint32_t head_dim,
                                                       atb::Operation** operation) {
  uint32_t tensor_idx = 0;
  uint32_t input_qkv_tensor = tensor_idx++;  // [ntokens, 3 * head_num * head_dim] or [ntokens,
                                             // (head_num + 2 * kv_head_num) * head_dim]
  uint32_t output_q_tensor = tensor_idx++;   // [ntokens, head_num * head_dim]
  uint32_t output_k_tensor = tensor_idx++;
  uint32_t output_v_tensor = tensor_idx++;

  auto kv_head_num = (kv_head_size > 0 && kv_head_size != head_size) ? kv_head_size : 0;

  uint32_t node_idx = 0;
  atb::GraphParam op_graph;
  op_graph.name = "SplitQKV";
  op_graph.inTensorNum = 1;
  op_graph.outTensorNum = 3;
  op_graph.internalTensorNum = 0;
  op_graph.nodes.resize(kv_head_num > 0 ? 3 : 1);

  if (kv_head_num > 0) {
    {
      atb::Node& op_node = op_graph.nodes.at(node_idx++);
      atb::infer::SliceParam op_param;
      op_param.offsets.resize(2);
      op_param.size.resize(2);
      op_param.offsets[0] = 0;
      op_param.offsets[1] = 0;
      op_param.size[0] = -1;
      op_param.size[1] = head_size * head_dim;
      atb::CreateOperation(op_param, &op_node.operation);
      op_node.inTensorIds = {input_qkv_tensor};
      op_node.outTensorIds = {output_q_tensor};
    }
    {
      atb::Node& op_node = op_graph.nodes.at(node_idx++);
      atb::infer::SliceParam op_param;
      op_param.offsets.resize(2);
      op_param.size.resize(2);
      op_param.offsets[0] = 0;
      op_param.offsets[1] = head_size * head_dim;
      op_param.size[0] = -1;
      op_param.size[1] = kv_head_size * head_dim;
      atb::CreateOperation(op_param, &op_node.operation);
      op_node.inTensorIds = {input_qkv_tensor};
      op_node.outTensorIds = {output_k_tensor};
    }
    {
      atb::Node& op_node = op_graph.nodes.at(node_idx++);
      atb::infer::SliceParam op_param;
      op_param.offsets.resize(2);
      op_param.size.resize(2);
      op_param.offsets[0] = 0;
      op_param.offsets[1] = (head_size + kv_head_size) * head_dim;
      op_param.size[0] = -1;
      op_param.size[1] = kv_head_size * head_dim;
      atb::CreateOperation(op_param, &op_node.operation);
      op_node.inTensorIds = {input_qkv_tensor};
      op_node.outTensorIds = {output_v_tensor};
    }
    op_graph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc>& input_tensor_descs,
                                  atb::SVector<atb::TensorDesc>& output_tensor_descs) {
      output_tensor_descs.resize(3);
      output_tensor_descs.at(0) = input_tensor_descs.at(0);
      output_tensor_descs.at(0).shape.dims[0] = input_tensor_descs.at(0).shape.dims[0];
      output_tensor_descs.at(0).shape.dims[1] = head_size * head_dim;
      output_tensor_descs.at(1) = input_tensor_descs.at(0);
      output_tensor_descs.at(1).shape.dims[0] = input_tensor_descs.at(0).shape.dims[0];
      output_tensor_descs.at(1).shape.dims[1] = kv_head_size * head_dim;
      output_tensor_descs.at(2) = input_tensor_descs.at(0);
      output_tensor_descs.at(2).shape.dims[0] = input_tensor_descs.at(0).shape.dims[0];
      output_tensor_descs.at(2).shape.dims[1] = kv_head_size * head_dim;
      return atb::NO_ERROR;
    };
  } else {
    atb::Node& op_node = op_graph.nodes.at(node_idx++);
    atb::infer::SplitParam op_param;
    op_param.splitDim = 1;
    op_param.splitNum = 3;  // only fp16
    atb::CreateOperation(op_param, &op_node.operation);
    op_node.inTensorIds = {input_qkv_tensor};
    op_node.outTensorIds = {output_q_tensor, output_k_tensor, output_v_tensor};
    op_graph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc>& input_tensor_descs,
                                  atb::SVector<atb::TensorDesc>& output_tensor_descs) {
      output_tensor_descs.resize(3);
      output_tensor_descs.at(0) = input_tensor_descs.at(0);
      output_tensor_descs.at(0).shape.dims[0] = input_tensor_descs.at(0).shape.dims[0];
      output_tensor_descs.at(0).shape.dims[1] = head_size * head_dim;
      output_tensor_descs.at(1) = input_tensor_descs.at(0);
      output_tensor_descs.at(1).shape.dims[0] = input_tensor_descs.at(0).shape.dims[0];
      output_tensor_descs.at(1).shape.dims[1] = head_size * head_dim;
      output_tensor_descs.at(2) = input_tensor_descs.at(0);
      output_tensor_descs.at(2).shape.dims[0] = input_tensor_descs.at(0).shape.dims[0];
      output_tensor_descs.at(2).shape.dims[1] = head_size * head_dim;
      return atb::NO_ERROR;
    };
  }
  atb::CreateOperation(op_graph, operation);
}

template class ATBPagedAttention<aclFloat16>;
template class ATBPagedAttention<float>;
#endif

}  // namespace ascend
}  // namespace llm_kernels
