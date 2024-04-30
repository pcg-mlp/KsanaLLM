/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include "attention.h"

#include <cmath>

#include "aclnnop/aclnn_incre_flash_attention_v2.h"
#include "aclnnop/aclnn_prompt_flash_attention_v2.h"

#include "csrc/kernels/ascend/cat/cat.h"
#include "csrc/kernels/ascend/elementwise/elementwise.h"
#include "csrc/kernels/ascend/layernorm/layernorm.h"
#include "csrc/kernels/ascend/matmul/matmul.h"
#include "csrc/kernels/ascend/permute/permute.h"
#include "csrc/kernels/ascend/reshape/reshape.h"
#include "csrc/kernels/ascend/rotary_embedding/rotary_embedding.h"
#include "csrc/kernels/ascend/slice/slice.h"
#include "csrc/kernels/ascend/transpose/transpose.h"

#include "csrc/utils/ascend/common.h"

using namespace llm_kernels::utils;

namespace llm_kernels {
namespace ascend {

void FlashAttentionACL::Init(const int max_position_embeddings, const int head_dims, const int q_heads,
                             const int kv_heads, const float rope_theta, const float rope_scaling_factor,
                             aclDataType dtype, aclrtStream& stream, void (*ws_func)(size_t, void**)) {
  q_heads_ = q_heads;
  kv_heads_ = kv_heads;
  head_dims_ = head_dims;
  dtype_ = dtype;

  rope_ptr_ = std::make_unique<RotaryEmbeddingACL>();
  rope_ptr_->Init(max_position_embeddings, head_dims_, rope_theta, rope_scaling_factor, dtype_, stream, ws_func);

  InitAttnMask(max_position_embeddings, dtype);
}

void FlashAttentionACL::InitAttnMask(int max_tokens_num, aclDataType dtype) {
  // only support fp16 yet
  ACL_CHECK_EQ(dtype, aclDataType::ACL_FLOAT16);
  ACL_CHECK_GT(max_tokens_num, 1);
  constexpr uint16_t one_in_fp16 = 0b11110000000000;
  // constexpr uint16_t zero_in_fp16 = 0b0;

  std::vector<int64_t> attn_mask_shape = {max_tokens_num, max_tokens_num};
  auto elem_nums = max_tokens_num * max_tokens_num;
  std::vector<uint16_t> host_data(elem_nums, 0u);
  for (int i = 0; i < max_tokens_num; ++i) {
    for (int j = i + 1; j < max_tokens_num; ++j) {
      host_data[i * max_tokens_num + j] = one_in_fp16;
    }
  }
  auto fmt = aclFormat::ACL_FORMAT_ND;
  auto byte_size = elem_nums * DT2LONG.at(dtype_);
  ACL_CHECK_RET(aclrtMalloc(&attn_mask_dev_, byte_size, ACL_MEM_MALLOC_NORMAL_ONLY));
  ACL_CHECK_RET(aclrtMemcpy(attn_mask_dev_, byte_size, host_data.data(), byte_size, ACL_MEMCPY_HOST_TO_DEVICE));
  CreateAclTensorWithData(attn_mask_shape, &attn_mask_dev_, dtype_, fmt, &attn_mask_);
}

FlashAttentionACL::~FlashAttentionACL() {
  if (attn_mask_) {
    aclDestroyTensor(attn_mask_);
  }
  if (attn_mask_dev_) {
    aclrtFree(attn_mask_dev_);
    attn_mask_dev_ = nullptr;
  }
}

void FlashAttentionACL::GetSliceAndPermute(const aclTensor* input, int one_of_qkv_index,
                                           const std::vector<int64_t>& output_shape, void** tmp_buffer_dev,
                                           void** output_dev, aclTensor** output, aclrtStream& stream,
                                           void (*ws_func)(size_t, void**)) {
  auto fmt = aclFormat::ACL_FORMAT_ND;
  ACL_CHECK_EQ(output_shape.size(), 4);
  const auto& bs = output_shape[0];
  const auto& num_heads = output_shape[1];
  const auto& seq_len = output_shape[2];
  const auto& head_dims_ = output_shape[3];

  auto hidden_size = num_heads * head_dims_;
  // input(bs, seq_len, 3 * hidden_size) => output(bs, heads, seq_len, head_dim)
  ACL_CHECK_LT(one_of_qkv_index, 3);

  // input(bs, seq_len, hiddens) - > slice(bs, seq_len, hiddens)
  int64_t slice_dim = -1;
  int64_t slice_start = one_of_qkv_index * hidden_size;
  int64_t slice_end = slice_start + hidden_size;
  int64_t slice_step = 1;
  std::vector<int64_t> input_shape = {bs, seq_len, hidden_size};
  aclTensor* slice_output = nullptr;
  CreateAclTensorWithData(input_shape, tmp_buffer_dev, dtype_, fmt, &slice_output);
  Slice(input, slice_dim, slice_start, slice_end, slice_step, &slice_output, stream, ws_func);

  // slice(bs, seq_len, hiddens) -> reshape(bs, seq_len, num_heads, head_dims_)
  aclTensor* reshape_output = nullptr;
  std::vector<int64_t> reshape_output_shape = {bs, seq_len, num_heads, head_dims_};
  Reshape(slice_output, tmp_buffer_dev, reshape_output_shape, &reshape_output, stream, ws_func);
  aclDestroyTensor(slice_output);

  // reshape(bs, seq_len, num_heads, head_dims_) -> permute(bs, heads, seq_len, head_dims_)
  aclTensor* permute_output = nullptr;
  std::vector<int64_t> permute_dims = {0, 2, 1, 3};
  Permute(reshape_output, tmp_buffer_dev, &permute_output, permute_dims, stream, ws_func);
  aclDestroyTensor(reshape_output);

  // copy to contigous
  // TODO: maybe do not need copy
  CreateAclTensorWithData(output_shape, output_dev, dtype_, fmt, output);
  Copy(permute_output, output, stream, ws_func);
  aclDestroyTensor(permute_output);
}

void FlashAttentionACL::PromptFlashAttention(const aclTensor* query, const aclTensor* key, const aclTensor* attnInputV,
                                             aclTensor** attentionOut, aclrtStream& stream,
                                             void (*ws_func)(size_t, void**)) {
  aclTensor* paddingMask = nullptr;
  aclIntArray* actualSeqLengths = nullptr;
  aclIntArray* actualSeqLengthsKv = nullptr;
  aclTensor* deqScale1 = nullptr;
  aclTensor* quantScale1 = nullptr;
  aclTensor* deqScale2 = nullptr;
  aclTensor* quantScale2 = nullptr;
  aclTensor* quantOffset2 = nullptr;

  double scaleValue = 1.0 / sqrt((double)head_dims_);
  int64_t preTokens = 214748647;
  int64_t nextTokens = 0;
  char inputLayout[] = "BNSD";
  int64_t sparseMode = 2;

  uint64_t ws_size = 0ull;
  void* workspace = nullptr;
  aclOpExecutor* executor = nullptr;
  ACL_CHECK_RET(aclnnPromptFlashAttentionV2GetWorkspaceSize(
      query, key, attnInputV, paddingMask, attn_mask_, actualSeqLengths, actualSeqLengthsKv, deqScale1, quantScale1,
      deqScale2, quantScale2, quantOffset2, q_heads_, scaleValue, preTokens, nextTokens, inputLayout, kv_heads_,
      sparseMode, *attentionOut, &ws_size, &executor));
  ws_func(ws_size, &workspace);

  ACL_CHECK_RET(aclnnPromptFlashAttentionV2(workspace, ws_size, executor, stream));
  ACL_CHECK_RET(aclrtSynchronizeStream(stream));
}

void FlashAttentionACL::IncFlashAttention(const aclTensor* query, const aclTensor* key, const aclTensor* attnInputV,
                                          aclTensor** attentionOut, aclrtStream& stream,
                                          void (*ws_func)(size_t, void**)) {
  std::vector<const aclTensor*> keyTmp{key};
  aclTensorList* keyList = aclCreateTensorList(keyTmp.data(), keyTmp.size());
  std::vector<const aclTensor*> valueTmp{attnInputV};
  aclTensorList* valueList = aclCreateTensorList(valueTmp.data(), valueTmp.size());
  aclTensor* paddingMask = nullptr;
  aclTensor* attnInputMask = nullptr;
  aclIntArray* actualSeqLengths = nullptr;
  aclTensor* dequantScale1 = nullptr;
  aclTensor* quantScale1 = nullptr;
  aclTensor* dequantScale2 = nullptr;
  aclTensor* quantScale2 = nullptr;
  aclTensor* quantOffset2 = nullptr;
  double scaleValue = 1.0 / sqrt((double)head_dims_);
  char inputLayout[] = "BNSD";

  uint64_t ws_size = 0ull;
  void* workspace = nullptr;
  aclOpExecutor* executor = nullptr;

  ACL_CHECK_RET(aclnnIncreFlashAttentionV2GetWorkspaceSize(query, keyList, valueList, paddingMask, attnInputMask,
                                                           actualSeqLengths, dequantScale1, quantScale1, dequantScale2,
                                                           quantScale2, quantOffset2, q_heads_, scaleValue, inputLayout,
                                                           kv_heads_, *attentionOut, &ws_size, &executor));
  ws_func(ws_size, &workspace);
  ACL_CHECK_RET(aclnnIncreFlashAttentionV2(workspace, ws_size, executor, stream));
  ACL_CHECK_RET(aclrtSynchronizeStream(stream));
}

void FlashAttentionACL::PrepareRopeIndex(const int bs, const int seq_len, const int64_t token_pos,
                                         const bool is_context_stage, void** rope_index_dev, aclTensor** rope_index) {
  std::vector<int64_t> rope_index_shape = {bs, seq_len};
  auto byte_size = bs * seq_len * sizeof(int64_t);
  ACL_CHECK_RET(aclrtMalloc(rope_index_dev, byte_size, ACL_MEM_MALLOC_HUGE_FIRST));
  std::vector<int64_t> rope_index_host(bs * seq_len, token_pos);
  if (is_context_stage) {
    // generate it by seq_len
    for (int b = 0; b < bs; ++b) {
      for (int i = 0; i < seq_len; ++i) {
        rope_index_host[b * seq_len + i] = i;
      }
    }
  } else {
    ACL_CHECK_EQ(seq_len, 1);
  }
  ACL_CHECK_RET(aclrtMemcpy(*rope_index_dev, byte_size, rope_index_host.data(), byte_size, ACL_MEMCPY_HOST_TO_DEVICE));
  CreateAclTensorWithData(rope_index_shape, rope_index_dev, aclDataType::ACL_INT64, aclFormat::ACL_FORMAT_ND,
                          rope_index);
}

// output is tmp_buffers[1]
void FlashAttentionACL::Forward(const aclTensor* matmulQKVOutput, const int64_t token_pos, void** key_cache,
                                void** val_cache, std::vector<void*>& tmp_buffers, aclTensor** output,
                                const bool is_context_stage, aclrtStream& stream, void (*ws_func)(size_t, void**),
                                void* workspace_buf_ptr) {
  auto fmt = aclFormat::ACL_FORMAT_ND;

  int64_t* input_shape = nullptr;
  uint64_t input_shape_num = 0;
  ACL_CHECK_RET(aclGetViewShape(matmulQKVOutput, &input_shape, &input_shape_num));
  ACL_CHECK_EQ(input_shape_num, 3);
  auto& bs = input_shape[0];
  auto& seq_len = input_shape[1];
  auto num_heads = input_shape[2] / 3 / head_dims_;
  ACL_CHECK_EQ(q_heads_, num_heads);

  aclTensor* attnInputQ = nullptr;
  aclTensor* attnInputK = nullptr;
  aclTensor* attnInputV = nullptr;
  std::vector<int64_t> attnInputShape = {bs, num_heads, seq_len, head_dims_};
  std::vector<int64_t> kvStrides;
  utils::CalShapeStrides(attnInputShape, kvStrides);

  aclTensor* ropeIndex = nullptr;
  void* ropeIndexDev = nullptr;
  PrepareRopeIndex(bs, seq_len, token_pos, is_context_stage, &ropeIndexDev, &ropeIndex);

  // / Rope Query
  // input(bs, seq_len, 3 * hidden_size) => output(bs, heads, seq_len, head_dim)
  aclTensor* ropeQueryInput = nullptr;
  GetSliceAndPermute(matmulQKVOutput, 0, attnInputShape, &tmp_buffers[1], &tmp_buffers[2], &ropeQueryInput, stream,
                     ws_func);
  // PrintTensor(ropeQueryInput, stream, "ropeQueryInput");
  CreateAclTensorWithData(attnInputShape, &tmp_buffers[3], dtype_, fmt, &attnInputQ);
  rope_ptr_->Forward(ropeQueryInput, ropeIndex, &attnInputQ, stream, ws_func, workspace_buf_ptr);
  aclDestroyTensor(ropeQueryInput);

  // / Rope Key
  // input(bs, seq_len, 3 * hidden_size) => output(bs, heads, seq_len, head_dim)
  aclTensor* ropeKeyInput = nullptr;
  GetSliceAndPermute(matmulQKVOutput, 1, attnInputShape, &tmp_buffers[1], &tmp_buffers[2], &ropeKeyInput, stream,
                     ws_func);
  if (is_context_stage) {
    // CreateAclTensorWithData(attnInputShape, &key_cache, dtype_, fmt, &attnInputK);
    attnInputK = aclCreateTensor(attnInputShape.data(), attnInputShape.size(), dtype_, kvStrides.data(), 0, fmt,
                                 attnInputShape.data(), attnInputShape.size(), *key_cache);
    rope_ptr_->Forward(ropeKeyInput, ropeIndex, &attnInputK, stream, ws_func, workspace_buf_ptr);
    aclDestroyTensor(ropeKeyInput);
  } else {
    aclTensor* ropeKeyOutput = nullptr;
    CreateAclTensorWithData(attnInputShape, &tmp_buffers[1], dtype_, fmt, &ropeKeyOutput);
    rope_ptr_->Forward(ropeKeyInput, ropeIndex, &ropeKeyOutput, stream, ws_func, workspace_buf_ptr);
    aclDestroyTensor(ropeKeyInput);

    // read last k
    // lastKeyStates + ropeKeyOutput -> catKeyOutput(attnInputK)
    aclTensor* lastKeyStates = nullptr;
    std::vector<int64_t> lastKeyStatesShape = {bs, num_heads, token_pos, head_dims_};
    std::vector<int64_t> catKeyOutputShape = {bs, num_heads, token_pos + 1, head_dims_};
    CreateAclTensorWithData(lastKeyStatesShape, key_cache, dtype_, fmt, &lastKeyStates);
    CreateAclTensorWithData(catKeyOutputShape, &tmp_buffers[4], dtype_, fmt, &attnInputK);
    int64_t catKeyDim = 2;
    std::vector<const aclTensor*> catKeyTmp{lastKeyStates, ropeKeyOutput};
    Cat(catKeyTmp, catKeyDim, &attnInputK, stream, ws_func);

    // catKeyOutput -> key_cache
    auto catKeySize = GetShapeSize(catKeyOutputShape) * DT2LONG.at(dtype_);
    ACL_CHECK_RET(aclrtMemcpy(*key_cache, catKeySize, tmp_buffers[4], catKeySize, ACL_MEMCPY_DEVICE_TO_DEVICE));
    aclDestroyTensor(lastKeyStates);
    aclDestroyTensor(ropeKeyOutput);
  }

  // / V
  if (is_context_stage) {
    GetSliceAndPermute(matmulQKVOutput, 2, attnInputShape, &tmp_buffers[1], val_cache, &attnInputV, stream, ws_func);
  } else {
    aclTensor* sliceOutputV = nullptr;
    GetSliceAndPermute(matmulQKVOutput, 2, attnInputShape, &tmp_buffers[1], &tmp_buffers[2], &sliceOutputV, stream,
                       ws_func);

    // read last v
    // lastValueStates + sliceOutputV -> catValueOutput(attnInputV)
    aclTensor* lastValueStates = nullptr;
    std::vector<int64_t> lastValueStatesShape = {bs, num_heads, token_pos, head_dims_};
    std::vector<int64_t> catValueOutputShape = {bs, num_heads, token_pos + 1, head_dims_};
    CreateAclTensorWithData(lastValueStatesShape, val_cache, dtype_, fmt, &lastValueStates);
    CreateAclTensorWithData(catValueOutputShape, &tmp_buffers[1], dtype_, fmt, &attnInputV);
    int64_t catValueDim = 2;
    std::vector<const aclTensor*> catValueTmp{lastValueStates, sliceOutputV};
    Cat(catValueTmp, catValueDim, &attnInputV, stream, ws_func);

    // catValueOutput -> val_cache
    auto catValueSize = GetShapeSize(catValueOutputShape) * DT2LONG.at(dtype_);
    ACL_CHECK_RET(aclrtMemcpy(*val_cache, catValueSize, tmp_buffers[1], catValueSize, ACL_MEMCPY_DEVICE_TO_DEVICE));
    aclDestroyTensor(lastValueStates);
    aclDestroyTensor(sliceOutputV);
  }

  aclDestroyTensor(ropeIndex);
  aclrtFree(ropeIndexDev);
  ropeIndexDev = nullptr;

  // attnInputQ + attnInputK + attnInputV + attn_mask_ = attentionOut
  aclTensor* attentionOut = nullptr;
  std::vector<int64_t> attentionOutShape = {bs, num_heads, seq_len, head_dims_};
  CreateAclTensorWithData(attentionOutShape, &tmp_buffers[2], dtype_, fmt, &attentionOut);
  int q_heads_ = num_heads;
  int kv_heads_ = num_heads;
  // PrintTensor(attnInputQ, stream, "attnInputQ");
  // PrintTensor(attnInputK, stream, "attnInputK");
  // PrintTensor(attnInputV, stream, "attnInputV");
  if (is_context_stage) {
    PromptFlashAttention(attnInputQ, attnInputK, attnInputV, &attentionOut, stream, ws_func);
  } else {
    IncFlashAttention(attnInputQ, attnInputK, attnInputV, &attentionOut, stream, ws_func);
  }
  aclDestroyTensor(attnInputQ);
  aclDestroyTensor(attnInputK);
  aclDestroyTensor(attnInputV);
  // PrintTensor(attentionOut, stream, "attentionOut");

  // attentionOut(bs, heads, seq_len, head_dims_) -> trans and reshape to output(bs, seq_len, heads * head_dims_)
  aclTensor* permuteOutput = nullptr;
  aclTensor* copyOutput = nullptr;
  const std::vector<int64_t> dims = {0, 2, 1, 3};
  Permute(attentionOut, &tmp_buffers[2], &permuteOutput, dims, stream, ws_func);
  aclDestroyTensor(attentionOut);
  std::vector<int64_t> transpose4OutputShape = {bs, seq_len, num_heads, head_dims_};
  CreateAclTensorWithData(transpose4OutputShape, &tmp_buffers[1], dtype_, fmt, &copyOutput);
  Copy(permuteOutput, &copyOutput, stream, ws_func);
  aclDestroyTensor(permuteOutput);

  std::vector<int64_t> reshape4OutputShape = {bs, seq_len, num_heads * head_dims_};
  Reshape(copyOutput, &tmp_buffers[1], reshape4OutputShape, output, stream, ws_func);
  aclDestroyTensor(copyOutput);
}

}  // namespace ascend
}  // namespace llm_kernels
