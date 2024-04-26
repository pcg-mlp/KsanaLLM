/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */
#include <math.h>

#include "rotary_embedding.h"

#include "csrc/kernels/ascend/cat/cat.h"
#include "csrc/kernels/ascend/elementwise/elementwise.h"
#include "csrc/kernels/ascend/gather/gather.h"
#include "csrc/kernels/ascend/pointwise/pointwise.h"
#include "csrc/kernels/ascend/slice/slice.h"

#include "csrc/utils/ascend/common.h"

using namespace llm_kernels::utils;

namespace llm_kernels {
namespace ascend {

void RotaryEmbeddingACL::Init(const int max_position_embeddings, const int head_dims, const float rope_theta,
                              const float rope_scaling_factor, aclDataType dtype, aclrtStream& stream,
                              void (*ws_func)(size_t, void**)) {
  max_position_embeddings_ = max_position_embeddings;
  rope_theta_ = rope_theta;
  head_dims_ = head_dims;
  rope_scaling_factor_ = rope_scaling_factor;
  dtype_ = dtype;

  InitSinCos(max_position_embeddings, head_dims, rope_theta, dtype, stream, ws_func);
}

void RotaryEmbeddingACL::InitSinCos(const int max_position_embeddings, const int head_dims, const float rope_theta,
                                    aclDataType dtype, aclrtStream& stream, void (*ws_func)(size_t, void**)) {
  // only support fp16 yet
  ACL_CHECK_EQ(dtype, aclDataType::ACL_FLOAT16);
  ACL_CHECK_GT(max_position_embeddings, 1);

  std::vector<int64_t> shape = {max_position_embeddings, head_dims};
  auto elem_nums = max_position_embeddings * head_dims;
  std::vector<float> host_cos(elem_nums, 0u);
  std::vector<float> host_sin(elem_nums, 0u);
  for (int seq_idx = 0; seq_idx < max_position_embeddings; ++seq_idx) {
    for (int i = 0; i < head_dims / 2; ++i) {
      float inv_freq = 1.0 / pow(rope_theta, i * 2 / (float)head_dims);
      float freq = seq_idx * inv_freq;
      float fp32_cos = cos(freq);
      float fp32_sin = sin(freq);

      host_cos[seq_idx * head_dims + i] = fp32_cos;
      host_sin[seq_idx * head_dims + i] = fp32_sin;

      host_cos[seq_idx * head_dims + head_dims / 2 + i] = fp32_cos;
      host_sin[seq_idx * head_dims + head_dims / 2 + i] = fp32_sin;
    }
  }

  auto fmt = aclFormat::ACL_FORMAT_ND;
  auto byte_size_fp32 = elem_nums * sizeof(float);
  void* sin_dev_fp32 = nullptr;
  void* cos_dev_fp32 = nullptr;
  ACL_CHECK_RET(aclrtMalloc(&sin_dev_fp32, byte_size_fp32, ACL_MEM_MALLOC_NORMAL_ONLY));
  ACL_CHECK_RET(aclrtMalloc(&cos_dev_fp32, byte_size_fp32, ACL_MEM_MALLOC_NORMAL_ONLY));
  ACL_CHECK_RET(aclrtMemcpy(sin_dev_fp32, byte_size_fp32, host_sin.data(), byte_size_fp32, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK_RET(aclrtMemcpy(cos_dev_fp32, byte_size_fp32, host_cos.data(), byte_size_fp32, ACL_MEMCPY_HOST_TO_DEVICE));
  aclTensor* sin_fp32 = nullptr;
  aclTensor* cos_fp32 = nullptr;
  CreateAclTensorWithData(shape, &sin_dev_fp32, aclDataType::ACL_FLOAT, fmt, &sin_fp32);
  CreateAclTensorWithData(shape, &cos_dev_fp32, aclDataType::ACL_FLOAT, fmt, &cos_fp32);

  ACL_CHECK_RET(aclrtMalloc(&sin_dev_, byte_size_fp32 / 2, ACL_MEM_MALLOC_NORMAL_ONLY));
  ACL_CHECK_RET(aclrtMalloc(&cos_dev_, byte_size_fp32 / 2, ACL_MEM_MALLOC_NORMAL_ONLY));
  CreateAclTensorWithData(shape, &sin_dev_, dtype_, fmt, &sin_);
  CreateAclTensorWithData(shape, &cos_dev_, dtype_, fmt, &cos_);

  Cast(sin_fp32, dtype_, &sin_, stream, ws_func);
  Cast(cos_fp32, dtype_, &cos_, stream, ws_func);

  aclDestroyTensor(sin_fp32);
  aclDestroyTensor(cos_fp32);
  aclrtFree(sin_dev_fp32);
  aclrtFree(cos_dev_fp32);
}

RotaryEmbeddingACL::~RotaryEmbeddingACL() {
  if (sin_) {
    aclDestroyTensor(sin_);
  }
  if (cos_) {
    aclDestroyTensor(cos_);
  }
  if (sin_dev_) {
    aclrtFree(sin_dev_);
    sin_dev_ = nullptr;
  }
  if (cos_dev_) {
    aclrtFree(cos_dev_);
    cos_dev_ = nullptr;
  }
}

void RotaryEmbeddingACL::RotarySplit(const aclTensor* input, const int bs, const int64_t seq_len, const int num_heads,
                                     void** maxDev_a, void** maxDev_b, void** maxDev_c, aclTensor** catOutput,
                                     aclrtStream& stream, void (*ws_func)(size_t, void**)) {
  auto fmt = aclFormat::ACL_FORMAT_ND;
  // /* part b */
  // input - > slice1Output
  int64_t slice1Dim = -1;
  int64_t slice1Start = 0;
  int64_t slice1End = head_dims_ / 2;
  int64_t slice1Step = 1;
  std::vector<int64_t> slice1OutputShape = {bs, num_heads, seq_len, head_dims_ / 2};
  aclTensor* slice1Output = nullptr;
  CreateAclTensorWithData(slice1OutputShape, maxDev_a, dtype_, fmt, &slice1Output);
  Slice(input, slice1Dim, slice1Start, slice1End, slice1Step, &slice1Output, stream, ws_func);

  // /* part c */
  // input - > slice2Output
  int64_t slice2Dim = -1;
  int64_t slice2Start = head_dims_ / 2;
  int64_t slice2End = head_dims_;
  int64_t slice2Step = 1;
  std::vector<int64_t> slice2OutputShape = {bs, num_heads, seq_len, head_dims_ / 2};
  aclTensor* slice2Output = nullptr;
  CreateAclTensorWithData(slice2OutputShape, maxDev_b, dtype_, fmt, &slice2Output);
  Slice(input, slice2Dim, slice2Start, slice2End, slice2Step, &slice2Output, stream, ws_func);

  // slice2Output - > negOutput
  std::vector<int64_t> negOutputShape = {bs, num_heads, seq_len, head_dims_ / 2};
  aclTensor* negOutput = nullptr;
  CreateAclTensorWithData(negOutputShape, maxDev_c, dtype_, fmt, &negOutput);
  Neg(slice2Output, &negOutput, stream, ws_func);
  aclDestroyTensor(slice2Output);

  // negOutput + slice1Output -> catOutput
  std::vector<const aclTensor*> tmp{negOutput, slice1Output};
  int64_t catDim = -1;
  Cat(tmp, catDim, catOutput, stream, ws_func);
  aclDestroyTensor(slice1Output);
  aclDestroyTensor(negOutput);
}

void RotaryEmbeddingACL::Forward(const aclTensor* input, const aclTensor* ropeIndex, aclTensor** output,
                                 aclrtStream& stream, void (*ws_func)(size_t, void**), void* workspace_buf_ptr) {
  aclOpExecutor* executor;
  auto dtype_ = aclDataType::ACL_FLOAT16;
  auto fmt = aclFormat::ACL_FORMAT_ND;

  int64_t* input_shape = nullptr;
  uint64_t input_shape_num = 0;
  ACL_CHECK_RET(aclGetViewShape(input, &input_shape, &input_shape_num));
  ACL_CHECK_EQ(input_shape_num, 4);

  auto bs = input_shape[0];
  auto num_heads = input_shape[1];
  auto seq_len = input_shape[2];
  auto head_dims = input_shape[3];

  // TODO: opt this mem
  auto maxDevSize = seq_len * num_heads * head_dims * sizeof(uint16_t);
  void* maxDev_a = nullptr;
  void* maxDev_b = nullptr;
  void* maxDev_c = nullptr;
  if (workspace_buf_ptr == nullptr) {
    ACL_CHECK_RET(aclrtMalloc(&maxDev_a, maxDevSize, ACL_MEM_MALLOC_NORMAL_ONLY));
    ACL_CHECK_RET(aclrtMalloc(&maxDev_b, maxDevSize, ACL_MEM_MALLOC_NORMAL_ONLY));
    ACL_CHECK_RET(aclrtMalloc(&maxDev_c, maxDevSize, ACL_MEM_MALLOC_NORMAL_ONLY));
  } else {
    maxDev_a = workspace_buf_ptr;
    maxDev_b = workspace_buf_ptr + maxDevSize;
    maxDev_c = workspace_buf_ptr + maxDevSize * 2;
  }

  std::vector<int64_t> catOutputShape = {bs, num_heads, seq_len, head_dims};
  aclTensor* catOutput = nullptr;
  CreateAclTensorWithData(catOutputShape, &maxDev_b, dtype_, fmt, &catOutput);
  RotarySplit(input, bs, seq_len, num_heads, &maxDev_a, &maxDev_b, &maxDev_c, &catOutput, stream, ws_func);

  // /* part a -sin */
  // sin_ + sinIndex - > sinOutput
  std::vector<int64_t> sinOutputShape = {bs, seq_len, head_dims};
  aclTensor* sinOutput = nullptr;
  CreateAclTensorWithData(sinOutputShape, &maxDev_a, dtype_, fmt, &sinOutput);
  int64_t sinDim = 0;
  Gather(sin_, sinDim, ropeIndex, &sinOutput, stream, ws_func);

  // sinOutput reshape to -> usq1Output
  aclTensor* usq1Output = nullptr;
  std::vector<int64_t> usq1OutputShape = {bs, 1, seq_len, head_dims};
  CreateAclTensorWithData(usq1OutputShape, &maxDev_a, dtype_, fmt, &usq1Output);

  // catOutput * usq1Output - > mul1Output
  std::vector<int64_t> mul1OutputShape = {bs, num_heads, seq_len, head_dims};
  aclTensor* mul1Output = nullptr;
  CreateAclTensorWithData(mul1OutputShape, &maxDev_c, dtype_, fmt, &mul1Output);
  Mul(catOutput, usq1Output, &mul1Output, stream, ws_func);
  aclDestroyTensor(catOutput);
  aclDestroyTensor(usq1Output);

  // /* part d - cos */
  // cos_ + cosIndex - > cosOutput
  std::vector<int64_t> cosIndexShape = {bs, seq_len};
  std::vector<int64_t> cosOutputShape = {bs, seq_len, head_dims};
  aclTensor* cosOutput = nullptr;
  CreateAclTensorWithData(cosOutputShape, &maxDev_a, dtype_, fmt, &cosOutput);
  int64_t cosDim = 0;
  Gather(cos_, cosDim, ropeIndex, &cosOutput, stream, ws_func);

  // cosOutput - > usq2Output
  std::vector<int64_t> usq2OutputShape = {bs, 1, seq_len, head_dims};
  aclTensor* usq2Output = nullptr;
  CreateAclTensorWithData(usq2OutputShape, &maxDev_a, dtype_, fmt, &usq2Output);

  // input * usq2Output -> mul2Output
  std::vector<int64_t> mul2OutputShape = {bs, num_heads, seq_len, head_dims};
  aclTensor* mul2Output = nullptr;
  CreateAclTensorWithData(mul2OutputShape, &maxDev_b, dtype_, fmt, &mul2Output);
  Mul(input, usq2Output, &mul2Output, stream, ws_func);
  aclDestroyTensor(usq2Output);
  aclDestroyTensor(cosOutput);

  // mul1Output + mul2Output - > output
  uint16_t one_in_fp16 = 0b11110000000000;
  aclScalar* addAlpha = aclCreateScalar(&one_in_fp16, dtype_);
  Add(mul1Output, mul2Output, addAlpha, output, stream, ws_func);
  aclDestroyTensor(mul1Output);
  aclDestroyTensor(mul2Output);
  aclDestroyScalar(addAlpha);

  // release dev mem for infer
  if (workspace_buf_ptr == nullptr) {
    aclrtFree(maxDev_a);
    maxDev_a = nullptr;
    aclrtFree(maxDev_b);
    maxDev_b = nullptr;
    aclrtFree(maxDev_c);
    maxDev_c = nullptr;
  }
}

}  // namespace ascend
}  // namespace llm_kernels
