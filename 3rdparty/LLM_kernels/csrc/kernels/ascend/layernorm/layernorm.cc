/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include "layernorm.h"

#include "csrc/kernels/ascend/elementwise/elementwise.h"
#include "csrc/kernels/ascend/pointwise/pointwise.h"

#include "csrc/utils/ascend/common.h"

using namespace llm_kernels::utils;
namespace llm_kernels {
namespace ascend {

void RMSLayerNorm(const aclTensor* input, const aclTensor* weight, aclTensor** output, aclrtStream& stream,
                  void (*ws_func)(size_t, void**)) {
  auto dtype_fp32 = aclDataType::ACL_FLOAT;
  auto dtype_fp16 = aclDataType::ACL_FLOAT16;
  auto fmt = aclFormat::ACL_FORMAT_ND;

  aclDataType src_dtype;
  ACL_CHECK_RET(aclGetDataType(input, &src_dtype));
  ACL_CHECK_EQ(src_dtype, dtype_fp16);
  int64_t* input_shape = nullptr;
  uint64_t input_dim = 0;
  ACL_CHECK_RET(aclGetViewShape(input, &input_shape, &input_dim));
  ACL_CHECK_EQ(input_dim, 3);
  auto bs = input_shape[0];
  auto seq_len = input_shape[1];
  auto hidden_size = input_shape[2];

  // malloc dev mem for infer
  // TODO: alloc outside
  auto maxDevSize = seq_len * hidden_size * sizeof(float);
  void* maxDev_a = nullptr;
  void* maxDev_b = nullptr;
  void* maxDev_c = nullptr;
  ACL_CHECK_RET(aclrtMalloc(&maxDev_a, maxDevSize, ACL_MEM_MALLOC_NORMAL_ONLY));
  ACL_CHECK_RET(aclrtMalloc(&maxDev_b, maxDevSize, ACL_MEM_MALLOC_NORMAL_ONLY));
  ACL_CHECK_RET(aclrtMalloc(&maxDev_c, maxDevSize, ACL_MEM_MALLOC_NORMAL_ONLY));

  // input - > cast1Output
  std::vector<int64_t> cast1OutputShape = {bs, seq_len, hidden_size};
  aclTensor* cast1Output = nullptr;
  CreateAclTensorWithData(cast1OutputShape, &maxDev_a, dtype_fp32, fmt, &cast1Output);
  Cast(input, dtype_fp32, &cast1Output, stream, ws_func);

  // cast1Output - > powOutput
  float powExponentValue = 2.0f;
  std::vector<int64_t> powOutputShape = {bs, seq_len, hidden_size};
  aclTensor* powOutput = nullptr;
  CreateAclTensorWithData(powOutputShape, &maxDev_b, dtype_fp32, fmt, &powOutput);
  Pow(cast1Output, powExponentValue, &powOutput, stream, ws_func);

  // powOutput - > meanOutput
  aclTensor* meanOutput = nullptr;
  std::vector<int64_t> meanOutputShape = {bs, seq_len, 1};
  CreateAclTensorWithData(meanOutputShape, &maxDev_c, dtype_fp32, fmt, &meanOutput);
  std::vector<int64_t> meanDimData = {-1};
  bool keepdim = true;
  Mean(powOutput, meanDimData, keepdim, dtype_fp32, &meanOutput, stream, ws_func);
  aclDestroyTensor(powOutput);

  // meanOutput - > addOutput
  float addAlphaValue = 0.000001;
  float addConstValue = 1;
  std::vector<int64_t> addOutputShape = {bs, seq_len, 1};
  aclScalar* addConst = nullptr;
  aclTensor* addOutput = nullptr;
  aclScalar* addAlpha = nullptr;
  addAlpha = aclCreateScalar(&addAlphaValue, dtype_fp32);
  addConst = aclCreateScalar(&addConstValue, dtype_fp32);
  CreateAclTensorWithData(addOutputShape, &maxDev_b, dtype_fp32, fmt, &addOutput);
  Adds(meanOutput, addAlpha, addConst, &addOutput, stream, ws_func);
  // here use Adds?
  aclDestroyTensor(meanOutput);
  aclDestroyScalar(addConst);
  aclDestroyScalar(addAlpha);

  // addOutput - > addOutput
  InplaceSqrt(&addOutput, stream, ws_func);

  // cast1Output / addOutput - > cast1Output
  InplaceDiv(addOutput, &cast1Output, stream, ws_func);
  aclDestroyTensor(addOutput);

  // cast1Output - > cast2Output
  std::vector<int64_t> cast2OutputShape = {bs, seq_len, hidden_size};
  aclTensor* cast2Output = nullptr;
  CreateAclTensorWithData(cast2OutputShape, &maxDev_b, dtype_fp16, fmt, &cast2Output);
  Cast(cast1Output, dtype_fp16, &cast2Output, stream, ws_func);
  aclDestroyTensor(cast1Output);

  // cast2Output - > mulOutput
  // PrintTensor(cast2Output, stream, "cast2Output");
  Mul(cast2Output, weight, output, stream, ws_func);
  aclDestroyTensor(cast2Output);

  // release dev mem for infer
  aclrtFree(maxDev_a);
  maxDev_a = nullptr;
  aclrtFree(maxDev_b);
  maxDev_b = nullptr;
  aclrtFree(maxDev_c);
  maxDev_c = nullptr;
}

}  // namespace ascend
}  // namespace llm_kernels
