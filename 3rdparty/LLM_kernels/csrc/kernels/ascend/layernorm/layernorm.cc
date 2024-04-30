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

void RMSLayerNorm(const aclTensor* input, const aclTensor* weight, float eps, aclTensor** output, aclrtStream& stream,
                  void (*ws_func)(size_t, void**), void* workspace_buf_ptr) {
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
  auto dev_buf_size = bs * seq_len * hidden_size * sizeof(float);
  void* dev_buf_a = nullptr;
  void* dev_buf_b = nullptr;
  void* dev_buf_c = nullptr;
  if (workspace_buf_ptr == nullptr) {
    ACL_CHECK_RET(aclrtMalloc(&dev_buf_a, dev_buf_size, ACL_MEM_MALLOC_NORMAL_ONLY));
    ACL_CHECK_RET(aclrtMalloc(&dev_buf_b, dev_buf_size, ACL_MEM_MALLOC_NORMAL_ONLY));
    ACL_CHECK_RET(aclrtMalloc(&dev_buf_c, dev_buf_size, ACL_MEM_MALLOC_NORMAL_ONLY));
  } else {
    dev_buf_a = workspace_buf_ptr;
    dev_buf_b = workspace_buf_ptr + dev_buf_size;
    dev_buf_c = workspace_buf_ptr + dev_buf_size * 2;
  }

  // input - > fp32_buf_tensor
  std::vector<int64_t> fp16fp32_cast_shape = {bs, seq_len, hidden_size};
  aclTensor* fp32_buf_tensor = nullptr;
  CreateAclTensorWithData(fp16fp32_cast_shape, &dev_buf_a, dtype_fp32, fmt, &fp32_buf_tensor);
  Cast(input, dtype_fp32, &fp32_buf_tensor, stream, ws_func);

  // fp32_buf_tensor - > pow_output_tensor
  constexpr float pow_exponent_v = 2.0f;
  std::vector<int64_t> pow_output_shape = {bs, seq_len, hidden_size};
  aclTensor* pow_output_tensor = nullptr;
  CreateAclTensorWithData(pow_output_shape, &dev_buf_b, dtype_fp32, fmt, &pow_output_tensor);
  Pow(fp32_buf_tensor, pow_exponent_v, &pow_output_tensor, stream, ws_func);

  // pow_output_tensor - > mean_output_tensor
  aclTensor* mean_output_tensor = nullptr;
  std::vector<int64_t> mean_output_shape = {bs, seq_len, 1};
  CreateAclTensorWithData(mean_output_shape, &dev_buf_c, dtype_fp32, fmt, &mean_output_tensor);
  std::vector<int64_t> mean_dim = {-1};
  bool keepdim = true;
  Mean(pow_output_tensor, mean_dim, keepdim, dtype_fp32, &mean_output_tensor, stream, ws_func);
  ACL_CHECK_RET(aclDestroyTensor(pow_output_tensor));

  // mean_output_tensor - > add_output_tensor
  float add_const_v = 1;
  std::vector<int64_t> add_output_shape = {bs, seq_len, 1};
  aclScalar* add_const_scalar = nullptr;
  aclTensor* add_output_tensor = nullptr;
  aclScalar* add_alpha_scalar = nullptr;
  add_alpha_scalar = aclCreateScalar(&eps, dtype_fp32);
  add_const_scalar = aclCreateScalar(&add_const_v, dtype_fp32);
  CreateAclTensorWithData(add_output_shape, &dev_buf_b, dtype_fp32, fmt, &add_output_tensor);
  Adds(mean_output_tensor, add_alpha_scalar, add_const_scalar, &add_output_tensor, stream, ws_func);
  ACL_CHECK_RET(aclDestroyTensor(mean_output_tensor));
  ACL_CHECK_RET(aclDestroyScalar(add_const_scalar));
  ACL_CHECK_RET(aclDestroyScalar(add_alpha_scalar));

  // add_output_tensor - > add_output_tensor
  InplaceSqrt(&add_output_tensor, stream, ws_func);

  // fp32_buf_tensor / add_output_tensor - > fp32_buf_tensor
  InplaceDiv(add_output_tensor, &fp32_buf_tensor, stream, ws_func);
  ACL_CHECK_RET(aclDestroyTensor(add_output_tensor));

  // fp32_buf_tensor - > fp16_buf_tensor
  std::vector<int64_t> fp32fp16_cast_shape = {bs, seq_len, hidden_size};
  aclTensor* fp16_buf_tensor = nullptr;
  CreateAclTensorWithData(fp32fp16_cast_shape, &dev_buf_b, dtype_fp16, fmt, &fp16_buf_tensor);
  Cast(fp32_buf_tensor, dtype_fp16, &fp16_buf_tensor, stream, ws_func);
  ACL_CHECK_RET(aclDestroyTensor(fp32_buf_tensor));

  // fp16_buf_tensor - > muloutput
  Mul(fp16_buf_tensor, weight, output, stream, ws_func);
  ACL_CHECK_RET(aclDestroyTensor(fp16_buf_tensor));

  if (workspace_buf_ptr == nullptr) {
    // release dev mem for infer
    ACL_CHECK_RET(aclrtFree(dev_buf_a));
    dev_buf_a = nullptr;
    ACL_CHECK_RET(aclrtFree(dev_buf_b));
    dev_buf_b = nullptr;
    ACL_CHECK_RET(aclrtFree(dev_buf_c));
    dev_buf_c = nullptr;
  }
}

}  // namespace ascend
}  // namespace llm_kernels
