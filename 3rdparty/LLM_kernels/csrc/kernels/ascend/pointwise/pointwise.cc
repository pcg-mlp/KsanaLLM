/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include "pointwise.h"

#include "aclnnop/aclnn_cast.h"
#include "aclnnop/aclnn_div.h"
#include "aclnnop/aclnn_mean.h"
#include "aclnnop/aclnn_neg.h"
#include "aclnnop/aclnn_pow.h"
#include "aclnnop/aclnn_sqrt.h"

#include "csrc/utils/ascend/common.h"

namespace llm_kernels {
namespace ascend {

void Cast(const aclTensor* castInput, const aclDataType castToType, aclTensor** castOutput, aclrtStream& stream,
          void (*ws_func)(size_t, void**)) {
  uint64_t ws_size = 0ull;
  void* workspace = nullptr;
  aclOpExecutor* executor = nullptr;

  ACL_CHECK_RET(aclnnCastGetWorkspaceSize(castInput, castToType, *castOutput, &ws_size, &executor));
  ws_func(ws_size, &workspace);
  ACL_CHECK_RET(aclnnCast(workspace, ws_size, executor, stream));
  ACL_CHECK_RET(aclrtSynchronizeStream(stream));
}

void Pow(const aclTensor* input, float powExponentValue, aclTensor** output, aclrtStream& stream,
         void (*ws_func)(size_t, void**)) {
  uint64_t ws_size = 0ull;
  void* workspace = nullptr;
  aclOpExecutor* executor = nullptr;

  aclScalar* powExponent = nullptr;
  powExponent = aclCreateScalar(&powExponentValue, aclDataType::ACL_FLOAT);
  ACL_CHECK_RET(aclnnPowTensorScalarGetWorkspaceSize(input, powExponent, *output, &ws_size, &executor));
  ws_func(ws_size, &workspace);
  ACL_CHECK_RET(aclnnPowTensorScalar(workspace, ws_size, executor, stream));
  ACL_CHECK_RET(aclDestroyScalar(powExponent));
}

void Mean(const aclTensor* input, std::vector<int64_t>& meanDimData, const bool keepdim, aclDataType dtype,
          aclTensor** output, aclrtStream& stream, void (*ws_func)(size_t, void**)) {
  uint64_t ws_size = 0ull;
  void* workspace = nullptr;
  aclOpExecutor* executor = nullptr;

  aclIntArray* meanDim = nullptr;
  meanDim = aclCreateIntArray(meanDimData.data(), meanDimData.size());
  ACL_CHECK_RET(aclnnMeanGetWorkspaceSize(input, meanDim, keepdim, dtype, *output, &ws_size, &executor));
  ws_func(ws_size, &workspace);
  ACL_CHECK_RET(aclnnMean(workspace, ws_size, executor, stream));
  ACL_CHECK_RET(aclDestroyIntArray(meanDim));
}

void Neg(const aclTensor* input, aclTensor** output, aclrtStream& stream, void (*ws_func)(size_t, void**)) {
  uint64_t ws_size = 0ull;
  void* workspace = nullptr;
  aclOpExecutor* executor = nullptr;

  ACL_CHECK_RET(aclnnNegGetWorkspaceSize(input, *output, &ws_size, &executor));
  ws_func(ws_size, &workspace);
  ACL_CHECK_RET(aclnnNeg(workspace, ws_size, executor, stream));
}

void InplaceDiv(const aclTensor* input, aclTensor** output, aclrtStream& stream, void (*ws_func)(size_t, void**)) {
  uint64_t ws_size = 0ull;
  void* workspace = nullptr;
  aclOpExecutor* executor = nullptr;

  ACL_CHECK_RET(aclnnInplaceDivGetWorkspaceSize(*output, input, &ws_size, &executor));
  ws_func(ws_size, &workspace);
  ACL_CHECK_RET(aclnnInplaceDiv(workspace, ws_size, executor, stream));
}

void InplaceSqrt(aclTensor** output, aclrtStream& stream, void (*ws_func)(size_t, void**)) {
  uint64_t ws_size = 0ull;
  void* workspace = nullptr;
  aclOpExecutor* executor = nullptr;

  ACL_CHECK_RET(aclnnInplaceSqrtGetWorkspaceSize(*output, &ws_size, &executor));
  ws_func(ws_size, &workspace);
  ACL_CHECK_RET(aclnnInplaceSqrt(workspace, ws_size, executor, stream));
}

}  // namespace ascend
}  // namespace llm_kernels
