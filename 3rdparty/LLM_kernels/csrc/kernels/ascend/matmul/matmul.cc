/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include "matmul.h"

#include "aclnnop/aclnn_matmul.h"
#include "csrc/kernels/ascend/matmul/matmul.h"
#include "csrc/utils/ascend/common.h"
#include "tests/references/matmul.h"
#include "tiling/tiling_api.h"

#ifdef ENABLE_ACL_ATB
#  include "atb/infer_op_params.h"
#endif

namespace llm_kernels {
namespace ascend {

aclError InvokeAclNNMatMul(const aclTensor* input, const aclTensor* weight,
                           const llm_kernels::utils::ACLNNMatmulComputeType cube_math_type, aclTensor** output,
                           aclrtStream& stream, void (*ws_func)(size_t, void**)) {
  uint64_t ws_size = 0ull;
  void* workspace = nullptr;
  aclOpExecutor* executor = nullptr;

  ACL_CHECK_RET(
      aclnnMatmulGetWorkspaceSize(input, weight, *output, static_cast<int8_t>(cube_math_type), &ws_size, &executor));
  ws_func(ws_size, &workspace);
  ACL_CHECK_RET(aclnnMatmul(workspace, ws_size, executor, stream));
  ACL_CHECK_RET(aclrtSynchronizeStream(stream));
  return ACL_SUCCESS;
}

#ifdef ENABLE_ACL_ATB
template <typename DTYPE>
void CreateATBMatMulOperator(const size_t m, const size_t n, const size_t k, DTYPE* input_device, DTYPE* weight_device,
                     DTYPE* bias_device, DTYPE* output_device, aclrtStream& stream, void (*ws_func)(size_t, void**)) {
  atb::infer::LinearParam linear_param;
  linear_param.transposeA = false;
  linear_param.transposeB = false;
  if (bias_device == nullptr) {
    linear_param.hasBias = false;
  } else {
    linear_param.hasBias = true;
  }

  if (std::is_same<DTYPE, float>::value) {
    linear_param.outDataType = ACL_FLOAT;
  } else if (std::is_same<DTYPE, aclFloat16>::value) {
    linear_param.outDataType = ACL_FLOAT16;
  } else {
    throw std::invalid_argument("Not support matmul dtype, only support float16 and float32");
  }
}
#endif

}  // namespace ascend
}  // namespace llm_kernels
