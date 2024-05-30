/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include "matmul.h"

#include "aclnnop/aclnn_matmul.h"
#include "aclrtlaunch_InvokeMatmulFloatKernel.h"
#include "aclrtlaunch_InvokeMatmulHalfKernel.h"
#include "csrc/kernels/ascend/matmul/matmul.h"
#include "csrc/utils/ascend/common.h"
#include "tests/references/matmul.h"
#include "tiling/tiling_api.h"

namespace llm_kernels {
namespace ascend {

aclError MatMul(const aclTensor* input, const aclTensor* weight, const int8_t matmulCubeMathType, aclTensor** output,
                aclrtStream& stream, void (*ws_func)(size_t, void**)) {
  uint64_t ws_size = 0ull;
  void* workspace = nullptr;
  aclOpExecutor* executor = nullptr;

  ACL_CHECK_RET(aclnnMatmulGetWorkspaceSize(input, weight, *output, matmulCubeMathType, &ws_size, &executor));
  ws_func(ws_size, &workspace);
  ACL_CHECK_RET(aclnnMatmul(workspace, ws_size, executor, stream));
  ACL_CHECK_RET(aclrtSynchronizeStream(stream));
  return ACL_SUCCESS;
}

template <typename DTYPE>
void InvokeMatMul(const size_t m, const size_t n, const size_t k, DTYPE* input_device, DTYPE* weight_device,
                  DTYPE* bias_device, DTYPE* output_device, aclrtStream& stream, void (*ws_func)(size_t, void**)) {
  if (bias_device != nullptr) {
    throw std::invalid_argument("Not support matmul with bias.");
  }

  size_t a_size = m * k * sizeof(DTYPE);
  size_t b_size = k * n * sizeof(DTYPE);
  size_t c_size = m * n * sizeof(DTYPE);
  uint32_t block_dim = CUBE_CORE_NUM;

  matmul_tiling::DataType computing_dtype = matmul_tiling::DataType::DT_FLOAT16;
  if (std::is_same<DTYPE, aclFloat16>::value) {
    computing_dtype = matmul_tiling::DataType::DT_FLOAT16;
  } else if (std::is_same<DTYPE, float>::value) {
    computing_dtype = matmul_tiling::DataType::DT_FLOAT;
  } else {
    throw std::invalid_argument("Not support matmul dtype, only support float16 and float32.");
  }

  matmul_tiling::TPosition left_pos = matmul_tiling::TPosition::GM;
  matmul_tiling::CubeFormat left_format = matmul_tiling::CubeFormat::ND;
  matmul_tiling::DataType left_dtype = computing_dtype;
  bool transpose_a = false;
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

  optiling::TCubeTiling tiling_data;
  tiling_data.set_usedCoreNum(block_dim);
  matmul_tiling::MatmulApiTiling tiling_api;
  tiling_api.SetAType(left_pos, left_format, left_dtype, transpose_a);
  tiling_api.SetBType(right_pos, right_format, right_dtype, transpose_b);
  tiling_api.SetCType(res_pos, res_format, res_dtype);
  tiling_api.SetBiasType(bias_pos, bias_format, bias_dtype);
  tiling_api.SetOrgShape(m, n, k);
  tiling_api.SetShape(m, n, k);
  tiling_api.SetBias(is_bias);
  tiling_api.SetBufferSpace(-1, -1, -1);
  if (tiling_api.GetTiling(tiling_data) == -1) {
    throw std::runtime_error("Init matmul tiling error: matmul_tiling::MatmulApiTiling::GetTiling.");
  }
  matmul_tiling::SysTilingTempBufSize buf_size;
  if (MultiCoreMatmulGetTmpBufSize(tiling_data, buf_size) == -1) {
    throw std::runtime_error("Init matmul tiling error: MultiCoreMatmulGetTmpBufSize.");
  }
  uint32_t tiling_size = tiling_data.GetDataSize();
  std::vector<char> buf(tiling_size);
  tiling_data.SaveToBuffer((void*)(buf.data()), tiling_size);

  void* tiling_device;
  ws_func(tiling_size, &tiling_device);
  ACL_CHECK_RET(aclrtMemcpyAsync(tiling_device, tiling_size, (void*)(buf.data()), tiling_size,
                                 ACL_MEMCPY_HOST_TO_DEVICE, stream));
  ACL_CHECK_RET(aclrtSynchronizeStream(stream));
  if (std::is_same<DTYPE, aclFloat16>::value) {
    ACL_CHECK_RET(ACLRT_LAUNCH_KERNEL(InvokeMatmulHalfKernel)(block_dim, stream, (uint8_t*)input_device,
                                                              (uint8_t*)weight_device, (uint8_t*)output_device,
                                                              (uint8_t*)tiling_device));
  } else if (std::is_same<DTYPE, float>::value) {
    ACL_CHECK_RET(ACLRT_LAUNCH_KERNEL(InvokeMatmulFloatKernel)(block_dim, stream, (uint8_t*)input_device,
                                                               (uint8_t*)weight_device, (uint8_t*)output_device,
                                                               (uint8_t*)tiling_device));
  } else {
    throw std::invalid_argument("Not support matmul dtype, only support float16 and float32");
  }
}

template void InvokeMatMul(const size_t m, const size_t n, const size_t k, aclFloat16* input_device,
                           aclFloat16* weight_device, aclFloat16* bias_device, aclFloat16* output_device,
                           aclrtStream& stream, void (*ws_func)(size_t, void**));
template void InvokeMatMul(const size_t m, const size_t n, const size_t k, float* input_device, float* weight_device,
                           float* bias_device, float* output_device, aclrtStream& stream,
                           void (*ws_func)(size_t, void**));

}  // namespace ascend
}  // namespace llm_kernels
