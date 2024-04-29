/**
 * Copyright 2024 Tencent Inc.  All rights reserved.
 * Modify by karlluo@tencent.com
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
using namespace AscendC;
using namespace matmul;

__aicore__ inline void InitDefaultTiling(TCubeTiling& tiling) {
  tiling.shareMode = 0;
  tiling.shareL1Size = TOTAL_L1_SIZE;
  tiling.shareL0CSize = TOTAL_L0C_SIZE;
  tiling.shareUbSize = 0;
}

template <typename aType, typename bType, typename cType, typename biasType>
class MatmulSiluKernel {
 public:
  __aicore__ inline MatmulSiluKernel() {}
  __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR c, GM_ADDR workspace, GM_ADDR tiling);
  __aicore__ inline void Process();

 private:
  __aicore__ inline void MamtulCompute();
  __aicore__ inline void SiluCompute();
  __aicore__ inline void CopyOut(uint32_t count);
  __aicore__ inline void CalcOffset(int32_t blockIdx, const TCubeTiling& tiling, int32_t& offset_a, int32_t& offset_b,
                                    int32_t& offset_c, int32_t& offset_bias);

  Matmul<MatmulType<TPosition::GM, CubeFormat::ND, aType>, MatmulType<TPosition::GM, CubeFormat::ND, bType>,
         MatmulType<TPosition::LCM, CubeFormat::ND, cType>, MatmulType<TPosition::GM, CubeFormat::ND, biasType>>
      matmul_obj;
  GlobalTensor<aType> a_global;
  GlobalTensor<bType> b_global;
  GlobalTensor<cType> c_global;
  GlobalTensor<biasType> bias_global;
  LocalTensor<cType> silu_output_local;
  TPipe pipe;
  TCubeTiling tiling;
  TQue<QuePosition::VECOUT, 1> silu_out_queue_;
};

template <typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void MatmulSiluKernel<aType, bType, cType, biasType>::Init(GM_ADDR a, GM_ADDR b, GM_ADDR bias,
                                                                             GM_ADDR c, GM_ADDR workspace,
                                                                             GM_ADDR tiling_gm) {
  auto tmp_tiling_gm = (__gm__ uint32_t*)tiling_gm;
  auto tmp_tiling = (uint32_t*)&tiling;
  for (int32_t i = 0; i < sizeof(TCubeTiling) / sizeof(int32_t); ++i, ++tmp_tiling_gm, ++tmp_tiling) {
    *tmp_tiling = *tmp_tiling_gm;
  }
  a_global.SetGlobalBuffer(reinterpret_cast<__gm__ aType*>(a), tiling.M * tiling.Ka);
  b_global.SetGlobalBuffer(reinterpret_cast<__gm__ bType*>(b), tiling.Kb * tiling.N);
  c_global.SetGlobalBuffer(reinterpret_cast<__gm__ cType*>(c), tiling.M * tiling.N);
  bias_global.SetGlobalBuffer(reinterpret_cast<__gm__ biasType*>(bias), tiling.N);

  int32_t offset_a, offset_b, offset_c, offset_bias;
  CalcOffset(GetBlockIdx(), tiling, offset_a, offset_b, offset_c, offset_bias);
  a_global = a_global[offset_a];
  b_global = b_global[offset_b];
  c_global = c_global[offset_c];
  bias_global = bias_global[offset_bias];
  pipe.InitBuffer(silu_out_queue_, 1, TOTAL_VEC_LOCAL_SIZE);
  InitDefaultTiling(tiling);

  SetSysWorkspace(workspace);
  if (GetSysWorkSpacePtr() == nullptr) {
    return;
  }
}

template <typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void MatmulSiluKernel<aType, bType, cType, biasType>::Process() {
  uint32_t compute_round = 0;
  REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), matmul_obj);
  matmul_obj.Init(&tiling);
  matmul_obj.SetTensorA(a_global);
  matmul_obj.SetTensorB(b_global);
  matmul_obj.SetBias(bias_global);

  while (matmul_obj.template Iterate<true>()) {
    MamtulCompute();
    SiluCompute();
    CopyOut(compute_round);
    compute_round++;
  }
  matmul_obj.End();
}

template <typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void MatmulSiluKernel<aType, bType, cType, biasType>::MamtulCompute() {
  silu_output_local = silu_out_queue_.AllocTensor<cType>();
  matmul_obj.template GetTensorC<true>(silu_output_local, false, true);
}

template <typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void MatmulSiluKernel<aType, bType, cType, biasType>::SiluCompute() {
  Silu(silu_output_local, silu_output_local, tiling.baseM * tiling.baseN);
  silu_out_queue_.EnQue(silu_output_local);
}

template <typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void MatmulSiluKernel<aType, bType, cType, biasType>::CopyOut(uint32_t count) {
  silu_out_queue_.DeQue<cType>();
  const uint32_t round_m = tiling.singleCoreM / tiling.baseM;
  const uint32_t round_n = tiling.singleCoreN / tiling.baseN;
  uint32_t start_offset = (count % round_m * tiling.baseM * tiling.N + count / round_m * tiling.baseN);
  DataCopyParams copy_param = {(uint16_t)tiling.baseM, (uint16_t)(tiling.baseN * sizeof(cType) / DEFAULT_C0_SIZE), 0,
                               (uint16_t)((tiling.N - tiling.baseN) * sizeof(cType) / DEFAULT_C0_SIZE)};
  DataCopy(c_global[start_offset], silu_output_local, copy_param);
  silu_out_queue_.FreeTensor(silu_output_local);
}

template <typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void MatmulSiluKernel<aType, bType, cType, biasType>::CalcOffset(int32_t blockIdx,
                                                                                   const TCubeTiling& tiling,
                                                                                   int32_t& offset_a, int32_t& offset_b,
                                                                                   int32_t& offset_c,
                                                                                   int32_t& offset_bias) {
  auto temp0 = Ceil(tiling.M, tiling.singleCoreM);
  auto temp1 = Ceil(tiling.N, tiling.singleCoreN);
  auto m_core_idx = blockIdx % temp0;
  auto n_core_idx = blockIdx / temp0;
  offset_a = m_core_idx * tiling.Ka * tiling.singleCoreM;
  offset_b = n_core_idx * tiling.singleCoreN;
  offset_c = m_core_idx * tiling.N * tiling.singleCoreM + n_core_idx * tiling.singleCoreN;
  offset_bias = n_core_idx * tiling.singleCoreN;
}

extern "C" __global__ __aicore__ void InvokeMatmulSiluKernel(GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR c,
                                                             GM_ADDR workspace, GM_ADDR tiling) {
  MatmulSiluKernel<half, half, float, float> matmul_silu_kernel;
  matmul_silu_kernel.Init(a, b, bias, c, workspace, tiling);
  matmul_silu_kernel.Process();
}
