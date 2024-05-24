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
#include "lib/matrix/matmul/matmul.h"
using namespace AscendC;
using namespace matmul;

__aicore__ inline void CopyTiling(TCubeTiling* tiling, GM_ADDR tilingGM) {
  uint32_t* ptr = reinterpret_cast<uint32_t*>(tiling);
  auto tiling32 = reinterpret_cast<__gm__ uint32_t*>(tilingGM);

  for (int i = 0; i < sizeof(TCubeTiling) / sizeof(uint32_t); i++, ptr++) {
    *ptr = *(tiling32 + i);
  }
  return;
}

__aicore__ inline void CalcGMOffset(int blockIdx, int usedCoreNum, TCubeTiling& param, int& offsetA, int& offsetB,
                                    int& offsetC) {
  ASSERT(blockIdx < usedCoreNum);
  uint32_t mIterSize = Ceil(param.M, param.singleCoreM);
  ASSERT(mIterSize != 0);
  uint32_t mCoreIndx = blockIdx % mIterSize;
  uint32_t nCoreIndx = blockIdx / mIterSize;

  offsetA = mCoreIndx * param.Ka * param.singleCoreM;
  offsetB = nCoreIndx * param.singleCoreN;
  offsetC = mCoreIndx * param.N * param.singleCoreM + nCoreIndx * param.singleCoreN;

  // tail M
  int gmUseM = param.M - mCoreIndx * param.singleCoreM;
  param.singleCoreM = gmUseM < param.singleCoreM ? gmUseM : param.singleCoreM;

  // tail N
  int gmUseN = param.N - nCoreIndx * param.singleCoreN;
  param.singleCoreN = gmUseN < param.singleCoreN ? gmUseN : param.singleCoreN;

  // tail K
  int gmUseK = param.Ka;
  param.singleCoreK = gmUseK < param.singleCoreK ? gmUseK : param.singleCoreK;
}

extern "C" __global__ __aicore__ void InvokeMatmulHalfKernel(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR tiling_gm) {
  // cube core cases, ignore vector core
  if (g_coreType == AIV) {
    return;
  }
  using A_T = half;
  using B_T = half;
  using C_T = half;
  using BiasT = half;

  TPipe que;
  TCubeTiling tiling;
  CopyTiling(&tiling, tiling_gm);

  if (GetBlockIdx() >= tiling.usedCoreNum) {
    return;
  }

  GlobalTensor<A_T> a_global;
  GlobalTensor<B_T> b_global;
  GlobalTensor<C_T> c_global;

  a_global.SetGlobalBuffer(reinterpret_cast<__gm__ A_T*>(a), tiling.M * tiling.Ka);
  b_global.SetGlobalBuffer(reinterpret_cast<__gm__ B_T*>(b), tiling.Kb * tiling.N);
  c_global.SetGlobalBuffer(reinterpret_cast<__gm__ C_T*>(c), tiling.M * tiling.N);

  int offset_a = 0;
  int offset_b = 0;
  int offset_c = 0;

  auto gm_a = a_global[offset_a];
  auto gm_b = b_global[offset_b];
  auto gm_c = c_global[offset_c];

  typedef MatmulType<AscendC::TPosition::GM, CubeFormat::ND, A_T> AType;
  typedef MatmulType<AscendC::TPosition::GM, CubeFormat::ND, B_T> BType;
  typedef MatmulType<AscendC::TPosition::GM, CubeFormat::ND, C_T> CType;
  typedef MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BiasT> BiasType;
  MatmulImpl<AType, BType, CType, BiasType> mm;
  mm.SetSubBlockIdx(0);
  mm.Init(&tiling, &que);

  mm.SetTensorA(gm_a);
  mm.SetTensorB(gm_b);
  mm.IterateAll(gm_c);
}

extern "C" __global__ __aicore__ void InvokeMatmulFloatKernel(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR tiling_gm) {
  // cube core cases, ignore vector core
  if (g_coreType == AIV) {
    return;
  }
  using A_T = float;
  using B_T = float;
  using C_T = float;
  using BiasT = float;

  TPipe que;
  TCubeTiling tiling;
  CopyTiling(&tiling, tiling_gm);

  if (GetBlockIdx() >= tiling.usedCoreNum) {
    return;
  }

  GlobalTensor<A_T> a_global;
  GlobalTensor<B_T> b_global;
  GlobalTensor<C_T> c_global;

  a_global.SetGlobalBuffer(reinterpret_cast<__gm__ A_T*>(a), tiling.M * tiling.Ka);
  b_global.SetGlobalBuffer(reinterpret_cast<__gm__ B_T*>(b), tiling.Kb * tiling.N);
  c_global.SetGlobalBuffer(reinterpret_cast<__gm__ C_T*>(c), tiling.M * tiling.N);

  int offset_a = 0;
  int offset_b = 0;
  int offset_c = 0;

  auto gm_a = a_global[offset_a];
  auto gm_b = b_global[offset_b];
  auto gm_c = c_global[offset_c];

  typedef MatmulType<AscendC::TPosition::GM, CubeFormat::ND, A_T> AType;
  typedef MatmulType<AscendC::TPosition::GM, CubeFormat::ND, B_T> BType;
  typedef MatmulType<AscendC::TPosition::GM, CubeFormat::ND, C_T> CType;
  typedef MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BiasT> BiasType;
  MatmulImpl<AType, BType, CType, BiasType> mm;
  mm.SetSubBlockIdx(0);
  mm.Init(&tiling, &que);

  mm.SetTensorA(gm_a);
  mm.SetTensorB(gm_b);
  mm.IterateAll(gm_c);
}
