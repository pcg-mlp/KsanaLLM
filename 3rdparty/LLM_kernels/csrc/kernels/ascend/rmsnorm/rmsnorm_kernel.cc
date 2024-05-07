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
#include "kernel_tiling/kernel_tiling.h"
#include "lib/rmsnorm/kernel_operator_rmsnorm_intf.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;  // tensor num for each queue

template <typename T>
class RmsNormKernel {
 public:
  __aicore__ inline RmsNormKernel() {}
  __aicore__ inline void Init(GM_ADDR input, GM_ADDR gamma, GM_ADDR output, GM_ADDR tiling_gm);
  __aicore__ inline void Process();

 private:
  __aicore__ inline void CopyIn();
  __aicore__ inline void RmsNormCompute();
  __aicore__ inline void CopyOut();

  GlobalTensor<T> input_global;
  GlobalTensor<T> gamma_global;
  GlobalTensor<T> output_global;
  TPipe pipe;
  RmsNormTiling tiling;

  // create queues for input, in this case depth is equal to buffer num
  TQue<QuePosition::VECIN, BUFFER_NUM> input_queue, gamma_queue;
  // create queue for output, in this case depth is equal to buffer num
  TQue<QuePosition::VECOUT, BUFFER_NUM> output_queue;
};

template <typename T>
__aicore__ inline void RmsNormKernel<T>::Init(GM_ADDR input, GM_ADDR gamma, GM_ADDR output, GM_ADDR tiling_gm) {
  ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
  auto tmp_tiling_gm = (__gm__ uint32_t*)tiling_gm;
  auto tmp_tiling = (uint32_t*)&tiling;
  for (int32_t i = 0; i < sizeof(RmsNormTiling) / sizeof(uint32_t); ++i, ++tmp_tiling_gm, ++tmp_tiling) {
    *tmp_tiling = *tmp_tiling_gm;
  }

  if (GetBlockIdx() >= tiling.sLength) {
    return;
  }

  // get start index for current core, core parallel
  input_global.SetGlobalBuffer((__gm__ T*)input + tiling.hLength * GetBlockIdx(), tiling.hLength);
  gamma_global.SetGlobalBuffer((__gm__ T*)gamma + tiling.hLength * GetBlockIdx(), tiling.hLength);
  output_global.SetGlobalBuffer((__gm__ T*)output + tiling.hLength * GetBlockIdx(), tiling.hLength);

  pipe.InitBuffer(input_queue, BUFFER_NUM, tiling.hLength * sizeof(T));
  pipe.InitBuffer(gamma_queue, BUFFER_NUM, tiling.hLength * sizeof(T));
  pipe.InitBuffer(output_queue, BUFFER_NUM, tiling.hLength * sizeof(T));
}

template <typename T>
__aicore__ inline void RmsNormKernel<T>::Process() {
  CopyIn();
  RmsNormCompute();
  CopyOut();
}

template <typename T>
__aicore__ inline void RmsNormKernel<T>::CopyIn() {
  // alloc tensor from queue memory
  LocalTensor<T> input_local = input_queue.AllocTensor<T>();
  LocalTensor<T> gamma_local = gamma_queue.AllocTensor<T>();
  // copy progress_th tile from global tensor to local tensor
  DataCopy(input_local, input_global[0], tiling.hLength);
  DataCopy(gamma_local, gamma_global[0], tiling.hLength);
  // enque input tensors to VECIN queue
  input_queue.EnQue(input_local);
  gamma_queue.EnQue(gamma_local);
}

template <typename T>
__aicore__ inline void RmsNormKernel<T>::CopyOut() {
  // deque output tensor from VECOUT queue
  LocalTensor<T> output_local = output_queue.DeQue<half>();
  // copy progress_th tile from local tensor to global tensor
  DataCopy(output_global[0], output_local, tiling.hLength);
  // free output tensor for reuse
  output_queue.FreeTensor(output_local);
}

template <typename T>
__aicore__ inline void RmsNormKernel<T>::RmsNormCompute() {
  // deque input tensors from VECIN queue
  LocalTensor<T> input_local = input_queue.DeQue<T>();
  LocalTensor<T> gamma_local = gamma_queue.DeQue<T>();
  LocalTensor<T> output_local = output_queue.AllocTensor<T>();
  // call RmsNorm instr for computation
  RmsNorm<T>(output_local, input_local,gamma_local, (T)1e-6, tiling);
  // enque the output tensor to VECOUT queue
  output_queue.EnQue<T>(output_local);
  // free input tensors for reuse
  input_queue.FreeTensor(input_local);
  gamma_queue.FreeTensor(gamma_local);
}

extern "C" __global__ __aicore__ void InvokeRmsNormKernel(GM_ADDR input, GM_ADDR gamma, GM_ADDR output,
                                                          GM_ADDR tiling_gm) {
  RmsNormKernel<half> rms_norm_kernel;
  rms_norm_kernel.Init(input, gamma, output, tiling_gm);
  rms_norm_kernel.Process();
}
