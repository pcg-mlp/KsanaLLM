/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

#include "csrc/kernels/ascend/permute/permute_tiling.h"

using namespace AscendC;
using namespace llm_kernels::ascend;

constexpr int32_t BUFFER_NUM = 1;  // tensor num for each queue

__aicore__ inline void CopyTiling(PermuteTilingData* tiling, GM_ADDR tiling_gm) {
  uint32_t* tiling_ptr = reinterpret_cast<uint32_t*>(tiling);
  __gm__ uint32_t* tiling_gm_ptr = reinterpret_cast<__gm__ uint32_t*>(tiling_gm);

  for (int i = 0; i < sizeof(PermuteTilingData) / sizeof(uint32_t); i++, tiling_ptr++) {
    *tiling_ptr = *(tiling_gm_ptr + i);
  }
}

// A implement of paged attention.
template <typename T>
class PermuteKernel {
 public:
  __aicore__ inline PermuteKernel() {}

  __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, PermuteTilingData* tiling);

  __aicore__ inline void Process();

 private:
  // The process.
  __aicore__ inline uint32_t GetInputIndexPos(uint32_t i, uint32_t j, uint32_t k, uint32_t x, uint32_t y, uint32_t z);

  __aicore__ inline uint32_t GetNewIndexPos(uint32_t i, uint32_t j, uint32_t k, uint32_t x, uint32_t y, uint32_t z);

  // The tiling config.
  PermuteTilingData* tiling_;

  GlobalTensor<T> input_gm_;
  GlobalTensor<T> output_gm_;

  // The input and output queue.
  TQue<QuePosition::VECIN, BUFFER_NUM> input_queue_;

  int32_t block_idx_ = 0;
  int32_t block_dim_ = 0;
};

template <typename T>
__aicore__ uint32_t PermuteKernel<T>::GetInputIndexPos(uint32_t i, uint32_t j, uint32_t k, uint32_t x, uint32_t y,
                                                       uint32_t z) {
  return (i * tiling_->stride0 + j * tiling_->stride1 + k * tiling_->stride2 + x * tiling_->stride3 +
          y * tiling_->stride4 + z * tiling_->stride5);
}

template <typename T>
__aicore__ uint32_t PermuteKernel<T>::GetNewIndexPos(uint32_t i, uint32_t j, uint32_t k, uint32_t x, uint32_t y,
                                                     uint32_t z) {
  uint32_t indexes[6] = {i, j, k, x, y, z};
  return (indexes[tiling_->new_idx0] * tiling_->new_stride0 + indexes[tiling_->new_idx1] * tiling_->new_stride1 +
          indexes[tiling_->new_idx2] * tiling_->new_stride2 + indexes[tiling_->new_idx3] * tiling_->new_stride3 +
          indexes[tiling_->new_idx4] * tiling_->new_stride4 + indexes[tiling_->new_idx5] * tiling_->new_stride5);
}

template <typename T>
__aicore__ void PermuteKernel<T>::Init(GM_ADDR input, GM_ADDR output, PermuteTilingData* tiling) {
  tiling_ = tiling;

  block_idx_ = GetBlockIdx();
  block_dim_ = tiling_->used_core_num;

  input_gm_.SetGlobalBuffer((__gm__ T*)input);
  output_gm_.SetGlobalBuffer((__gm__ T*)output);
}

template <typename T>
__aicore__ void PermuteKernel<T>::Process() {
  bool should_break = false;
  for (size_t i = 0; i < tiling_->dim0; ++i) {
    if (should_break) break;
    for (size_t j = 0; j < tiling_->dim1; ++j) {
      if (should_break) break;
      for (size_t k = 0; k < tiling_->dim2; ++k) {
        if (should_break) break;
        for (size_t x = 0; x < tiling_->dim3; ++x) {
          if (should_break) break;
          for (size_t y = 0; y < tiling_->dim4; ++y) {
            if (should_break) break;
            for (size_t z = 0; z < tiling_->dim5; ++z) {
              uint64_t src_pos = GetInputIndexPos(i, j, k, x, y, z);
              if (src_pos >= tiling_->block_length * (block_idx_ + 1) || src_pos >= tiling_->total_length) {
                should_break = true;
                break;
              }

              if (src_pos >= tiling_->block_length * block_idx_) {
                uint64_t dst_pos = GetNewIndexPos(i, j, k, x, y, z);
                *(const_cast<__gm__ T*>(output_gm_.GetPhyAddr()) + dst_pos) =
                    *(const_cast<__gm__ T*>(input_gm_.GetPhyAddr()) + src_pos);
              }
            }
          }
        }
      }
    }
  }
}

extern "C" __global__ __aicore__ void InvokePermuteKernel(GM_ADDR input, GM_ADDR output, GM_ADDR workspace,
                                                          GM_ADDR tiling_gm) {
  PermuteTilingData tiling;
  CopyTiling(&tiling, tiling_gm);
  if (GetBlockIdx() >= tiling.used_core_num) {
    return;
  }

  if (workspace == nullptr) {
    return;
  }

  SetSysWorkspace(workspace);
  GM_ADDR usr_workspace = GetUserWorkspace(workspace);
  if (usr_workspace == nullptr) {
    return;
  }

  if (tiling.tiling_key == 0) {
    PermuteKernel<half> op;
    op.Init(input, output, &tiling);
    op.Process();
  } else if (tiling.tiling_key == 1) {
    PermuteKernel<float> op;
    op.Init(input, output, &tiling);
    op.Process();
  }
}
