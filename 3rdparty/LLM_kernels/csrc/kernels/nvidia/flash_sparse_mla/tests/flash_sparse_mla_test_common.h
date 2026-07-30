/*
 * Copyright 2026 Tencent Inc.  All rights reserved.
 */

#pragma once

#include "flash_sparse_mla_test_config.h"

#include "csrc/kernels/nvidia/flash_sparse_mla/flash_sparse_mla.h"
#include "csrc/utils/nvidia/cuda_utils.h"
#include "params.h"
#include "tests/kernels/nvidia/utils/testsuit_base.h"

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/torch.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <iomanip>
#include <random>
#include <sstream>

namespace llm_kernels {
namespace nvidia {
namespace test {

using namespace llm_kernels::utils;

struct AccuracyResult;
struct MlaPerfRow;

class LlamaNvidiaFlashSparseMlaTestSuit;

class LlamaNvidiaFlashSparseMlaTestSuit : public NvidiaTestSuitBase {
 public:
  friend AccuracyResult VerifyAccuracy(const LlamaNvidiaFlashSparseMlaTestSuit& fx, const ModelConfig& cfg,
                                       size_t batch_size, size_t seqlen_q, size_t num_heads_q, size_t seqlen_k,
                                       IndicesStrategy idx_strategy, const torch::Tensor& d_out);

  friend void MeasureAndSave(LlamaNvidiaFlashSparseMlaTestSuit& fx, const ModelConfig& cfg, size_t batch_size,
                             size_t seqlen_q, size_t seqlen_k, size_t num_heads_q, IndicesStrategy idx_strategy,
                             const MlaTestRange& range, size_t backend_idx,
                             std::vector<std::vector<MlaPerfRow>>& all_results);

  friend void RunMlaBenchmark(LlamaNvidiaFlashSparseMlaTestSuit& fx, ModelType type, const MlaTestRange& range);

  void SetUp() override {
    NvidiaTestSuitBase::SetUp();
    c10::cuda::CUDAStream torch_stream = c10::cuda::getStreamFromExternal(stream, device);
    c10::cuda::setCurrentCUDAStream(torch_stream);
    torch::manual_seed(0);
    torch::cuda::manual_seed_all(0);
    selected_backends = ParseMlaBackends();
    std::cout << "[MLA_BACKENDS] running " << selected_backends.size() << " backend(s):";
    for (size_t i = 0; i < selected_backends.size(); ++i) {
      std::cout << (i == 0 ? " " : ", ") << selected_backends[i].name;
    }
    std::cout << std::endl;
  }

  void TearDown() override {
    c10::cuda::setCurrentCUDAStream(c10::cuda::getDefaultCUDAStream(device));
    NvidiaTestSuitBase::TearDown();
  }

 protected:
  void GenerateIndices(IndicesStrategy strategy, int* indices_ptr, size_t batch_size, size_t seqlen_q, size_t topk,
                       size_t num_blocks, size_t seqlen_k) {
    std::mt19937 rng(42);
    size_t valid_blocks = num_blocks / batch_size;
    valid_blocks = std::max(valid_blocks, size_t(1));
    for (size_t i = 0; i < batch_size; i++) {
      const size_t batch_block_offset = i * valid_blocks;
      for (size_t j = 0; j < seqlen_q; j++) {
        size_t max_valid = std::min(topk, seqlen_k - seqlen_q + j + 1);
        for (size_t k = 0; k < topk; k++) {
          if (k < max_valid) {
            int block_id;
            if (strategy == IndicesStrategy::kSequential) {
              block_id = static_cast<int>(k % valid_blocks);
            } else if (strategy == IndicesStrategy::kRandom) {
              block_id = static_cast<int>(rng() % valid_blocks);
            } else {
              size_t window_start = (j / kBlockSize) % valid_blocks;
              if (k < topk * 3 / 4) {
                block_id = static_cast<int>((window_start + k) % valid_blocks);
              } else {
                block_id = static_cast<int>(rng() % valid_blocks);
              }
            }
            const int abs_block_id = static_cast<int>(batch_block_offset) + block_id;
            indices_ptr[i * seqlen_q * topk + j * topk + k] = abs_block_id * kBlockSize + (k % kBlockSize);
          } else {
            indices_ptr[i * seqlen_q * topk + j * topk + k] = -1;
          }
        }
      }
    }
  }

  void PrepareData(const ModelConfig& cfg, const size_t batch_size, const size_t seqlen_q, const size_t seqlen_k,
                   const size_t num_heads_q, IndicesStrategy idx_strategy = IndicesStrategy::kSequential) {
    current_cfg = &cfg;
    kcache_batch_stride_fp8 = kBlockSize * (cfg.head_size_k + kCacheBlockPadding);

    auto gpu = torch::TensorOptions().device(torch::kCUDA, device);
    auto cpu = torch::TensorOptions().device(torch::kCPU);

    d_tile_scheduler_metadata = torch::empty({static_cast<int64_t>(GetNumSmParts(num_heads_q, seqlen_q)),
                                              static_cast<int64_t>(GetDecodingSchedMetaSize() / sizeof(int))},
                                             gpu.dtype(torch::kInt32));
    d_num_splits = torch::empty({static_cast<int64_t>(batch_size + 1)}, gpu.dtype(torch::kInt32));

    const size_t valid_blocks = (seqlen_k + kBlockSize - 1) / kBlockSize;
    num_blocks = batch_size * valid_blocks;

    h_indices = torch::empty({static_cast<int64_t>(batch_size * seqlen_q), static_cast<int64_t>(cfg.topk)},
                             cpu.dtype(torch::kInt32));
    GenerateIndices(idx_strategy, h_indices.data_ptr<int>(), batch_size, seqlen_q, cfg.topk, num_blocks, seqlen_k);
    d_indices = h_indices.to(torch::kCUDA);

    if (cfg.IsModel1()) {
      h_topk_length = torch::empty({static_cast<int64_t>(batch_size * seqlen_q)}, cpu.dtype(torch::kInt32));
      int* topk_length_data = h_topk_length.data_ptr<int>();
      for (size_t i = 0; i < batch_size; i++) {
        for (size_t j = 0; j < seqlen_q; j++) {
          topk_length_data[i * seqlen_q + j] = std::min(cfg.topk, seqlen_k - seqlen_q + j + 1);
        }
      }
      d_topk_length = h_topk_length.to(torch::kCUDA);
    }

    if (cfg.IsModel1()) {
      extra_num_blocks = num_blocks;
      d_extra_kcache =
          torch::rand({static_cast<int64_t>(extra_num_blocks), static_cast<int64_t>(cfg.kcache_batch_stride)},
                      gpu.dtype(torch::kFloat32));
      d_extra_kcache.mul_(2.0f).sub_(1.0f);
      d_extra_kcache = d_extra_kcache.to(torch::kFloat8_e4m3fn);
      d_extra_kcache_fp8 =
          torch::rand({static_cast<int64_t>(extra_num_blocks), static_cast<int64_t>(kcache_batch_stride_fp8)},
                      gpu.dtype(torch::kFloat32));
      d_extra_kcache_fp8.mul_(2.0f).sub_(1.0f);
      d_extra_kcache_fp8 = d_extra_kcache_fp8.to(torch::kFloat8_e4m3fn);
      d_extra_indices = h_indices.to(torch::kCUDA);
      d_extra_topk_length = h_topk_length.to(torch::kCUDA);
      d_attn_sink = torch::rand({static_cast<int64_t>(std::max<size_t>(64, num_heads_q))}, gpu.dtype(torch::kFloat32));
      d_attn_sink.mul_(2.0f).sub_(1.0f);
    }

    d_q = torch::rand({static_cast<int64_t>(batch_size), static_cast<int64_t>(seqlen_q),
                       static_cast<int64_t>(num_heads_q), static_cast<int64_t>(cfg.head_size_k)},
                      gpu.dtype(torch::kFloat32));
    d_q.mul_(2.0f).sub_(1.0f);
    d_q = d_q.to(torch::kBFloat16);
    d_q_fp8 = torch::rand({static_cast<int64_t>(batch_size), static_cast<int64_t>(seqlen_q),
                           static_cast<int64_t>(num_heads_q), static_cast<int64_t>(cfg.head_size_k)},
                          gpu.dtype(torch::kFloat32));
    d_q_fp8.mul_(2.0f).sub_(1.0f);
    d_q_fp8 = d_q_fp8.to(torch::kFloat8_e4m3fn);
    d_kcache = torch::rand({static_cast<int64_t>(num_blocks), static_cast<int64_t>(cfg.kcache_batch_stride)},
                           gpu.dtype(torch::kFloat32));
    d_kcache.mul_(2.0f).sub_(1.0f);
    d_kcache = d_kcache.to(torch::kFloat8_e4m3fn);
    d_kcache_fp8 = torch::rand({static_cast<int64_t>(num_blocks), static_cast<int64_t>(kcache_batch_stride_fp8)},
                               gpu.dtype(torch::kFloat32));
    d_kcache_fp8.mul_(2.0f).sub_(1.0f);
    d_kcache_fp8 = d_kcache_fp8.to(torch::kFloat8_e4m3fn);
    d_workspace = torch::empty({1 << 28}, gpu.dtype(torch::kFloat32));
    d_out = torch::empty({static_cast<int64_t>(batch_size), static_cast<int64_t>(seqlen_q),
                          static_cast<int64_t>(num_heads_q), static_cast<int64_t>(cfg.head_size_v)},
                         gpu.dtype(torch::kBFloat16));
  }

  int* GetTopkLengthPtr() const { return current_cfg->IsModel1() ? d_topk_length.data_ptr<int>() : nullptr; }
  float* GetAttnSinkPtr() const { return current_cfg->IsModel1() ? d_attn_sink.data_ptr<float>() : nullptr; }
  uint8_t* GetExtraKcachePtr() const {
    return current_cfg->IsModel1() ? reinterpret_cast<uint8_t*>(d_extra_kcache.data_ptr()) : nullptr;
  }
  uint8_t* GetExtraKcacheFp8Ptr() const {
    return current_cfg->IsModel1() ? reinterpret_cast<uint8_t*>(d_extra_kcache_fp8.data_ptr()) : nullptr;
  }
  int* GetExtraIndicesPtr() const { return current_cfg->IsModel1() ? d_extra_indices.data_ptr<int>() : nullptr; }
  int* GetExtraTopkLengthPtr() const { return current_cfg->IsModel1() ? d_extra_topk_length.data_ptr<int>() : nullptr; }

  using NvidiaTestSuitBase::device;
  using NvidiaTestSuitBase::stream;
  using T = __nv_bfloat16;

  std::vector<MlaBackendSpec> selected_backends;

  const size_t num_heads_k = 1;
  const float softmax_scale = 0.1147213867929261;

  const ModelConfig* current_cfg = nullptr;
  size_t kcache_batch_stride_fp8 = 0;
  size_t num_blocks = 0;
  size_t extra_num_blocks = 0;

  torch::Tensor d_tile_scheduler_metadata;
  torch::Tensor d_num_splits;
  torch::Tensor h_indices;
  torch::Tensor d_indices;
  torch::Tensor h_topk_length;
  torch::Tensor d_topk_length;
  torch::Tensor d_extra_kcache;
  torch::Tensor d_extra_kcache_fp8;
  torch::Tensor d_extra_indices;
  torch::Tensor d_extra_topk_length;
  torch::Tensor d_attn_sink;
  torch::Tensor d_q;
  torch::Tensor d_q_fp8;
  torch::Tensor d_kcache;
  torch::Tensor d_kcache_fp8;
  torch::Tensor d_workspace;
  torch::Tensor d_out;
};

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels
