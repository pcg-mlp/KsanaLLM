/*
 * Copyright 2024 Tencent Inc.  All rights reserved.
 */

#include <random>
#include <sstream>

#include <gtest/gtest.h>

#include "csrc/kernels/nvidia/bgmv/bgmv.h"
#include "csrc/kernels/nvidia/bgmv/bgmv_config.h"
#include "tests/kernels/nvidia/utils/testsuit_base.h"

namespace llm_kernels {
namespace nvidia {
namespace test {
class LlamaNvidiaBGMVTestSuit : public NvidiaTestSuitBase {
 public:
  void SetUp() override { NvidiaTestSuitBase::SetUp(); }

  void TearDown() override { NvidiaTestSuitBase::TearDown(); }

 protected:
  using NvidiaTestSuitBase::stream;

  template <typename T>
  void PytorchReference(const size_t layer_idx, const float scale) {
    // NOTE(karlluo): original implement reference:
    //   def _lora_ref_impl(
    //     y_final: torch.Tensor,
    //     x: torch.Tensor,
    //     wa_T_all: torch.Tensor,
    //     wb_T_all: torch.Tensor,
    //     indices: torch.LongTensor,
    //     layer_idx: int32_t,
    //     scale: float,
    // ):
    //     y_stage_1 = torch.empty(
    //         (x.size(0), wa_T_all.size(-2)),
    //         dtype=torch.float32,
    //         device=x.device,
    //     )
    //     bs = x.shape[0]
    //     s = torch.tensor(scale, dtype=torch.float32, device=x.device)
    //     for i, lora_idx in zip(range(bs), indices.cpu().tolist()):
    //         xi = x[i].unsqueeze(0).to(torch.float32)
    //         wa = wa_T_all[lora_idx, layer_idx].transpose(-1, -2).to(torch.float32)
    //         wb = wb_T_all[lora_idx, layer_idx].transpose(-1, -2).to(torch.float32)

    //         tmp = xi @ wa
    //         y_stage_1[i] = tmp.squeeze(0)
    //         y_final[i] += (tmp @ wb).squeeze(0) * s
    //     return y_final, y_stage_1
    std::stringstream ss;
    ss << "python bgmv_test.py --layer_idx=" << layer_idx << " --scale=" << scale;
    system(ss.str().c_str());
  }

  template <typename T>
  void TestLoraCorrectness(const size_t h1, const size_t h2) {
    constexpr size_t num_loras = 4;
    constexpr size_t num_layers = 1;
    constexpr size_t r = 8;
    constexpr size_t bs = 32;
    constexpr float scale = 0.123f;
    constexpr int32_t default_rank = 0;
    const std::string y_ref_str = "y_ref.npy";
    std::random_device dev;
    std::mt19937 rng(dev());

    BufferMeta wa_T_all =
        CreateBuffer<T>(MemoryType::MEMORY_GPU, {num_loras, num_layers, r, h1}, /*is_random_init*/ true);
    BufferMeta wb_T_all =
        CreateBuffer<T>(MemoryType::MEMORY_GPU, {num_loras, num_layers, h2, r}, /*is_random_init*/ true);
    BufferMeta indices_host = CreateBuffer<int64_t>(MemoryType::MEMORY_CPU, {bs}, /*is_random_init*/ false);
    std::uniform_int_distribution<int64_t> dist_indices(0, num_loras - 1);
    for (size_t bs_idx = 0; bs_idx < bs; ++bs_idx) {
      reinterpret_cast<int64_t*>(indices_host.data_ptr)[bs_idx] = dist_indices(rng);
    }
    BufferMeta indices = CopyToDevice<int64_t>(indices_host);
    BufferMeta x = CreateBuffer<T>(MemoryType::MEMORY_GPU, {bs, h1}, /*is_random_init*/ true);
    BufferMeta y = CreateBuffer<T>(MemoryType::MEMORY_GPU, {bs, h2}, /*is_random_init*/ false);
    BufferMeta y_ref = CreateBuffer<T>(MemoryType::MEMORY_GPU, {bs, h2}, /*is_random_init*/ false);
    BufferMeta y_stage_1 = CreateBuffer<float>(MemoryType::MEMORY_GPU, {bs, r}, /*is_random_init*/ false);
    BufferMeta y_stage_1_ref = CreateBuffer<float>(MemoryType::MEMORY_GPU, {bs, r}, /*is_random_init*/ false);

    wa_T_all.SaveNpy<T>("wa_T_all.npy");
    wb_T_all.SaveNpy<T>("wb_T_all.npy");
    x.SaveNpy<T>("x.npy");
    indices.SaveNpy<int64_t>("indices.npy");
    y_ref.SaveNpy<T>("y_ref.npy");
    y_stage_1_ref.SaveNpy<T>("y_stage_1.npy");

    for (size_t layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
      // run python original reference
      PytorchReference<T>(layer_idx, scale);
      // load back reference result
      LoadNpy<T>("y_ref.npy", MemoryType::MEMORY_GPU, y_ref);
      LoadNpy<float>("y_stage_1.npy", MemoryType::MEMORY_GPU, y_stage_1_ref);

      int64_t* indices_ptr = reinterpret_cast<int64_t*>(indices.data_ptr);
      T* y_stage_1_ptr = reinterpret_cast<T*>(y_stage_1.data_ptr);
      T* y_ptr = reinterpret_cast<T*>(y.data_ptr);
      T* x_ptr = reinterpret_cast<T*>(x.data_ptr);
      T* w_ptr = reinterpret_cast<T*>(wa_T_all.data_ptr);

      // y_stage_1 = compute input * lora_weight_a
      // y_stage_1: [bs, r]
      // x: [bs, h1]
      // w: [num_loras, num_layers, r, h1]
      InvokeBGMV<T, T, T>(y_stage_1_ptr, x_ptr, w_ptr, indices_ptr, layer_idx, 1.0f, bs, num_layers, h1, r, 0, r,
                          stream);

      w_ptr = reinterpret_cast<T*>(wb_T_all.data_ptr);
      // y = compute y_stage_1 * lora_weight_b
      // y: [bs, h2]
      // y_stage_1: [bs, r]
      // w: [num_loras, num_layers, h2, r]
      InvokeBGMV<T, T, T>(y_ptr, y_stage_1_ptr, w_ptr, indices_ptr, layer_idx, scale, bs, num_layers, r, h2, 0, h2,
                          stream);
      CHECK_NVIDIA_CUDA_ERROR(cudaStreamSynchronize(stream));

      EXPECT_TRUE(CheckResult<T>("bgmv_test_half_h1" + std::to_string(h1) + "_h2_" + std::to_string(h2), y, y_ref,
                                 1e-5f, 1e-5f));
    }
  }
};

TEST_F(LlamaNvidiaBGMVTestSuit, CommonTest) {
  const std::vector<size_t> h1_test_list = {128,   256,   512,   1024,  1280,  2048,  2560,  2752, 3072, 3456,
                                            3584,  4096,  5120,  5504,  5632,  6912,  7168,  8192, 9216, 10240,
                                            11008, 13824, 14336, 32000, 32256, 32512, 32768, 33024};
  const std::vector<size_t> h2_test_list = h1_test_list;
  for (const auto& h1 : h1_test_list) {
    for (const auto& h2 : h2_test_list) {
      TestLoraCorrectness<nv_half>(h1, h2);
    }
    break;
  }
}

}  // namespace test
}  // namespace nvidia
}  // namespace llm_kernels
