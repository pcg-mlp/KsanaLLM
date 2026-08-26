/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

#include "device/nvidia/cuda_utils.h"
#include "kernels/nvidia/triton_wrapper.h"
#include "layers/kda_attention_layer.h"
#include "layers/layer_test_base.h"
#include "layers/layer_workspace_manager.h"
#include "utils/singleton.h"

namespace ksana_llm {
namespace {

constexpr size_t kHeadDim = 128;
constexpr size_t kNumHeads = 12;
constexpr size_t kTokenHeadDim = kNumHeads * kHeadDim;
constexpr size_t kQkvDim = 3 * kTokenHeadDim;

template <typename T>
Tensor UploadTensor(DataType dtype, const std::vector<size_t>& shape, const std::vector<T>& host) {
  Tensor tensor(MemoryLocation::LOCATION_DEVICE, dtype, shape);
  CUDA_CHECK(cudaMemcpy(tensor.GetPtr<void>(), host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
  return tensor;
}

Tensor ZeroTensor(DataType dtype, const std::vector<size_t>& shape) {
  Tensor tensor(MemoryLocation::LOCATION_DEVICE, dtype, shape);
  CUDA_CHECK(cudaMemset(tensor.GetPtr<void>(), 0, tensor.GetTotalBytes()));
  return tensor;
}

Tensor MakeCurrentTokenConvWeight() {
  std::vector<float> weight(kQkvDim * 4, 0.0f);
  for (size_t channel = 0; channel < kQkvDim; ++channel) {
    weight[channel * 4 + 3] = 1.0f;
  }
  return UploadTensor(DataType::TYPE_FP32, {kQkvDim, 4}, weight);
}

KdaAttentionInputs MakeCommonInputs(Tensor& qkv, Tensor& forget_gate, Tensor& beta,
                                    Tensor& output_gate, Tensor& conv_weight, Tensor& a_log,
                                    Tensor& dt_bias, Tensor& norm_weight, Tensor& conv_state,
                                    Tensor& ssm_state, Tensor& state_indices) {
  KdaAttentionInputs inputs;
  inputs.qkv = &qkv;
  inputs.forget_gate = &forget_gate;
  inputs.beta = &beta;
  inputs.output_gate = &output_gate;
  inputs.conv_weight = &conv_weight;
  inputs.a_log = &a_log;
  inputs.dt_bias = &dt_bias;
  inputs.norm_weight = &norm_weight;
  inputs.conv_state = &conv_state;
  inputs.ssm_state = &ssm_state;
  inputs.state_indices = &state_indices;
  return inputs;
}

}  // namespace

class KdaAttentionLayerTest : public LayerTest {
 protected:
  void SetUp() override {
    LayerTest::SetUp();

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    int major = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
    if (major < 9) {
      GTEST_SKIP() << "KDA AOT cubins target SM90";
    }

    const std::string kernels_dir = Singleton<TritonWrapper>::GetInstance()->GetKernelsDir();
    const std::string canary = kernels_dir +
                               "/kda/fused_recurrent_kda_packed_decode_kernel_H_12_BK_128_BV_32_LB_0_SSM_fp32";
    if (!std::filesystem::exists(canary + ".json") || !std::filesystem::exists(canary + ".cubin")) {
      GTEST_SKIP() << "KDA AOT artifacts not found under " << kernels_dir;
    }

    runtime_config.inter_data_type = DataType::TYPE_BF16;
    runtime_config.max_step_token_num = 64;
    runtime_config.max_batch_size = 2;
    ASSERT_TRUE(layer_.Init({KdaAttentionLayerParameters{
                                 .num_heads = static_cast<uint32_t>(kNumHeads),
                                 .head_dim = static_cast<uint32_t>(kHeadDim)}},
                            runtime_config, context_, 0)
                    .OK());
    workspace_manager_ = std::make_shared<LayerWorkspaceManager>(0);
    ASSERT_TRUE(layer_.SetWorkspaceBuffer(workspace_manager_->GetWorkspace(layer_.GetWorkspaceSize())).OK());
  }

  KdaAttentionLayer layer_;
  std::shared_ptr<LayerWorkspaceManager> workspace_manager_;
};

TEST_F(KdaAttentionLayerTest, PrefillLaunches) {
  constexpr size_t kTokens = 1;
  Tensor qkv = ZeroTensor(DataType::TYPE_BF16, {kTokens, kQkvDim});
  Tensor forget_gate = ZeroTensor(DataType::TYPE_BF16, {kTokens, kNumHeads, kHeadDim});
  Tensor beta = ZeroTensor(DataType::TYPE_BF16, {kTokens, kNumHeads});
  Tensor output_gate = ZeroTensor(DataType::TYPE_BF16, {kTokens, kNumHeads, kHeadDim});
  Tensor conv_weight = MakeCurrentTokenConvWeight();
  Tensor a_log = ZeroTensor(DataType::TYPE_FP32, {kNumHeads});
  Tensor dt_bias = ZeroTensor(DataType::TYPE_FP32, {kNumHeads, kHeadDim});
  Tensor norm_weight = ZeroTensor(DataType::TYPE_BF16, {kHeadDim});
  Tensor conv_state = ZeroTensor(DataType::TYPE_BF16, {1, 3, kQkvDim});
  Tensor ssm_state = ZeroTensor(DataType::TYPE_FP32, {1, kNumHeads, kHeadDim, kHeadDim});
  Tensor state_indices = UploadTensor<int32_t>(DataType::TYPE_INT32, {1}, {0});
  Tensor cu_seqlens = UploadTensor<uint64_t>(DataType::TYPE_UINT64, {2}, {0, 1});
  Tensor chunk_indices = UploadTensor<int32_t>(DataType::TYPE_INT32, {1, 2}, {0, 0});
  Tensor chunk_offsets = UploadTensor<int32_t>(DataType::TYPE_INT32, {2}, {0, 1});
  Tensor has_initial_state = UploadTensor<uint8_t>(DataType::TYPE_BOOL, {1}, {0});
  Tensor output = ZeroTensor(DataType::TYPE_BF16, {kTokens, kNumHeads, kHeadDim});

  KdaPrefillInputs inputs;
  inputs.common = MakeCommonInputs(qkv, forget_gate, beta, output_gate, conv_weight, a_log,
                                   dt_bias, norm_weight, conv_state, ssm_state, state_indices);
  inputs.max_seq_len = 1;
  inputs.cu_seqlens = &cu_seqlens;
  inputs.chunk_indices = &chunk_indices;
  inputs.chunk_offsets = &chunk_offsets;
  inputs.has_initial_state = &has_initial_state;

  ASSERT_TRUE(layer_.ForwardPrefill(inputs, &output).OK());
  EXPECT_EQ(cudaStreamSynchronize(context_->device->GetPrimaryComputeStream().Get()), cudaSuccess);
  EXPECT_EQ(output.shape, (std::vector<size_t>{kTokens, kNumHeads, kHeadDim}));
}

TEST_F(KdaAttentionLayerTest, DecodeLaunches) {
  constexpr size_t kBatch = 2;
  Tensor qkv = ZeroTensor(DataType::TYPE_BF16, {kBatch, kQkvDim});
  Tensor forget_gate = ZeroTensor(DataType::TYPE_BF16, {kBatch, kNumHeads, kHeadDim});
  Tensor beta = ZeroTensor(DataType::TYPE_BF16, {kBatch, kNumHeads});
  Tensor output_gate = ZeroTensor(DataType::TYPE_BF16, {kBatch, kNumHeads, kHeadDim});
  Tensor conv_weight = MakeCurrentTokenConvWeight();
  Tensor a_log = ZeroTensor(DataType::TYPE_FP32, {kNumHeads});
  Tensor dt_bias = ZeroTensor(DataType::TYPE_FP32, {kNumHeads, kHeadDim});
  Tensor norm_weight = ZeroTensor(DataType::TYPE_BF16, {kHeadDim});
  Tensor conv_state = ZeroTensor(DataType::TYPE_BF16, {kBatch, 3, kQkvDim});
  Tensor ssm_state = ZeroTensor(DataType::TYPE_FP32, {kBatch, kNumHeads, kHeadDim, kHeadDim});
  Tensor state_indices = UploadTensor<int32_t>(DataType::TYPE_INT32, {kBatch}, {0, 1});
  Tensor output = ZeroTensor(DataType::TYPE_BF16, {kBatch, kNumHeads, kHeadDim});

  KdaDecodeInputs inputs;
  inputs.common = MakeCommonInputs(qkv, forget_gate, beta, output_gate, conv_weight, a_log,
                                   dt_bias, norm_weight, conv_state, ssm_state, state_indices);

  ASSERT_TRUE(layer_.ForwardDecode(inputs, &output).OK());
  EXPECT_EQ(cudaStreamSynchronize(context_->device->GetPrimaryComputeStream().Get()), cudaSuccess);
  EXPECT_EQ(output.shape, (std::vector<size_t>{kBatch, kNumHeads, kHeadDim}));
}

}  // namespace ksana_llm
