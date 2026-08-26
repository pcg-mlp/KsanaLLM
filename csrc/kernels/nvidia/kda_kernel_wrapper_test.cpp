/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "test.h"

#include "device/nvidia/cuda_utils.h"
#include "kernels/nvidia/kda_kernel_wrapper.h"
#include "kernels/nvidia/triton_wrapper.h"
#include "utils/singleton.h"

namespace ksana_llm {
namespace {

constexpr int kHeadDim = 128;
constexpr int kNumHeads = 12;
constexpr int kNumTokens = 128;
constexpr int kNumChunks = 2;
constexpr size_t kBf16Bytes = 2;
// The recurrent state and its verify checkpoints are FP32; the BF16 pool is
// exercised through its own artifact in RecurrentDecodeBf16StateLaunches.
constexpr size_t kSsmBytes = sizeof(float);
constexpr size_t kTokenHeadDim = static_cast<size_t>(kNumHeads) * kHeadDim;
constexpr size_t kQkvDim = 3 * kTokenHeadDim;
constexpr size_t kTokenVectorElements = static_cast<size_t>(kNumTokens) * kTokenHeadDim;
constexpr size_t kTokenHeads = static_cast<size_t>(kNumTokens) * kNumHeads;
constexpr size_t kStateElements = static_cast<size_t>(kNumHeads) * kHeadDim * kHeadDim;

class DeviceBuffer {
 public:
  explicit DeviceBuffer(size_t bytes) : bytes_(bytes) {
    CUDA_CHECK(cudaMalloc(&data_, bytes_));
    CUDA_CHECK(cudaMemset(data_, 0, bytes_));
  }

  ~DeviceBuffer() {
    if (data_ != nullptr) cudaFree(data_);
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  void* data() const { return data_; }

  // Typed view of the allocation, matching the wrapper's parameter types.
  template <typename T>
  T* As() const {
    return static_cast<T*>(data_);
  }

  template <typename T>
  void Upload(const std::vector<T>& host) {
    ASSERT_LE(host.size() * sizeof(T), bytes_);
    CUDA_CHECK(cudaMemcpy(data_, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
  }

 private:
  void* data_{nullptr};
  size_t bytes_{0};
};

}  // namespace

class KdaKernelWrapperTest : public testing::Test {
 protected:
  void SetUp() override {
    CUDA_CHECK(cudaStreamCreate(&stream_));
    int device = 0;
    int major = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
    if (major < 9) {
      GTEST_SKIP() << "KDA AOT cubins target SM90; skipping on compute capability " << major;
    }
    const std::string dir = Singleton<TritonWrapper>::GetInstance()->GetKernelsDir();
    const std::string canary = dir + "/kda/l2norm_fwd_kernel_D_128_BT_16";
    if (!std::filesystem::exists(canary + ".json") || !std::filesystem::exists(canary + ".cubin")) {
      GTEST_SKIP() << "KDA AOT artifacts not found under " << dir;
    }
  }

  void TearDown() override {
    if (stream_ != nullptr) {
      cudaStreamSynchronize(stream_);
      cudaStreamDestroy(stream_);
    }
  }

  void ExpectLaunchOk(const std::function<void()>& launch) {
    launch();
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
    EXPECT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
  }

  cudaStream_t stream_{nullptr};
};

TEST_F(KdaKernelWrapperTest, ChunkKdaSmallGridLaunches) {
  DeviceBuffer q(kTokenVectorElements * kBf16Bytes);
  DeviceBuffer k(kTokenVectorElements * kBf16Bytes);
  DeviceBuffer v(kTokenVectorElements * kBf16Bytes);
  DeviceBuffer raw_gate(kTokenVectorElements * kBf16Bytes);
  DeviceBuffer beta(kTokenHeads * sizeof(float));
  DeviceBuffer a_log(kNumHeads * sizeof(float));
  DeviceBuffer dt_bias(kTokenHeadDim * sizeof(float));
  DeviceBuffer cu_seqlens(2 * sizeof(int32_t));
  DeviceBuffer chunk_indices(kNumChunks * 2 * sizeof(int32_t));
  DeviceBuffer chunk_offsets(2 * sizeof(int32_t));
  DeviceBuffer state_indices(sizeof(int32_t));
  DeviceBuffer state(kStateElements * kSsmBytes);
  DeviceBuffer output(kTokenVectorElements * kBf16Bytes);
  cu_seqlens.Upload<int32_t>({0, kNumTokens});
  chunk_indices.Upload<int32_t>({0, 0, 0, 1});
  chunk_offsets.Upload<int32_t>({0, kNumChunks});
  state_indices.Upload<int32_t>({0});

  DeviceBuffer workspace(GetKdaWorkspaceSize(kNumTokens, kNumHeads, kNumChunks));

  KdaChunkParams params;
  params.q = q.As<__nv_bfloat16>();
  params.k = k.As<__nv_bfloat16>();
  params.v = v.As<__nv_bfloat16>();
  params.forget_gate = raw_gate.As<__nv_bfloat16>();
  params.beta = beta.As<float>();
  params.a_log = a_log.As<float>();
  params.dt_bias = dt_bias.As<float>();
  params.cu_seqlens = cu_seqlens.As<int32_t>();
  params.chunk_indices = chunk_indices.As<int32_t>();
  params.chunk_offsets = chunk_offsets.As<int32_t>();
  params.state = state.data();
  params.state_indices = state_indices.As<int32_t>();
  params.output = output.As<__nv_bfloat16>();
  params.scratch = BuildKdaWorkspaceLayout(workspace.data(), kNumTokens, kNumHeads, kNumChunks);
  params.num_tokens = kNumTokens;
  params.num_sequences = 1;
  params.num_chunks = kNumChunks;
  params.num_heads = kNumHeads;

  ExpectLaunchOk([&] { InvokeChunkKda(params, stream_); });
}

TEST_F(KdaKernelWrapperTest, ChunkKdaLargeGridLaunches) {
  constexpr int kSequences = 22;
  constexpr int kTokens = 22;
  const size_t token_vector_elements = static_cast<size_t>(kTokens) * kTokenHeadDim;
  const size_t token_heads = static_cast<size_t>(kTokens) * kNumHeads;
  DeviceBuffer q(token_vector_elements * kBf16Bytes);
  DeviceBuffer k(token_vector_elements * kBf16Bytes);
  DeviceBuffer v(token_vector_elements * kBf16Bytes);
  DeviceBuffer raw_gate(token_vector_elements * kBf16Bytes);
  DeviceBuffer beta(token_heads * sizeof(float));
  DeviceBuffer a_log(kNumHeads * sizeof(float));
  DeviceBuffer dt_bias(kTokenHeadDim * sizeof(float));
  DeviceBuffer cu_seqlens((kSequences + 1) * sizeof(int32_t));
  DeviceBuffer chunk_indices(kSequences * 2 * sizeof(int32_t));
  DeviceBuffer chunk_offsets((kSequences + 1) * sizeof(int32_t));
  DeviceBuffer state_indices(kSequences * sizeof(int32_t));
  DeviceBuffer state(static_cast<size_t>(kSequences) * kStateElements * kSsmBytes);
  DeviceBuffer output(token_vector_elements * kBf16Bytes);

  std::vector<int32_t> cu(kSequences + 1);
  std::vector<int32_t> chunks(kSequences * 2);
  std::vector<int32_t> offsets(kSequences + 1);
  std::vector<int32_t> indices(kSequences);
  for (int i = 0; i <= kSequences; ++i) {
    cu[i] = i;
    offsets[i] = i;
    if (i < kSequences) {
      chunks[2 * i] = i;
      chunks[2 * i + 1] = 0;
      indices[i] = i;
    }
  }
  cu_seqlens.Upload(cu);
  chunk_indices.Upload(chunks);
  chunk_offsets.Upload(offsets);
  state_indices.Upload(indices);

  DeviceBuffer workspace(GetKdaWorkspaceSize(kTokens, kNumHeads, kSequences));

  KdaChunkParams params;
  params.q = q.As<__nv_bfloat16>();
  params.k = k.As<__nv_bfloat16>();
  params.v = v.As<__nv_bfloat16>();
  params.forget_gate = raw_gate.As<__nv_bfloat16>();
  params.beta = beta.As<float>();
  params.a_log = a_log.As<float>();
  params.dt_bias = dt_bias.As<float>();
  params.cu_seqlens = cu_seqlens.As<int32_t>();
  params.chunk_indices = chunk_indices.As<int32_t>();
  params.chunk_offsets = chunk_offsets.As<int32_t>();
  params.state = state.data();
  params.state_indices = state_indices.As<int32_t>();
  params.output = output.As<__nv_bfloat16>();
  params.scratch = BuildKdaWorkspaceLayout(workspace.data(), kTokens, kNumHeads, kSequences);
  params.num_tokens = kTokens;
  params.num_sequences = kSequences;
  params.num_chunks = kSequences;
  params.num_heads = kNumHeads;

  ExpectLaunchOk([&] { InvokeChunkKda(params, stream_); });
}

TEST_F(KdaKernelWrapperTest, PackedDecodeLaunches) {
  constexpr int kBatch = 2;
  DeviceBuffer qkv(static_cast<size_t>(kBatch) * kQkvDim * kBf16Bytes);
  DeviceBuffer a(static_cast<size_t>(kBatch) * kTokenHeadDim * kBf16Bytes);
  DeviceBuffer b(static_cast<size_t>(kBatch) * kNumHeads * kBf16Bytes);
  DeviceBuffer a_log(kNumHeads * sizeof(float));
  DeviceBuffer dt_bias(kTokenHeadDim * sizeof(float));
  DeviceBuffer output(static_cast<size_t>(kBatch) * kTokenHeadDim * kBf16Bytes);
  DeviceBuffer state(static_cast<size_t>(kBatch) * kStateElements * kSsmBytes);
  DeviceBuffer indices(kBatch * sizeof(int32_t));
  indices.Upload<int32_t>({0, 1});

  KdaPackedDecodeParams params;
  params.mixed_qkv = qkv.As<__nv_bfloat16>();
  params.forget_gate = a.As<__nv_bfloat16>();
  params.beta = b.As<__nv_bfloat16>();
  params.a_log = a_log.As<float>();
  params.dt_bias = dt_bias.As<float>();
  params.output = output.As<__nv_bfloat16>();
  params.state = state.data();
  params.state_indices = indices.As<int32_t>();
  params.batch_size = kBatch;
  params.num_heads = kNumHeads;

  ExpectLaunchOk([&] { InvokeFusedRecurrentKdaPackedDecode(params, stream_); });
}

TEST_F(KdaKernelWrapperTest, RecurrentDecodeAndVerifyLaunch) {
  constexpr int kTokens = 2;
  DeviceBuffer q(kTokens * kTokenHeadDim * kBf16Bytes);
  DeviceBuffer k(kTokens * kTokenHeadDim * kBf16Bytes);
  DeviceBuffer v(kTokens * kTokenHeadDim * kBf16Bytes);
  DeviceBuffer a(kTokens * kTokenHeadDim * kBf16Bytes);
  DeviceBuffer b(kTokens * kNumHeads * kBf16Bytes);
  DeviceBuffer a_log(kNumHeads * sizeof(float));
  DeviceBuffer dt_bias(kTokenHeadDim * sizeof(float));
  DeviceBuffer output(kTokens * kTokenHeadDim * kBf16Bytes);
  DeviceBuffer state(kStateElements * kSsmBytes);
  DeviceBuffer state_indices(sizeof(int32_t));
  DeviceBuffer cu_seqlens(2 * sizeof(int32_t));
  DeviceBuffer intermediate_indices(sizeof(int32_t));
  DeviceBuffer intermediate(static_cast<size_t>(kTokens) * kStateElements * kSsmBytes);
  DeviceBuffer parents(kTokens * sizeof(int32_t));
  state_indices.Upload<int32_t>({0});
  cu_seqlens.Upload<int32_t>({0, kTokens});
  intermediate_indices.Upload<int32_t>({0});

  auto launch = [&](KdaForwardMode mode, void* intermediate_states, const int32_t* intermediate_state_indices,
                    const int32_t* retrieve_parent_token, int cache_steps) {
    KdaRecurrentUpdateParams params;
    params.mode = mode;
    params.a_log = a_log.As<float>();
    params.forget_gate = a.As<__nv_bfloat16>();
    params.dt_bias = dt_bias.As<float>();
    params.q = q.As<__nv_bfloat16>();
    params.k = k.As<__nv_bfloat16>();
    params.v = v.As<__nv_bfloat16>();
    params.beta = b.As<__nv_bfloat16>();
    params.output = output.As<__nv_bfloat16>();
    params.state = state.data();
    params.state_indices = state_indices.As<int32_t>();
    params.cu_seqlens = cu_seqlens.As<int32_t>();
    params.intermediate_states = intermediate_states;
    params.intermediate_state_indices = intermediate_state_indices;
    params.retrieve_parent_token = retrieve_parent_token;
    params.cache_steps = cache_steps;
    params.tree_metadata_stride = kTokens;
    params.qkv_token_stride = static_cast<int>(kTokenHeadDim);
    params.num_tokens = kTokens;
    params.num_sequences = 1;
    params.num_heads = kNumHeads;
    InvokeFusedSigmoidGatingDeltaRuleUpdate(params, stream_);
  };
  ExpectLaunchOk([&] { launch(KdaForwardMode::kDecode, nullptr, nullptr, nullptr, 0); });
  ExpectLaunchOk([&] {
    launch(KdaForwardMode::kTargetVerifyChain, intermediate.data(), intermediate_indices.As<int32_t>(), nullptr,
           kTokens);
  });
  ExpectLaunchOk([&] {
    launch(KdaForwardMode::kTargetVerifyTree, intermediate.data(), intermediate_indices.As<int32_t>(),
           parents.As<int32_t>(), kTokens);
  });
}

// Same launch through the BF16 state artifact, which a build narrowed with
// KLLM_KDA_SSM_DTYPE may not contain.
TEST_F(KdaKernelWrapperTest, RecurrentDecodeBf16StateLaunches) {
  const std::string dir = Singleton<TritonWrapper>::GetInstance()->GetKernelsDir();
  const std::string canary =
      dir + "/kda/fused_sigmoid_gating_delta_rule_update_kernel_H_12_MODE_decode_NP2_1_BK_128_BV_32_LB_0_SSM_bf16";
  if (!std::filesystem::exists(canary + ".json") || !std::filesystem::exists(canary + ".cubin")) {
    GTEST_SKIP() << "BF16 recurrent-state artifacts were not generated";
  }

  constexpr int kTokens = 2;
  DeviceBuffer q(kTokens * kTokenHeadDim * kBf16Bytes);
  DeviceBuffer k(kTokens * kTokenHeadDim * kBf16Bytes);
  DeviceBuffer v(kTokens * kTokenHeadDim * kBf16Bytes);
  DeviceBuffer a(kTokens * kTokenHeadDim * kBf16Bytes);
  DeviceBuffer b(kTokens * kNumHeads * kBf16Bytes);
  DeviceBuffer a_log(kNumHeads * sizeof(float));
  DeviceBuffer dt_bias(kTokenHeadDim * sizeof(float));
  DeviceBuffer output(kTokens * kTokenHeadDim * kBf16Bytes);
  DeviceBuffer state(kStateElements * kBf16Bytes);
  DeviceBuffer state_indices(sizeof(int32_t));
  DeviceBuffer cu_seqlens(2 * sizeof(int32_t));
  state_indices.Upload<int32_t>({0});
  cu_seqlens.Upload<int32_t>({0, kTokens});

  KdaRecurrentUpdateParams params;
  params.mode = KdaForwardMode::kDecode;
  params.a_log = a_log.As<float>();
  params.forget_gate = a.As<__nv_bfloat16>();
  params.dt_bias = dt_bias.As<float>();
  params.q = q.As<__nv_bfloat16>();
  params.k = k.As<__nv_bfloat16>();
  params.v = v.As<__nv_bfloat16>();
  params.beta = b.As<__nv_bfloat16>();
  params.output = output.As<__nv_bfloat16>();
  params.state = state.data();
  params.ssm_dtype = DataType::TYPE_BF16;
  params.state_indices = state_indices.As<int32_t>();
  params.cu_seqlens = cu_seqlens.As<int32_t>();
  params.qkv_token_stride = static_cast<int>(kTokenHeadDim);
  params.num_tokens = kTokens;
  params.num_sequences = 1;
  params.num_heads = kNumHeads;

  ExpectLaunchOk([&] { InvokeFusedSigmoidGatingDeltaRuleUpdate(params, stream_); });
}

TEST_F(KdaKernelWrapperTest, CausalConvAndGatedNormLaunch) {
  constexpr int kTokens = 2;
  DeviceBuffer x(kTokens * kTokenHeadDim * kBf16Bytes);
  DeviceBuffer qkv(kTokens * kQkvDim * kBf16Bytes);
  DeviceBuffer weight(kQkvDim * 4 * sizeof(float));
  DeviceBuffer conv_state(9 * kTokenHeadDim * kBf16Bytes);
  DeviceBuffer indices(sizeof(int32_t));
  DeviceBuffer has_initial(sizeof(uint8_t));
  DeviceBuffer cu_seqlens(2 * sizeof(int32_t));
  DeviceBuffer output(kTokens * kQkvDim * kBf16Bytes);
  DeviceBuffer cache_seqlens(sizeof(int32_t));
  indices.Upload<int32_t>({0});
  has_initial.Upload<uint8_t>({0});
  cu_seqlens.Upload<int32_t>({0, kTokens});
  cache_seqlens.Upload<int32_t>({0});

  KdaCausalConv1dParams prefill_conv;
  prefill_conv.x = x.As<__nv_bfloat16>();
  prefill_conv.weight = weight.As<float>();
  prefill_conv.conv_state = conv_state.As<__nv_bfloat16>();
  prefill_conv.state_indices = indices.As<int32_t>();
  prefill_conv.has_initial_state = has_initial.As<bool>();
  prefill_conv.cu_seqlens = cu_seqlens.As<int32_t>();
  prefill_conv.output = output.As<__nv_bfloat16>();
  prefill_conv.num_cache_lines = 1;
  prefill_conv.num_sequences = 1;
  prefill_conv.max_seq_len = kTokens;
  prefill_conv.num_channels = static_cast<int>(kTokenHeadDim);
  prefill_conv.x_strides = {0, 1, static_cast<int>(kTokenHeadDim)};
  prefill_conv.state_strides = {kKdaConvStateLen * static_cast<int>(kTokenHeadDim), 1,
                                static_cast<int>(kTokenHeadDim)};
  prefill_conv.output_strides = {0, 1, static_cast<int>(kTokenHeadDim)};
  ExpectLaunchOk([&] { InvokeCausalConv1dFn(prefill_conv, stream_); });

  KdaCausalConv1dUpdateParams decode_conv;
  decode_conv.mode = KdaForwardMode::kDecode;
  decode_conv.x = qkv.As<__nv_bfloat16>();
  decode_conv.weight = weight.As<float>();
  decode_conv.conv_state = conv_state.As<__nv_bfloat16>();
  decode_conv.state_indices = indices.As<int32_t>();
  decode_conv.output = output.As<__nv_bfloat16>();
  decode_conv.batch_size = 1;
  decode_conv.seqlen = 1;
  decode_conv.num_cache_lines = 1;
  decode_conv.num_channels = static_cast<int>(kQkvDim);
  decode_conv.cache_seqlens = cache_seqlens.As<int32_t>();
  decode_conv.x_strides = {static_cast<int>(kQkvDim), 1, static_cast<int>(kQkvDim)};
  decode_conv.state_strides = {kKdaConvStateLen * static_cast<int>(kQkvDim), 1, static_cast<int>(kQkvDim)};
  decode_conv.output_strides = decode_conv.x_strides;
  ExpectLaunchOk([&] { InvokeCausalConv1dUpdate(decode_conv, stream_); });

  DeviceBuffer gate(kTokenHeadDim * kBf16Bytes);
  DeviceBuffer norm_weight(kHeadDim * kBf16Bytes);
  DeviceBuffer norm_output(kTokenHeadDim * kBf16Bytes);
  DeviceBuffer rstd(kNumHeads * sizeof(float));

  KdaGatedRmsNormParams norm;
  norm.x = x.As<__nv_bfloat16>();
  norm.gate = gate.As<__nv_bfloat16>();
  norm.weight = norm_weight.As<__nv_bfloat16>();
  norm.output = norm_output.As<__nv_bfloat16>();
  norm.rstd = rstd.As<float>();
  norm.num_rows = kNumHeads;
  ExpectLaunchOk([&] { InvokeKdaGatedRmsNorm(norm, stream_); });
}

}  // namespace ksana_llm
