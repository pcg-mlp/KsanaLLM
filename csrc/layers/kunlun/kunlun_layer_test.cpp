/* Copyright 2026 Tencent Inc.  All rights reserved.

 * kunlun_layer_test.cpp — 在 XPU 上通过 KsanaLLM 框架真正调用 Layer::Forward()
 *
 * 这是真正的框架级验证：
 *   DeviceContext 初始化 → Tensor 分配 → AddLayer::Init() → AddLayer::Forward() → 验证
 *
 * 编译：
 *   cd /data/KsanaLLM/build
 *   cmake .. -DWITH_KUNLUN=ON -DWITH_TESTING=ON
 *   make kunlun_layer_test
 *
 * 运行：
 *   ./bin/kunlun_layer_test

==============================================================================*/

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

#include "device/common_device.h"
#include "device/device_types.h"
#include "device/kunlun/kunlun_device_context_impl.h"
#include "kernels/argmax.h"
#include "kernels/kunlun/basic_kernel_wrapper.h"
#include "layers/activation_layer.h"
#include "layers/add_layer.h"
#include "layers/add_norm_layer.h"
#include "layers/assemble_tokens_hidden_layer.h"
#include "layers/cast_layer.h"
#include "layers/emb_lookup_layer.h"
#include "layers/layernorm_layer.h"
#include "layers/matmul_layer.h"
#include "layers/mul_layer.h"
#include "layers/silu_mul_layer.h"
#include "layers/split_layer.h"
#include "test.h"
#include "utils/logger.h"

namespace ksana_llm {

// Test helper: expose protected members for direct setup (bypass Init which needs GlobalContext)
class TestableAddLayer : public AddLayer {
 public:
  void SetupForTest(std::shared_ptr<Context> ctx, DataType dtype) {
    context_ = ctx;
    rank_ = 0;
    inter_data_type_ = dtype;
    tp_size_ = 1;
    dp_size_ = 1;
    attn_dp_atp_size_ = 1;
    attn_dp_group_id_ = 0;
    attn_dp_rank_id_ = 0;
  }
};

// Testable helpers for layers whose Init() requires GlobalContext / parameters.
class TestableMatMulLayer : public MatMulLayer {
 public:
  void SetupForTest(std::shared_ptr<Context> ctx, DataType dtype) {
    context_ = ctx;
    rank_ = 0;
    inter_data_type_ = dtype;
  }
};

class TestableSiluMulLayer : public SiluMulLayer {
 public:
  void SetupForTest(std::shared_ptr<Context> ctx, DataType dtype) {
    context_ = ctx;
    rank_ = 0;
    inter_data_type_ = dtype;
  }
};

class TestableLayernormLayer : public LayernormLayer {
 public:
  void SetupForTest(std::shared_ptr<Context> ctx, DataType dtype, float eps) {
    context_ = ctx;
    rank_ = 0;
    inter_data_type_ = dtype;
    rms_norm_eps_ = eps;
  }
};

class TestableActivationLayer : public ActivationLayer {
 public:
  void SetupForTest(std::shared_ptr<Context> ctx, DataType dtype, ActivationType act_type) {
    context_ = ctx;
    rank_ = 0;
    inter_data_type_ = dtype;
    activation_type_ = act_type;
  }
};

class TestableSigmoidLayer : public SigmoidLayer {
 public:
  void SetupForTest(std::shared_ptr<Context> ctx, DataType dtype) {
    context_ = ctx;
    rank_ = 0;
    inter_data_type_ = dtype;
  }
};

class TestableMulLayer : public MulLayer {
 public:
  void SetupForTest(std::shared_ptr<Context> ctx, DataType dtype) {
    context_ = ctx;
    rank_ = 0;
    inter_data_type_ = dtype;
  }
};

class TestableCastLayer : public CastLayer {
 public:
  void SetupForTest(std::shared_ptr<Context> ctx, DataType dtype) {
    context_ = ctx;
    rank_ = 0;
    inter_data_type_ = dtype;
  }
};

class TestableAddNormLayer : public AddNormLayer {
 public:
  void SetupForTest(std::shared_ptr<Context> ctx, DataType dtype, float eps) {
    context_ = ctx;
    rank_ = 0;
    inter_data_type_ = dtype;
    rms_norm_eps_ = eps;
  }
};

class TestableSplitLayer : public SplitLayer {
 public:
  void SetupForTest(std::shared_ptr<Context> ctx, DataType dtype) {
    context_ = ctx;
    rank_ = 0;
    inter_data_type_ = dtype;
  }
};

class TestableAssembleTokensHiddenLayer : public AssembleTokensHiddenLayer {
 public:
  void SetupForTest(std::shared_ptr<Context> ctx, DataType dtype) {
    context_ = ctx;
    rank_ = 0;
    inter_data_type_ = dtype;
  }
};

class TestableEmbLookupLayer : public EmbLookupLayer {
 public:
  void SetupForTest(std::shared_ptr<Context> ctx, DataType dtype, size_t max_token_num = 4096) {
    context_ = ctx;
    rank_ = 0;
    inter_data_type_ = dtype;
    workspace_buffer_ = std::make_shared<Tensor>();
    workspace_buffer_->ReallocateMemory(MemoryLocation::LOCATION_DEVICE, DataType::TYPE_INT64, {max_token_num});
  }
};

// Minimal Context setup for single-card XPU test (no Environment/GlobalContext needed)
class KunlunLayerTest : public testing::Test {
 protected:
  void SetUp() override {
    // Create Context with minimal GlobalContext (tp=1)
    context_ = std::make_shared<Context>();

    // We need a minimal GlobalContext. Use a simple mock approach:
    // GlobalContext needs Environment, but we can work around it by
    // just creating the DeviceContext (which manages streams).
    context_->device = std::make_shared<DeviceContext>(1);

    // Set runtime config
    runtime_config.parallel_basic_config.tensor_parallel_size = 1;
    runtime_config.inter_data_type = TYPE_FP32;
  }

  void TearDown() override {}

 protected:
  RuntimeConfig runtime_config;
  std::shared_ptr<Context> context_{nullptr};
};

TEST_F(KunlunLayerTest, AddLayerTest) {
  constexpr int kDeviceRank = 0;

  // 创建 Tensor（框架自动在 XPU 设备上分配内存）
  Tensor input_a = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {4, 256});
  Tensor input_b = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {4, 256});
  std::vector<Tensor> output(1);
  output[0] = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {4, 256});

  // 准备 host 数据
  size_t num_elements = 4 * 256;
  std::vector<float> h_a(num_elements), h_b(num_elements);
  for (size_t i = 0; i < num_elements; ++i) {
    h_a[i] = static_cast<float>(i) * 0.01f;
    h_b[i] = 1.0f;
  }

  // H2D 拷贝（通过框架的 MemcpyAsync）
  MemcpyAsync(input_a.GetPtr<void>(), h_a.data(), num_elements * sizeof(float), MEMCPY_HOST_TO_DEVICE,
              context_->device->GetMemoryManageStream());
  MemcpyAsync(input_b.GetPtr<void>(), h_b.data(), num_elements * sizeof(float), MEMCPY_HOST_TO_DEVICE,
              context_->device->GetMemoryManageStream());
  StreamSynchronize(context_->device->GetMemoryManageStream());

  // 通过框架调用 AddLayer::Forward()
  // 使用 TestableAddLayer 绕过 Init()（它需要 GlobalContext）
  TestableAddLayer add_layer;
  add_layer.SetupForTest(context_, TYPE_FP32);
  add_layer.Forward(&input_a, &input_b, &output[0]);
  StreamSynchronize(context_->device->GetPrimaryComputeStream());

  // D2H 拷贝验证
  std::vector<float> h_out(num_elements);
  Memcpy(h_out.data(), output[0].GetPtr<void>(), num_elements * sizeof(float), MEMCPY_DEVICE_TO_HOST);

  // 验证
  float max_diff = 0.0f;
  for (size_t i = 0; i < num_elements; ++i) {
    float expected = h_a[i] + h_b[i];
    float diff = std::fabs(h_out[i] - expected);
    if (diff > max_diff) max_diff = diff;
  }
  KLLM_LOG_INFO << "AddLayer::Forward() on XPU: max_diff = " << max_diff;
  EXPECT_LT(max_diff, 1e-6);
}

// ── ArgMax precision test ─────────────────────────────────────────────────────
// ArgMax takes Stream& from the framework (not raw XPUStream).
// Construct input with known per-row argmax, verify output matches exactly.
TEST_F(KunlunLayerTest, ArgMaxTest) {
  constexpr int kBatch = 4;
  constexpr int kVocab = 16;

  // Expected argmax per row: spread across range to catch off-by-one bugs.
  const std::vector<int> expected_idx = {0, 5, 10, 15};

  std::vector<float> h_input(kBatch * kVocab, -1.0f);
  for (int b = 0; b < kBatch; ++b) {
    h_input[b * kVocab + expected_idx[b]] = 1.0f;
  }

  Tensor input_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32,
                      {static_cast<size_t>(kBatch), static_cast<size_t>(kVocab)});
  Tensor result_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_UINT32, {static_cast<size_t>(kBatch)});

  Stream& stream = context_->device->GetPrimaryComputeStream();

  MemcpyAsync(input_tensor.GetPtr<void>(), h_input.data(), h_input.size() * sizeof(float), MEMCPY_HOST_TO_DEVICE,
              context_->device->GetMemoryManageStream());
  StreamSynchronize(context_->device->GetMemoryManageStream());

  // ArgMax on Kunlun requires xpukernel::Context* (not xdnn::Context*) and a pre-allocated int64 buffer.
  void* raw_xpuctx = context_->device->ext->GetXpuKernelContext(stream.Get());
  Tensor buffer_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT64, {static_cast<size_t>(kBatch)});

  Status status = ArgMax<float>(input_tensor.GetPtr<float>(), kBatch, kVocab, result_tensor.GetPtr<uint32_t>(), stream,
                                raw_xpuctx, buffer_tensor.GetPtr<void>());
  ASSERT_TRUE(status.OK()) << "ArgMax failed: " << status.ToString();
  StreamSynchronize(stream);

  std::vector<uint32_t> h_result(kBatch);
  Memcpy(h_result.data(), result_tensor.GetPtr<void>(), kBatch * sizeof(uint32_t), MEMCPY_DEVICE_TO_HOST);

  for (int b = 0; b < kBatch; ++b) {
    EXPECT_EQ(static_cast<int>(h_result[b]), expected_idx[b]) << "ArgMax mismatch at batch=" << b;
  }
  KLLM_LOG_INFO << "ArgMax: batch=" << kBatch << " vocab=" << kVocab << " — all indices match";
}

// ── MatMulLayer test ──────────────────────────────────────────────────────────
// Validates y[m,n] = x[m,k] × w^T[n,k] (w stored as [n,k], w_trans=true)
TEST_F(KunlunLayerTest, MatMulLayerTest) {
  constexpr int64_t M = 8;
  constexpr int64_t K = 16;
  constexpr int64_t N = 12;

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

  std::vector<float> h_x(M * K), h_w(N * K), h_ref(M * N, 0.0f);
  for (auto& v : h_x) v = dist(rng);
  for (auto& v : h_w) v = dist(rng);

  // CPU reference: y[i,j] = sum_l x[i,l] * w[j,l]
  for (int64_t i = 0; i < M; ++i)
    for (int64_t j = 0; j < N; ++j) {
      float s = 0.0f;
      for (int64_t l = 0; l < K; ++l) s += h_x[i * K + l] * h_w[j * K + l];
      h_ref[i * N + j] = s;
    }

  Tensor x_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {M, K});
  Tensor w_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {N, K});
  Tensor y_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {M, N});

  MemcpyAsync(x_tensor.GetPtr<void>(), h_x.data(), h_x.size() * sizeof(float), MEMCPY_HOST_TO_DEVICE,
              context_->device->GetMemoryManageStream());
  MemcpyAsync(w_tensor.GetPtr<void>(), h_w.data(), h_w.size() * sizeof(float), MEMCPY_HOST_TO_DEVICE,
              context_->device->GetMemoryManageStream());
  StreamSynchronize(context_->device->GetMemoryManageStream());

  TestableMatMulLayer layer;
  layer.SetupForTest(context_, TYPE_FP32);
  ASSERT_TRUE(layer.Forward(&x_tensor, &w_tensor, &y_tensor).OK());
  StreamSynchronize(context_->device->GetPrimaryComputeStream());

  std::vector<float> h_out(M * N);
  Memcpy(h_out.data(), y_tensor.GetPtr<void>(), h_out.size() * sizeof(float), MEMCPY_DEVICE_TO_HOST);

  float max_diff = 0.0f;
  for (size_t i = 0; i < h_out.size(); ++i) max_diff = std::max(max_diff, std::fabs(h_out[i] - h_ref[i]));
  KLLM_LOG_INFO << "MatMulLayer: M=" << M << " K=" << K << " N=" << N << " max_diff=" << max_diff;
  EXPECT_LT(max_diff, 1e-4);
}

// ── SiluMulLayer (two-input mode) test ────────────────────────────────────────
// output = silu(input) * gate
TEST_F(KunlunLayerTest, SiluMulLayerTwoInputTest) {
  constexpr int64_t M = 4;
  constexpr int64_t N = 32;

  std::mt19937 rng(7);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

  std::vector<float> h_in(M * N), h_gate(M * N), h_ref(M * N);
  for (auto& v : h_in) v = dist(rng);
  for (auto& v : h_gate) v = dist(rng);

  for (size_t i = 0; i < h_in.size(); ++i) {
    float silu = h_in[i] / (1.0f + std::exp(-h_in[i]));
    h_ref[i] = silu * h_gate[i];
  }

  Tensor in_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {M, N});
  Tensor gate_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {M, N});
  Tensor out_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {M, N});

  MemcpyAsync(in_tensor.GetPtr<void>(), h_in.data(), h_in.size() * sizeof(float), MEMCPY_HOST_TO_DEVICE,
              context_->device->GetMemoryManageStream());
  MemcpyAsync(gate_tensor.GetPtr<void>(), h_gate.data(), h_gate.size() * sizeof(float), MEMCPY_HOST_TO_DEVICE,
              context_->device->GetMemoryManageStream());
  StreamSynchronize(context_->device->GetMemoryManageStream());

  TestableSiluMulLayer layer;
  layer.SetupForTest(context_, TYPE_FP32);
  ASSERT_TRUE(layer.Forward(&in_tensor, &gate_tensor, &out_tensor).OK());
  StreamSynchronize(context_->device->GetPrimaryComputeStream());

  std::vector<float> h_out(M * N);
  Memcpy(h_out.data(), out_tensor.GetPtr<void>(), h_out.size() * sizeof(float), MEMCPY_DEVICE_TO_HOST);

  float max_diff = 0.0f;
  for (size_t i = 0; i < h_out.size(); ++i) max_diff = std::max(max_diff, std::fabs(h_out[i] - h_ref[i]));
  KLLM_LOG_INFO << "SiluMulLayer(two-input): M=" << M << " N=" << N << " max_diff=" << max_diff;
  EXPECT_LT(max_diff, 1e-5);
}

// ── SiluMulLayer (row-based split mode) test ──────────────────────────────────
// input [m, 2n] → output[m,n]; output = silu(left) * right
TEST_F(KunlunLayerTest, SiluMulLayerRowBasedTest) {
  constexpr int64_t M = 4;
  constexpr int64_t N = 16;  // output cols; input cols = 2*N
  constexpr int64_t input_cols = 2 * N;

  std::mt19937 rng(11);
  std::uniform_real_distribution<float> dist(-3.0f, 3.0f);

  std::vector<float> h_in(M * input_cols), h_ref(M * N);
  for (auto& v : h_in) v = dist(rng);

  for (int64_t i = 0; i < M; ++i)
    for (int64_t j = 0; j < N; ++j) {
      float left = h_in[i * input_cols + j];
      float right = h_in[i * input_cols + N + j];
      h_ref[i * N + j] = (left / (1.0f + std::exp(-left))) * right;
    }

  Tensor in_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {M, input_cols});
  Tensor out_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {M, N});

  MemcpyAsync(in_tensor.GetPtr<void>(), h_in.data(), h_in.size() * sizeof(float), MEMCPY_HOST_TO_DEVICE,
              context_->device->GetMemoryManageStream());
  StreamSynchronize(context_->device->GetMemoryManageStream());

  TestableSiluMulLayer layer;
  layer.SetupForTest(context_, TYPE_FP32);
  ASSERT_TRUE(layer.Forward(&in_tensor, &out_tensor).OK());
  StreamSynchronize(context_->device->GetPrimaryComputeStream());

  std::vector<float> h_out(M * N);
  Memcpy(h_out.data(), out_tensor.GetPtr<void>(), h_out.size() * sizeof(float), MEMCPY_DEVICE_TO_HOST);

  float max_diff = 0.0f;
  for (size_t i = 0; i < h_out.size(); ++i) max_diff = std::max(max_diff, std::fabs(h_out[i] - h_ref[i]));
  KLLM_LOG_INFO << "SiluMulLayer(row-based): M=" << M << " N=" << N << " max_diff=" << max_diff;
  // fp32 silu involves exp; XPU math library may differ slightly from host libm.
  EXPECT_LT(max_diff, 2e-5);
}

// ── LayernormLayer (RMSNorm path, no bias) test ───────────────────────────────
TEST_F(KunlunLayerTest, RmsNormLayerTest) {
  constexpr int64_t M = 4;
  constexpr int64_t N = 32;
  const float eps = 1e-6f;

  std::mt19937 rng(5);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  std::vector<float> h_in(M * N), h_w(N, 1.0f), h_ref(M * N);
  for (auto& v : h_in) v = dist(rng);
  for (int64_t i = 0; i < N; ++i) h_w[i] = 0.5f + 0.01f * static_cast<float>(i);

  for (int64_t i = 0; i < M; ++i) {
    float ms = 0.0f;
    for (int64_t j = 0; j < N; ++j) ms += h_in[i * N + j] * h_in[i * N + j];
    float rms = std::sqrt(ms / static_cast<float>(N) + eps);
    for (int64_t j = 0; j < N; ++j) h_ref[i * N + j] = (h_in[i * N + j] / rms) * h_w[j];
  }

  Tensor in_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {M, N});
  Tensor w_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {N});
  Tensor out_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {M, N});

  MemcpyAsync(in_tensor.GetPtr<void>(), h_in.data(), h_in.size() * sizeof(float), MEMCPY_HOST_TO_DEVICE,
              context_->device->GetMemoryManageStream());
  MemcpyAsync(w_tensor.GetPtr<void>(), h_w.data(), h_w.size() * sizeof(float), MEMCPY_HOST_TO_DEVICE,
              context_->device->GetMemoryManageStream());
  StreamSynchronize(context_->device->GetMemoryManageStream());

  TestableLayernormLayer layer;
  layer.SetupForTest(context_, TYPE_FP32, eps);
  ASSERT_TRUE(layer.Forward(&in_tensor, &w_tensor, &out_tensor).OK());
  StreamSynchronize(context_->device->GetPrimaryComputeStream());

  std::vector<float> h_out(M * N);
  Memcpy(h_out.data(), out_tensor.GetPtr<void>(), h_out.size() * sizeof(float), MEMCPY_DEVICE_TO_HOST);

  float max_diff = 0.0f;
  for (size_t i = 0; i < h_out.size(); ++i) max_diff = std::max(max_diff, std::fabs(h_out[i] - h_ref[i]));
  KLLM_LOG_INFO << "RmsNormLayer: M=" << M << " N=" << N << " max_diff=" << max_diff;
  EXPECT_LT(max_diff, 1e-4);
}

// ── LayernormLayer (LayerNorm path, with bias) test ───────────────────────────
TEST_F(KunlunLayerTest, LayerNormWithBiasTest) {
  constexpr int64_t M = 4;
  constexpr int64_t N = 32;
  const float eps = 1e-5f;

  std::mt19937 rng(9);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  std::vector<float> h_in(M * N), h_w(N), h_b(N), h_ref(M * N);
  for (auto& v : h_in) v = dist(rng);
  for (int64_t i = 0; i < N; ++i) {
    h_w[i] = 0.5f + 0.01f * static_cast<float>(i);
    h_b[i] = 0.1f * static_cast<float>(i);
  }

  for (int64_t i = 0; i < M; ++i) {
    float mean = 0.0f;
    for (int64_t j = 0; j < N; ++j) mean += h_in[i * N + j];
    mean /= static_cast<float>(N);
    float var = 0.0f;
    for (int64_t j = 0; j < N; ++j) {
      float d = h_in[i * N + j] - mean;
      var += d * d;
    }
    var /= static_cast<float>(N);
    float inv = 1.0f / std::sqrt(var + eps);
    for (int64_t j = 0; j < N; ++j) h_ref[i * N + j] = ((h_in[i * N + j] - mean) * inv) * h_w[j] + h_b[j];
  }

  Tensor in_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {M, N});
  Tensor w_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {N});
  Tensor b_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {N});
  Tensor out_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {M, N});

  MemcpyAsync(in_tensor.GetPtr<void>(), h_in.data(), h_in.size() * sizeof(float), MEMCPY_HOST_TO_DEVICE,
              context_->device->GetMemoryManageStream());
  MemcpyAsync(w_tensor.GetPtr<void>(), h_w.data(), h_w.size() * sizeof(float), MEMCPY_HOST_TO_DEVICE,
              context_->device->GetMemoryManageStream());
  MemcpyAsync(b_tensor.GetPtr<void>(), h_b.data(), h_b.size() * sizeof(float), MEMCPY_HOST_TO_DEVICE,
              context_->device->GetMemoryManageStream());
  StreamSynchronize(context_->device->GetMemoryManageStream());

  TestableLayernormLayer layer;
  layer.SetupForTest(context_, TYPE_FP32, eps);
  ASSERT_TRUE(layer.Forward(&in_tensor, &w_tensor, &out_tensor, &b_tensor).OK());
  StreamSynchronize(context_->device->GetPrimaryComputeStream());

  std::vector<float> h_out(M * N);
  Memcpy(h_out.data(), out_tensor.GetPtr<void>(), h_out.size() * sizeof(float), MEMCPY_DEVICE_TO_HOST);

  float max_diff = 0.0f;
  for (size_t i = 0; i < h_out.size(); ++i) max_diff = std::max(max_diff, std::fabs(h_out[i] - h_ref[i]));
  KLLM_LOG_INFO << "LayerNorm(with bias): M=" << M << " N=" << N << " max_diff=" << max_diff;
  EXPECT_LT(max_diff, 1e-4);
}

// ── ActivationLayer (Gelu) test ────────────────────────────────────────────────
TEST_F(KunlunLayerTest, ActivationGeluTest) {
  constexpr int64_t M = 4;
  constexpr int64_t N = 32;

  std::mt19937 rng(3);
  std::uniform_real_distribution<float> dist(-3.0f, 3.0f);

  std::vector<float> h_in(M * N), h_ref(M * N);
  for (auto& v : h_in) v = dist(rng);

  // gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
  for (size_t i = 0; i < h_in.size(); ++i) {
    float x = h_in[i];
    h_ref[i] = 0.5f * x * (1.0f + std::erf(x / std::sqrt(2.0f)));
  }

  Tensor in_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {M, N});
  Tensor out_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {M, N});

  MemcpyAsync(in_tensor.GetPtr<void>(), h_in.data(), h_in.size() * sizeof(float), MEMCPY_HOST_TO_DEVICE,
              context_->device->GetMemoryManageStream());
  StreamSynchronize(context_->device->GetMemoryManageStream());

  TestableActivationLayer layer;
  layer.SetupForTest(context_, TYPE_FP32, ActivationType::Gelu);
  ASSERT_TRUE(layer.Forward(&in_tensor, &out_tensor).OK());
  StreamSynchronize(context_->device->GetPrimaryComputeStream());

  std::vector<float> h_out(M * N);
  Memcpy(h_out.data(), out_tensor.GetPtr<void>(), h_out.size() * sizeof(float), MEMCPY_DEVICE_TO_HOST);

  float max_diff = 0.0f;
  for (size_t i = 0; i < h_out.size(); ++i) max_diff = std::max(max_diff, std::fabs(h_out[i] - h_ref[i]));
  KLLM_LOG_INFO << "Activation(Gelu): M=" << M << " N=" << N << " max_diff=" << max_diff;
  EXPECT_LT(max_diff, 1e-4);
}

// ── ActivationLayer (Swiglu, two-input) test ──────────────────────────────────
TEST_F(KunlunLayerTest, ActivationSwigluTest) {
  constexpr int64_t M = 4;
  constexpr int64_t N = 32;

  std::mt19937 rng(15);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

  std::vector<float> h_in(M * N), h_gate(M * N), h_ref(M * N);
  for (auto& v : h_in) v = dist(rng);
  for (auto& v : h_gate) v = dist(rng);

  for (size_t i = 0; i < h_in.size(); ++i) {
    float silu = h_in[i] / (1.0f + std::exp(-h_in[i]));
    h_ref[i] = silu * h_gate[i];
  }

  Tensor in_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {M, N});
  Tensor gate_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {M, N});
  Tensor out_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {M, N});

  MemcpyAsync(in_tensor.GetPtr<void>(), h_in.data(), h_in.size() * sizeof(float), MEMCPY_HOST_TO_DEVICE,
              context_->device->GetMemoryManageStream());
  MemcpyAsync(gate_tensor.GetPtr<void>(), h_gate.data(), h_gate.size() * sizeof(float), MEMCPY_HOST_TO_DEVICE,
              context_->device->GetMemoryManageStream());
  StreamSynchronize(context_->device->GetMemoryManageStream());

  TestableActivationLayer layer;
  layer.SetupForTest(context_, TYPE_FP32, ActivationType::Swiglu);
  ASSERT_TRUE(layer.Forward(&in_tensor, &out_tensor, &gate_tensor).OK());
  StreamSynchronize(context_->device->GetPrimaryComputeStream());

  std::vector<float> h_out(M * N);
  Memcpy(h_out.data(), out_tensor.GetPtr<void>(), h_out.size() * sizeof(float), MEMCPY_DEVICE_TO_HOST);

  float max_diff = 0.0f;
  for (size_t i = 0; i < h_out.size(); ++i) max_diff = std::max(max_diff, std::fabs(h_out[i] - h_ref[i]));
  KLLM_LOG_INFO << "Activation(Swiglu): M=" << M << " N=" << N << " max_diff=" << max_diff;
  EXPECT_LT(max_diff, 1e-5);
}

// ── MulLayer (element-wise) test ─────────────────────────────────────────────
TEST_F(KunlunLayerTest, MulLayerElementwiseTest) {
  constexpr int64_t M = 4;
  constexpr int64_t N = 64;

  std::mt19937 rng(20);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  std::vector<float> h_a(M * N), h_b(M * N), h_ref(M * N);
  for (size_t i = 0; i < h_a.size(); ++i) {
    h_a[i] = dist(rng);
    h_b[i] = dist(rng);
    h_ref[i] = h_a[i] * h_b[i];
  }

  Tensor a_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {M, N});
  Tensor b_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {M, N});
  Tensor out_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {M, N});

  MemcpyAsync(a_tensor.GetPtr<void>(), h_a.data(), h_a.size() * sizeof(float), MEMCPY_HOST_TO_DEVICE,
              context_->device->GetMemoryManageStream());
  MemcpyAsync(b_tensor.GetPtr<void>(), h_b.data(), h_b.size() * sizeof(float), MEMCPY_HOST_TO_DEVICE,
              context_->device->GetMemoryManageStream());
  StreamSynchronize(context_->device->GetMemoryManageStream());

  TestableMulLayer layer;
  layer.SetupForTest(context_, TYPE_FP32);
  ASSERT_TRUE(layer.Forward(&a_tensor, &b_tensor, &out_tensor).OK());
  StreamSynchronize(context_->device->GetPrimaryComputeStream());

  std::vector<float> h_out(M * N);
  Memcpy(h_out.data(), out_tensor.GetPtr<void>(), h_out.size() * sizeof(float), MEMCPY_DEVICE_TO_HOST);

  float max_diff = 0.0f;
  for (size_t i = 0; i < h_out.size(); ++i) max_diff = std::max(max_diff, std::fabs(h_out[i] - h_ref[i]));
  KLLM_LOG_INFO << "MulLayer(element-wise): M=" << M << " N=" << N << " max_diff=" << max_diff;
  EXPECT_LT(max_diff, 1e-6);
}

// ── MulLayer (broadcast [m,1] × [m,n]) test ───────────────────────────────────
TEST_F(KunlunLayerTest, MulLayerBroadcastTest) {
  constexpr int64_t M = 4;
  constexpr int64_t N = 64;

  std::mt19937 rng(25);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  std::vector<float> h_gate(M, 1.0f), h_data(M * N), h_ref(M * N);
  for (int64_t i = 0; i < M; ++i) h_gate[i] = dist(rng);
  for (auto& v : h_data) v = dist(rng);

  for (int64_t i = 0; i < M; ++i)
    for (int64_t j = 0; j < N; ++j) h_ref[i * N + j] = h_gate[i] * h_data[i * N + j];

  // input[0] is gate [m,1], input[1] is data [m,n]  (per MulLayer broadcast path)
  Tensor gate_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {M, 1});
  Tensor data_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {M, N});
  Tensor out_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {M, N});

  MemcpyAsync(gate_tensor.GetPtr<void>(), h_gate.data(), h_gate.size() * sizeof(float), MEMCPY_HOST_TO_DEVICE,
              context_->device->GetMemoryManageStream());
  MemcpyAsync(data_tensor.GetPtr<void>(), h_data.data(), h_data.size() * sizeof(float), MEMCPY_HOST_TO_DEVICE,
              context_->device->GetMemoryManageStream());
  StreamSynchronize(context_->device->GetMemoryManageStream());

  TestableMulLayer layer;
  layer.SetupForTest(context_, TYPE_FP32);
  ASSERT_TRUE(layer.Forward(&gate_tensor, &data_tensor, &out_tensor).OK());
  StreamSynchronize(context_->device->GetPrimaryComputeStream());

  std::vector<float> h_out(M * N);
  Memcpy(h_out.data(), out_tensor.GetPtr<void>(), h_out.size() * sizeof(float), MEMCPY_DEVICE_TO_HOST);

  float max_diff = 0.0f;
  for (size_t i = 0; i < h_out.size(); ++i) max_diff = std::max(max_diff, std::fabs(h_out[i] - h_ref[i]));
  KLLM_LOG_INFO << "MulLayer(broadcast): M=" << M << " N=" << N << " max_diff=" << max_diff;
  EXPECT_LT(max_diff, 1e-6);
}

// ── CastLayer test (bf16 → fp32) ──────────────────────────────────────────────
// CastLayer::ForwardT dispatches by inter_data_type_ which must be FP16/BF16/FP32
// (DISPATCH_BY_3_DTYPE does not support INT32). Cast is always SRC→FP32.
TEST_F(KunlunLayerTest, CastLayerBf16ToFp32Test) {
  constexpr int64_t LEN = 64;

  std::mt19937 rng(88);
  std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

  std::vector<float> h_in_fp32(LEN);
  for (auto& v : h_in_fp32) v = dist(rng);

  // CPU reference: bf16→fp32 is lossy (~7-bit mantissa), so compare against
  // the bf16-truncated value, not the original fp32.
  auto bf16_from_float = [](float f) -> uint16_t {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(float));
    // Round-to-nearest-even truncation to bf16 (upper 16 bits)
    uint32_t lsb = (bits >> 16) & 1;
    uint32_t rounding_bias = 0x7FFF + lsb;
    uint32_t rounded = bits + rounding_bias;
    return static_cast<uint16_t>(rounded >> 16);
  };
  auto float_from_bf16 = [](uint16_t bf) -> float {
    uint32_t bits = static_cast<uint32_t>(bf) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(float));
    return f;
  };

  std::vector<float> h_ref(LEN);
  for (int64_t i = 0; i < LEN; ++i) h_ref[i] = float_from_bf16(bf16_from_float(h_in_fp32[i]));

  // Step 1: Prepare bf16 input on device via XpuCast from fp32
  namespace xpukernel = xpytorch::xpu::api;
  void* stream_ptr = context_->device->GetPrimaryComputeStream().Get();
  auto* xpuctx = static_cast<xpukernel::Context*>(context_->device->ext->GetXpuKernelContext(stream_ptr));

  Tensor in_fp32_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {LEN});
  MemcpyAsync(in_fp32_tensor.GetPtr<void>(), h_in_fp32.data(), LEN * sizeof(float), MEMCPY_HOST_TO_DEVICE,
              context_->device->GetMemoryManageStream());
  StreamSynchronize(context_->device->GetMemoryManageStream());

  Tensor in_bf16_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16, {LEN});
  int ret_cast =
      XpuCast<float, bfloat16>(xpuctx, in_fp32_tensor.GetPtr<float>(), in_bf16_tensor.GetPtr<bfloat16>(), LEN);
  ASSERT_EQ(ret_cast, 0);
  StreamSynchronize(context_->device->GetPrimaryComputeStream());

  // Step 2: Run CastLayer (bf16 → fp32)
  // Typed interface: cast whole input at byte_offset=0. vocab_size/vocab_size_pad
  // are unused by the kunlun impl (length taken from input element count).
  Tensor out_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {LEN});

  TestableCastLayer layer;
  layer.SetupForTest(context_, TYPE_BF16);
  ASSERT_TRUE(layer.Forward(&in_bf16_tensor, /*byte_offset=*/0, /*vocab_size=*/LEN,
                            /*vocab_size_pad=*/LEN, &out_tensor)
                  .OK());
  StreamSynchronize(context_->device->GetPrimaryComputeStream());

  std::vector<float> h_out(LEN);
  Memcpy(h_out.data(), out_tensor.GetPtr<void>(), LEN * sizeof(float), MEMCPY_DEVICE_TO_HOST);

  float max_diff = 0.0f;
  for (int64_t i = 0; i < LEN; ++i) max_diff = std::max(max_diff, std::fabs(h_out[i] - h_ref[i]));
  KLLM_LOG_INFO << "CastLayer(bf16→fp32): len=" << LEN << " max_diff=" << max_diff;
  // bf16→fp32 cast is exact (fp32 can represent all bf16 values exactly)
  EXPECT_LT(max_diff, 1e-6);
}

// ── AddNormLayer test (in-place fused add+rmsnorm) ────────────────────────────
// Uses N=128 to match kernel's SIMD alignment requirements (same as
// FusedAddRmsNormTest in kunlun_kernel_test.cpp).
TEST_F(KunlunLayerTest, AddNormLayerTest) {
  constexpr int64_t M = 4;
  constexpr int64_t N = 128;
  const float eps = 1e-6f;

  std::mt19937 rng(99);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  std::vector<float> h_input(M * N), h_residual(M * N), h_w(N, 1.0f), h_ref(M * N);
  for (auto& v : h_input) v = dist(rng);
  for (auto& v : h_residual) v = dist(rng);

  // CPU ref: residual_after_add = input + residual; output = residual_after_add * rms_inv * weight
  // where rms_inv = 1 / sqrt(ss/n + eps)
  // NOTE: must not modify h_residual in-place — it's used for H2D copy below.
  std::vector<float> ref_residual(M * N);
  for (int64_t i = 0; i < M; ++i) {
    float ss = 0.0f;
    for (int64_t j = 0; j < N; ++j) {
      ref_residual[i * N + j] = h_input[i * N + j] + h_residual[i * N + j];
      ss += ref_residual[i * N + j] * ref_residual[i * N + j];
    }
    float rms_inv = 1.0f / std::sqrt(ss / static_cast<float>(N) + eps);
    for (int64_t j = 0; j < N; ++j) h_ref[i * N + j] = ref_residual[i * N + j] * rms_inv * h_w[j];
  }

  Tensor input_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {M, N});
  Tensor residual_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {M, N});
  Tensor w_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {N});

  // H2D: h_residual is the ORIGINAL (unmodified) residual here.
  MemcpyAsync(input_tensor.GetPtr<void>(), h_input.data(), h_input.size() * sizeof(float), MEMCPY_HOST_TO_DEVICE,
              context_->device->GetMemoryManageStream());
  MemcpyAsync(residual_tensor.GetPtr<void>(), h_residual.data(), h_residual.size() * sizeof(float),
              MEMCPY_HOST_TO_DEVICE, context_->device->GetMemoryManageStream());
  MemcpyAsync(w_tensor.GetPtr<void>(), h_w.data(), h_w.size() * sizeof(float), MEMCPY_HOST_TO_DEVICE,
              context_->device->GetMemoryManageStream());
  StreamSynchronize(context_->device->GetMemoryManageStream());

  TestableAddNormLayer layer;
  layer.SetupForTest(context_, TYPE_FP32, eps);
  ASSERT_TRUE(layer.Forward(&input_tensor, &residual_tensor, &w_tensor, &input_tensor).OK());
  StreamSynchronize(context_->device->GetPrimaryComputeStream());

  std::vector<float> h_out(M * N);
  Memcpy(h_out.data(), input_tensor.GetPtr<void>(), h_out.size() * sizeof(float), MEMCPY_DEVICE_TO_HOST);

  float max_diff = 0.0f;
  for (size_t i = 0; i < h_out.size(); ++i) max_diff = std::max(max_diff, std::fabs(h_out[i] - h_ref[i]));
  KLLM_LOG_INFO << "AddNormLayer: M=" << M << " N=" << N << " max_diff=" << max_diff;
  EXPECT_LT(max_diff, 1e-4);
}

// ── SplitLayer test ───────────────────────────────────────────────────────────
// Split input [M, total_cols] into multiple outputs along axis=1
TEST_F(KunlunLayerTest, SplitLayerTest) {
  constexpr int64_t M = 4;
  constexpr int64_t part0_cols = 16;
  constexpr int64_t part1_cols = 24;
  constexpr int64_t total_cols = part0_cols + part1_cols;

  std::mt19937 rng(44);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  std::vector<float> h_in(M * total_cols);
  for (auto& v : h_in) v = dist(rng);

  Tensor in_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {M, total_cols});
  Tensor out0(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {M, part0_cols});
  Tensor out1(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {M, part1_cols});

  MemcpyAsync(in_tensor.GetPtr<void>(), h_in.data(), h_in.size() * sizeof(float), MEMCPY_HOST_TO_DEVICE,
              context_->device->GetMemoryManageStream());
  StreamSynchronize(context_->device->GetMemoryManageStream());

  TestableSplitLayer layer;
  layer.SetupForTest(context_, TYPE_FP32);
  ASSERT_TRUE(layer.Forward(&in_tensor, {static_cast<size_t>(part0_cols), static_cast<size_t>(part1_cols)},
                            {&out0, &out1})
                  .OK());
  StreamSynchronize(context_->device->GetPrimaryComputeStream());

  std::vector<float> h_out0(M * part0_cols), h_out1(M * part1_cols);
  Memcpy(h_out0.data(), out0.GetPtr<void>(), h_out0.size() * sizeof(float), MEMCPY_DEVICE_TO_HOST);
  Memcpy(h_out1.data(), out1.GetPtr<void>(), h_out1.size() * sizeof(float), MEMCPY_DEVICE_TO_HOST);

  float max_diff = 0.0f;
  for (int64_t i = 0; i < M; ++i) {
    for (int64_t j = 0; j < part0_cols; ++j)
      max_diff = std::max(max_diff, std::fabs(h_out0[i * part0_cols + j] - h_in[i * total_cols + j]));
    for (int64_t j = 0; j < part1_cols; ++j)
      max_diff = std::max(max_diff, std::fabs(h_out1[i * part1_cols + j] - h_in[i * total_cols + part0_cols + j]));
  }
  KLLM_LOG_INFO << "SplitLayer: M=" << M << " cols=[" << part0_cols << "," << part1_cols << "] max_diff=" << max_diff;
  EXPECT_LT(max_diff, 1e-6);
}

// ── AssembleTokensHiddenLayer test ───────────────────────────────────────────
// Gather rows from hidden_states by index → output[num_logits, hidden_size]
TEST_F(KunlunLayerTest, AssembleTokensHiddenLayerTest) {
  constexpr int64_t total_tokens = 16;
  constexpr int64_t hidden_size = 32;
  constexpr int64_t num_logits = 5;

  std::mt19937 rng(55);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  std::vector<float> h_hidden(total_tokens * hidden_size);
  for (auto& v : h_hidden) v = dist(rng);

  // Select indices: spread across the range, including edge values.
  std::vector<int64_t> h_idx = {0, 3, 7, 15, 10};

  // CPU reference
  std::vector<float> h_ref(num_logits * hidden_size);
  for (int64_t i = 0; i < num_logits; ++i) {
    int64_t row = h_idx[i];
    for (int64_t j = 0; j < hidden_size; ++j) h_ref[i * hidden_size + j] = h_hidden[row * hidden_size + j];
  }

  Tensor hidden_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {total_tokens, hidden_size});
  // indices are uint64_t (bit-compatible int64)
  Tensor idx_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_UINT64, {num_logits});
  Tensor out_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {num_logits, hidden_size});

  MemcpyAsync(hidden_tensor.GetPtr<void>(), h_hidden.data(), h_hidden.size() * sizeof(float), MEMCPY_HOST_TO_DEVICE,
              context_->device->GetMemoryManageStream());
  MemcpyAsync(idx_tensor.GetPtr<void>(), h_idx.data(), h_idx.size() * sizeof(int64_t), MEMCPY_HOST_TO_DEVICE,
              context_->device->GetMemoryManageStream());
  StreamSynchronize(context_->device->GetMemoryManageStream());

  TestableAssembleTokensHiddenLayer layer;
  layer.SetupForTest(context_, TYPE_FP32);
  ASSERT_TRUE(layer.Forward(&hidden_tensor, &idx_tensor, &out_tensor).OK());
  StreamSynchronize(context_->device->GetPrimaryComputeStream());

  std::vector<float> h_out(num_logits * hidden_size);
  Memcpy(h_out.data(), out_tensor.GetPtr<void>(), h_out.size() * sizeof(float), MEMCPY_DEVICE_TO_HOST);

  float max_diff = 0.0f;
  for (size_t i = 0; i < h_out.size(); ++i) max_diff = std::max(max_diff, std::fabs(h_out[i] - h_ref[i]));
  KLLM_LOG_INFO << "AssembleTokensHidden: total=" << total_tokens << " hidden=" << hidden_size
                << " num_logits=" << num_logits << " max_diff=" << max_diff;
  EXPECT_LT(max_diff, 1e-6);
}

// ── EmbLookupLayer test ──────────────────────────────────────────────────────
// Look up token embeddings by id from a weight table.
TEST_F(KunlunLayerTest, EmbLookupLayerTest) {
  constexpr int64_t vocab_size = 16;
  constexpr int64_t hidden_size = 32;
  constexpr int64_t token_num = 5;

  std::mt19937 rng(66);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  // Build embedding table [vocab_size, hidden_size]
  std::vector<float> h_weight(vocab_size * hidden_size);
  for (auto& v : h_weight) v = dist(rng);

  // Token IDs to look up
  std::vector<int32_t> h_ids = {0, 5, 10, 15, 3};
  std::vector<float> h_ref(token_num * hidden_size);
  for (int64_t i = 0; i < token_num; ++i) {
    int32_t id = h_ids[i];
    for (int64_t j = 0; j < hidden_size; ++j) h_ref[i * hidden_size + j] = h_weight[id * hidden_size + j];
  }

  // Input tensors: [ids, ids_offsets, prefix_offsets, weight, steps]
  // ids_offsets and prefix_offsets are not used by the kunlun impl (only ids + weight).
  // Create minimal dummy tensors for them.
  Tensor ids_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {token_num});
  Tensor offsets_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {2});
  Tensor prefix_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {2});
  Tensor weight_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {vocab_size, hidden_size});
  Tensor out_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {token_num, hidden_size});

  MemcpyAsync(ids_tensor.GetPtr<void>(), h_ids.data(), h_ids.size() * sizeof(int32_t), MEMCPY_HOST_TO_DEVICE,
              context_->device->GetMemoryManageStream());
  MemcpyAsync(weight_tensor.GetPtr<void>(), h_weight.data(), h_weight.size() * sizeof(float), MEMCPY_HOST_TO_DEVICE,
              context_->device->GetMemoryManageStream());
  StreamSynchronize(context_->device->GetMemoryManageStream());

  TestableEmbLookupLayer layer;
  layer.SetupForTest(context_, TYPE_FP32, /*max_token_num=*/4096);
  ASSERT_TRUE(layer.Forward(&ids_tensor, &offsets_tensor, &prefix_tensor, &weight_tensor, &out_tensor).OK());
  StreamSynchronize(context_->device->GetPrimaryComputeStream());

  std::vector<float> h_out(token_num * hidden_size);
  Memcpy(h_out.data(), out_tensor.GetPtr<void>(), h_out.size() * sizeof(float), MEMCPY_DEVICE_TO_HOST);

  float max_diff = 0.0f;
  for (size_t i = 0; i < h_out.size(); ++i) max_diff = std::max(max_diff, std::fabs(h_out[i] - h_ref[i]));
  KLLM_LOG_INFO << "EmbLookup: vocab=" << vocab_size << " hidden=" << hidden_size << " tokens=" << token_num
                << " max_diff=" << max_diff;
  EXPECT_LT(max_diff, 1e-6);
}

// ── SigmoidLayer test (in-place) ──────────────────────────────────────────────
TEST_F(KunlunLayerTest, SigmoidLayerTest) {
  constexpr int64_t M = 4;
  constexpr int64_t N = 64;
  const size_t num_elements = M * N;

  std::mt19937 rng(77);
  std::uniform_real_distribution<float> dist(-5.0f, 5.0f);

  std::vector<float> h_in(num_elements), h_ref(num_elements);
  for (size_t i = 0; i < num_elements; ++i) {
    h_in[i] = dist(rng);
    h_ref[i] = 1.0f / (1.0f + std::exp(-h_in[i]));
  }

  // SigmoidLayer operates in-place: output[0] == input[0].
  Tensor data_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {M, N});

  MemcpyAsync(data_tensor.GetPtr<void>(), h_in.data(), num_elements * sizeof(float), MEMCPY_HOST_TO_DEVICE,
              context_->device->GetMemoryManageStream());
  StreamSynchronize(context_->device->GetMemoryManageStream());

  TestableSigmoidLayer layer;
  layer.SetupForTest(context_, TYPE_FP32);
  // Sigmoid operates in-place on the tensor.
  Tensor out_tensor = data_tensor;
  ASSERT_TRUE(layer.Forward(&data_tensor).OK());
  StreamSynchronize(context_->device->GetPrimaryComputeStream());

  std::vector<float> h_out(num_elements);
  Memcpy(h_out.data(), out_tensor.GetPtr<void>(), num_elements * sizeof(float), MEMCPY_DEVICE_TO_HOST);

  float max_diff = 0.0f;
  for (size_t i = 0; i < num_elements; ++i) max_diff = std::max(max_diff, std::fabs(h_out[i] - h_ref[i]));
  KLLM_LOG_INFO << "SigmoidLayer: M=" << M << " N=" << N << " max_diff=" << max_diff;
  EXPECT_LT(max_diff, 1e-5);
}

}  // namespace ksana_llm
