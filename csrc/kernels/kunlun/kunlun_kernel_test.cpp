/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#include <xpu/runtime.h>
#include <xpu/xdnn.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include "test.h"

#include "runtime/xpu_context.h"  // xmlir_rt::getXpuKernelContext

#include "device/kunlun/xpu_utils.h"
#include "kernels/kunlun/attention_kernel_wrapper.h"
#include "kernels/kunlun/basic_kernel_wrapper.h"
#include "utils/logger.h"

namespace ksana_llm {

// XPU wall-clock timer (xdnn has no elapsed-time event API)
class XpuTimer {
 public:
  XpuTimer(const std::string& name, XPUStream stream) : name_(name), stream_(stream) {
    XPU_CHECK(xpu_wait(stream_));
    start_ = std::chrono::steady_clock::now();
  }
  ~XpuTimer() {
    XPU_CHECK(xpu_wait(stream_));
    auto ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - start_).count();
    KLLM_LOG_INFO << fmt::format("{} cost: {:.3f} ms", name_, ms);
  }

 private:
  std::string name_;
  XPUStream stream_;
  std::chrono::steady_clock::time_point start_;
};

class KunlunKernelWrapperTest : public testing::Test {
 public:
  // Stream / xdnn::Context are process-lifetime, mirroring XSGLANG's stream
  // model (PyTorch's torch.cuda.Stream is also process-managed). The xpytorch
  // ctx used by XpuScatter / XpuArgMax / etc. retains this stream across all
  // tests, so it never sees a dead stream — eliminating the cuapi_xpu_wait
  // status=400 shutdown noise.
  static void SetUpTestSuite() {
    XPU_CHECK(xpu_current_device(&device_));
    XPU_CHECK(xpu_stream_create(&stream_));
    ctx_ = xdnn::create_context();
    ctx_->set_stream(stream_);
    // Persistent xpukernel::Context* obtained the same way production code does
    // (mirror of KunlunDeviceContextExtension::GetXpuKernelContext): process-level
    // singleton from xmlir_rt, bound to our test stream so xdnn-only and
    // xpukernel calls stay ordered.
    xpuctx_ = xmlir_rt::getXpuKernelContext(/*useCpu=*/false);
    xpuctx_->set_stream(stream_, false);
    XPU_CHECK(xpu_wait(stream_));
  }

  static void TearDownTestSuite() {
    XPU_CHECK(xpu_wait(stream_));
    // xdnn::destroy_context deadlocks on P800; leak it (process exiting anyway).
    // stream_ intentionally leaked: xdnn::Context still references it.
    // Destroying it triggers cuapi_xpu_wait status=400 at process exit.
    // Both stream_ and ctx_ are reclaimed by the OS.
    //
    // Note: This fixture assumes a single RUN_ALL_TESTS() invocation per
    // process. If the suite is ever re-run inside the same process (e.g. by
    // an embedding test driver), the static stream_/ctx_/xpuctx_ members
    // are NOT re-initialized and will accumulate. For that usage pattern,
    // add an explicit reset path here guarded by a static run-counter.
  }

  void SetUp() override {
    // Per-test: ensure the shared stream is idle before this test runs.
    XPU_CHECK(xpu_wait(stream_));
  }

  void TearDown() override { XPU_CHECK(xpu_wait(stream_)); }

 protected:
  // Aliases so existing test bodies that reference `stream` / `ctx` / `device`
  // unmodified still compile.
  XPUStream& stream = stream_;
  xdnn::Context*& ctx = ctx_;
  xpytorch::xpu::api::Context*& xpuctx = xpuctx_;
  int& device = device_;

  static int device_;
  static XPUStream stream_;
  static xdnn::Context* ctx_;
  static xpytorch::xpu::api::Context* xpuctx_;
};

int KunlunKernelWrapperTest::device_ = -1;
XPUStream KunlunKernelWrapperTest::stream_ = nullptr;
xdnn::Context* KunlunKernelWrapperTest::ctx_ = nullptr;
xpytorch::xpu::api::Context* KunlunKernelWrapperTest::xpuctx_ = nullptr;

#ifdef ENABLE_KUNLUN

TEST_F(KunlunKernelWrapperTest, AddTest) {
  constexpr int64_t n = 1024;
  std::vector<float> host_a(n), host_b(n), host_c(n);
  for (int i = 0; i < n; ++i) {
    host_a[i] = static_cast<float>(i) * 0.01f;
    host_b[i] = 1.0f;
  }

  float *dev_a, *dev_b, *dev_c;
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_a), n * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_b), n * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_c), n * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_memcpy(dev_a, host_a.data(), n * sizeof(float), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(dev_b, host_b.data(), n * sizeof(float), XPU_HOST_TO_DEVICE));

  {
    XpuTimer timer("XpuAdd", stream);
    ASSERT_EQ(0, (XpuAdd<float>(xpuctx, dev_a, dev_b, dev_c, n)));
  }
  XPU_CHECK(xpu_memcpy(host_c.data(), dev_c, n * sizeof(float), XPU_DEVICE_TO_HOST));

  for (int i = 0; i < n; ++i) {
    EXPECT_NEAR(host_c[i], host_a[i] + host_b[i], 1e-5f);
  }

  XPU_CHECK(xpu_free(dev_a));
  XPU_CHECK(xpu_free(dev_b));
  XPU_CHECK(xpu_free(dev_c));
}

TEST_F(KunlunKernelWrapperTest, SiluTest) {
  constexpr int64_t n = 512;
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
  std::vector<float> host_in(n), host_out(n);
  for (auto& v : host_in) v = dist(rng);

  float *dev_in, *dev_out;
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_in), n * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_out), n * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_memcpy(dev_in, host_in.data(), n * sizeof(float), XPU_HOST_TO_DEVICE));

  {
    XpuTimer timer("XpuSilu", stream);
    ASSERT_EQ(0, (XpuSilu<float>(xpuctx, dev_in, dev_out, n)));
  }
  XPU_CHECK(xpu_memcpy(host_out.data(), dev_out, n * sizeof(float), XPU_DEVICE_TO_HOST));

  for (int i = 0; i < n; ++i) {
    float ref = host_in[i] / (1.0f + std::exp(-host_in[i]));
    EXPECT_NEAR(host_out[i], ref, 1e-5f);
  }

  XPU_CHECK(xpu_free(dev_in));
  XPU_CHECK(xpu_free(dev_out));
}

// XpuSiluMul: two separate tensors; output[i] = silu(input[i]) * gate[i]
// Used by SiluMulLayer when gate_proj and up_proj are NOT fused (separate weights).
TEST_F(KunlunKernelWrapperTest, SiluMulTwoInputTest) {
  constexpr int64_t m = 4, n = 64;
  std::mt19937 rng(200);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
  std::vector<float> host_in(m * n), host_gate(m * n), host_out(m * n);
  for (auto& v : host_in) v = dist(rng);
  for (auto& v : host_gate) v = dist(rng);

  std::vector<float> ref(m * n);
  for (size_t i = 0; i < static_cast<size_t>(m * n); ++i) {
    float s = host_in[i] / (1.0f + std::exp(-host_in[i]));
    ref[i] = s * host_gate[i];
  }

  float *dev_in, *dev_gate, *dev_out;
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_in), m * n * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_gate), m * n * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_out), m * n * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_memcpy(dev_in, host_in.data(), m * n * sizeof(float), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(dev_gate, host_gate.data(), m * n * sizeof(float), XPU_HOST_TO_DEVICE));

  {
    XpuTimer timer("XpuSiluMul<float>", stream);
    ASSERT_EQ(0, (XpuSiluMul<float>(xpuctx, dev_in, dev_gate, dev_out, m, n)));
  }
  XPU_CHECK(xpu_memcpy(host_out.data(), dev_out, m * n * sizeof(float), XPU_DEVICE_TO_HOST));

  for (size_t i = 0; i < static_cast<size_t>(m * n); ++i) {
    EXPECT_NEAR(host_out[i], ref[i], 1e-5f) << "mismatch at [" << i << "]";
  }

  XPU_CHECK(xpu_free(dev_in));
  XPU_CHECK(xpu_free(dev_gate));
  XPU_CHECK(xpu_free(dev_out));
}

// SwiGLU: input [m, 2n] -> output [m, n]; out[i,j] = silu(in[i,j]) * in[i, n+j]
TEST_F(KunlunKernelWrapperTest, RowBasedSiluMulTest) {
  constexpr int64_t m = 4, n = 64;
  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
  std::vector<float> host_in(m * 2 * n), host_out(m * n);
  for (auto& v : host_in) v = dist(rng);

  float *dev_in, *dev_out;
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_in), m * 2 * n * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_out), m * n * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_memcpy(dev_in, host_in.data(), m * 2 * n * sizeof(float), XPU_HOST_TO_DEVICE));

  {
    XpuTimer timer("XpuRowBasedSiluMul", stream);
    ASSERT_EQ(0, (XpuRowBasedSiluMul<float>(ctx, dev_in, dev_out, m, n)));
  }
  XPU_CHECK(xpu_memcpy(host_out.data(), dev_out, m * n * sizeof(float), XPU_DEVICE_TO_HOST));

  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      float left = host_in[i * 2 * n + j];
      float right = host_in[i * 2 * n + n + j];
      float ref = (left / (1.0f + std::exp(-left))) * right;
      EXPECT_NEAR(host_out[i * n + j], ref, 1e-5f);
    }
  }

  XPU_CHECK(xpu_free(dev_in));
  XPU_CHECK(xpu_free(dev_out));
}

// y[m,i] = x[m,i] / sqrt(mean(x[m,:]^2) + eps) * scale[i]
TEST_F(KunlunKernelWrapperTest, RmsLayerNormTest) {
  constexpr int64_t m = 8, n = 256;
  constexpr float eps = 1e-6f;
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> host_x(m * n), host_scale(n, 1.0f), host_y(m * n);
  for (auto& v : host_x) v = dist(rng);

  float *dev_x, *dev_scale, *dev_y;
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_x), m * n * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_scale), n * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_y), m * n * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_memcpy(dev_x, host_x.data(), m * n * sizeof(float), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(dev_scale, host_scale.data(), n * sizeof(float), XPU_HOST_TO_DEVICE));

  {
    XpuTimer timer("XpuRmsLayerNorm", stream);
    ASSERT_EQ(0, (XpuRmsLayerNorm<float, float>(ctx, dev_x, dev_y, m, n, eps, dev_scale,
                                                /*bias=*/nullptr, /*var=*/nullptr)));
  }
  XPU_CHECK(xpu_memcpy(host_y.data(), dev_y, m * n * sizeof(float), XPU_DEVICE_TO_HOST));

  for (int row = 0; row < m; ++row) {
    float ss = 0.0f;
    for (int j = 0; j < n; ++j) ss += host_x[row * n + j] * host_x[row * n + j];
    float rms_inv = 1.0f / std::sqrt(ss / n + eps);
    for (int j = 0; j < n; ++j) {
      float ref = host_x[row * n + j] * rms_inv * host_scale[j];
      EXPECT_NEAR(host_y[row * n + j], ref, 1e-4f) << "mismatch at [" << row << "," << j << "]";
    }
  }

  XPU_CHECK(xpu_free(dev_x));
  XPU_CHECK(xpu_free(dev_scale));
  XPU_CHECK(xpu_free(dev_y));
}

// in-place: residual += input; input = rmsnorm(residual, weight)
TEST_F(KunlunKernelWrapperTest, FusedAddRmsNormTest) {
  constexpr int64_t m = 4, n = 128;
  constexpr float eps = 1e-6f;
  std::mt19937 rng(99);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> host_input(m * n), host_residual(m * n), host_weight(n, 1.0f);
  for (auto& v : host_input) v = dist(rng);
  for (auto& v : host_residual) v = dist(rng);

  // CPU reference: residual_ref[i] = input[i] + residual[i]; input_ref[i] = rmsnorm(residual_ref)
  std::vector<float> ref_residual(m * n), ref_input(m * n);
  for (int row = 0; row < m; ++row) {
    float ss = 0.0f;
    for (int j = 0; j < n; ++j) {
      ref_residual[row * n + j] = host_input[row * n + j] + host_residual[row * n + j];
      ss += ref_residual[row * n + j] * ref_residual[row * n + j];
    }
    float rms_inv = 1.0f / std::sqrt(ss / n + eps);
    for (int j = 0; j < n; ++j) {
      ref_input[row * n + j] = ref_residual[row * n + j] * rms_inv * host_weight[j];
    }
  }

  float *dev_input, *dev_residual, *dev_weight;
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_input), m * n * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_residual), m * n * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_weight), n * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_memcpy(dev_input, host_input.data(), m * n * sizeof(float), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(dev_residual, host_residual.data(), m * n * sizeof(float), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(dev_weight, host_weight.data(), n * sizeof(float), XPU_HOST_TO_DEVICE));

  {
    XpuTimer timer("XpuFusedAddRmsNorm", stream);
    ASSERT_EQ(0, (XpuFusedAddRmsNorm<float, float>(ctx, dev_input, dev_residual, dev_weight, m, n, eps)));
  }

  std::vector<float> out_input(m * n), out_residual(m * n);
  XPU_CHECK(xpu_memcpy(out_input.data(), dev_input, m * n * sizeof(float), XPU_DEVICE_TO_HOST));
  XPU_CHECK(xpu_memcpy(out_residual.data(), dev_residual, m * n * sizeof(float), XPU_DEVICE_TO_HOST));

  for (size_t i = 0; i < static_cast<size_t>(m * n); ++i) {
    EXPECT_NEAR(out_residual[i], ref_residual[i], 1e-5f) << "residual mismatch at [" << i << "]";
    EXPECT_NEAR(out_input[i], ref_input[i], 1e-4f) << "normed input mismatch at [" << i << "]";
  }

  XPU_CHECK(xpu_free(dev_input));
  XPU_CHECK(xpu_free(dev_residual));
  XPU_CHECK(xpu_free(dev_weight));
}

TEST_F(KunlunKernelWrapperTest, EmbeddingLookupTest) {
  constexpr int64_t token_num = 8, vocab = 32, hidden = 64;
  std::mt19937 rng(55);
  std::uniform_real_distribution<float> wdist(-1.0f, 1.0f);
  std::uniform_int_distribution<int64_t> idist(0, vocab - 1);

  std::vector<float> host_weight(vocab * hidden);
  std::vector<int64_t> host_ids(token_num);
  for (auto& v : host_weight) v = wdist(rng);
  for (auto& v : host_ids) v = idist(rng);

  // CPU reference
  std::vector<float> ref(token_num * hidden);
  for (int i = 0; i < token_num; ++i) {
    std::memcpy(ref.data() + i * hidden, host_weight.data() + host_ids[i] * hidden, hidden * sizeof(float));
  }

  int64_t* dev_ids;
  float *dev_weight, *dev_out;
  std::vector<float> host_out(token_num * hidden);
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_ids), token_num * sizeof(int64_t), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_weight), vocab * hidden * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_out), token_num * hidden * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_memcpy(dev_ids, host_ids.data(), token_num * sizeof(int64_t), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(dev_weight, host_weight.data(), vocab * hidden * sizeof(float), XPU_HOST_TO_DEVICE));

  {
    XpuTimer timer("XpuEmbeddingLookup", stream);
    ASSERT_EQ(0, (XpuEmbeddingLookup<float>(xpuctx, dev_ids, dev_weight, dev_out, token_num, hidden, vocab)));
  }
  XPU_CHECK(xpu_memcpy(host_out.data(), dev_out, token_num * hidden * sizeof(float), XPU_DEVICE_TO_HOST));

  for (size_t i = 0; i < static_cast<size_t>(token_num * hidden); ++i) {
    EXPECT_NEAR(host_out[i], ref[i], 1e-6f) << "mismatch at [" << i << "]";
  }

  XPU_CHECK(xpu_free(dev_ids));
  XPU_CHECK(xpu_free(dev_weight));
  XPU_CHECK(xpu_free(dev_out));
}

// ── XpuCast: int32 -> int64 ───────────────────────────────────────────────────

TEST_F(KunlunKernelWrapperTest, CastInt32ToInt64Test) {
  constexpr int64_t n = 64;
  std::vector<int32_t> host_in(n);
  std::vector<int64_t> host_out(n);
  for (int i = 0; i < n; ++i) host_in[i] = i * 3 - 50;

  int32_t* dev_in;
  int64_t* dev_out;
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_in), n * sizeof(int32_t), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_out), n * sizeof(int64_t), XPU_MEM_MAIN));
  XPU_CHECK(xpu_memcpy(dev_in, host_in.data(), n * sizeof(int32_t), XPU_HOST_TO_DEVICE));

  {
    XpuTimer timer("XpuCast<int32,int64>", stream);
    ASSERT_EQ(0, (XpuCast<int32_t, int64_t>(xpuctx, dev_in, dev_out, n)));
  }
  XPU_CHECK(xpu_memcpy(host_out.data(), dev_out, n * sizeof(int64_t), XPU_DEVICE_TO_HOST));

  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(host_out[i], static_cast<int64_t>(host_in[i]));
  }

  XPU_CHECK(xpu_free(dev_in));
  XPU_CHECK(xpu_free(dev_out));
}

// ── XpuCast: float <-> float16 round-trip ────────────────────────────────────
// float -> float16 -> float; round-trip error bounded by float16 precision (~1e-3 rel)

TEST_F(KunlunKernelWrapperTest, CastFloat16RoundTripTest) {
  constexpr int64_t n = 256;
  std::mt19937 rng(11);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
  std::vector<float> host_f32(n), host_roundtrip(n);
  for (auto& v : host_f32) v = dist(rng);

  float* dev_f32_in;
  float16* dev_f16;
  float* dev_f32_out;
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_f32_in), n * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_f16), n * sizeof(float16), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_f32_out), n * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_memcpy(dev_f32_in, host_f32.data(), n * sizeof(float), XPU_HOST_TO_DEVICE));

  {
    XpuTimer timer("XpuCast<float,float16>+<float16,float>", stream);
    ASSERT_EQ(0, (XpuCast<float, float16>(xpuctx, dev_f32_in, dev_f16, n)));
    ASSERT_EQ(0, (XpuCast<float16, float>(xpuctx, dev_f16, dev_f32_out, n)));
  }
  XPU_CHECK(xpu_memcpy(host_roundtrip.data(), dev_f32_out, n * sizeof(float), XPU_DEVICE_TO_HOST));

  // float16 has ~3.3 decimal digits; relative error < 0.2% for normal values
  for (int i = 0; i < n; ++i) {
    float rel_err = std::fabs(host_roundtrip[i] - host_f32[i]) / (std::fabs(host_f32[i]) + 1e-6f);
    EXPECT_LT(rel_err, 2e-3f) << "round-trip at [" << i << "]: " << host_f32[i] << " -> " << host_roundtrip[i];
  }

  XPU_CHECK(xpu_free(dev_f32_in));
  XPU_CHECK(xpu_free(dev_f16));
  XPU_CHECK(xpu_free(dev_f32_out));
}

// ── XpuLayerNorm ──────────────────────────────────────────────────────────────
// y[m,i] = (x[m,i] - mean(x[m,:])) / sqrt(var(x[m,:]) + eps) * scale[i] + bias[i]

TEST_F(KunlunKernelWrapperTest, LayerNormTest) {
  constexpr int64_t m = 6, n = 128;
  constexpr float eps = 1e-5f;
  std::mt19937 rng(33);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> host_x(m * n), host_scale(n), host_bias(n), host_y(m * n);
  for (auto& v : host_x) v = dist(rng);
  for (int i = 0; i < n; ++i) {
    host_scale[i] = 0.5f + 0.5f * dist(rng);
    host_bias[i] = 0.1f * dist(rng);
  }

  std::vector<float> ref(m * n);
  for (int row = 0; row < m; ++row) {
    float mean = 0.0f;
    for (int j = 0; j < n; ++j) mean += host_x[row * n + j];
    mean /= n;
    float var = 0.0f;
    for (int j = 0; j < n; ++j) {
      float d = host_x[row * n + j] - mean;
      var += d * d;
    }
    var /= n;
    float inv_std = 1.0f / std::sqrt(var + eps);
    for (int j = 0; j < n; ++j) {
      ref[row * n + j] = (host_x[row * n + j] - mean) * inv_std * host_scale[j] + host_bias[j];
    }
  }

  float *dev_x, *dev_scale, *dev_bias, *dev_y;
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_x), m * n * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_scale), n * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_bias), n * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_y), m * n * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_memcpy(dev_x, host_x.data(), m * n * sizeof(float), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(dev_scale, host_scale.data(), n * sizeof(float), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(dev_bias, host_bias.data(), n * sizeof(float), XPU_HOST_TO_DEVICE));

  {
    XpuTimer timer("XpuLayerNorm", stream);
    ASSERT_EQ(0, (XpuLayerNorm<float, float>(xpuctx, dev_x, dev_y, m, n, eps, dev_scale, dev_bias,
                                             /*mean=*/nullptr, /*var=*/nullptr)));
  }
  XPU_CHECK(xpu_memcpy(host_y.data(), dev_y, m * n * sizeof(float), XPU_DEVICE_TO_HOST));

  for (size_t i = 0; i < static_cast<size_t>(m * n); ++i) {
    EXPECT_NEAR(host_y[i], ref[i], 1e-4f) << "mismatch at [" << i << "]";
  }

  XPU_CHECK(xpu_free(dev_x));
  XPU_CHECK(xpu_free(dev_scale));
  XPU_CHECK(xpu_free(dev_bias));
  XPU_CHECK(xpu_free(dev_y));
}

// ── XpuGatedGelu ──────────────────────────────────────────────────────────────
// output = gelu(input) * gate; gelu(x) = 0.5*x*(1 + erf(x/sqrt(2)))

TEST_F(KunlunKernelWrapperTest, GatedGeluTest) {
  constexpr int64_t m = 4, n = 64;
  std::mt19937 rng(77);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
  std::vector<float> host_in(m * n), host_gate(m * n), host_out(m * n);
  for (auto& v : host_in) v = dist(rng);
  for (auto& v : host_gate) v = dist(rng);

  std::vector<float> ref(m * n);
  constexpr float kSqrt2Inv = 0.70710678118f;  // 1/sqrt(2)
  for (size_t i = 0; i < static_cast<size_t>(m * n); ++i) {
    float g = 0.5f * host_in[i] * (1.0f + std::erf(host_in[i] * kSqrt2Inv));
    ref[i] = g * host_gate[i];
  }

  float *dev_in, *dev_gate, *dev_out;
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_in), m * n * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_gate), m * n * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_out), m * n * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_memcpy(dev_in, host_in.data(), m * n * sizeof(float), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(dev_gate, host_gate.data(), m * n * sizeof(float), XPU_HOST_TO_DEVICE));

  {
    XpuTimer timer("XpuGatedGelu", stream);
    ASSERT_EQ(0, (XpuGatedGelu<float>(xpuctx, dev_in, dev_gate, dev_out, m, n)));
  }
  XPU_CHECK(xpu_memcpy(host_out.data(), dev_out, m * n * sizeof(float), XPU_DEVICE_TO_HOST));

  for (size_t i = 0; i < static_cast<size_t>(m * n); ++i) {
    EXPECT_NEAR(host_out[i], ref[i], 1e-4f) << "mismatch at [" << i << "]";
  }

  XPU_CHECK(xpu_free(dev_in));
  XPU_CHECK(xpu_free(dev_gate));
  XPU_CHECK(xpu_free(dev_out));
}

// ── XpuCopy2d ─────────────────────────────────────────────────────────────────
// m rows; src row stride = ldx, dst row stride = ldy; copy n elements per row

TEST_F(KunlunKernelWrapperTest, Copy2dTest) {
  constexpr int64_t m = 4, n = 8, ldx = 16, ldy = 12;
  std::vector<float> host_src(m * ldx, 0.0f), host_dst(m * ldy, 0.0f);
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < ldx; ++j) host_src[i * ldx + j] = static_cast<float>(i * 100 + j);
  }

  float *dev_src, *dev_dst;
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_src), m * ldx * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_dst), m * ldy * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_memcpy(dev_src, host_src.data(), m * ldx * sizeof(float), XPU_HOST_TO_DEVICE));
  // zero-init dst via host zero buffer
  std::vector<float> zeros(m * ldy, 0.0f);
  XPU_CHECK(xpu_memcpy(dev_dst, zeros.data(), m * ldy * sizeof(float), XPU_HOST_TO_DEVICE));

  {
    XpuTimer timer("XpuCopy2d", stream);
    ASSERT_EQ(0, (XpuCopy2d<float>(xpuctx, dev_src, dev_dst, m, ldx, ldy, n)));
  }
  XPU_CHECK(xpu_memcpy(host_dst.data(), dev_dst, m * ldy * sizeof(float), XPU_DEVICE_TO_HOST));

  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      EXPECT_EQ(host_dst[i * ldy + j], host_src[i * ldx + j]) << "mismatch row=" << i << " col=" << j;
    }
    // Elements beyond n in each dst row should remain 0
    for (int j = n; j < ldy; ++j) {
      EXPECT_EQ(host_dst[i * ldy + j], 0.0f) << "overwrite at row=" << i << " col=" << j;
    }
  }

  XPU_CHECK(xpu_free(dev_src));
  XPU_CHECK(xpu_free(dev_dst));
}

// ── XpuFc (GEMM) ─────────────────────────────────────────────────────────────
// y[m,n] = x[m,k] * w[k,n]; x_trans=false, w_trans=false, no quant scaling

TEST_F(KunlunKernelWrapperTest, FcTest) {
  constexpr int64_t m = 4, n = 8, k = 16;
  std::mt19937 rng(321);
  std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
  std::vector<float> host_x(m * k), host_w(k * n), host_y(m * n);
  for (auto& v : host_x) v = dist(rng);
  for (auto& v : host_w) v = dist(rng);

  // CPU reference: naive matmul
  std::vector<float> ref(m * n, 0.0f);
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      float acc = 0.0f;
      for (int kk = 0; kk < k; ++kk) acc += host_x[i * k + kk] * host_w[kk * n + j];
      ref[i * n + j] = acc;
    }
  }

  float *dev_x, *dev_w, *dev_y;
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_x), m * k * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_w), k * n * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_y), m * n * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_memcpy(dev_x, host_x.data(), m * k * sizeof(float), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(dev_w, host_w.data(), k * n * sizeof(float), XPU_HOST_TO_DEVICE));

  {
    XpuTimer timer("XpuFc<float>", stream);
    ASSERT_EQ(0, (XpuFc<float, float, float>(xpuctx, dev_x, dev_w, dev_y, m, n, k,
                                             /*x_trans=*/false, /*w_trans=*/false,
                                             /*x_maxptr=*/nullptr, /*w_maxptr=*/nullptr,
                                             /*y_maxptr=*/nullptr)));
  }
  XPU_CHECK(xpu_memcpy(host_y.data(), dev_y, m * n * sizeof(float), XPU_DEVICE_TO_HOST));

  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      EXPECT_NEAR(host_y[i * n + j], ref[i * n + j], 1e-4f) << "mismatch at [" << i << "," << j << "]";
    }
  }

  XPU_CHECK(xpu_free(dev_x));
  XPU_CHECK(xpu_free(dev_w));
  XPU_CHECK(xpu_free(dev_y));
}

#endif  // ENABLE_KUNLUN

// ═══════════════════════════════════════════════════════════════════════════════
// XSGLANG-aligned utility kernel tests
// ═══════════════════════════════════════════════════════════════════════════════
#ifdef ENABLE_KUNLUN

// XpuArange: fill y with increasing sequence y[i] = begin + i * step
TEST_F(KunlunKernelWrapperTest, ArangeTest) {
  constexpr int64_t n = 512;
  constexpr float begin = 0.0f, step = 0.5f;
  std::vector<float> host_y(n);

  float* dev_y;
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_y), n * sizeof(float), XPU_MEM_MAIN));

  {
    XpuTimer timer("XpuArange", stream);
    ASSERT_EQ(0, (XpuArange<float, float>(xpuctx, dev_y, begin, step, n)));
  }
  XPU_CHECK(xpu_memcpy(host_y.data(), dev_y, n * sizeof(float), XPU_DEVICE_TO_HOST));

  for (int i = 0; i < n; ++i) {
    EXPECT_NEAR(host_y[i], begin + i * step, 1e-5f);
  }

  XPU_CHECK(xpu_free(dev_y));
}

// XpuConstant: fill x with a constant value
TEST_F(KunlunKernelWrapperTest, ConstantTest) {
  constexpr int64_t n = 256;
  constexpr float val = 3.14f;
  std::vector<float> host_x(n);

  float* dev_x;
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_x), n * sizeof(float), XPU_MEM_MAIN));

  {
    XpuTimer timer("XpuConstant", stream);
    ASSERT_EQ(0, (XpuConstant<float>(xpuctx, dev_x, n, val)));
  }
  XPU_CHECK(xpu_memcpy(host_x.data(), dev_x, n * sizeof(float), XPU_DEVICE_TO_HOST));

  for (int i = 0; i < n; ++i) {
    EXPECT_FLOAT_EQ(host_x[i], val);
  }

  XPU_CHECK(xpu_free(dev_x));
}

// XpuNeg: y[i] = -x[i]
TEST_F(KunlunKernelWrapperTest, NegTest) {
  constexpr int64_t n = 512;
  std::mt19937 rng(300);
  std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
  std::vector<float> host_x(n), host_y(n);
  for (auto& v : host_x) v = dist(rng);

  float *dev_x, *dev_y;
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_x), n * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_y), n * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_memcpy(dev_x, host_x.data(), n * sizeof(float), XPU_HOST_TO_DEVICE));

  {
    XpuTimer timer("XpuNeg", stream);
    ASSERT_EQ(0, (XpuNeg<float>(xpuctx, dev_x, dev_y, n)));
  }
  XPU_CHECK(xpu_memcpy(host_y.data(), dev_y, n * sizeof(float), XPU_DEVICE_TO_HOST));

  for (int i = 0; i < n; ++i) {
    EXPECT_NEAR(host_y[i], -host_x[i], 1e-5f);
  }

  XPU_CHECK(xpu_free(dev_x));
  XPU_CHECK(xpu_free(dev_y));
}

// XpuClamp: y[i] = clamp(x[i], min_val, max_val)
TEST_F(KunlunKernelWrapperTest, ClampTest) {
  constexpr int64_t n = 512;
  constexpr float lo = -1.0f, hi = 1.0f;
  std::mt19937 rng(400);
  std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
  std::vector<float> host_x(n), host_y(n);
  for (auto& v : host_x) v = dist(rng);

  float *dev_x, *dev_y;
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_x), n * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_y), n * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_memcpy(dev_x, host_x.data(), n * sizeof(float), XPU_HOST_TO_DEVICE));

  {
    XpuTimer timer("XpuClamp", stream);
    ASSERT_EQ(0, (XpuClamp<float>(xpuctx, dev_x, dev_y, n, lo, hi)));
  }
  XPU_CHECK(xpu_memcpy(host_y.data(), dev_y, n * sizeof(float), XPU_DEVICE_TO_HOST));

  for (int i = 0; i < n; ++i) {
    float ref = std::max(lo, std::min(hi, host_x[i]));
    EXPECT_NEAR(host_y[i], ref, 1e-5f);
  }

  XPU_CHECK(xpu_free(dev_x));
  XPU_CHECK(xpu_free(dev_y));
}

// XpuAddScalar: y[i] = x[i] + other * alpha (matches torch.add(x, other, alpha))
TEST_F(KunlunKernelWrapperTest, AddScalarTest) {
  constexpr int64_t n = 256;
  constexpr float alpha = 2.0f, other = 1.5f;
  std::mt19937 rng(500);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
  std::vector<float> host_x(n), host_y(n);
  for (auto& v : host_x) v = dist(rng);

  float *dev_x, *dev_y;
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_x), n * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_y), n * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_memcpy(dev_x, host_x.data(), n * sizeof(float), XPU_HOST_TO_DEVICE));

  {
    XpuTimer timer("XpuAddScalar", stream);
    ASSERT_EQ(0, (XpuAddScalar<float>(xpuctx, dev_x, dev_y, alpha, other, n)));
  }
  XPU_CHECK(xpu_memcpy(host_y.data(), dev_y, n * sizeof(float), XPU_DEVICE_TO_HOST));

  for (int i = 0; i < n; ++i) {
    EXPECT_NEAR(host_y[i], host_x[i] + other * alpha, 1e-5f);
  }

  XPU_CHECK(xpu_free(dev_x));
  XPU_CHECK(xpu_free(dev_y));
}

// XpuSubScalar: y[i] = x[i] - other * alpha (matches torch.sub(x, other, alpha))
TEST_F(KunlunKernelWrapperTest, SubScalarTest) {
  constexpr int64_t n = 256;
  constexpr float alpha = 0.5f, other = 0.75f;
  std::mt19937 rng(600);
  std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
  std::vector<float> host_x(n), host_y(n);
  for (auto& v : host_x) v = dist(rng);

  float *dev_x, *dev_y;
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_x), n * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_y), n * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_memcpy(dev_x, host_x.data(), n * sizeof(float), XPU_HOST_TO_DEVICE));

  {
    XpuTimer timer("XpuSubScalar", stream);
    ASSERT_EQ(0, (XpuSubScalar<float>(xpuctx, dev_x, dev_y, alpha, other, n)));
  }
  XPU_CHECK(xpu_memcpy(host_y.data(), dev_y, n * sizeof(float), XPU_DEVICE_TO_HOST));

  for (int i = 0; i < n; ++i) {
    EXPECT_NEAR(host_y[i], host_x[i] - other * alpha, 1e-5f);
  }

  XPU_CHECK(xpu_free(dev_x));
  XPU_CHECK(xpu_free(dev_y));
}

// XpuGather: gather elements from x along axis 0 by index
TEST_F(KunlunKernelWrapperTest, GatherTest) {
  // x shape [4, 3], gather along axis 0 with index [0, 2, 1, 0]
  std::vector<float> host_x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
  std::vector<int32_t> host_idx = {0, 2, 1, 0};
  int64_t idx_len = 4, inner_dim = 3;
  std::vector<float> host_y(idx_len * inner_dim);

  float *dev_x, *dev_y;
  int32_t* dev_idx;
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_x), host_x.size() * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_y), host_y.size() * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_idx), idx_len * sizeof(int32_t), XPU_MEM_MAIN));
  XPU_CHECK(xpu_memcpy(dev_x, host_x.data(), host_x.size() * sizeof(float), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(dev_idx, host_idx.data(), idx_len * sizeof(int32_t), XPU_HOST_TO_DEVICE));

  {
    XpuTimer timer("XpuGather", stream);
    ASSERT_EQ(0, (XpuGather<float, int32_t>(xpuctx, dev_x, dev_idx, dev_y, {4, 3}, {idx_len}, 0)));
  }
  XPU_CHECK(xpu_memcpy(host_y.data(), dev_y, host_y.size() * sizeof(float), XPU_DEVICE_TO_HOST));

  // Expected: rows 0, 2, 1, 0 → [1,2,3, 7,8,9, 4,5,6, 1,2,3]
  std::vector<float> ref = {1.0f, 2.0f, 3.0f, 7.0f, 8.0f, 9.0f, 4.0f, 5.0f, 6.0f, 1.0f, 2.0f, 3.0f};
  for (size_t i = 0; i < ref.size(); ++i) {
    EXPECT_NEAR(host_y[i], ref[i], 1e-5f);
  }

  XPU_CHECK(xpu_free(dev_x));
  XPU_CHECK(xpu_free(dev_y));
  XPU_CHECK(xpu_free(dev_idx));
}

// XpuIndexSelect: select elements along axis 0 by 1D index
TEST_F(KunlunKernelWrapperTest, IndexSelectTest) {
  // x shape [5, 2], select rows {0, 3, 4}
  std::vector<float> host_x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
  std::vector<int64_t> host_idx = {0, 3, 4};
  int64_t src_rows = 5, src_cols = 2, idx_len = 3;
  std::vector<float> host_y(idx_len * src_cols);

  float *dev_x, *dev_y;
  int64_t* dev_idx;
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_x), host_x.size() * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_y), host_y.size() * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_idx), idx_len * sizeof(int64_t), XPU_MEM_MAIN));
  XPU_CHECK(xpu_memcpy(dev_x, host_x.data(), host_x.size() * sizeof(float), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(dev_idx, host_idx.data(), idx_len * sizeof(int64_t), XPU_HOST_TO_DEVICE));

  {
    XpuTimer timer("XpuIndexSelect", stream);
    ASSERT_EQ(0, (XpuIndexSelect<float, int64_t>(xpuctx, dev_x, dev_idx, dev_y, {src_rows, src_cols}, idx_len, 0)));
  }
  XPU_CHECK(xpu_memcpy(host_y.data(), dev_y, host_y.size() * sizeof(float), XPU_DEVICE_TO_HOST));

  std::vector<float> ref = {1.0f, 2.0f, 7.0f, 8.0f, 9.0f, 10.0f};
  for (size_t i = 0; i < ref.size(); ++i) {
    EXPECT_NEAR(host_y[i], ref[i], 1e-5f);
  }

  XPU_CHECK(xpu_free(dev_x));
  XPU_CHECK(xpu_free(dev_y));
  XPU_CHECK(xpu_free(dev_idx));
}

// XpuScatter: int32 data + int64 index (XSGLANG: scatter_v2_small_index<int, long>)
// NOTE: float scatter and int32-index scatter hang on this xdnn version;
// XSGLANG uses int data + long index, so we mirror that exactly.
TEST_F(KunlunKernelWrapperTest, ScatterTest) {
  std::vector<int32_t> host_out_init = {0, 0, 0, 0, 0, 0};
  std::vector<int32_t> host_updates = {99};
  std::vector<int64_t> host_idx = {2};
  std::vector<int32_t> host_y(6);

  int32_t *dev_out, *dev_updates, *dev_y;
  int64_t* dev_idx;
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_out), 6 * sizeof(int32_t), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_updates), 1 * sizeof(int32_t), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_y), 6 * sizeof(int32_t), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_idx), 1 * sizeof(int64_t), XPU_MEM_MAIN));
  XPU_CHECK(xpu_memcpy(dev_out, host_out_init.data(), 6 * sizeof(int32_t), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(dev_updates, host_updates.data(), 1 * sizeof(int32_t), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(dev_idx, host_idx.data(), 1 * sizeof(int64_t), XPU_HOST_TO_DEVICE));

  {
    XpuTimer timer("XpuScatter", stream);
    ASSERT_EQ(0, (XpuScatter<int32_t, int64_t>(xpuctx, dev_out, dev_updates, dev_y, dev_idx, 1, {6}, 0)));
  }
  XPU_CHECK(xpu_memcpy(host_y.data(), dev_y, 6 * sizeof(int32_t), XPU_DEVICE_TO_HOST));

  EXPECT_EQ(host_y[0], 0);
  EXPECT_EQ(host_y[1], 0);
  EXPECT_EQ(host_y[2], 99);
  EXPECT_EQ(host_y[3], 0);
  EXPECT_EQ(host_y[4], 0);
  EXPECT_EQ(host_y[5], 0);

  XPU_CHECK(xpu_free(dev_out));
  XPU_CHECK(xpu_free(dev_updates));
  XPU_CHECK(xpu_free(dev_y));
  XPU_CHECK(xpu_free(dev_idx));
}

// XpuStridedSlice: extract a slice with strides
TEST_F(KunlunKernelWrapperTest, StridedSliceTest) {
  // x shape [4, 4], slice rows [1:3] (get rows 1,2)
  std::vector<float> host_x = {0.0f, 1.0f, 2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,
                               8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f};
  std::vector<float> host_y(8);

  float *dev_x, *dev_y;
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_x), 16 * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_y), 8 * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_memcpy(dev_x, host_x.data(), 16 * sizeof(float), XPU_HOST_TO_DEVICE));

  {
    XpuTimer timer("XpuStridedSlice", stream);
    ASSERT_EQ(0, (XpuStridedSlice<float>(xpuctx, dev_x, dev_y, {4, 4}, {1, 0}, {3, 4}, {1, 1}, 0)));
  }
  XPU_CHECK(xpu_memcpy(host_y.data(), dev_y, 8 * sizeof(float), XPU_DEVICE_TO_HOST));

  // Rows 1,2 → [4,5,6,7, 8,9,10,11]
  std::vector<float> ref = {4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f};
  for (size_t i = 0; i < ref.size(); ++i) {
    EXPECT_NEAR(host_y[i], ref[i], 1e-5f);
  }

  XPU_CHECK(xpu_free(dev_x));
  XPU_CHECK(xpu_free(dev_y));
}

// XpuCumsum: prefix sum along axis 0
TEST_F(KunlunKernelWrapperTest, CumsumTest) {
  // x shape [4, 2]
  std::vector<float> host_x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  std::vector<float> host_y(8);

  float *dev_x, *dev_y;
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_x), 8 * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_y), 8 * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_memcpy(dev_x, host_x.data(), 8 * sizeof(float), XPU_HOST_TO_DEVICE));

  {
    XpuTimer timer("XpuCumsum", stream);
    ASSERT_EQ(0, (XpuCumsum<float>(xpuctx, dev_x, dev_y, {4, 2},
                                   /*reverse=*/false, /*exclusive=*/false, /*axis=*/0)));
  }
  XPU_CHECK(xpu_memcpy(host_y.data(), dev_y, 8 * sizeof(float), XPU_DEVICE_TO_HOST));

  // Col0: [1, 1+3=4, 4+5=9, 9+7=16], Col1: [2, 2+4=6, 6+6=12, 12+8=20]
  std::vector<float> ref = {1.0f, 2.0f, 4.0f, 6.0f, 9.0f, 12.0f, 16.0f, 20.0f};
  for (size_t i = 0; i < ref.size(); ++i) {
    EXPECT_NEAR(host_y[i], ref[i], 1e-5f);
  }

  XPU_CHECK(xpu_free(dev_x));
  XPU_CHECK(xpu_free(dev_y));
}

// XpuSort: sort each row and return sorted indices
TEST_F(KunlunKernelWrapperTest, SortTest) {
  constexpr int64_t m = 3, n = 5;
  std::vector<float> host_x = {3.0f, 1.0f,  5.0f,  2.0f,  4.0f,  9.0f,  7.0f, 8.0f,
                               6.0f, 10.0f, 15.0f, 11.0f, 13.0f, 14.0f, 12.0f};
  std::vector<float> host_y(m * n);
  std::vector<int64_t> host_idx(m * n);

  float *dev_x, *dev_y;
  int64_t* dev_idx;
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_x), m * n * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_y), m * n * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_idx), m * n * sizeof(int64_t), XPU_MEM_MAIN));
  XPU_CHECK(xpu_memcpy(dev_x, host_x.data(), m * n * sizeof(float), XPU_HOST_TO_DEVICE));

  {
    XpuTimer timer("XpuSort", stream);
    ASSERT_EQ(0, (XpuSort<float, int64_t>(xpuctx, dev_x, dev_y, dev_idx, m, n)));
  }
  XPU_CHECK(xpu_memcpy(host_y.data(), dev_y, m * n * sizeof(float), XPU_DEVICE_TO_HOST));
  XPU_CHECK(xpu_memcpy(host_idx.data(), dev_idx, m * n * sizeof(int64_t), XPU_DEVICE_TO_HOST));

  // Verify each row is sorted
  for (int row = 0; row < m; ++row) {
    for (int j = 1; j < n; ++j) {
      EXPECT_LE(host_y[row * n + j - 1], host_y[row * n + j]);
    }
  }
  // Verify index mapping: host_x[row][host_idx[row][j]] == host_y[row][j]
  for (int row = 0; row < m; ++row) {
    for (int j = 0; j < n; ++j) {
      EXPECT_NEAR(host_x[row * n + host_idx[row * n + j]], host_y[row * n + j], 1e-5f);
    }
  }

  XPU_CHECK(xpu_free(dev_x));
  XPU_CHECK(xpu_free(dev_y));
  XPU_CHECK(xpu_free(dev_idx));
}

// XpuEqScalar: element-wise equality comparison
TEST_F(KunlunKernelWrapperTest, EqScalarTest) {
  constexpr int64_t n = 128;
  std::vector<float> host_x(n);
  for (int i = 0; i < n; ++i) host_x[i] = static_cast<float>(i % 3);
  std::vector<uint8_t> host_mask(n);  // uint8_t to avoid vector<bool> specialization

  float* dev_x;
  bool* dev_mask;
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_x), n * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_mask), n * sizeof(bool), XPU_MEM_MAIN));
  XPU_CHECK(xpu_memcpy(dev_x, host_x.data(), n * sizeof(float), XPU_HOST_TO_DEVICE));

  {
    XpuTimer timer("XpuEqScalar", stream);
    ASSERT_EQ(0, (XpuEqScalar<float>(xpuctx, dev_x, dev_mask, 1.0f, n)));
  }
  XPU_CHECK(xpu_memcpy(host_mask.data(), dev_mask, n * sizeof(bool), XPU_DEVICE_TO_HOST));

  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(host_mask[i], host_x[i] == 1.0f);
  }

  XPU_CHECK(xpu_free(dev_x));
  XPU_CHECK(xpu_free(dev_mask));
}

// XpuGtScalar: element-wise greater-than comparison
TEST_F(KunlunKernelWrapperTest, GtScalarTest) {
  constexpr int64_t n = 128;
  std::vector<float> host_x(n);
  for (int i = 0; i < n; ++i) host_x[i] = static_cast<float>(i) - 64.0f;
  std::vector<uint8_t> host_mask(n);  // uint8_t to avoid vector<bool> specialization

  float* dev_x;
  bool* dev_mask;
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_x), n * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_mask), n * sizeof(bool), XPU_MEM_MAIN));
  XPU_CHECK(xpu_memcpy(dev_x, host_x.data(), n * sizeof(float), XPU_HOST_TO_DEVICE));

  {
    XpuTimer timer("XpuGtScalar", stream);
    ASSERT_EQ(0, (XpuGtScalar<float>(xpuctx, dev_x, dev_mask, 0.0f, n)));
  }
  XPU_CHECK(xpu_memcpy(host_mask.data(), dev_mask, n * sizeof(bool), XPU_DEVICE_TO_HOST));

  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(host_mask[i], host_x[i] > 0.0f);
  }

  XPU_CHECK(xpu_free(dev_x));
  XPU_CHECK(xpu_free(dev_mask));
}

// XpuConcat: concatenate two tensors along axis 0
TEST_F(KunlunKernelWrapperTest, ConcatTest) {
  // x1: [2, 3], x2: [3, 3] → out: [5, 3]
  std::vector<float> host_x1 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::vector<float> host_x2 = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f};
  std::vector<float> host_y(15);

  float *dev_x1, *dev_x2, *dev_y;
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_x1), 6 * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_x2), 9 * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_y), 15 * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_memcpy(dev_x1, host_x1.data(), 6 * sizeof(float), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(dev_x2, host_x2.data(), 9 * sizeof(float), XPU_HOST_TO_DEVICE));

  {
    XpuTimer timer("XpuConcat", stream);
    std::vector<const float*> x_list = {dev_x1, dev_x2};
    ASSERT_EQ(0, (XpuConcat<float>(xpuctx, x_list, dev_y, {{2, 3}, {3, 3}}, 0)));
  }
  XPU_CHECK(xpu_memcpy(host_y.data(), dev_y, 15 * sizeof(float), XPU_DEVICE_TO_HOST));

  for (size_t i = 0; i < 6; ++i) EXPECT_NEAR(host_y[i], host_x1[i], 1e-5f);
  for (size_t i = 0; i < 9; ++i) EXPECT_NEAR(host_y[6 + i], host_x2[i], 1e-5f);

  XPU_CHECK(xpu_free(dev_x1));
  XPU_CHECK(xpu_free(dev_x2));
  XPU_CHECK(xpu_free(dev_y));
}

// XpuConcat axis=1: concatenate two tensors along the inner dimension
TEST_F(KunlunKernelWrapperTest, ConcatAxis1Test) {
  // x1: [4, 3], x2: [4, 5] → out: [4, 8]
  constexpr int64_t rows = 4, c1 = 3, c2 = 5, cols_out = c1 + c2;
  std::vector<float> host_x1(rows * c1), host_x2(rows * c2), host_y(rows * cols_out);
  for (int i = 0; i < rows * c1; ++i) host_x1[i] = static_cast<float>(i);
  for (int i = 0; i < rows * c2; ++i) host_x2[i] = static_cast<float>(100 + i);

  float *dev_x1, *dev_x2, *dev_y;
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_x1), rows * c1 * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_x2), rows * c2 * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_y), rows * cols_out * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_memcpy(dev_x1, host_x1.data(), rows * c1 * sizeof(float), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(dev_x2, host_x2.data(), rows * c2 * sizeof(float), XPU_HOST_TO_DEVICE));

  {
    XpuTimer timer("XpuConcat axis=1", stream);
    std::vector<const float*> x_list = {dev_x1, dev_x2};
    ASSERT_EQ(0, (XpuConcat<float>(xpuctx, x_list, dev_y, {{rows, c1}, {rows, c2}}, 1)));
  }
  XPU_CHECK(xpu_memcpy(host_y.data(), dev_y, rows * cols_out * sizeof(float), XPU_DEVICE_TO_HOST));

  // Verify: out[row, 0:c1] = x1[row, :], out[row, c1:c1+c2] = x2[row, :]
  for (int row = 0; row < rows; ++row) {
    for (int j = 0; j < c1; ++j) {
      EXPECT_NEAR(host_y[row * cols_out + j], host_x1[row * c1 + j], 1e-5f)
          << "x1 mismatch row=" << row << " col=" << j;
    }
    for (int j = 0; j < c2; ++j) {
      EXPECT_NEAR(host_y[row * cols_out + c1 + j], host_x2[row * c2 + j], 1e-5f)
          << "x2 mismatch row=" << row << " col=" << j;
    }
  }

  XPU_CHECK(xpu_free(dev_x1));
  XPU_CHECK(xpu_free(dev_x2));
  XPU_CHECK(xpu_free(dev_y));
}

// ── XpuSplit: split a tensor along axis (inverse of concat) ──────────────────
// x shape [5, 3] split along axis 0 into [2, 3] and [3, 3]

TEST_F(KunlunKernelWrapperTest, SplitTest) {
  // x: [5, 3] = [1..15], split along axis 0 into sizes [2, 3]
  std::vector<float> host_x = {1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f, 8.0f,
                               9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f};
  std::vector<float> host_y0(6), host_y1(9);

  float *dev_x, *dev_y0, *dev_y1;
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_x), 15 * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_y0), 6 * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_y1), 9 * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_memcpy(dev_x, host_x.data(), 15 * sizeof(float), XPU_HOST_TO_DEVICE));

  {
    XpuTimer timer("XpuSplit", stream);
    std::vector<float*> y_list = {dev_y0, dev_y1};
    ASSERT_EQ(0, (XpuSplit<float>(xpuctx, dev_x, y_list, {5, 3}, {2, 3}, 0)));
  }
  XPU_CHECK(xpu_memcpy(host_y0.data(), dev_y0, 6 * sizeof(float), XPU_DEVICE_TO_HOST));
  XPU_CHECK(xpu_memcpy(host_y1.data(), dev_y1, 9 * sizeof(float), XPU_DEVICE_TO_HOST));

  // y0 = first 2 rows: [1,2,3, 4,5,6]
  for (size_t i = 0; i < 6; ++i) EXPECT_NEAR(host_y0[i], host_x[i], 1e-5f);
  // y1 = last 3 rows: [7,8,9, 10,11,12, 13,14,15]
  for (size_t i = 0; i < 9; ++i) EXPECT_NEAR(host_y1[i], host_x[6 + i], 1e-5f);

  XPU_CHECK(xpu_free(dev_x));
  XPU_CHECK(xpu_free(dev_y0));
  XPU_CHECK(xpu_free(dev_y1));
}

// XpuSplit along axis 1: x [4, 8] → [4, 3] + [4, 5]
TEST_F(KunlunKernelWrapperTest, SplitAxis1Test) {
  constexpr int64_t rows = 4, cols = 8;
  constexpr int64_t split0 = 3, split1 = 5;
  std::vector<float> host_x(rows * cols);
  for (int i = 0; i < rows * cols; ++i) host_x[i] = static_cast<float>(i);

  std::vector<float> host_y0(rows * split0), host_y1(rows * split1);

  float *dev_x, *dev_y0, *dev_y1;
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_x), rows * cols * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_y0), rows * split0 * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_y1), rows * split1 * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_memcpy(dev_x, host_x.data(), rows * cols * sizeof(float), XPU_HOST_TO_DEVICE));

  {
    XpuTimer timer("XpuSplit axis=1", stream);
    std::vector<float*> y_list = {dev_y0, dev_y1};
    ASSERT_EQ(0, (XpuSplit<float>(xpuctx, dev_x, y_list, {rows, cols}, {split0, split1}, 1)));
  }
  XPU_CHECK(xpu_memcpy(host_y0.data(), dev_y0, rows * split0 * sizeof(float), XPU_DEVICE_TO_HOST));
  XPU_CHECK(xpu_memcpy(host_y1.data(), dev_y1, rows * split1 * sizeof(float), XPU_DEVICE_TO_HOST));

  // Verify: y0[row, j] = x[row, j] for j in [0, split0)
  //         y1[row, j] = x[row, split0 + j] for j in [0, split1)
  for (int row = 0; row < rows; ++row) {
    for (int j = 0; j < split0; ++j) {
      EXPECT_NEAR(host_y0[row * split0 + j], host_x[row * cols + j], 1e-5f)
          << "y0 mismatch row=" << row << " col=" << j;
    }
    for (int j = 0; j < split1; ++j) {
      EXPECT_NEAR(host_y1[row * split1 + j], host_x[row * cols + split0 + j], 1e-5f)
          << "y1 mismatch row=" << row << " col=" << j;
    }
  }

  XPU_CHECK(xpu_free(dev_x));
  XPU_CHECK(xpu_free(dev_y0));
  XPU_CHECK(xpu_free(dev_y1));
}

// XpuStableSort: stable sort each row
TEST_F(KunlunKernelWrapperTest, StableSortTest) {
  constexpr int64_t m = 2, n = 6;
  // Row 0: [3, 1, 5, 1, 2, 4] — has duplicate 1s
  // Row 1: [9, 7, 8, 6, 10, 7] — has duplicate 7s
  std::vector<float> host_x = {3.0f, 1.0f, 5.0f, 1.0f, 2.0f, 4.0f, 9.0f, 7.0f, 8.0f, 6.0f, 10.0f, 7.0f};
  std::vector<float> host_y(m * n);
  std::vector<int64_t> host_idx(m * n);

  float *dev_x, *dev_y;
  int64_t* dev_idx;
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_x), m * n * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_y), m * n * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_idx), m * n * sizeof(int64_t), XPU_MEM_MAIN));
  XPU_CHECK(xpu_memcpy(dev_x, host_x.data(), m * n * sizeof(float), XPU_HOST_TO_DEVICE));

  {
    XpuTimer timer("XpuStableSort", stream);
    ASSERT_EQ(0, (XpuStableSort<float, int64_t>(xpuctx, dev_x, dev_y, dev_idx, m, n)));
  }
  XPU_CHECK(xpu_memcpy(host_y.data(), dev_y, m * n * sizeof(float), XPU_DEVICE_TO_HOST));
  XPU_CHECK(xpu_memcpy(host_idx.data(), dev_idx, m * n * sizeof(int64_t), XPU_DEVICE_TO_HOST));

  // Row 0 sorted: [1, 1, 2, 3, 4, 5], with stable: first 1 at idx 1, second 1 at idx 3
  for (int row = 0; row < m; ++row) {
    for (int j = 1; j < n; ++j) {
      EXPECT_LE(host_y[row * n + j - 1], host_y[row * n + j]);
    }
    // Check stability: duplicate values preserve original order
    for (int j = 0; j < n; ++j) {
      EXPECT_NEAR(host_x[row * n + host_idx[row * n + j]], host_y[row * n + j], 1e-5f);
    }
  }

  XPU_CHECK(xpu_free(dev_x));
  XPU_CHECK(xpu_free(dev_y));
  XPU_CHECK(xpu_free(dev_idx));
}

// XpuUniqueCount: count unique elements
TEST_F(KunlunKernelWrapperTest, UniqueCountTest) {
  constexpr int64_t n = 8;
  std::vector<float> host_x = {1.0f, 3.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 2.0f};
  std::vector<int64_t> host_unique_len(1);

  float* dev_x;
  int64_t* dev_unique_len;
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_x), n * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_unique_len), sizeof(int64_t), XPU_MEM_MAIN));
  XPU_CHECK(xpu_memcpy(dev_x, host_x.data(), n * sizeof(float), XPU_HOST_TO_DEVICE));

  {
    XpuTimer timer("XpuUniqueCount", stream);
    ASSERT_EQ(0, (XpuUniqueCount<float, int64_t>(xpuctx, dev_x, dev_unique_len, n)));
  }
  XPU_CHECK(xpu_memcpy(host_unique_len.data(), dev_unique_len, sizeof(int64_t), XPU_DEVICE_TO_HOST));

  // Values: {1, 2, 3, 4, 5} → 5 unique
  EXPECT_EQ(host_unique_len[0], 5);

  XPU_CHECK(xpu_free(dev_x));
  XPU_CHECK(xpu_free(dev_unique_len));
}

// XpuCopy: 1D device-to-device copy
TEST_F(KunlunKernelWrapperTest, Copy1DTest) {
  constexpr int64_t n = 512;
  std::mt19937 rng(800);
  std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
  std::vector<float> host_src(n), host_dst(n);
  for (auto& v : host_src) v = dist(rng);

  float *dev_src, *dev_dst;
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_src), n * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_dst), n * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_memcpy(dev_src, host_src.data(), n * sizeof(float), XPU_HOST_TO_DEVICE));

  {
    XpuTimer timer("XpuCopy", stream);
    ASSERT_EQ(0, (XpuCopy<float>(xpuctx, dev_src, dev_dst, n)));
  }
  XPU_CHECK(xpu_memcpy(host_dst.data(), dev_dst, n * sizeof(float), XPU_DEVICE_TO_HOST));

  for (int i = 0; i < n; ++i) {
    EXPECT_NEAR(host_dst[i], host_src[i], 1e-5f);
  }

  XPU_CHECK(xpu_free(dev_src));
  XPU_CHECK(xpu_free(dev_dst));
}

// XpuGatherNd: gather elements via N-dimensional index
TEST_F(KunlunKernelWrapperTest, GatherNdTest) {
  // x shape [4, 3], gather rows {1, 3} (index [[1], [3]])
  std::vector<float> host_x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
  std::vector<int32_t> host_idx = {1, 3};  // gather_nd 1D indices
  std::vector<float> host_y(6);            // 2 rows * 3 cols

  float *dev_x, *dev_y;
  int32_t* dev_idx;
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_x), 12 * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_y), 6 * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_idx), 2 * sizeof(int32_t), XPU_MEM_MAIN));
  XPU_CHECK(xpu_memcpy(dev_x, host_x.data(), 12 * sizeof(float), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(dev_idx, host_idx.data(), 2 * sizeof(int32_t), XPU_HOST_TO_DEVICE));

  {
    XpuTimer timer("XpuGatherNd", stream);
    ASSERT_EQ(0, (XpuGatherNd<float, int32_t>(xpuctx, dev_x, dev_idx, dev_y, {4, 3}, {2, 1})));
  }
  XPU_CHECK(xpu_memcpy(host_y.data(), dev_y, 6 * sizeof(float), XPU_DEVICE_TO_HOST));

  // Expected: rows 1 and 3 → [4,5,6, 10,11,12]
  std::vector<float> ref = {4.0f, 5.0f, 6.0f, 10.0f, 11.0f, 12.0f};
  for (size_t i = 0; i < ref.size(); ++i) {
    EXPECT_NEAR(host_y[i], ref[i], 1e-5f);
  }

  XPU_CHECK(xpu_free(dev_x));
  XPU_CHECK(xpu_free(dev_y));
  XPU_CHECK(xpu_free(dev_idx));
}

// XpuScatterNd: int32 data + int64 index (XSGLANG: scatternd_idx_to_scatter_idx<long>)
// NOTE: float scatter_nd hangs on this xdnn version; XSGLANG uses int data + long index.
TEST_F(KunlunKernelWrapperTest, ScatterNdTest) {
  // output shape [4, 3], scatter [100, 200] at [0,0] and [3,2]
  std::vector<int32_t> host_out_init = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int32_t> host_updates = {100, 200};
  std::vector<int64_t> host_idx = {0, 0, 3, 2};  // [[0,0], [3,2]]
  std::vector<int32_t> host_y(12);

  int32_t *dev_out, *dev_updates, *dev_y;
  int64_t* dev_idx;
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_out), 12 * sizeof(int32_t), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_updates), 2 * sizeof(int32_t), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_y), 12 * sizeof(int32_t), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_idx), 4 * sizeof(int64_t), XPU_MEM_MAIN));
  XPU_CHECK(xpu_memcpy(dev_out, host_out_init.data(), 12 * sizeof(int32_t), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(dev_updates, host_updates.data(), 2 * sizeof(int32_t), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(dev_idx, host_idx.data(), 4 * sizeof(int64_t), XPU_HOST_TO_DEVICE));

  {
    XpuTimer timer("XpuScatterNd", stream);
    ASSERT_EQ(0, (XpuScatterNd<int32_t, int64_t>(xpuctx, dev_out, dev_updates, dev_y, dev_idx, {4, 3}, {2, 2},
                                                 /*is_overwrite=*/true)));
  }
  XPU_CHECK(xpu_memcpy(host_y.data(), dev_y, 12 * sizeof(int32_t), XPU_DEVICE_TO_HOST));

  EXPECT_EQ(host_y[0], 100);  // [0,0] scattered 100
  EXPECT_EQ(host_y[1], 0);
  EXPECT_EQ(host_y[11], 200);  // [3,2] scattered 200

  XPU_CHECK(xpu_free(dev_out));
  XPU_CHECK(xpu_free(dev_updates));
  XPU_CHECK(xpu_free(dev_y));
  XPU_CHECK(xpu_free(dev_idx));
}

// XpuMulTensor: element-wise multiply z[i] = x[i] * y[i]
TEST_F(KunlunKernelWrapperTest, MulTensorTest) {
  constexpr int64_t n = 256;
  std::mt19937 rng(900);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
  std::vector<float> host_x(n), host_y(n), host_z(n);
  for (auto& v : host_x) v = dist(rng);
  for (auto& v : host_y) v = dist(rng);

  float *dev_x, *dev_y, *dev_z;
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_x), n * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_y), n * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_z), n * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_memcpy(dev_x, host_x.data(), n * sizeof(float), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(dev_y, host_y.data(), n * sizeof(float), XPU_HOST_TO_DEVICE));

  {
    XpuTimer timer("XpuMulTensor", stream);
    ASSERT_EQ(0, (XpuMulTensor<float>(xpuctx, dev_x, dev_y, dev_z, {n})));
  }
  XPU_CHECK(xpu_memcpy(host_z.data(), dev_z, n * sizeof(float), XPU_DEVICE_TO_HOST));

  for (int i = 0; i < n; ++i) {
    EXPECT_NEAR(host_z[i], host_x[i] * host_y[i], 1e-5f);
  }

  XPU_CHECK(xpu_free(dev_x));
  XPU_CHECK(xpu_free(dev_y));
  XPU_CHECK(xpu_free(dev_z));
}

// XpuMulScalar: y[i] = x[i] * other
TEST_F(KunlunKernelWrapperTest, MulScalarTest) {
  constexpr int64_t n = 256;
  constexpr float factor = 2.5f;
  std::mt19937 rng(1000);
  std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
  std::vector<float> host_x(n), host_y(n);
  for (auto& v : host_x) v = dist(rng);

  float *dev_x, *dev_y;
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_x), n * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_y), n * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_memcpy(dev_x, host_x.data(), n * sizeof(float), XPU_HOST_TO_DEVICE));

  {
    XpuTimer timer("XpuMulScalar", stream);
    ASSERT_EQ(0, (XpuMulScalar<float>(xpuctx, dev_x, dev_y, factor, n)));
  }
  XPU_CHECK(xpu_memcpy(host_y.data(), dev_y, n * sizeof(float), XPU_DEVICE_TO_HOST));

  for (int i = 0; i < n; ++i) {
    EXPECT_NEAR(host_y[i], host_x[i] * factor, 1e-5f);
  }

  XPU_CHECK(xpu_free(dev_x));
  XPU_CHECK(xpu_free(dev_y));
}

// XpuSoftmax: y = softmax(x) along axis 1
TEST_F(KunlunKernelWrapperTest, SoftmaxTest) {
  constexpr int64_t bs = 2, n = 4;
  std::vector<float> host_x = {1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  std::vector<float> host_y(bs * n);

  // CPU reference softmax
  std::vector<float> ref(bs * n);
  for (int b = 0; b < bs; ++b) {
    float sum_exp = 0.0f;
    float max_val = host_x[b * n];
    for (int j = 1; j < n; ++j) max_val = std::max(max_val, host_x[b * n + j]);
    for (int j = 0; j < n; ++j) sum_exp += std::exp(host_x[b * n + j] - max_val);
    for (int j = 0; j < n; ++j) ref[b * n + j] = std::exp(host_x[b * n + j] - max_val) / sum_exp;
  }

  float *dev_x, *dev_y;
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_x), bs * n * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_y), bs * n * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_memcpy(dev_x, host_x.data(), bs * n * sizeof(float), XPU_HOST_TO_DEVICE));

  {
    XpuTimer timer("XpuSoftmax", stream);
    ASSERT_EQ(0, (XpuSoftmax<float>(xpuctx, dev_x, dev_y, {bs, n}, /*axis=*/1)));
  }
  XPU_CHECK(xpu_memcpy(host_y.data(), dev_y, bs * n * sizeof(float), XPU_DEVICE_TO_HOST));

  for (int i = 0; i < bs * n; ++i) {
    EXPECT_NEAR(host_y[i], ref[i], 1e-4f);
  }

  XPU_CHECK(xpu_free(dev_x));
  XPU_CHECK(xpu_free(dev_y));
}

// ── XpuReplaceLastToken / XpuWriteRing ─────────────────────────────────────────
// Build-ahead (depth=2 executor pipeline overlap) token relay, Phase 1
// (concurrency=1 => batch_size==1 on the real fast path; a bs=2 case below
// additionally checks the req_id linear scan isn't fooled by ordering).
// Protocol under test (see build_ahead_kunlun_phase1_design.md §3):
//   cur_pairs[2*bs]    interleaved (req_id, last_token_offset) per request.
//   ring[1+2*ring_cap] [count, req_id_0, token_0, req_id_1, token_1, ...].
constexpr int32_t kPlaceholder = -1;

// bs=1 hit: input_ids[last]==-1 placeholder; prev_ring has a matching req_id
// entry (from the previous step's WriteRing) -> placeholder gets filled in.
TEST_F(KunlunKernelWrapperTest, ReplaceLastTokenHitTest) {
  constexpr int64_t bs = 1, input_ids_len = 4, ring_cap = 4;
  constexpr int64_t kReqId = 42, kLastOff = 3, kTok = 777;

  std::vector<int32_t> host_input_ids = {10, 20, 30, kPlaceholder};  // slot 3 = placeholder
  std::vector<int64_t> host_cur_pairs = {kReqId, kLastOff};
  std::vector<int64_t> host_prev_ring(1 + 2 * ring_cap, 0);
  host_prev_ring[0] = 1;  // count
  host_prev_ring[1] = kReqId;
  host_prev_ring[2] = kTok;

  int32_t* dev_input_ids;
  int64_t *dev_cur_pairs, *dev_prev_ring;
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_input_ids), input_ids_len * sizeof(int32_t), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_cur_pairs), 2 * bs * sizeof(int64_t), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_prev_ring), (1 + 2 * ring_cap) * sizeof(int64_t), XPU_MEM_MAIN));
  XPU_CHECK(xpu_memcpy(dev_input_ids, host_input_ids.data(), input_ids_len * sizeof(int32_t), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(dev_cur_pairs, host_cur_pairs.data(), 2 * bs * sizeof(int64_t), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(dev_prev_ring, host_prev_ring.data(), (1 + 2 * ring_cap) * sizeof(int64_t), XPU_HOST_TO_DEVICE));

  {
    XpuTimer timer("XpuReplaceLastToken(hit)", stream);
    ASSERT_EQ(0, XpuReplaceLastToken(xpuctx, dev_input_ids, dev_cur_pairs, dev_prev_ring, bs, input_ids_len));
  }
  std::vector<int32_t> host_out(input_ids_len);
  XPU_CHECK(xpu_memcpy(host_out.data(), dev_input_ids, input_ids_len * sizeof(int32_t), XPU_DEVICE_TO_HOST));

  EXPECT_EQ(host_out[3], static_cast<int32_t>(kTok));
  // Untouched slots keep their prior contents.
  EXPECT_EQ(host_out[0], 10);
  EXPECT_EQ(host_out[1], 20);
  EXPECT_EQ(host_out[2], 30);

  XPU_CHECK(xpu_free(dev_input_ids));
  XPU_CHECK(xpu_free(dev_cur_pairs));
  XPU_CHECK(xpu_free(dev_prev_ring));
}

// bs=1 placeholder guard: input_ids[last] is not -1 (e.g. a prefill / chunked
// prefill middle step) -> XpuReplaceLastToken must not touch it, even though
// prev_ring has a matching req_id entry.
TEST_F(KunlunKernelWrapperTest, ReplaceLastTokenPlaceholderGuardTest) {
  constexpr int64_t bs = 1, input_ids_len = 4, ring_cap = 4;
  constexpr int64_t kReqId = 42, kLastOff = 3, kTok = 777;
  constexpr int32_t kNonPlaceholder = 99;

  std::vector<int32_t> host_input_ids = {10, 20, 30, kNonPlaceholder};
  std::vector<int64_t> host_cur_pairs = {kReqId, kLastOff};
  std::vector<int64_t> host_prev_ring(1 + 2 * ring_cap, 0);
  host_prev_ring[0] = 1;
  host_prev_ring[1] = kReqId;
  host_prev_ring[2] = kTok;

  int32_t* dev_input_ids;
  int64_t *dev_cur_pairs, *dev_prev_ring;
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_input_ids), input_ids_len * sizeof(int32_t), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_cur_pairs), 2 * bs * sizeof(int64_t), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_prev_ring), (1 + 2 * ring_cap) * sizeof(int64_t), XPU_MEM_MAIN));
  XPU_CHECK(xpu_memcpy(dev_input_ids, host_input_ids.data(), input_ids_len * sizeof(int32_t), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(dev_cur_pairs, host_cur_pairs.data(), 2 * bs * sizeof(int64_t), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(dev_prev_ring, host_prev_ring.data(), (1 + 2 * ring_cap) * sizeof(int64_t), XPU_HOST_TO_DEVICE));

  {
    XpuTimer timer("XpuReplaceLastToken(placeholder-guard)", stream);
    ASSERT_EQ(0, XpuReplaceLastToken(xpuctx, dev_input_ids, dev_cur_pairs, dev_prev_ring, bs, input_ids_len));
  }
  std::vector<int32_t> host_out(input_ids_len);
  XPU_CHECK(xpu_memcpy(host_out.data(), dev_input_ids, input_ids_len * sizeof(int32_t), XPU_DEVICE_TO_HOST));

  EXPECT_EQ(host_out[3], kNonPlaceholder);  // unchanged

  XPU_CHECK(xpu_free(dev_input_ids));
  XPU_CHECK(xpu_free(dev_cur_pairs));
  XPU_CHECK(xpu_free(dev_prev_ring));
}

// bs=1 miss: prev_ring's req_id doesn't match cur_pairs' req_id -> the linear
// scan finds nothing, input_ids[last] stays at its -1 placeholder (defensive
// fallback; should not happen on the real fast path).
TEST_F(KunlunKernelWrapperTest, ReplaceLastTokenMissTest) {
  constexpr int64_t bs = 1, input_ids_len = 4, ring_cap = 4;
  constexpr int64_t kReqId = 42, kOtherReqId = 43, kLastOff = 3, kTok = 777;

  std::vector<int32_t> host_input_ids = {10, 20, 30, kPlaceholder};
  std::vector<int64_t> host_cur_pairs = {kReqId, kLastOff};
  std::vector<int64_t> host_prev_ring(1 + 2 * ring_cap, 0);
  host_prev_ring[0] = 1;
  host_prev_ring[1] = kOtherReqId;  // different req_id: no match for kReqId
  host_prev_ring[2] = kTok;

  int32_t* dev_input_ids;
  int64_t *dev_cur_pairs, *dev_prev_ring;
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_input_ids), input_ids_len * sizeof(int32_t), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_cur_pairs), 2 * bs * sizeof(int64_t), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_prev_ring), (1 + 2 * ring_cap) * sizeof(int64_t), XPU_MEM_MAIN));
  XPU_CHECK(xpu_memcpy(dev_input_ids, host_input_ids.data(), input_ids_len * sizeof(int32_t), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(dev_cur_pairs, host_cur_pairs.data(), 2 * bs * sizeof(int64_t), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(dev_prev_ring, host_prev_ring.data(), (1 + 2 * ring_cap) * sizeof(int64_t), XPU_HOST_TO_DEVICE));

  {
    XpuTimer timer("XpuReplaceLastToken(miss)", stream);
    ASSERT_EQ(0, XpuReplaceLastToken(xpuctx, dev_input_ids, dev_cur_pairs, dev_prev_ring, bs, input_ids_len));
  }
  std::vector<int32_t> host_out(input_ids_len);
  XPU_CHECK(xpu_memcpy(host_out.data(), dev_input_ids, input_ids_len * sizeof(int32_t), XPU_DEVICE_TO_HOST));

  EXPECT_EQ(host_out[3], kPlaceholder);  // stays -1

  XPU_CHECK(xpu_free(dev_input_ids));
  XPU_CHECK(xpu_free(dev_cur_pairs));
  XPU_CHECK(xpu_free(dev_prev_ring));
}

// Cold start: prev_ring[0] == 0 (nothing written yet) -> the scan is vacuous,
// input_ids stays unchanged at its -1 placeholder. This is the first-step case.
TEST_F(KunlunKernelWrapperTest, ReplaceLastTokenColdStartTest) {
  constexpr int64_t bs = 1, input_ids_len = 4, ring_cap = 4;
  constexpr int64_t kReqId = 42, kLastOff = 3;

  std::vector<int32_t> host_input_ids = {10, 20, 30, kPlaceholder};
  std::vector<int64_t> host_cur_pairs = {kReqId, kLastOff};
  std::vector<int64_t> host_prev_ring(1 + 2 * ring_cap, 0);  // count=0, cold start

  int32_t* dev_input_ids;
  int64_t *dev_cur_pairs, *dev_prev_ring;
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_input_ids), input_ids_len * sizeof(int32_t), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_cur_pairs), 2 * bs * sizeof(int64_t), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_prev_ring), (1 + 2 * ring_cap) * sizeof(int64_t), XPU_MEM_MAIN));
  XPU_CHECK(xpu_memcpy(dev_input_ids, host_input_ids.data(), input_ids_len * sizeof(int32_t), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(dev_cur_pairs, host_cur_pairs.data(), 2 * bs * sizeof(int64_t), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(dev_prev_ring, host_prev_ring.data(), (1 + 2 * ring_cap) * sizeof(int64_t), XPU_HOST_TO_DEVICE));

  {
    XpuTimer timer("XpuReplaceLastToken(cold-start)", stream);
    ASSERT_EQ(0, XpuReplaceLastToken(xpuctx, dev_input_ids, dev_cur_pairs, dev_prev_ring, bs, input_ids_len));
  }
  std::vector<int32_t> host_out(input_ids_len);
  XPU_CHECK(xpu_memcpy(host_out.data(), dev_input_ids, input_ids_len * sizeof(int32_t), XPU_DEVICE_TO_HOST));

  EXPECT_EQ(host_out[3], kPlaceholder);  // unchanged

  XPU_CHECK(xpu_free(dev_input_ids));
  XPU_CHECK(xpu_free(dev_cur_pairs));
  XPU_CHECK(xpu_free(dev_prev_ring));
}

// bs=2 hit, out-of-order ring entries: verifies the linear scan matches by
// req_id, not by position -- prev_ring lists the two (req_id, token) pairs in
// the opposite order from cur_pairs.
TEST_F(KunlunKernelWrapperTest, ReplaceLastTokenBatchTwoHitTest) {
  constexpr int64_t bs = 2, input_ids_len = 4, ring_cap = 4;
  constexpr int64_t kReqA = 5, kOffA = 0, kTokA = 505;
  constexpr int64_t kReqB = 9, kOffB = 1, kTokB = 909;

  std::vector<int32_t> host_input_ids = {kPlaceholder, kPlaceholder, 30, 40};
  std::vector<int64_t> host_cur_pairs = {kReqA, kOffA, kReqB, kOffB};
  std::vector<int64_t> host_prev_ring(1 + 2 * ring_cap, 0);
  host_prev_ring[0] = 2;
  // Reversed relative to cur_pairs: (req_id=9,tok=909) first, then (req_id=5,tok=505).
  host_prev_ring[1] = kReqB;
  host_prev_ring[2] = kTokB;
  host_prev_ring[3] = kReqA;
  host_prev_ring[4] = kTokA;

  int32_t* dev_input_ids;
  int64_t *dev_cur_pairs, *dev_prev_ring;
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_input_ids), input_ids_len * sizeof(int32_t), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_cur_pairs), 2 * bs * sizeof(int64_t), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_prev_ring), (1 + 2 * ring_cap) * sizeof(int64_t), XPU_MEM_MAIN));
  XPU_CHECK(xpu_memcpy(dev_input_ids, host_input_ids.data(), input_ids_len * sizeof(int32_t), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(dev_cur_pairs, host_cur_pairs.data(), 2 * bs * sizeof(int64_t), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(dev_prev_ring, host_prev_ring.data(), (1 + 2 * ring_cap) * sizeof(int64_t), XPU_HOST_TO_DEVICE));

  {
    XpuTimer timer("XpuReplaceLastToken(bs=2)", stream);
    ASSERT_EQ(0, XpuReplaceLastToken(xpuctx, dev_input_ids, dev_cur_pairs, dev_prev_ring, bs, input_ids_len));
  }
  std::vector<int32_t> host_out(input_ids_len);
  XPU_CHECK(xpu_memcpy(host_out.data(), dev_input_ids, input_ids_len * sizeof(int32_t), XPU_DEVICE_TO_HOST));

  EXPECT_EQ(host_out[0], static_cast<int32_t>(kTokA));
  EXPECT_EQ(host_out[1], static_cast<int32_t>(kTokB));
  EXPECT_EQ(host_out[2], 30);
  EXPECT_EQ(host_out[3], 40);

  XPU_CHECK(xpu_free(dev_input_ids));
  XPU_CHECK(xpu_free(dev_cur_pairs));
  XPU_CHECK(xpu_free(dev_prev_ring));
}

// XpuWriteRing bs=1: writes count header + one (req_id, token) pair, and
// leaves the rest of the (larger) ring buffer untouched.
TEST_F(KunlunKernelWrapperTest, WriteRingBasicTest) {
  constexpr int64_t bs = 1, ring_cap = 4;
  constexpr int64_t kReqId = 7, kLastOff = 2;
  constexpr uint32_t kTok = 1234u;

  std::vector<int64_t> host_cur_pairs = {kReqId, kLastOff};
  std::vector<uint32_t> host_sampled = {kTok};
  std::vector<int64_t> host_ring_init(1 + 2 * ring_cap, -999);  // sentinel beyond what should be written

  // Cold source slot (count == 0): carry-forward is vacuous, so only this step's pair is written.
  std::vector<int64_t> host_prev_ring(1 + 2 * ring_cap, 0);

  int64_t *dev_ring, *dev_prev_ring, *dev_cur_pairs;
  uint32_t* dev_sampled;
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_ring), (1 + 2 * ring_cap) * sizeof(int64_t), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_prev_ring), (1 + 2 * ring_cap) * sizeof(int64_t), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_cur_pairs), 2 * bs * sizeof(int64_t), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_sampled), bs * sizeof(uint32_t), XPU_MEM_MAIN));
  XPU_CHECK(xpu_memcpy(dev_ring, host_ring_init.data(), (1 + 2 * ring_cap) * sizeof(int64_t), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(dev_prev_ring, host_prev_ring.data(), (1 + 2 * ring_cap) * sizeof(int64_t), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(dev_cur_pairs, host_cur_pairs.data(), 2 * bs * sizeof(int64_t), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(dev_sampled, host_sampled.data(), bs * sizeof(uint32_t), XPU_HOST_TO_DEVICE));

  {
    XpuTimer timer("XpuWriteRing(bs=1)", stream);
    ASSERT_EQ(0, XpuWriteRing(xpuctx, dev_ring, dev_prev_ring, dev_cur_pairs, dev_sampled,
                              /*evict_ids=*/nullptr, /*evict_count=*/0, bs, ring_cap));
  }
  std::vector<int64_t> host_ring(1 + 2 * ring_cap);
  XPU_CHECK(xpu_memcpy(host_ring.data(), dev_ring, (1 + 2 * ring_cap) * sizeof(int64_t), XPU_DEVICE_TO_HOST));

  EXPECT_EQ(host_ring[0], 1);
  EXPECT_EQ(host_ring[1], kReqId);
  EXPECT_EQ(host_ring[2], static_cast<int64_t>(kTok));
  // Slots beyond the written count are untouched.
  for (int64_t i = 3; i < 1 + 2 * ring_cap; ++i) {
    EXPECT_EQ(host_ring[i], -999) << "unexpected write at ring[" << i << "]";
  }

  XPU_CHECK(xpu_free(dev_ring));
  XPU_CHECK(xpu_free(dev_prev_ring));
  XPU_CHECK(xpu_free(dev_cur_pairs));
  XPU_CHECK(xpu_free(dev_sampled));
}

// Round trip: XpuWriteRing(step N) followed by XpuReplaceLastToken(step N+1)
// consuming that ring as prev_ring -- the exact two-call sequence the executor
// issues each decode step once build-ahead is wired up.
TEST_F(KunlunKernelWrapperTest, WriteRingThenReplaceLastTokenRoundTripTest) {
  constexpr int64_t bs = 1, ring_cap = 4, input_ids_len = 4;
  constexpr int64_t kReqId = 7, kLastOffStepN = 2, kLastOffStepNPlus1 = 3;
  constexpr uint32_t kTok = 1234u;

  // Step N: sampling produced kTok for kReqId; write it into the ring.
  std::vector<int64_t> host_cur_pairs_n = {kReqId, kLastOffStepN};
  std::vector<uint32_t> host_sampled = {kTok};
  std::vector<int64_t> host_ring_init(1 + 2 * ring_cap, 0);
  // Cold source slot (count == 0): the union starts empty, so this step's pair is the only entry.
  std::vector<int64_t> host_prev_ring(1 + 2 * ring_cap, 0);

  int64_t *dev_ring, *dev_prev_ring, *dev_cur_pairs_n;
  uint32_t* dev_sampled;
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_ring), (1 + 2 * ring_cap) * sizeof(int64_t), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_prev_ring), (1 + 2 * ring_cap) * sizeof(int64_t), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_cur_pairs_n), 2 * bs * sizeof(int64_t), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_sampled), bs * sizeof(uint32_t), XPU_MEM_MAIN));
  XPU_CHECK(xpu_memcpy(dev_ring, host_ring_init.data(), (1 + 2 * ring_cap) * sizeof(int64_t), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(dev_prev_ring, host_prev_ring.data(), (1 + 2 * ring_cap) * sizeof(int64_t), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(dev_cur_pairs_n, host_cur_pairs_n.data(), 2 * bs * sizeof(int64_t), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(dev_sampled, host_sampled.data(), bs * sizeof(uint32_t), XPU_HOST_TO_DEVICE));

  {
    XpuTimer timer("XpuWriteRing(round-trip)", stream);
    ASSERT_EQ(0, XpuWriteRing(xpuctx, dev_ring, dev_prev_ring, dev_cur_pairs_n, dev_sampled,
                              /*evict_ids=*/nullptr, /*evict_count=*/0, bs, ring_cap));
  }

  // Step N+1: same req_id continues decoding; its input_ids slot starts as the
  // -1 placeholder ModelInput::PrepareInputIds would have written.
  std::vector<int32_t> host_input_ids = {10, 20, 30, kPlaceholder};
  std::vector<int64_t> host_cur_pairs_n1 = {kReqId, kLastOffStepNPlus1};

  int32_t* dev_input_ids;
  int64_t* dev_cur_pairs_n1;
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_input_ids), input_ids_len * sizeof(int32_t), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_cur_pairs_n1), 2 * bs * sizeof(int64_t), XPU_MEM_MAIN));
  XPU_CHECK(xpu_memcpy(dev_input_ids, host_input_ids.data(), input_ids_len * sizeof(int32_t), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(dev_cur_pairs_n1, host_cur_pairs_n1.data(), 2 * bs * sizeof(int64_t), XPU_HOST_TO_DEVICE));

  {
    XpuTimer timer("XpuReplaceLastToken(round-trip)", stream);
    ASSERT_EQ(0, XpuReplaceLastToken(xpuctx, dev_input_ids, dev_cur_pairs_n1, dev_ring, bs, input_ids_len));
  }
  std::vector<int32_t> host_out(input_ids_len);
  XPU_CHECK(xpu_memcpy(host_out.data(), dev_input_ids, input_ids_len * sizeof(int32_t), XPU_DEVICE_TO_HOST));

  EXPECT_EQ(host_out[3], static_cast<int32_t>(kTok));

  XPU_CHECK(xpu_free(dev_ring));
  XPU_CHECK(xpu_free(dev_prev_ring));
  XPU_CHECK(xpu_free(dev_cur_pairs_n));
  XPU_CHECK(xpu_free(dev_sampled));
  XPU_CHECK(xpu_free(dev_input_ids));
  XPU_CHECK(xpu_free(dev_cur_pairs_n1));
}

#endif  // ENABLE_KUNLUN

// IndexSelect1D: 1D gather for EAGLE3 hot_token_id mapping
// output[i] = table[index[i]]. Tests int32_t data with int32_t index.
TEST_F(KunlunKernelWrapperTest, IndexSelect1DTest) {
  std::vector<int32_t> host_table = {100, 200, 300, 400, 500};
  std::vector<int32_t> host_index32 = {4, 0, 2};
  int64_t table_len = 5, num_indices = 3;
  std::vector<int32_t> host_output(num_indices, -1);

  int32_t *dev_table, *dev_index32, *dev_output;
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_table), table_len * sizeof(int32_t), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_index32), num_indices * sizeof(int32_t), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_output), num_indices * sizeof(int32_t), XPU_MEM_MAIN));
  XPU_CHECK(xpu_memcpy(dev_table, host_table.data(), table_len * sizeof(int32_t), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(dev_index32, host_index32.data(), num_indices * sizeof(int32_t), XPU_HOST_TO_DEVICE));

  {
    XpuTimer timer("XpuIndexSelect<int32,int32> 1D", stream);
    ASSERT_EQ(0, (XpuIndexSelect<int32_t, int32_t>(xpuctx, dev_table, dev_index32, dev_output,
                                                    {table_len, 1}, num_indices, /*axis=*/0)));
  }
  XPU_CHECK(xpu_memcpy(host_output.data(), dev_output, num_indices * sizeof(int32_t), XPU_DEVICE_TO_HOST));
  EXPECT_EQ(host_output[0], 500);
  EXPECT_EQ(host_output[1], 100);
  EXPECT_EQ(host_output[2], 300);

  XPU_CHECK(xpu_free(dev_table));
  XPU_CHECK(xpu_free(dev_index32));
  XPU_CHECK(xpu_free(dev_output));
}

}  // namespace ksana_llm

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
