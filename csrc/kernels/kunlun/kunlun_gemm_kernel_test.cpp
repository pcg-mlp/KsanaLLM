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

#include "runtime/xpu_context.h"

#include "device/kunlun/xpu_utils.h"
#include "kernels/kunlun/basic_kernel_wrapper.h"
#include "kernels/kunlun/batched_matmul_wrapper.h"
#include "kernels/kunlun/quant_gemm_wrapper.h"
#include "utils/logger.h"

namespace ksana_llm {

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

class KunlunGemmKernelTest : public testing::Test {
 public:
  static void SetUpTestSuite() {
    XPU_CHECK(xpu_current_device(&device_));
    XPU_CHECK(xpu_stream_create(&stream_));
    ctx_ = xdnn::create_context();
    ctx_->set_stream(stream_);
    xpuctx_ = xmlir_rt::getXpuKernelContext(/*useCpu=*/false);
    xpuctx_->set_stream(stream_, false);
    XPU_CHECK(xpu_wait(stream_));
  }

  static void TearDownTestSuite() { XPU_CHECK(xpu_wait(stream_)); }

  void SetUp() override { XPU_CHECK(xpu_wait(stream_)); }
  void TearDown() override { XPU_CHECK(xpu_wait(stream_)); }

 protected:
  XPUStream& stream = stream_;
  xdnn::Context*& ctx = ctx_;
  xpytorch::xpu::api::Context*& xpuctx = xpuctx_;
  int& device = device_;

  static int device_;
  static XPUStream stream_;
  static xdnn::Context* ctx_;
  static xpytorch::xpu::api::Context* xpuctx_;
};

int KunlunGemmKernelTest::device_ = -1;
XPUStream KunlunGemmKernelTest::stream_ = nullptr;
xdnn::Context* KunlunGemmKernelTest::ctx_ = nullptr;
xpytorch::xpu::api::Context* KunlunGemmKernelTest::xpuctx_ = nullptr;

#ifdef ENABLE_KUNLUN

// ═══════════════════════════════════════════════════════════════════════════════
// XpuBatchedMatMul<bfloat16>
// C[M, B, N] = A[M, B, K] × B[B, N, K]^T
// ═══════════════════════════════════════════════════════════════════════════════

TEST_F(KunlunGemmKernelTest, BatchedMatMulBf16) {
  constexpr int64_t m = 4, n = 8, k = 16, batch = 2;
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

  std::vector<float> host_a(m * batch * k), host_b(batch * n * k);
  for (auto& v : host_a) v = dist(rng);
  for (auto& v : host_b) v = dist(rng);

  // CPU reference: C[i, b, j] = sum_kk A[i, b, kk] * B[b, j, kk]
  std::vector<float> ref(m * batch * n, 0.0f);
  for (int i = 0; i < m; ++i) {
    for (int b = 0; b < batch; ++b) {
      for (int j = 0; j < n; ++j) {
        float acc = 0.0f;
        for (int kk = 0; kk < k; ++kk) {
          acc += host_a[i * batch * k + b * k + kk] * host_b[b * n * k + j * k + kk];
        }
        ref[i * batch * n + b * n + j] = acc;
      }
    }
  }

  // Upload float data to device, then cast to bf16
  float *dev_fa, *dev_fb, *dev_fc;
  bfloat16 *dev_a, *dev_b, *dev_c;
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_fa), m * batch * k * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_fb), batch * n * k * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_fc), m * batch * n * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_a), m * batch * k * sizeof(bfloat16), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_b), batch * n * k * sizeof(bfloat16), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_c), m * batch * n * sizeof(bfloat16), XPU_MEM_MAIN));

  XPU_CHECK(xpu_memcpy(dev_fa, host_a.data(), m * batch * k * sizeof(float), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(dev_fb, host_b.data(), batch * n * k * sizeof(float), XPU_HOST_TO_DEVICE));
  ASSERT_EQ(0, (XpuCast<float, bfloat16>(xpuctx, dev_fa, dev_a, m * batch * k)));
  ASSERT_EQ(0, (XpuCast<float, bfloat16>(xpuctx, dev_fb, dev_b, batch * n * k)));

  {
    XpuTimer timer("XpuBatchedMatMul<bf16>", stream);
    ASSERT_EQ(0, (XpuBatchedMatMul<bfloat16>(ctx, dev_a, dev_b, dev_c, m, n, k, batch)));
  }

  ASSERT_EQ(0, (XpuCast<bfloat16, float>(xpuctx, dev_c, dev_fc, m * batch * n)));
  std::vector<float> host_c(m * batch * n);
  XPU_CHECK(xpu_memcpy(host_c.data(), dev_fc, m * batch * n * sizeof(float), XPU_DEVICE_TO_HOST));

  for (int i = 0; i < m * batch * n; ++i) {
    EXPECT_NEAR(host_c[i], ref[i], 0.1f) << "mismatch at [" << i << "]";
  }

  XPU_CHECK(xpu_free(dev_fa));
  XPU_CHECK(xpu_free(dev_fb));
  XPU_CHECK(xpu_free(dev_fc));
  XPU_CHECK(xpu_free(dev_a));
  XPU_CHECK(xpu_free(dev_b));
  XPU_CHECK(xpu_free(dev_c));
}

// ═══════════════════════════════════════════════════════════════════════════════
// XpuBatchedMatMulInt8<bfloat16>
// C[M, B, N] = dequant(quant(A[M, B, K]) × B_int8[B, N, K]^T, a_scale, b_scale)
// scale format: max_abs (i.e., pre-multiplied by 127)
// ═══════════════════════════════════════════════════════════════════════════════

TEST_F(KunlunGemmKernelTest, BatchedMatMulInt8Bf16) {
  constexpr int64_t m = 4, n = 8, k = 64, batch = 2;
  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  // Activation in bf16 (stored as float for host reference)
  std::vector<float> host_a(m * batch * k);
  for (auto& v : host_a) v = dist(rng);

  // Generate float weights [B, N, K], then quantize to int8 + max_abs scale
  std::vector<float> host_b_float(batch * n * k);
  for (auto& v : host_b_float) v = dist(rng);

  std::vector<int8_t> host_b(batch * n * k);
  std::vector<float> host_b_scales(batch * n);  // max_abs format

  for (int b = 0; b < batch; ++b) {
    for (int j = 0; j < n; ++j) {
      float max_abs = 0.0f;
      for (int kk = 0; kk < k; ++kk) {
        max_abs = std::max(max_abs, std::fabs(host_b_float[b * n * k + j * k + kk]));
      }
      if (max_abs < 1e-8f) max_abs = 1e-8f;
      host_b_scales[b * n + j] = max_abs;
      for (int kk = 0; kk < k; ++kk) {
        float val = host_b_float[b * n * k + j * k + kk];
        int q = static_cast<int>(std::round(val / max_abs * 127.0f));
        q = std::max(-127, std::min(127, q));
        host_b[b * n * k + j * k + kk] = static_cast<int8_t>(q);
      }
    }
  }

  // CPU reference: fp32 matmul (the kernel should approximate this)
  // C[i, b, j] = sum_kk A[i, b, kk] * B_float[b, j, kk]
  std::vector<float> ref(m * batch * n, 0.0f);
  for (int i = 0; i < m; ++i) {
    for (int b = 0; b < batch; ++b) {
      for (int j = 0; j < n; ++j) {
        float acc = 0.0f;
        for (int kk = 0; kk < k; ++kk) {
          acc += host_a[i * batch * k + b * k + kk] * host_b_float[b * n * k + j * k + kk];
        }
        ref[i * batch * n + b * n + j] = acc;
      }
    }
  }

  float* dev_fa;
  bfloat16* dev_a;
  int8_t* dev_b;
  float* dev_b_scales;
  bfloat16* dev_c;
  float* dev_fc;
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_fa), m * batch * k * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_a), m * batch * k * sizeof(bfloat16), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_b), batch * n * k * sizeof(int8_t), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_b_scales), batch * n * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_c), m * batch * n * sizeof(bfloat16), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_fc), m * batch * n * sizeof(float), XPU_MEM_MAIN));

  XPU_CHECK(xpu_memcpy(dev_fa, host_a.data(), m * batch * k * sizeof(float), XPU_HOST_TO_DEVICE));
  ASSERT_EQ(0, (XpuCast<float, bfloat16>(xpuctx, dev_fa, dev_a, m * batch * k)));
  XPU_CHECK(xpu_memcpy(dev_b, host_b.data(), batch * n * k * sizeof(int8_t), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(dev_b_scales, host_b_scales.data(), batch * n * sizeof(float), XPU_HOST_TO_DEVICE));

  {
    XpuTimer timer("XpuBatchedMatMulInt8<bf16>", stream);
    ASSERT_EQ(0, (XpuBatchedMatMulInt8<bfloat16>(ctx, dev_a, dev_b, dev_b_scales, dev_c, m, n, k, batch)));
  }

  ASSERT_EQ(0, (XpuCast<bfloat16, float>(xpuctx, dev_c, dev_fc, m * batch * n)));
  std::vector<float> host_c(m * batch * n);
  XPU_CHECK(xpu_memcpy(host_c.data(), dev_fc, m * batch * n * sizeof(float), XPU_DEVICE_TO_HOST));

  for (int i = 0; i < m * batch * n; ++i) {
    float abs_err = std::fabs(host_c[i] - ref[i]);
    float scale = std::fabs(ref[i]) + 1.0f;
    EXPECT_LT(abs_err / scale, 0.3f) << "mismatch at [" << i << "] got=" << host_c[i] << " ref=" << ref[i];
  }

  XPU_CHECK(xpu_free(dev_fa));
  XPU_CHECK(xpu_free(dev_a));
  XPU_CHECK(xpu_free(dev_b));
  XPU_CHECK(xpu_free(dev_b_scales));
  XPU_CHECK(xpu_free(dev_c));
  XPU_CHECK(xpu_free(dev_fc));
}

// ═══════════════════════════════════════════════════════════════════════════════
// XpuInt8ScaledMM<bfloat16>
// y[m,n] = dequant(quant(x[m,k]) × w_int8[n,k]^T, scale_x, scale_w)
// scale_w format: max_abs per channel (pre-multiplied by 127)
// ═══════════════════════════════════════════════════════════════════════════════

TEST_F(KunlunGemmKernelTest, Int8ScaledMMBf16) {
  constexpr int64_t m = 4, n = 16, k = 64;
  std::mt19937 rng(200);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  // Activation x [m, k] in bf16
  std::vector<float> host_x(m * k);
  for (auto& v : host_x) v = dist(rng);

  // Generate float weights [N, K], quantize to int8 + max_abs scale
  std::vector<float> host_w_float(n * k);
  for (auto& v : host_w_float) v = dist(rng);

  std::vector<int8_t> host_w(n * k);
  std::vector<float> host_scale_w(n);  // max_abs format

  for (int j = 0; j < n; ++j) {
    float max_abs = 0.0f;
    for (int kk = 0; kk < k; ++kk) {
      max_abs = std::max(max_abs, std::fabs(host_w_float[j * k + kk]));
    }
    if (max_abs < 1e-8f) max_abs = 1e-8f;
    host_scale_w[j] = max_abs;
    for (int kk = 0; kk < k; ++kk) {
      float val = host_w_float[j * k + kk];
      int q = static_cast<int>(std::round(val / max_abs * 127.0f));
      q = std::max(-127, std::min(127, q));
      host_w[j * k + kk] = static_cast<int8_t>(q);
    }
  }

  // CPU reference: y[i,j] = sum_kk(x[i,kk] * w_float[j,kk])
  std::vector<float> ref(m * n, 0.0f);
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      float acc = 0.0f;
      for (int kk = 0; kk < k; ++kk) {
        acc += host_x[i * k + kk] * host_w_float[j * k + kk];
      }
      ref[i * n + j] = acc;
    }
  }

  float* dev_fx;
  float* dev_fy;
  bfloat16 *dev_x, *dev_y;
  int8_t* dev_w;
  float* dev_scale_w;
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_fx), m * k * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_fy), m * n * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_x), m * k * sizeof(bfloat16), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_y), m * n * sizeof(bfloat16), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_w), n * k * sizeof(int8_t), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_scale_w), n * sizeof(float), XPU_MEM_MAIN));

  XPU_CHECK(xpu_memcpy(dev_fx, host_x.data(), m * k * sizeof(float), XPU_HOST_TO_DEVICE));
  ASSERT_EQ(0, (XpuCast<float, bfloat16>(xpuctx, dev_fx, dev_x, m * k)));
  XPU_CHECK(xpu_memcpy(dev_w, host_w.data(), n * k * sizeof(int8_t), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(dev_scale_w, host_scale_w.data(), n * sizeof(float), XPU_HOST_TO_DEVICE));

  {
    XpuTimer timer("XpuInt8ScaledMM<bf16>", stream);
    ASSERT_EQ(0, (XpuInt8ScaledMM<bfloat16>(ctx, dev_x, dev_w, dev_scale_w, dev_y, m, n, k, /*bias=*/nullptr)));
  }

  ASSERT_EQ(0, (XpuCast<bfloat16, float>(xpuctx, dev_y, dev_fy, m * n)));
  std::vector<float> host_y(m * n);
  XPU_CHECK(xpu_memcpy(host_y.data(), dev_fy, m * n * sizeof(float), XPU_DEVICE_TO_HOST));

  for (int i = 0; i < m * n; ++i) {
    float abs_err = std::fabs(host_y[i] - ref[i]);
    float scale = std::fabs(ref[i]) + 1.0f;
    EXPECT_LT(abs_err / scale, 0.3f) << "mismatch at [" << i << "] got=" << host_y[i] << " ref=" << ref[i];
  }

  XPU_CHECK(xpu_free(dev_fx));
  XPU_CHECK(xpu_free(dev_fy));
  XPU_CHECK(xpu_free(dev_x));
  XPU_CHECK(xpu_free(dev_y));
  XPU_CHECK(xpu_free(dev_w));
  XPU_CHECK(xpu_free(dev_scale_w));
}

// ═══════════════════════════════════════════════════════════════════════════════
// XpuInt8ScaledMM<bfloat16> with bias
// ═══════════════════════════════════════════════════════════════════════════════

TEST_F(KunlunGemmKernelTest, Int8ScaledMMBf16WithBias) {
  constexpr int64_t m = 4, n = 16, k = 64;
  std::mt19937 rng(300);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  std::vector<float> host_x(m * k);
  for (auto& v : host_x) v = dist(rng);

  // Generate float weights [N, K], quantize to int8 + max_abs scale
  std::vector<float> host_w_float(n * k);
  for (auto& v : host_w_float) v = dist(rng);

  std::vector<int8_t> host_w(n * k);
  std::vector<float> host_scale_w(n);

  for (int j = 0; j < n; ++j) {
    float max_abs = 0.0f;
    for (int kk = 0; kk < k; ++kk) {
      max_abs = std::max(max_abs, std::fabs(host_w_float[j * k + kk]));
    }
    if (max_abs < 1e-8f) max_abs = 1e-8f;
    host_scale_w[j] = max_abs;
    for (int kk = 0; kk < k; ++kk) {
      float val = host_w_float[j * k + kk];
      int q = static_cast<int>(std::round(val / max_abs * 127.0f));
      q = std::max(-127, std::min(127, q));
      host_w[j * k + kk] = static_cast<int8_t>(q);
    }
  }

  std::vector<float> host_bias(n);
  for (auto& v : host_bias) v = dist(rng) * 0.1f;

  // CPU reference with bias: y[i,j] = sum_kk(x[i,kk] * w_float[j,kk]) + bias[j]
  std::vector<float> ref(m * n, 0.0f);
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      float acc = 0.0f;
      for (int kk = 0; kk < k; ++kk) {
        acc += host_x[i * k + kk] * host_w_float[j * k + kk];
      }
      ref[i * n + j] = acc + host_bias[j];
    }
  }

  float *dev_fx, *dev_fy;
  bfloat16 *dev_x, *dev_y;
  int8_t* dev_w;
  float *dev_scale_w, *dev_bias;
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_fx), m * k * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_fy), m * n * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_x), m * k * sizeof(bfloat16), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_y), m * n * sizeof(bfloat16), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_w), n * k * sizeof(int8_t), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_scale_w), n * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_bias), n * sizeof(float), XPU_MEM_MAIN));

  XPU_CHECK(xpu_memcpy(dev_fx, host_x.data(), m * k * sizeof(float), XPU_HOST_TO_DEVICE));
  ASSERT_EQ(0, (XpuCast<float, bfloat16>(xpuctx, dev_fx, dev_x, m * k)));
  XPU_CHECK(xpu_memcpy(dev_w, host_w.data(), n * k * sizeof(int8_t), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(dev_scale_w, host_scale_w.data(), n * sizeof(float), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(dev_bias, host_bias.data(), n * sizeof(float), XPU_HOST_TO_DEVICE));

  {
    XpuTimer timer("XpuInt8ScaledMM<bf16> with bias", stream);
    ASSERT_EQ(0, (XpuInt8ScaledMM<bfloat16>(ctx, dev_x, dev_w, dev_scale_w, dev_y, m, n, k, dev_bias)));
  }

  ASSERT_EQ(0, (XpuCast<bfloat16, float>(xpuctx, dev_y, dev_fy, m * n)));
  std::vector<float> host_y(m * n);
  XPU_CHECK(xpu_memcpy(host_y.data(), dev_fy, m * n * sizeof(float), XPU_DEVICE_TO_HOST));

  for (int i = 0; i < m * n; ++i) {
    float abs_err = std::fabs(host_y[i] - ref[i]);
    float scale = std::fabs(ref[i]) + 1.0f;
    EXPECT_LT(abs_err / scale, 0.3f) << "mismatch at [" << i << "] got=" << host_y[i] << " ref=" << ref[i];
  }

  XPU_CHECK(xpu_free(dev_fx));
  XPU_CHECK(xpu_free(dev_fy));
  XPU_CHECK(xpu_free(dev_x));
  XPU_CHECK(xpu_free(dev_y));
  XPU_CHECK(xpu_free(dev_w));
  XPU_CHECK(xpu_free(dev_scale_w));
  XPU_CHECK(xpu_free(dev_bias));
}

// ═══════════════════════════════════════════════════════════════════════════════
// XpuFcFusion<bfloat16>
// y[m,n] = x[m,k] * w[k,n] (no transpose, no quant scaling)
// ═══════════════════════════════════════════════════════════════════════════════

TEST_F(KunlunGemmKernelTest, FcFusionBf16NoTrans) {
  constexpr int64_t m = 4, n = 16, k = 32;
  std::mt19937 rng(400);
  std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

  std::vector<float> host_x(m * k), host_w(k * n);
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

  float *dev_fx, *dev_fw, *dev_fy;
  bfloat16 *dev_x, *dev_w, *dev_y;
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_fx), m * k * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_fw), k * n * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_fy), m * n * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_x), m * k * sizeof(bfloat16), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_w), k * n * sizeof(bfloat16), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_y), m * n * sizeof(bfloat16), XPU_MEM_MAIN));

  XPU_CHECK(xpu_memcpy(dev_fx, host_x.data(), m * k * sizeof(float), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(dev_fw, host_w.data(), k * n * sizeof(float), XPU_HOST_TO_DEVICE));
  ASSERT_EQ(0, (XpuCast<float, bfloat16>(xpuctx, dev_fx, dev_x, m * k)));
  ASSERT_EQ(0, (XpuCast<float, bfloat16>(xpuctx, dev_fw, dev_w, k * n)));

  {
    XpuTimer timer("XpuFcFusion<bf16> no-trans", stream);
    ASSERT_EQ(0, (XpuFcFusion<bfloat16, bfloat16, bfloat16>(ctx, dev_x, dev_w, dev_y, m, n, k,
                                                             /*x_trans=*/false, /*w_trans=*/false,
                                                             /*x_maxptr=*/nullptr, /*w_maxptr=*/nullptr,
                                                             /*y_maxptr=*/nullptr)));
  }

  ASSERT_EQ(0, (XpuCast<bfloat16, float>(xpuctx, dev_y, dev_fy, m * n)));
  std::vector<float> host_y(m * n);
  XPU_CHECK(xpu_memcpy(host_y.data(), dev_fy, m * n * sizeof(float), XPU_DEVICE_TO_HOST));

  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      EXPECT_NEAR(host_y[i * n + j], ref[i * n + j], 0.1f) << "mismatch at [" << i << "," << j << "]";
    }
  }

  XPU_CHECK(xpu_free(dev_fx));
  XPU_CHECK(xpu_free(dev_fw));
  XPU_CHECK(xpu_free(dev_fy));
  XPU_CHECK(xpu_free(dev_x));
  XPU_CHECK(xpu_free(dev_w));
  XPU_CHECK(xpu_free(dev_y));
}

// ═══════════════════════════════════════════════════════════════════════════════
// XpuFcFusion<bfloat16> with w_trans=true
// y[m,n] = x[m,k] * w^T where w is [n,k]
// ═══════════════════════════════════════════════════════════════════════════════

TEST_F(KunlunGemmKernelTest, FcFusionBf16WTrans) {
  constexpr int64_t m = 4, n = 16, k = 32;
  std::mt19937 rng(500);
  std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

  std::vector<float> host_x(m * k), host_w(n * k);  // w stored as [n, k]
  for (auto& v : host_x) v = dist(rng);
  for (auto& v : host_w) v = dist(rng);

  // CPU reference: y[i,j] = sum_kk x[i,kk] * w[j,kk]
  std::vector<float> ref(m * n, 0.0f);
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      float acc = 0.0f;
      for (int kk = 0; kk < k; ++kk) acc += host_x[i * k + kk] * host_w[j * k + kk];
      ref[i * n + j] = acc;
    }
  }

  float *dev_fx, *dev_fw, *dev_fy;
  bfloat16 *dev_x, *dev_w, *dev_y;
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_fx), m * k * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_fw), n * k * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_fy), m * n * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_x), m * k * sizeof(bfloat16), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_w), n * k * sizeof(bfloat16), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dev_y), m * n * sizeof(bfloat16), XPU_MEM_MAIN));

  XPU_CHECK(xpu_memcpy(dev_fx, host_x.data(), m * k * sizeof(float), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(dev_fw, host_w.data(), n * k * sizeof(float), XPU_HOST_TO_DEVICE));
  ASSERT_EQ(0, (XpuCast<float, bfloat16>(xpuctx, dev_fx, dev_x, m * k)));
  ASSERT_EQ(0, (XpuCast<float, bfloat16>(xpuctx, dev_fw, dev_w, n * k)));

  {
    XpuTimer timer("XpuFcFusion<bf16> w_trans", stream);
    ASSERT_EQ(0, (XpuFcFusion<bfloat16, bfloat16, bfloat16>(ctx, dev_x, dev_w, dev_y, m, n, k,
                                                             /*x_trans=*/false, /*w_trans=*/true,
                                                             /*x_maxptr=*/nullptr, /*w_maxptr=*/nullptr,
                                                             /*y_maxptr=*/nullptr)));
  }

  ASSERT_EQ(0, (XpuCast<bfloat16, float>(xpuctx, dev_y, dev_fy, m * n)));
  std::vector<float> host_y(m * n);
  XPU_CHECK(xpu_memcpy(host_y.data(), dev_fy, m * n * sizeof(float), XPU_DEVICE_TO_HOST));

  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      EXPECT_NEAR(host_y[i * n + j], ref[i * n + j], 0.1f) << "mismatch at [" << i << "," << j << "]";
    }
  }

  XPU_CHECK(xpu_free(dev_fx));
  XPU_CHECK(xpu_free(dev_fw));
  XPU_CHECK(xpu_free(dev_fy));
  XPU_CHECK(xpu_free(dev_x));
  XPU_CHECK(xpu_free(dev_w));
  XPU_CHECK(xpu_free(dev_y));
}

#endif  // ENABLE_KUNLUN

}  // namespace ksana_llm
