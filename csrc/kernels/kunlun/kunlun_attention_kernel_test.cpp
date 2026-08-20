/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
// Unit tests for attention kernel wrappers in attention_kernel_wrapper.
// MLA shapes aligned with DeepSeek V2 config:
//   kv_lora_rank = 512, qk_rope_head_dim = 64, qk_nope_head_dim = 128, v_head_dim = 128
// Uses minimal token/batch counts with pre-computed baselines to avoid CPU cost.

#include <xpu/runtime.h>
#include <xpu/xdnn.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "test.h"

#include "device/kunlun/xpu_utils.h"
#include "kernels/kunlun/attention_kernel_wrapper.h"
#include "kernels/kunlun/basic_kernel_wrapper.h"
#include "runtime/xpu_context.h"
#include "utils/logger.h"

// xbfloat16 (Kunlunxin "fast bf16 / InITTE" 2-byte type) lives in the xfa SDK,
// not in KsanaLLM. flash_api.h (pulled in via attention_kernel_wrapper.h)
// transitively exposes it, but include the header explicitly so the direct
// xfa::speculative_multi_latent_attention<xbfloat16,...> call below compiles
// even if that transitive path changes. `namespace xfa` alias mirrors
// attention_kernel_wrapper.cpp.
#include "xpu/refactor/util/xtdk_xbf16.h"
namespace xfa = baidu::xpu::xfa;

// ── Lightweight .npy loader (inlined from former npy_loader.h) ──────────────
// Supports float32, int32, int64 scalar and array data.
// Format reference: https://numpy.org/neps/nep-0001-npy-format.html
namespace ksana_llm {
namespace test_npy {

enum NpyDtype { NPY_F32, NPY_I32, NPY_I64, NPY_UNKNOWN };

struct npy_data_t {
  std::vector<char> raw;  // raw bytes (host)
  std::vector<int64_t> shape;
  NpyDtype dtype = NPY_UNKNOWN;
  int64_t numel = 0;

  // Typed accessors (no copy — pointers into raw)
  float* as_f32() { return reinterpret_cast<float*>(raw.data()); }
  int32_t* as_i32() { return reinterpret_cast<int32_t*>(raw.data()); }
  int64_t* as_i64() { return reinterpret_cast<int64_t*>(raw.data()); }

  // Scalar convenience
  float scalar_f32() { return *as_f32(); }
  int32_t scalar_i32() { return *as_i32(); }
  int64_t scalar_i64() { return *as_i64(); }
};

// Trim leading/trailing whitespace from a substring.
static inline std::string trim(const std::string& s) {
  size_t a = s.find_first_not_of(" \t\r\n'");
  size_t b = s.find_last_not_of(" \t\r\n',");
  if (a == std::string::npos) return "";
  return s.substr(a, b - a + 1);
}

// Parse the python dict header to extract descr and shape.
// Example header: {'descr': '<f4', 'fortran_order': False, 'shape': (8, 4, 192),}
static inline bool parse_npy_header(const std::string& header, NpyDtype& dtype, std::vector<int64_t>& shape) {
  // Find descr
  size_t pos = header.find("'descr'");
  if (pos == std::string::npos) return false;
  pos = header.find(':', pos);
  if (pos == std::string::npos) return false;
  pos = header.find('\'', pos);
  if (pos == std::string::npos) return false;
  ++pos;
  char byteorder = header[pos];     // '<' (little-endian) or '>' (big-endian)
  char typechar = header[pos + 1];  // 'f', 'i', 'u', ...
  std::string size_str = header.substr(pos + 2, header.find('\'', pos + 2) - (pos + 2));
  int itemsize = std::atoi(size_str.c_str());

  if (typechar == 'f' && itemsize == 4) {
    dtype = NPY_F32;
  } else if (typechar == 'i' && itemsize == 4) {
    dtype = NPY_I32;
  } else if (typechar == 'i' && itemsize == 8) {
    dtype = NPY_I64;
  } else {
    dtype = NPY_UNKNOWN;
  }

  // Find shape
  size_t sp = header.find("'shape'");
  if (sp == std::string::npos) return false;
  sp = header.find('(', sp);
  if (sp == std::string::npos) return false;
  size_t ep = header.find(')', sp);
  if (ep == std::string::npos) return false;
  std::string shape_str = header.substr(sp + 1, ep - sp - 1);

  shape.clear();
  std::stringstream ss(shape_str);
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    tok = trim(tok);
    if (tok.empty()) continue;
    shape.push_back(std::stoll(tok));
  }
  return dtype != NPY_UNKNOWN;
}

// Load a .npy file from disk. Returns empty data on failure.
static inline npy_data_t load_npy(const std::string& path) {
  npy_data_t result;
  std::ifstream f(path, std::ios::binary);
  if (!f.is_open()) return result;

  // Magic: \x93NUMPY
  char magic[6];
  f.read(magic, 6);
  if (std::memcmp(magic, "\x93NUMPY", 6) != 0) return result;

  uint8_t major, minor;
  f.read(reinterpret_cast<char*>(&major), 1);
  f.read(reinterpret_cast<char*>(&minor), 1);

  uint32_t header_len = 0;
  if (major == 1) {
    uint16_t hlen;
    f.read(reinterpret_cast<char*>(&hlen), 2);
    header_len = hlen;
  } else if (major == 2) {
    uint32_t hlen;
    f.read(reinterpret_cast<char*>(&hlen), 4);
    header_len = hlen;
  } else {
    return result;
  }

  std::string header(header_len, '\0');
  f.read(&header[0], header_len);

  if (!parse_npy_header(header, result.dtype, result.shape)) return result;

  // Compute numel
  result.numel = 1;
  for (int64_t d : result.shape) {
    if (d > 0) {
      result.numel *= d;
    } else if (d == 0) {
      result.numel = 0;
      break;
    }
  }

  // Read data
  int itemsize = 4;
  if (result.dtype == NPY_I64) itemsize = 8;
  result.raw.resize(result.numel * itemsize);
  if (result.numel > 0) {
    f.read(result.raw.data(), result.numel * itemsize);
  }
  return result;
}

// Helper: join path components
static inline std::string join(const std::string& a, const std::string& b) {
  if (a.empty()) return b;
  if (a.back() == '/') return a + b;
  return a + "/" + b;
}

}  // namespace test_npy
}  // namespace ksana_llm

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

class KunlunAttentionKernelTest : public testing::Test {
 public:
  static void SetUpTestSuite() {
    XPU_CHECK(xpu_current_device(&device_));
    XPU_CHECK(xpu_stream_create(&stream_));
    ctx_ = xdnn::create_context();
    ctx_->set_stream(stream_);
    xpuctx_ = xmlir_rt::getXpuKernelContext(/*useCpu=*/false);
    xpuctx_->set_stream(stream_, false);
    XPU_CHECK(xpu_wait(stream_));

    // Generate PyTorch baselines for MLA attention numeric tests.
    // The Python script writes .npy files under <source_dir>/mla_baselines/,
    // where <source_dir> is the directory of this .cpp file (via __FILE__).
    // Baselines are NOT stored in the repo — they are regenerated on every
    // test run and removed in TearDownTestSuite. Using __FILE__-based path
    // decouples the test from the runtime working directory (cwd).
    GenerateBaselines();
  }

  static void TearDownTestSuite() {
    XPU_CHECK(xpu_wait(stream_));
    CleanupBaselines();
  }

  void SetUp() override { XPU_CHECK(xpu_wait(stream_)); }
  void TearDown() override { XPU_CHECK(xpu_wait(stream_)); }

 protected:
  XPUStream& stream = stream_;
  xdnn::Context*& ctx = ctx_;
  xpytorch::xpu::api::Context*& xpuctx = xpuctx_;

  static int device_;
  static XPUStream stream_;
  static xdnn::Context* ctx_;
  static xpytorch::xpu::api::Context* xpuctx_;

  // Baselines directory is anchored to this source file's directory (via
  // __FILE__), NOT to the runtime cwd. This makes dump and load paths
  // consistent regardless of where the test binary is invoked from.
  // Layout: <source_dir>/mla_baselines/<case>/*.npy
  static std::string BaselinesDir() {
    std::filesystem::path src(__FILE__);
    return (src.parent_path() / "mla_baselines").string();
  }

 private:
  // Locate generate_mla_baselines.py relative to this source file:
  //   <repo>/csrc/kernels/kunlun/generate_mla_baselines.py
  static std::string FindPythonScript() {
    std::filesystem::path src(__FILE__);
    return (src.parent_path() / "generate_mla_baselines.py").string();
  }

  // Invoke the Python script to regenerate baselines under
  // <source_dir>/mla_baselines/. Fast (~2-3s with torch on CPU), so always
  // regenerate fresh — no caching.
  static void GenerateBaselines() {
    std::string baselines_dir = BaselinesDir();
    std::filesystem::remove_all(baselines_dir);

    std::string script = FindPythonScript();
    if (!std::filesystem::exists(script)) {
      KLLM_LOG_WARNING << "generate_mla_baselines.py not found at " << script << ", MLA numeric tests will be skipped.";
      return;
    }

    std::string command = fmt::format("python3 {} --outdir {}", script, baselines_dir);
    KLLM_LOG_INFO << "Generating MLA baselines: " << command;
    int rc = std::system(command.c_str());
    if (rc != 0 || !WIFEXITED(rc) || WEXITSTATUS(rc) != 0) {
      KLLM_LOG_WARNING << "Baseline generation failed (rc=" << rc << "), MLA numeric tests will be skipped.";
    } else {
      KLLM_LOG_INFO << "MLA baselines generated at " << baselines_dir;
    }
  }

  // Remove the generated baselines directory (called after all tests finish).
  static void CleanupBaselines() {
    std::string baselines_dir = BaselinesDir();
    if (!baselines_dir.empty() && std::filesystem::exists(baselines_dir)) {
      std::error_code ec;
      std::filesystem::remove_all(baselines_dir, ec);
      if (ec) {
        KLLM_LOG_WARNING << "Failed to remove baselines dir " << baselines_dir << ": " << ec.message();
      } else {
        KLLM_LOG_INFO << "Cleaned up MLA baselines: " << baselines_dir;
      }
    }
  }
};

int KunlunAttentionKernelTest::device_ = -1;
XPUStream KunlunAttentionKernelTest::stream_ = nullptr;
xdnn::Context* KunlunAttentionKernelTest::ctx_ = nullptr;
xpytorch::xpu::api::Context* KunlunAttentionKernelTest::xpuctx_ = nullptr;

#ifdef ENABLE_KUNLUN

// ── XpuRotaryEmbeddingNeox (float16 precision test) ──────────────────────────
// Cache layout (from host-fallback in flash_attention_layer.cpp):
//   cos_sin_cache[pos * rot_dim + i]            = cos(pos, dim_i)  i in [0, half_dim)
//   cos_sin_cache[pos * rot_dim + half_dim + i]  = sin(pos, dim_i)  i in [0, half_dim)
// Neox RoPE formula:
//   q_out[i]          = q[i]          * cos - q[i+half] * sin
//   q_out[i+half_dim] = q[i+half_dim] * cos + q[i]      * sin
// Compute float32 reference, cast to float16, compare (atol=3e-3 for fp16 precision).

TEST_F(KunlunAttentionKernelTest, RotaryEmbeddingNeoxTest) {
  constexpr int num_tokens = 4, num_heads = 2, num_kv_heads = 1;
  constexpr int head_size = 16, rot_dim = 16, half_dim = rot_dim / 2;
  constexpr int max_pos = num_tokens;
  constexpr int64_t q_size = static_cast<int64_t>(num_tokens) * num_heads * head_size;
  constexpr int64_t k_size = static_cast<int64_t>(num_tokens) * num_kv_heads * head_size;
  constexpr int64_t cs_size = static_cast<int64_t>(max_pos) * rot_dim;  // cos_sin_cache

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  // Build cos_sin_cache: simple freq = 1/(10000^(2j/rot_dim))
  std::vector<float> hcache_f(cs_size);
  for (int pos = 0; pos < max_pos; ++pos) {
    for (int j = 0; j < half_dim; ++j) {
      float freq = 1.0f / std::pow(10000.0f, 2.0f * j / rot_dim);
      hcache_f[pos * rot_dim + j] = std::cos(pos * freq);
      hcache_f[pos * rot_dim + half_dim + j] = std::sin(pos * freq);
    }
  }

  std::vector<float> hq_f(q_size), hk_f(k_size);
  std::vector<int64_t> hpos(num_tokens);
  for (auto& v : hq_f) v = dist(rng);
  for (auto& v : hk_f) v = dist(rng);
  for (int i = 0; i < num_tokens; ++i) hpos[i] = i;

  // CPU reference (float32)
  auto apply_rope = [&](std::vector<float>& h, int nheads) {
    std::vector<float> ref(h.size());
    for (int t = 0; t < num_tokens; ++t) {
      int pos = static_cast<int>(hpos[t]);
      const float* cs = hcache_f.data() + pos * rot_dim;
      for (int hh = 0; hh < nheads; ++hh) {
        float* head = h.data() + t * nheads * head_size + hh * head_size;
        float* rhead = ref.data() + t * nheads * head_size + hh * head_size;
        for (int j = 0; j < half_dim; ++j) {
          float cv = cs[j], sv = cs[half_dim + j];
          rhead[j] = head[j] * cv - head[half_dim + j] * sv;
          rhead[half_dim + j] = head[half_dim + j] * cv + head[j] * sv;
        }
      }
    }
    return ref;
  };
  std::vector<float> ref_q = apply_rope(hq_f, num_heads);
  std::vector<float> ref_k = apply_rope(hk_f, num_kv_heads);

  // Upload float, cast to float16 on device
  float *df_q, *df_k, *df_cache, *df_q_out, *df_k_out;
  float16 *dq, *dk, *dcache;
  int64_t* dpos;
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&df_q), q_size * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&df_k), k_size * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&df_cache), cs_size * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&df_q_out), q_size * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&df_k_out), k_size * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dq), q_size * sizeof(float16), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dk), k_size * sizeof(float16), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dcache), cs_size * sizeof(float16), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dpos), num_tokens * sizeof(int64_t), XPU_MEM_MAIN));

  XPU_CHECK(xpu_memcpy(df_q, hq_f.data(), q_size * sizeof(float), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(df_k, hk_f.data(), k_size * sizeof(float), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(df_cache, hcache_f.data(), cs_size * sizeof(float), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(dpos, hpos.data(), num_tokens * sizeof(int64_t), XPU_HOST_TO_DEVICE));
  ASSERT_EQ(0, (XpuCast<float, float16>(xpuctx, df_q, dq, q_size)));
  ASSERT_EQ(0, (XpuCast<float, float16>(xpuctx, df_k, dk, k_size)));
  ASSERT_EQ(0, (XpuCast<float, float16>(xpuctx, df_cache, dcache, cs_size)));
  XPU_CHECK(xpu_wait(stream));

  {
    XpuTimer timer("XpuRotaryEmbeddingNeox<float16>", stream);
    ASSERT_EQ(0, (XpuRotaryEmbeddingNeox<float16>(ctx, dpos, dq, dk, dcache, rot_dim, num_heads, num_kv_heads,
                                                  head_size, num_tokens)));
  }
  XPU_CHECK(xpu_wait(stream));

  // Cast output back to float, compare with reference
  ASSERT_EQ(0, (XpuCast<float16, float>(xpuctx, dq, df_q_out, q_size)));
  ASSERT_EQ(0, (XpuCast<float16, float>(xpuctx, dk, df_k_out, k_size)));
  XPU_CHECK(xpu_wait(stream));

  std::vector<float> out_q(q_size), out_k(k_size);
  XPU_CHECK(xpu_memcpy(out_q.data(), df_q_out, q_size * sizeof(float), XPU_DEVICE_TO_HOST));
  XPU_CHECK(xpu_memcpy(out_k.data(), df_k_out, k_size * sizeof(float), XPU_DEVICE_TO_HOST));

  for (int i = 0; i < q_size; ++i) EXPECT_NEAR(out_q[i], ref_q[i], 3e-3f) << "q mismatch at [" << i << "]";
  for (int i = 0; i < k_size; ++i) EXPECT_NEAR(out_k[i], ref_k[i], 3e-3f) << "k mismatch at [" << i << "]";

  XPU_CHECK(xpu_free(df_q));
  XPU_CHECK(xpu_free(df_k));
  XPU_CHECK(xpu_free(df_cache));
  XPU_CHECK(xpu_free(df_q_out));
  XPU_CHECK(xpu_free(df_k_out));
  XPU_CHECK(xpu_free(dq));
  XPU_CHECK(xpu_free(dk));
  XPU_CHECK(xpu_free(dcache));
  XPU_CHECK(xpu_free(dpos));
}

// ── XpuReshapeCachedKV (float16) ──────────────────────────────────────────────
// Scatter k/v into paged cache via slot_mapping.
// k: [num_tokens, num_heads, head_size]
// k_cache: [num_blocks, num_heads, block_size, head_size]
// slot s -> k_cache[s/block_size, h, s%block_size, d] = k[token, h, d]

TEST_F(KunlunAttentionKernelTest, ReshapeCachedKVTest) {
  constexpr int64_t num_tokens = 4, num_heads = 2, head_size = 8;
  constexpr int64_t block_size = 4, num_blocks = 2;
  // slot_mapping: token i -> slot i (all in block 0)
  std::vector<int32_t> hslots = {0, 1, 2, 3};

  std::mt19937 rng(88);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> hk_f(num_tokens * num_heads * head_size);
  std::vector<float> hv_f(num_tokens * num_heads * head_size);
  for (auto& v : hk_f) v = dist(rng);
  for (auto& v : hv_f) v = dist(rng);

  // Upload float, cast to float16
  float *dfk, *dfv;
  float16 *dk, *dv, *dk_cache, *dv_cache;
  int32_t* dslots;
  int64_t kv_elems = num_tokens * num_heads * head_size;
  int64_t cache_elems = num_blocks * num_heads * block_size * head_size;

  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dfk), kv_elems * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dfv), kv_elems * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dk), kv_elems * sizeof(float16), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dv), kv_elems * sizeof(float16), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dk_cache), cache_elems * sizeof(float16), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dv_cache), cache_elems * sizeof(float16), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dslots), num_tokens * sizeof(int32_t), XPU_MEM_MAIN));

  XPU_CHECK(xpu_memcpy(dfk, hk_f.data(), kv_elems * sizeof(float), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(dfv, hv_f.data(), kv_elems * sizeof(float), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(dslots, hslots.data(), num_tokens * sizeof(int32_t), XPU_HOST_TO_DEVICE));
  ASSERT_EQ(0, (XpuCast<float, float16>(xpuctx, dfk, dk, kv_elems)));
  ASSERT_EQ(0, (XpuCast<float, float16>(xpuctx, dfv, dv, kv_elems)));
  XPU_CHECK(xpu_wait(stream));

  {
    XpuTimer timer("XpuReshapeCachedKV<float16>", stream);
    ASSERT_EQ(0, (XpuReshapeCachedKV<float16>(ctx, dk, dv, dk_cache, dv_cache, dslots, num_tokens, num_blocks,
                                              num_heads, head_size, block_size)));
  }
  XPU_CHECK(xpu_wait(stream));

  // Read back cache as float16, cast to float, verify against input
  std::vector<float16> hk_cache_f16(cache_elems), hv_cache_f16(cache_elems);
  XPU_CHECK(xpu_memcpy(hk_cache_f16.data(), dk_cache, cache_elems * sizeof(float16), XPU_DEVICE_TO_HOST));

  // Cast float16 cache back to float for comparison
  float* dfk_cache_out;
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dfk_cache_out), cache_elems * sizeof(float), XPU_MEM_MAIN));
  ASSERT_EQ(0, (XpuCast<float16, float>(xpuctx, dk_cache, dfk_cache_out, cache_elems)));
  std::vector<float> hk_cache_f(cache_elems);
  XPU_CHECK(xpu_memcpy(hk_cache_f.data(), dfk_cache_out, cache_elems * sizeof(float), XPU_DEVICE_TO_HOST));

  // Verify: k_cache[blk, h, pos, d] == k[token, h, d]  (blk=0, pos=token for slots 0..3)
  for (int token = 0; token < num_tokens; ++token) {
    int slot = hslots[token];
    int blk = slot / block_size, pos = slot % block_size;
    for (int h = 0; h < num_heads; ++h) {
      for (int d = 0; d < head_size; ++d) {
        int cache_idx = ((blk * num_heads + h) * block_size + pos) * head_size + d;
        int kv_idx = (token * num_heads + h) * head_size + d;
        EXPECT_NEAR(hk_cache_f[cache_idx], hk_f[kv_idx], 2e-3f)
            << "k_cache mismatch token=" << token << " h=" << h << " d=" << d;
      }
    }
  }

  XPU_CHECK(xpu_free(dfk));
  XPU_CHECK(xpu_free(dfv));
  XPU_CHECK(xpu_free(dk));
  XPU_CHECK(xpu_free(dv));
  XPU_CHECK(xpu_free(dk_cache));
  XPU_CHECK(xpu_free(dv_cache));
  XPU_CHECK(xpu_free(dslots));
  XPU_CHECK(xpu_free(dfk_cache_out));
}

// ── XpuPagedAttention (float16 smoke test) ────────────────────────────────────
// Decode path: one query token per request, attends to paged KV cache.
// query:   [batch, num_heads * head_dim]
// k/v_cache: [num_blocks, num_kv_heads, block_size, head_dim]
// block_table: [batch, max_blocks_per_seq]
// context_len: [batch]  — number of KV tokens to attend to

TEST_F(KunlunAttentionKernelTest, PagedAttentionTest) {
  constexpr int64_t batch = 2;
  constexpr int64_t num_heads = 4, num_kv_heads = 2, head_dim = 16;
  constexpr int64_t block_size = 4, num_blocks = 4;
  constexpr int64_t ctx_len = 4;     // each seq has 4 KV tokens (fills 1 block)
  constexpr int64_t max_blocks = 2;  // max blocks per sequence
  constexpr float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  constexpr int64_t q_elems = batch * num_heads * head_dim;
  constexpr int64_t cache_elems = num_blocks * num_kv_heads * block_size * head_dim;
  constexpr int64_t out_elems = batch * num_heads * head_dim;

  std::mt19937 rng(77);
  std::uniform_real_distribution<float> flt(-0.5f, 0.5f);

  // Build float host data, cast to float16 on device
  std::vector<float> hq_f(q_elems), hkc_f(cache_elems), hvc_f(cache_elems);
  for (auto& v : hq_f) v = flt(rng);
  for (auto& v : hkc_f) v = flt(rng);
  for (auto& v : hvc_f) v = flt(rng);

  // block_table: seq0 → block 0, seq1 → block 1
  std::vector<int32_t> h_btable(batch * max_blocks, -1);
  h_btable[0 * max_blocks + 0] = 0;  // seq0, block0
  h_btable[1 * max_blocks + 0] = 1;  // seq1, block1

  std::vector<int32_t> h_ctxlen(batch, static_cast<int32_t>(ctx_len));

  float *df_q, *df_kc, *df_vc, *df_out;
  float16 *dq, *dk_cache, *dv_cache, *dout;
  int32_t *d_btable, *d_ctxlen;

  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&df_q), q_elems * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&df_kc), cache_elems * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&df_vc), cache_elems * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&df_out), out_elems * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dq), q_elems * sizeof(float16), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dk_cache), cache_elems * sizeof(float16), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dv_cache), cache_elems * sizeof(float16), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dout), out_elems * sizeof(float16), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&d_btable), batch * max_blocks * sizeof(int32_t), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&d_ctxlen), batch * sizeof(int32_t), XPU_MEM_MAIN));

  XPU_CHECK(xpu_memcpy(df_q, hq_f.data(), q_elems * sizeof(float), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(df_kc, hkc_f.data(), cache_elems * sizeof(float), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(df_vc, hvc_f.data(), cache_elems * sizeof(float), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(d_btable, h_btable.data(), batch * max_blocks * sizeof(int32_t), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(d_ctxlen, h_ctxlen.data(), batch * sizeof(int32_t), XPU_HOST_TO_DEVICE));
  ASSERT_EQ(0, (XpuCast<float, float16>(xpuctx, df_q, dq, q_elems)));
  ASSERT_EQ(0, (XpuCast<float, float16>(xpuctx, df_kc, dk_cache, cache_elems)));
  ASSERT_EQ(0, (XpuCast<float, float16>(xpuctx, df_vc, dv_cache, cache_elems)));
  XPU_CHECK(xpu_wait(stream));

  {
    XpuTimer timer("XpuPagedAttention<float16>", stream);
    ASSERT_EQ(0, (XpuPagedAttention<float16, int32_t>(ctx, dout, dq, dk_cache, dv_cache, num_kv_heads, scale, d_btable,
                                                      d_ctxlen, block_size, ctx_len, batch, num_heads, head_dim,
                                                      max_blocks)));
  }
  XPU_CHECK(xpu_wait(stream));

  // Cast output to float and verify finite + nonzero
  ASSERT_EQ(0, (XpuCast<float16, float>(xpuctx, dout, df_out, out_elems)));
  XPU_CHECK(xpu_wait(stream));
  std::vector<float> h_out(out_elems);
  XPU_CHECK(xpu_memcpy(h_out.data(), df_out, out_elems * sizeof(float), XPU_DEVICE_TO_HOST));

  int nonzero = 0;
  for (int i = 0; i < out_elems; ++i) {
    ASSERT_TRUE(std::isfinite(h_out[i])) << "NaN/Inf at output[" << i << "]";
    if (std::fabs(h_out[i]) > 1e-7f) ++nonzero;
  }
  EXPECT_GT(nonzero, 0) << "All outputs zero — paged attention likely not running";

  XPU_CHECK(xpu_free(df_q));
  XPU_CHECK(xpu_free(df_kc));
  XPU_CHECK(xpu_free(df_vc));
  XPU_CHECK(xpu_free(df_out));
  XPU_CHECK(xpu_free(dq));
  XPU_CHECK(xpu_free(dk_cache));
  XPU_CHECK(xpu_free(dv_cache));
  XPU_CHECK(xpu_free(dout));
  XPU_CHECK(xpu_free(d_btable));
  XPU_CHECK(xpu_free(d_ctxlen));
}

// ── XpuMlaReshapeCachedKV ────────────────────────────────────────────────────
// MLA stores [latent|k_pe] as single head (num_heads=1, block_size=1).
// kv_cache_dim = kv_lora_rank + qk_rope_head_dim = 576 for DeepSeek V2.

TEST_F(KunlunAttentionKernelTest, MlaReshapeCachedKVFp16) {
  constexpr int64_t kv_lora_rank = 512;
  constexpr int64_t qk_rope_head_dim = 64;
  constexpr int64_t kv_cache_dim = kv_lora_rank + qk_rope_head_dim;  // 576
  constexpr int64_t num_tokens = 2;
  constexpr int64_t num_blocks = 4;
  constexpr int64_t block_size = 1;

  // slot_mapping: token 0 → slot 2, token 1 → slot 0
  std::vector<int32_t> h_slots = {2, 0};

  // Input: sequential pattern so we can identify each token's data
  // token0: all 1.0, token1: all -1.0
  std::vector<float> h_kv_data(num_tokens * kv_cache_dim);
  std::fill(h_kv_data.begin(), h_kv_data.begin() + kv_cache_dim, 1.0f);
  std::fill(h_kv_data.begin() + kv_cache_dim, h_kv_data.end(), -1.0f);

  int64_t data_elems = num_tokens * kv_cache_dim;
  // cache layout: [num_blocks, 1, kv_cache_dim] (block_size=1 for MLA)
  int64_t cache_elems = num_blocks * kv_cache_dim;

  float *df_data, *df_cache;
  float16 *d_data, *d_cache;
  int32_t* d_slots;
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&df_data), data_elems * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&df_cache), cache_elems * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&d_data), data_elems * sizeof(float16), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&d_cache), cache_elems * sizeof(float16), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&d_slots), num_tokens * sizeof(int32_t), XPU_MEM_MAIN));

  // Zero-init cache
  std::vector<float> zeros(cache_elems, 0.0f);
  XPU_CHECK(xpu_memcpy(df_cache, zeros.data(), cache_elems * sizeof(float), XPU_HOST_TO_DEVICE));
  ASSERT_EQ(0, (XpuCast<float, float16>(xpuctx, df_cache, d_cache, cache_elems)));

  XPU_CHECK(xpu_memcpy(df_data, h_kv_data.data(), data_elems * sizeof(float), XPU_HOST_TO_DEVICE));
  ASSERT_EQ(0, (XpuCast<float, float16>(xpuctx, df_data, d_data, data_elems)));
  XPU_CHECK(xpu_memcpy(d_slots, h_slots.data(), num_tokens * sizeof(int32_t), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_wait(stream));

  ASSERT_EQ(0, (XpuMlaReshapeCachedKV<float16>(ctx, d_data, d_cache, d_slots, num_tokens, num_blocks, kv_cache_dim,
                                               block_size)));
  XPU_CHECK(xpu_wait(stream));

  // Read back
  ASSERT_EQ(0, (XpuCast<float16, float>(xpuctx, d_cache, df_cache, cache_elems)));
  XPU_CHECK(xpu_wait(stream));
  std::vector<float> h_cache(cache_elems);
  XPU_CHECK(xpu_memcpy(h_cache.data(), df_cache, cache_elems * sizeof(float), XPU_DEVICE_TO_HOST));

  // Verify: slot 2 (block 2) should contain token0 data (1.0)
  //         slot 0 (block 0) should contain token1 data (-1.0)
  for (int64_t d = 0; d < kv_cache_dim; ++d) {
    EXPECT_NEAR(h_cache[2 * kv_cache_dim + d], 1.0f, 2e-3f) << "slot2 mismatch at d=" << d;
    EXPECT_NEAR(h_cache[0 * kv_cache_dim + d], -1.0f, 2e-3f) << "slot0 mismatch at d=" << d;
  }
  // Untouched slots (1 and 3) should remain 0
  for (int64_t d = 0; d < kv_cache_dim; ++d) {
    EXPECT_NEAR(h_cache[1 * kv_cache_dim + d], 0.0f, 2e-3f) << "slot1 not zero at d=" << d;
    EXPECT_NEAR(h_cache[3 * kv_cache_dim + d], 0.0f, 2e-3f) << "slot3 not zero at d=" << d;
  }

  XPU_CHECK(xpu_free(df_data));
  XPU_CHECK(xpu_free(df_cache));
  XPU_CHECK(xpu_free(d_data));
  XPU_CHECK(xpu_free(d_cache));
  XPU_CHECK(xpu_free(d_slots));
}

// ── XpuKpeExpand ─────────────────────────────────────────────────────────────
// Broadcasts k_pe from [tokens, 1, k_rope_dim] → written at offset (v_head_dim + k_nope_dim)
// in output buffer [tokens, head_num, v_head_dim + k_nope_dim + k_rope_dim].
// DeepSeek V2: v_head_dim=128, k_nope_dim=128, k_rope_dim=64, total=320.

TEST_F(KunlunAttentionKernelTest, KpeExpandFp16) {
  constexpr int64_t tokens = 2;
  constexpr int64_t head_num = 4;
  constexpr int64_t v_head_dim = 128;
  constexpr int64_t k_nope_dim = 128;
  constexpr int64_t k_rope_dim = 64;
  constexpr int64_t total_head_dim = v_head_dim + k_nope_dim + k_rope_dim;  // 320

  // k_pe input: [tokens, 1, k_rope_dim]. Fill with distinct per-token pattern.
  constexpr int64_t kpe_elems = tokens * 1 * k_rope_dim;
  std::vector<float> h_kpe(kpe_elems);
  // token0: all 0.5, token1: all -0.5
  std::fill(h_kpe.begin(), h_kpe.begin() + k_rope_dim, 0.5f);
  std::fill(h_kpe.begin() + k_rope_dim, h_kpe.end(), -0.5f);

  // Output buffer: [tokens, head_num, total_head_dim]
  constexpr int64_t out_elems = tokens * head_num * total_head_dim;
  // Pre-fill output with zeros (the non-rope region should stay untouched)
  std::vector<float> h_vk_init(out_elems, 0.0f);

  float *df_kpe, *df_vk;
  float16 *d_kpe, *d_vk;
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&df_kpe), kpe_elems * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&df_vk), out_elems * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&d_kpe), kpe_elems * sizeof(float16), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&d_vk), out_elems * sizeof(float16), XPU_MEM_MAIN));

  XPU_CHECK(xpu_memcpy(df_kpe, h_kpe.data(), kpe_elems * sizeof(float), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(df_vk, h_vk_init.data(), out_elems * sizeof(float), XPU_HOST_TO_DEVICE));
  ASSERT_EQ(0, (XpuCast<float, float16>(xpuctx, df_kpe, d_kpe, kpe_elems)));
  ASSERT_EQ(0, (XpuCast<float, float16>(xpuctx, df_vk, d_vk, out_elems)));
  XPU_CHECK(xpu_wait(stream));

  ASSERT_EQ(0, (XpuKpeExpand<float16>(ctx, d_kpe, d_vk, tokens, head_num, v_head_dim, k_nope_dim, k_rope_dim)));
  XPU_CHECK(xpu_wait(stream));

  ASSERT_EQ(0, (XpuCast<float16, float>(xpuctx, d_vk, df_vk, out_elems)));
  XPU_CHECK(xpu_wait(stream));
  std::vector<float> h_out(out_elems);
  XPU_CHECK(xpu_memcpy(h_out.data(), df_vk, out_elems * sizeof(float), XPU_DEVICE_TO_HOST));

  // Verify: for each token t and each head h, the rope region at offset (v_head_dim + k_nope_dim)
  // should equal the token's k_pe value (broadcast from head 0 to all heads).
  constexpr int64_t rope_offset = v_head_dim + k_nope_dim;
  for (int64_t t = 0; t < tokens; ++t) {
    float expected = (t == 0) ? 0.5f : -0.5f;
    for (int64_t h = 0; h < head_num; ++h) {
      for (int64_t d = 0; d < k_rope_dim; ++d) {
        int64_t idx = (t * head_num + h) * total_head_dim + rope_offset + d;
        EXPECT_NEAR(h_out[idx], expected, 2e-3f) << "token=" << t << " head=" << h << " d=" << d;
      }
    }
  }

  XPU_CHECK(xpu_free(df_kpe));
  XPU_CHECK(xpu_free(df_vk));
  XPU_CHECK(xpu_free(d_kpe));
  XPU_CHECK(xpu_free(d_vk));
}

// ── XpuKpeExpand with v_head_dim=0 ──────────────────────────────────────────
// When v_head_dim=0, the output buffer is [tokens, head_num, k_nope_dim + k_rope_dim]
// and k_rope is written at offset k_nope_dim (=128).

TEST_F(KunlunAttentionKernelTest, KpeExpandFp16VHeadDimZero) {
  constexpr int64_t tokens = 2;
  constexpr int64_t head_num = 4;
  constexpr int64_t v_head_dim = 0;
  constexpr int64_t k_nope_dim = 128;
  constexpr int64_t k_rope_dim = 64;
  constexpr int64_t total_head_dim = v_head_dim + k_nope_dim + k_rope_dim;  // 192

  constexpr int64_t kpe_elems = tokens * 1 * k_rope_dim;
  std::vector<float> h_kpe(kpe_elems);
  std::fill(h_kpe.begin(), h_kpe.begin() + k_rope_dim, 0.5f);
  std::fill(h_kpe.begin() + k_rope_dim, h_kpe.end(), -0.5f);

  constexpr int64_t out_elems = tokens * head_num * total_head_dim;
  std::vector<float> h_vk_init(out_elems, 0.0f);

  float *df_kpe, *df_vk;
  float16 *d_kpe, *d_vk;
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&df_kpe), kpe_elems * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&df_vk), out_elems * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&d_kpe), kpe_elems * sizeof(float16), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&d_vk), out_elems * sizeof(float16), XPU_MEM_MAIN));

  XPU_CHECK(xpu_memcpy(df_kpe, h_kpe.data(), kpe_elems * sizeof(float), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(df_vk, h_vk_init.data(), out_elems * sizeof(float), XPU_HOST_TO_DEVICE));
  ASSERT_EQ(0, (XpuCast<float, float16>(xpuctx, df_kpe, d_kpe, kpe_elems)));
  ASSERT_EQ(0, (XpuCast<float, float16>(xpuctx, df_vk, d_vk, out_elems)));
  XPU_CHECK(xpu_wait(stream));

  ASSERT_EQ(0, (XpuKpeExpand<float16>(ctx, d_kpe, d_vk, tokens, head_num, v_head_dim, k_nope_dim, k_rope_dim)));
  XPU_CHECK(xpu_wait(stream));

  ASSERT_EQ(0, (XpuCast<float16, float>(xpuctx, d_vk, df_vk, out_elems)));
  XPU_CHECK(xpu_wait(stream));
  std::vector<float> h_out(out_elems);
  XPU_CHECK(xpu_memcpy(h_out.data(), df_vk, out_elems * sizeof(float), XPU_DEVICE_TO_HOST));

  constexpr int64_t rope_offset = v_head_dim + k_nope_dim;  // 128
  for (int64_t t = 0; t < tokens; ++t) {
    float expected = (t == 0) ? 0.5f : -0.5f;
    for (int64_t h = 0; h < head_num; ++h) {
      for (int64_t d = 0; d < k_rope_dim; ++d) {
        int64_t idx = (t * head_num + h) * total_head_dim + rope_offset + d;
        EXPECT_NEAR(h_out[idx], expected, 2e-3f) << "token=" << t << " head=" << h << " d=" << d;
      }
    }
  }

  XPU_CHECK(xpu_free(df_kpe));
  XPU_CHECK(xpu_free(df_vk));
  XPU_CHECK(xpu_free(d_kpe));
  XPU_CHECK(xpu_free(d_vk));
}

// ── XpuRmsnormSplitCol ───────────────────────────────────────────────────────
// Fused RMSNorm + column split on qkv_proj output:
//   input [rows, q_lora_rank + kv_lora_rank + qk_rope_head_dim]
// Outputs:
//   output_q: [rows, q_lora_rank] — RMSNorm(q_part) with weight0
//   output_kv: [rows, kv_lora_rank] — RMSNorm(kv_part) with weight1
//   output_latent: [rows, kv_lora_rank + qk_rope_head_dim] — normed_kv concat raw_rope
// DeepSeek V2: q_lora_rank=1536, kv_lora_rank=512, qk_rope_head_dim=64

TEST_F(KunlunAttentionKernelTest, RmsnormSplitColFp16) {
  constexpr int64_t rows = 2;
  constexpr int64_t q_dim = 1536;                   // q_lora_rank (DeepSeek V2)
  constexpr int64_t kv_dim = 512;                   // kv_lora_rank
  constexpr int64_t rot_dim = 64;                   // qk_rope_head_dim
  constexpr int64_t n = q_dim + kv_dim + rot_dim;   // 2112
  constexpr int64_t latent_dim = kv_dim + rot_dim;  // 576
  constexpr float eps = 1e-5f;

  // Input: all 1.0 → RMS = sqrt(mean(1^2)) = 1.0, rmsnorm(x) = x * weight / sqrt(1+eps) ≈ weight
  std::vector<float> h_input(rows * n, 1.0f);
  // weight0 (for q): all 0.5
  std::vector<float> h_weight0(q_dim, 0.5f);
  // weight1 (for kv): all 0.25
  std::vector<float> h_weight1(kv_dim, 0.25f);

  // Pre-computed baseline: RMSNorm(all-ones, dim=n=2112) with eps=1e-5
  // rms = sqrt(sum(1^2)/n + eps) = sqrt(1 + eps) ≈ 1.0
  // output_q[i] = 1.0 / sqrt(1 + eps) * weight0 ≈ 0.5
  // output_kv[i] = 1.0 / sqrt(1 + eps) * weight1 ≈ 0.25
  float rms_inv = 1.0f / std::sqrt(1.0f + eps);
  float expected_q = rms_inv * 0.5f;
  float expected_kv = rms_inv * 0.25f;
  // output_latent: [kv_dim + rot_dim] = cat(rmsnorm(kv_part) with weight1, raw_rope_part)
  // The kv part is rmsnorm with weight1, the rope part is raw (not normalized).
  // Based on the kernel signature: output_latent = [rmsnorm(kv), raw_rope]
  // raw_rope is the last rot_dim elements of input row = 1.0 (unchanged)
  float expected_latent_kv = expected_kv;
  float expected_latent_rope = 1.0f;  // raw, not normalized

  float *df_input, *df_w0, *df_w1, *df_oq, *df_okv, *df_olat;
  float16 *d_input, *d_oq, *d_okv, *d_olat;
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&df_input), rows * n * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&df_w0), q_dim * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&df_w1), kv_dim * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&df_oq), rows * q_dim * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&df_okv), rows * kv_dim * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&df_olat), rows * latent_dim * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&d_input), rows * n * sizeof(float16), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&d_oq), rows * q_dim * sizeof(float16), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&d_okv), rows * kv_dim * sizeof(float16), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&d_olat), rows * latent_dim * sizeof(float16), XPU_MEM_MAIN));

  XPU_CHECK(xpu_memcpy(df_input, h_input.data(), rows * n * sizeof(float), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(df_w0, h_weight0.data(), q_dim * sizeof(float), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(df_w1, h_weight1.data(), kv_dim * sizeof(float), XPU_HOST_TO_DEVICE));
  ASSERT_EQ(0, (XpuCast<float, float16>(xpuctx, df_input, d_input, rows * n)));
  XPU_CHECK(xpu_wait(stream));

  ASSERT_EQ(0, (XpuRmsnormSplitCol<float16, float>(ctx, d_input, df_w0, df_w1, d_oq, d_okv, d_olat, rows, q_dim, kv_dim,
                                                   rot_dim, eps)));
  XPU_CHECK(xpu_wait(stream));

  ASSERT_EQ(0, (XpuCast<float16, float>(xpuctx, d_oq, df_oq, rows * q_dim)));
  ASSERT_EQ(0, (XpuCast<float16, float>(xpuctx, d_okv, df_okv, rows * kv_dim)));
  ASSERT_EQ(0, (XpuCast<float16, float>(xpuctx, d_olat, df_olat, rows * latent_dim)));
  XPU_CHECK(xpu_wait(stream));

  std::vector<float> h_oq(rows * q_dim), h_okv(rows * kv_dim), h_olat(rows * latent_dim);
  XPU_CHECK(xpu_memcpy(h_oq.data(), df_oq, rows * q_dim * sizeof(float), XPU_DEVICE_TO_HOST));
  XPU_CHECK(xpu_memcpy(h_okv.data(), df_okv, rows * kv_dim * sizeof(float), XPU_DEVICE_TO_HOST));
  XPU_CHECK(xpu_memcpy(h_olat.data(), df_olat, rows * latent_dim * sizeof(float), XPU_DEVICE_TO_HOST));

  // Verify output_q
  for (int64_t i = 0; i < rows * q_dim; ++i) {
    EXPECT_NEAR(h_oq[i], expected_q, 5e-3f) << "output_q mismatch at " << i;
  }
  // Verify output_kv
  for (int64_t i = 0; i < rows * kv_dim; ++i) {
    EXPECT_NEAR(h_okv[i], expected_kv, 5e-3f) << "output_kv mismatch at " << i;
  }
  // Verify output_latent: first kv_dim elements should match kv norm, last rot_dim should be raw
  for (int64_t row = 0; row < rows; ++row) {
    for (int64_t d = 0; d < kv_dim; ++d) {
      EXPECT_NEAR(h_olat[row * latent_dim + d], expected_latent_kv, 5e-3f)
          << "latent_kv mismatch row=" << row << " d=" << d;
    }
    for (int64_t d = 0; d < rot_dim; ++d) {
      EXPECT_NEAR(h_olat[row * latent_dim + kv_dim + d], expected_latent_rope, 5e-3f)
          << "latent_rope mismatch row=" << row << " d=" << d;
    }
  }

  XPU_CHECK(xpu_free(df_input));
  XPU_CHECK(xpu_free(df_w0));
  XPU_CHECK(xpu_free(df_w1));
  XPU_CHECK(xpu_free(df_oq));
  XPU_CHECK(xpu_free(df_okv));
  XPU_CHECK(xpu_free(df_olat));
  XPU_CHECK(xpu_free(d_input));
  XPU_CHECK(xpu_free(d_oq));
  XPU_CHECK(xpu_free(d_okv));
  XPU_CHECK(xpu_free(d_olat));
}

// Same as RmsnormSplitColFp16 but with 16-bit (TW=T) norm weights — this mirrors the
// production path after commit 5416bef, where q_a/kv_a_layernorm weights keep their
// native fp16/bf16 dtype instead of being cast to fp32. Guards against the NaN caused by
// reading 16-bit weight bytes through a float*.
TEST_F(KunlunAttentionKernelTest, RmsnormSplitColFp16WeightFp16) {
  constexpr int64_t rows = 2;
  constexpr int64_t q_dim = 1536;                   // q_lora_rank (DeepSeek V2)
  constexpr int64_t kv_dim = 512;                   // kv_lora_rank
  constexpr int64_t rot_dim = 64;                   // qk_rope_head_dim
  constexpr int64_t n = q_dim + kv_dim + rot_dim;   // 2112
  constexpr int64_t latent_dim = kv_dim + rot_dim;  // 576
  constexpr float eps = 1e-5f;

  std::vector<float> h_input(rows * n, 1.0f);
  std::vector<float> h_weight0(q_dim, 0.5f);
  std::vector<float> h_weight1(kv_dim, 0.25f);

  float rms_inv = 1.0f / std::sqrt(1.0f + eps);
  float expected_q = rms_inv * 0.5f;
  float expected_kv = rms_inv * 0.25f;
  float expected_latent_kv = expected_kv;
  float expected_latent_rope = 1.0f;  // raw, not normalized

  // fp32 scratch used only to seed device buffers via XpuCast.
  float *df_input, *df_w0, *df_w1, *df_oq, *df_okv, *df_olat;
  // 16-bit device buffers: input, weights (TW=T), and outputs.
  float16 *d_input, *d_w0, *d_w1, *d_oq, *d_okv, *d_olat;
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&df_input), rows * n * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&df_w0), q_dim * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&df_w1), kv_dim * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&df_oq), rows * q_dim * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&df_okv), rows * kv_dim * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&df_olat), rows * latent_dim * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&d_input), rows * n * sizeof(float16), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&d_w0), q_dim * sizeof(float16), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&d_w1), kv_dim * sizeof(float16), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&d_oq), rows * q_dim * sizeof(float16), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&d_okv), rows * kv_dim * sizeof(float16), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&d_olat), rows * latent_dim * sizeof(float16), XPU_MEM_MAIN));

  XPU_CHECK(xpu_memcpy(df_input, h_input.data(), rows * n * sizeof(float), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(df_w0, h_weight0.data(), q_dim * sizeof(float), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(df_w1, h_weight1.data(), kv_dim * sizeof(float), XPU_HOST_TO_DEVICE));
  ASSERT_EQ(0, (XpuCast<float, float16>(xpuctx, df_input, d_input, rows * n)));
  ASSERT_EQ(0, (XpuCast<float, float16>(xpuctx, df_w0, d_w0, q_dim)));
  ASSERT_EQ(0, (XpuCast<float, float16>(xpuctx, df_w1, d_w1, kv_dim)));
  XPU_CHECK(xpu_wait(stream));

  // TW=T (16-bit) overload — the variant now used by KunlunRmsnormSplitColLayer.
  ASSERT_EQ(0, (XpuRmsnormSplitCol<float16, float16>(ctx, d_input, d_w0, d_w1, d_oq, d_okv, d_olat, rows, q_dim, kv_dim,
                                                     rot_dim, eps)));
  XPU_CHECK(xpu_wait(stream));

  ASSERT_EQ(0, (XpuCast<float16, float>(xpuctx, d_oq, df_oq, rows * q_dim)));
  ASSERT_EQ(0, (XpuCast<float16, float>(xpuctx, d_okv, df_okv, rows * kv_dim)));
  ASSERT_EQ(0, (XpuCast<float16, float>(xpuctx, d_olat, df_olat, rows * latent_dim)));
  XPU_CHECK(xpu_wait(stream));

  std::vector<float> h_oq(rows * q_dim), h_okv(rows * kv_dim), h_olat(rows * latent_dim);
  XPU_CHECK(xpu_memcpy(h_oq.data(), df_oq, rows * q_dim * sizeof(float), XPU_DEVICE_TO_HOST));
  XPU_CHECK(xpu_memcpy(h_okv.data(), df_okv, rows * kv_dim * sizeof(float), XPU_DEVICE_TO_HOST));
  XPU_CHECK(xpu_memcpy(h_olat.data(), df_olat, rows * latent_dim * sizeof(float), XPU_DEVICE_TO_HOST));

  // Inputs (1.0) and weights (0.5 / 0.25) are all exactly representable in fp16, so
  // the only error is the single fp16 rounding of each output. fp16 ULP near 0.5 is
  // ~2^-11 (~4.9e-4) and near 0.25 ~2^-12, so a few ULP of slack (2e-3) is plenty and
  // still tight enough to catch a several-percent precision regression.
  constexpr float kTol = 2e-3f;
  for (int64_t i = 0; i < rows * q_dim; ++i) {
    EXPECT_NEAR(h_oq[i], expected_q, kTol) << "output_q mismatch at " << i;
  }
  for (int64_t i = 0; i < rows * kv_dim; ++i) {
    EXPECT_NEAR(h_okv[i], expected_kv, kTol) << "output_kv mismatch at " << i;
  }
  for (int64_t row = 0; row < rows; ++row) {
    for (int64_t d = 0; d < kv_dim; ++d) {
      EXPECT_NEAR(h_olat[row * latent_dim + d], expected_latent_kv, kTol)
          << "latent_kv mismatch row=" << row << " d=" << d;
    }
    for (int64_t d = 0; d < rot_dim; ++d) {
      EXPECT_NEAR(h_olat[row * latent_dim + kv_dim + d], expected_latent_rope, kTol)
          << "latent_rope mismatch row=" << row << " d=" << d;
    }
  }

  XPU_CHECK(xpu_free(df_input));
  XPU_CHECK(xpu_free(df_w0));
  XPU_CHECK(xpu_free(df_w1));
  XPU_CHECK(xpu_free(df_oq));
  XPU_CHECK(xpu_free(df_okv));
  XPU_CHECK(xpu_free(df_olat));
  XPU_CHECK(xpu_free(d_input));
  XPU_CHECK(xpu_free(d_w0));
  XPU_CHECK(xpu_free(d_w1));
  XPU_CHECK(xpu_free(d_oq));
  XPU_CHECK(xpu_free(d_okv));
  XPU_CHECK(xpu_free(d_olat));
}

// ── XpuDpskDecodeRotaryEmbedding ─────────────────────────────────────────────
// In-place RoPE on query/key at specific offsets.
// Use position=0 where cos(0*freq)=1.0, sin(0*freq)=0.0 → output == input (identity).
// DeepSeek V2: rot_dim=64, query_stride=576 (kv_lora_rank+rope), key_stride=576

TEST_F(KunlunAttentionKernelTest, DpskDecodeRotaryEmbeddingIdentity) {
  constexpr int64_t kv_lora_rank = 512;
  constexpr int64_t qk_rope_head_dim = 64;
  constexpr int64_t rot_dim = qk_rope_head_dim;
  constexpr int64_t query_stride = kv_lora_rank + qk_rope_head_dim;  // 576
  constexpr int64_t key_stride = kv_lora_rank + qk_rope_head_dim;    // 576
  constexpr int64_t query_offset = kv_lora_rank;                     // rope applied at last 64 dims
  constexpr int64_t key_offset = kv_lora_rank;
  constexpr int64_t num_heads = 2;
  constexpr int64_t num_kv_heads = 1;
  constexpr int64_t num_tokens = 2;
  constexpr int64_t max_seq_len = 4;

  // rotary_pos_emb: [max_seq_len, rot_dim] stored as [cos(0..half), sin(0..half)]
  // For position 0: cos=1.0, sin=0.0 → identity rotation
  constexpr int64_t half_dim = rot_dim / 2;  // 32
  std::vector<float> h_emb(max_seq_len * rot_dim, 0.0f);
  // pos=0: cos all 1, sin all 0
  for (int64_t j = 0; j < half_dim; ++j) h_emb[0 * rot_dim + j] = 1.0f;
  // pos=1: cos(freq), sin(freq) — non-trivial
  for (int64_t j = 0; j < half_dim; ++j) {
    float freq = 1.0f / std::pow(10000.0f, 2.0f * j / rot_dim);
    h_emb[1 * rot_dim + j] = std::cos(freq);
    h_emb[1 * rot_dim + half_dim + j] = std::sin(freq);
  }

  // query: [num_tokens, num_heads, query_stride] — fill with 0.3
  constexpr int64_t q_elems = num_tokens * num_heads * query_stride;
  std::vector<float> h_query(q_elems, 0.3f);
  // key: [num_tokens, num_kv_heads, key_stride] — fill with 0.7
  constexpr int64_t k_elems = num_tokens * num_kv_heads * key_stride;
  std::vector<float> h_key(k_elems, 0.7f);

  // Positions: both tokens at position 0 → identity
  std::vector<int64_t> h_pos = {0, 0};

  float *df_emb, *df_q, *df_k;
  float16 *d_q, *d_k;
  int64_t* d_pos;
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&df_emb), max_seq_len * rot_dim * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&df_q), q_elems * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&df_k), k_elems * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&d_q), q_elems * sizeof(float16), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&d_k), k_elems * sizeof(float16), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&d_pos), num_tokens * sizeof(int64_t), XPU_MEM_MAIN));

  XPU_CHECK(xpu_memcpy(df_emb, h_emb.data(), max_seq_len * rot_dim * sizeof(float), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(df_q, h_query.data(), q_elems * sizeof(float), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(df_k, h_key.data(), k_elems * sizeof(float), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(d_pos, h_pos.data(), num_tokens * sizeof(int64_t), XPU_HOST_TO_DEVICE));
  ASSERT_EQ(0, (XpuCast<float, float16>(xpuctx, df_q, d_q, q_elems)));
  ASSERT_EQ(0, (XpuCast<float, float16>(xpuctx, df_k, d_k, k_elems)));
  XPU_CHECK(xpu_wait(stream));

  ASSERT_EQ(0, (XpuDpskDecodeRotaryEmbedding<float16, float, int64_t>(
                   ctx, d_q, d_k, reinterpret_cast<const float*>(df_emb), d_pos, max_seq_len, rot_dim, query_stride,
                   key_stride, query_offset, key_offset, num_heads, num_kv_heads, num_tokens)));
  XPU_CHECK(xpu_wait(stream));

  // Read back query
  ASSERT_EQ(0, (XpuCast<float16, float>(xpuctx, d_q, df_q, q_elems)));
  ASSERT_EQ(0, (XpuCast<float16, float>(xpuctx, d_k, df_k, k_elems)));
  XPU_CHECK(xpu_wait(stream));
  std::vector<float> h_q_out(q_elems), h_k_out(k_elems);
  XPU_CHECK(xpu_memcpy(h_q_out.data(), df_q, q_elems * sizeof(float), XPU_DEVICE_TO_HOST));
  XPU_CHECK(xpu_memcpy(h_k_out.data(), df_k, k_elems * sizeof(float), XPU_DEVICE_TO_HOST));

  // With position=0: cos=1, sin=0 → rope is identity, output should equal input
  // Non-rope region (0:query_offset) should also be unchanged
  for (int64_t i = 0; i < q_elems; ++i) {
    EXPECT_NEAR(h_q_out[i], 0.3f, 3e-3f) << "query changed at pos=0, idx=" << i;
  }
  for (int64_t i = 0; i < k_elems; ++i) {
    EXPECT_NEAR(h_k_out[i], 0.7f, 3e-3f) << "key changed at pos=0, idx=" << i;
  }

  XPU_CHECK(xpu_free(df_emb));
  XPU_CHECK(xpu_free(df_q));
  XPU_CHECK(xpu_free(df_k));
  XPU_CHECK(xpu_free(d_q));
  XPU_CHECK(xpu_free(d_k));
  XPU_CHECK(xpu_free(d_pos));
}

// ── XpuGatherMlaCache ────────────────────────────────────────────────────────
// Gathers kv_c and k_pe from paged cache.
// kv_cache: [NUM_BLOCKS, BLOCK_SIZE, kv_lora_rank + qk_rope_dim]
// Output: kv_c [sum_tokens, kv_lora_rank], k_pe [sum_tokens, qk_rope_dim]

TEST_F(KunlunAttentionKernelTest, GatherMlaCacheFp16) {
  constexpr int64_t kv_lora_rank = 512;
  constexpr int64_t qk_rope_dim = 64;
  constexpr int64_t cache_dim = kv_lora_rank + qk_rope_dim;  // 576
  constexpr int64_t block_size = 1;
  constexpr int64_t num_blocks = 4;
  constexpr int64_t max_num_blocks_per_seq = 2;

  // 2 sequences: seq0 has 1 token, seq1 has 1 token → sum_tokens = 2
  // seq_lod = [0, 1, 2] (CSR-style offsets)
  constexpr int batch_num = 2;
  constexpr int seq_lod_len = batch_num + 1;
  std::vector<int32_t> h_seq_lod = {0, 1, 2};
  // block_table: [batch, max_blocks_per_seq]
  // seq0 uses block 1, seq1 uses block 3
  std::vector<int32_t> h_block_table = {1, -1, 3, -1};

  constexpr int64_t sum_tokens = 2;

  // Fill cache: block[b] position [0] has (kv_c = b*0.1, k_pe = b*0.2) pattern
  int64_t cache_elems = num_blocks * block_size * cache_dim;
  std::vector<float> h_cache(cache_elems, 0.0f);
  for (int64_t b = 0; b < num_blocks; ++b) {
    float* row = h_cache.data() + b * cache_dim;
    for (int64_t d = 0; d < kv_lora_rank; ++d) row[d] = static_cast<float>(b) * 0.1f;
    for (int64_t d = 0; d < qk_rope_dim; ++d) row[kv_lora_rank + d] = static_cast<float>(b) * 0.2f;
  }

  float *df_cache, *df_kvc, *df_kpe;
  float16 *d_cache, *d_kvc, *d_kpe;
  int32_t *d_seq_lod, *d_block_table;
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&df_cache), cache_elems * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&df_kvc), sum_tokens * kv_lora_rank * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&df_kpe), sum_tokens * qk_rope_dim * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&d_cache), cache_elems * sizeof(float16), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&d_kvc), sum_tokens * kv_lora_rank * sizeof(float16), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&d_kpe), sum_tokens * qk_rope_dim * sizeof(float16), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&d_seq_lod), seq_lod_len * sizeof(int32_t), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&d_block_table), batch_num * max_num_blocks_per_seq * sizeof(int32_t),
                       XPU_MEM_MAIN));

  XPU_CHECK(xpu_memcpy(df_cache, h_cache.data(), cache_elems * sizeof(float), XPU_HOST_TO_DEVICE));
  ASSERT_EQ(0, (XpuCast<float, float16>(xpuctx, df_cache, d_cache, cache_elems)));
  XPU_CHECK(xpu_memcpy(d_seq_lod, h_seq_lod.data(), seq_lod_len * sizeof(int32_t), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(d_block_table, h_block_table.data(), batch_num * max_num_blocks_per_seq * sizeof(int32_t),
                       XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_wait(stream));

  ASSERT_EQ(0, (XpuGatherMlaCache<float16, int32_t>(ctx, d_cache, d_kvc, d_kpe, kv_lora_rank, qk_rope_dim,
                                                    h_seq_lod.data(), seq_lod_len, d_seq_lod, d_block_table, block_size,
                                                    max_num_blocks_per_seq)));
  XPU_CHECK(xpu_wait(stream));

  // Read back
  ASSERT_EQ(0, (XpuCast<float16, float>(xpuctx, d_kvc, df_kvc, sum_tokens * kv_lora_rank)));
  ASSERT_EQ(0, (XpuCast<float16, float>(xpuctx, d_kpe, df_kpe, sum_tokens * qk_rope_dim)));
  XPU_CHECK(xpu_wait(stream));

  std::vector<float> h_kvc(sum_tokens * kv_lora_rank), h_kpe(sum_tokens * qk_rope_dim);
  XPU_CHECK(xpu_memcpy(h_kvc.data(), df_kvc, sum_tokens * kv_lora_rank * sizeof(float), XPU_DEVICE_TO_HOST));
  XPU_CHECK(xpu_memcpy(h_kpe.data(), df_kpe, sum_tokens * qk_rope_dim * sizeof(float), XPU_DEVICE_TO_HOST));

  // Seq0 gathered from block 1: kv_c = 1*0.1 = 0.1, k_pe = 1*0.2 = 0.2
  for (int64_t d = 0; d < kv_lora_rank; ++d) {
    EXPECT_NEAR(h_kvc[0 * kv_lora_rank + d], 0.1f, 3e-3f) << "seq0 kv_c at d=" << d;
  }
  for (int64_t d = 0; d < qk_rope_dim; ++d) {
    EXPECT_NEAR(h_kpe[0 * qk_rope_dim + d], 0.2f, 3e-3f) << "seq0 k_pe at d=" << d;
  }
  // Seq1 gathered from block 3: kv_c = 3*0.1 = 0.3, k_pe = 3*0.2 = 0.6
  for (int64_t d = 0; d < kv_lora_rank; ++d) {
    EXPECT_NEAR(h_kvc[1 * kv_lora_rank + d], 0.3f, 3e-3f) << "seq1 kv_c at d=" << d;
  }
  for (int64_t d = 0; d < qk_rope_dim; ++d) {
    EXPECT_NEAR(h_kpe[1 * qk_rope_dim + d], 0.6f, 3e-3f) << "seq1 k_pe at d=" << d;
  }

  XPU_CHECK(xpu_free(df_cache));
  XPU_CHECK(xpu_free(df_kvc));
  XPU_CHECK(xpu_free(df_kpe));
  XPU_CHECK(xpu_free(d_cache));
  XPU_CHECK(xpu_free(d_kvc));
  XPU_CHECK(xpu_free(d_kpe));
  XPU_CHECK(xpu_free(d_seq_lod));
  XPU_CHECK(xpu_free(d_block_table));
}

// ── XpuAttentionMergeState ───────────────────────────────────────────────────
// Merges two attention outputs using their log-sum-exp values.
// Formula: merged = (exp(sa - smax)*va + exp(sb - smax)*vb) / (exp(sa-smax) + exp(sb-smax))
// When sa == sb: merged = (va + vb) / 2 (simple average)

TEST_F(KunlunAttentionKernelTest, AttentionMergeStateAverage) {
  constexpr int64_t num_tokens = 2;
  constexpr int64_t num_heads = 4;
  constexpr int64_t head_dim = 128;  // DeepSeek v_head_dim
  constexpr int64_t v_elems = num_tokens * num_heads * head_dim;
  constexpr int64_t s_elems = num_tokens * num_heads;

  // v_a: all 1.0, v_b: all 3.0 → average = 2.0 when LSEs are equal
  std::vector<float> h_va(v_elems, 1.0f);
  std::vector<float> h_vb(v_elems, 3.0f);
  // Equal LSEs → equal weights → average
  std::vector<float> h_sa(s_elems, 0.0f);
  std::vector<float> h_sb(s_elems, 0.0f);

  float *df_va, *df_vb, *d_sa, *d_sb, *df_merged, *d_s_merged;
  float16 *d_va, *d_vb, *d_merged;
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&df_va), v_elems * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&df_vb), v_elems * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&d_sa), s_elems * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&d_sb), s_elems * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&df_merged), v_elems * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&d_s_merged), s_elems * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&d_va), v_elems * sizeof(float16), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&d_vb), v_elems * sizeof(float16), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&d_merged), v_elems * sizeof(float16), XPU_MEM_MAIN));

  XPU_CHECK(xpu_memcpy(df_va, h_va.data(), v_elems * sizeof(float), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(df_vb, h_vb.data(), v_elems * sizeof(float), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(d_sa, h_sa.data(), s_elems * sizeof(float), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(d_sb, h_sb.data(), s_elems * sizeof(float), XPU_HOST_TO_DEVICE));
  ASSERT_EQ(0, (XpuCast<float, float16>(xpuctx, df_va, d_va, v_elems)));
  ASSERT_EQ(0, (XpuCast<float, float16>(xpuctx, df_vb, d_vb, v_elems)));
  XPU_CHECK(xpu_wait(stream));

  ASSERT_EQ(0, (XpuAttentionMergeState<float16>(ctx, d_va, d_sa, d_vb, d_sb, d_merged, d_s_merged, num_tokens,
                                                num_heads, head_dim)));
  XPU_CHECK(xpu_wait(stream));

  ASSERT_EQ(0, (XpuCast<float16, float>(xpuctx, d_merged, df_merged, v_elems)));
  XPU_CHECK(xpu_wait(stream));
  std::vector<float> h_merged(v_elems);
  XPU_CHECK(xpu_memcpy(h_merged.data(), df_merged, v_elems * sizeof(float), XPU_DEVICE_TO_HOST));

  // Pre-computed baseline: average of 1.0 and 3.0 = 2.0
  for (int64_t i = 0; i < v_elems; ++i) {
    EXPECT_NEAR(h_merged[i], 2.0f, 5e-3f) << "merge mismatch at " << i;
  }

  // Verify merged LSE: log(exp(0) + exp(0)) = log(2) ≈ 0.6931
  std::vector<float> h_s_merged(s_elems);
  XPU_CHECK(xpu_memcpy(h_s_merged.data(), d_s_merged, s_elems * sizeof(float), XPU_DEVICE_TO_HOST));
  float expected_lse = std::log(2.0f);
  for (int64_t i = 0; i < s_elems; ++i) {
    EXPECT_NEAR(h_s_merged[i], expected_lse, 1e-4f) << "merged LSE mismatch at " << i;
  }

  XPU_CHECK(xpu_free(df_va));
  XPU_CHECK(xpu_free(df_vb));
  XPU_CHECK(xpu_free(d_sa));
  XPU_CHECK(xpu_free(d_sb));
  XPU_CHECK(xpu_free(df_merged));
  XPU_CHECK(xpu_free(d_s_merged));
  XPU_CHECK(xpu_free(d_va));
  XPU_CHECK(xpu_free(d_vb));
  XPU_CHECK(xpu_free(d_merged));
}

// ── XpuAttentionMergeState with unequal LSE ──────────────────────────────────
// sa >> sb → merged ≈ va (va dominates)

TEST_F(KunlunAttentionKernelTest, AttentionMergeStateDominant) {
  constexpr int64_t num_tokens = 1;
  constexpr int64_t num_heads = 2;
  constexpr int64_t head_dim = 128;
  constexpr int64_t v_elems = num_tokens * num_heads * head_dim;
  constexpr int64_t s_elems = num_tokens * num_heads;

  std::vector<float> h_va(v_elems, 1.0f);
  std::vector<float> h_vb(v_elems, 99.0f);
  // sa = 10.0 >> sb = -10.0 → exp(10)/exp(-10) ratio ~e^20, va dominates
  std::vector<float> h_sa(s_elems, 10.0f);
  std::vector<float> h_sb(s_elems, -10.0f);

  float *df_va, *df_vb, *d_sa, *d_sb, *df_merged;
  float16 *d_va, *d_vb, *d_merged;
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&df_va), v_elems * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&df_vb), v_elems * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&d_sa), s_elems * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&d_sb), s_elems * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&df_merged), v_elems * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&d_va), v_elems * sizeof(float16), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&d_vb), v_elems * sizeof(float16), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&d_merged), v_elems * sizeof(float16), XPU_MEM_MAIN));

  XPU_CHECK(xpu_memcpy(df_va, h_va.data(), v_elems * sizeof(float), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(df_vb, h_vb.data(), v_elems * sizeof(float), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(d_sa, h_sa.data(), s_elems * sizeof(float), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(d_sb, h_sb.data(), s_elems * sizeof(float), XPU_HOST_TO_DEVICE));
  ASSERT_EQ(0, (XpuCast<float, float16>(xpuctx, df_va, d_va, v_elems)));
  ASSERT_EQ(0, (XpuCast<float, float16>(xpuctx, df_vb, d_vb, v_elems)));
  XPU_CHECK(xpu_wait(stream));

  ASSERT_EQ(0, (XpuAttentionMergeState<float16>(ctx, d_va, d_sa, d_vb, d_sb, d_merged, /*s_merged=*/nullptr, num_tokens,
                                                num_heads, head_dim)));
  XPU_CHECK(xpu_wait(stream));

  ASSERT_EQ(0, (XpuCast<float16, float>(xpuctx, d_merged, df_merged, v_elems)));
  XPU_CHECK(xpu_wait(stream));
  std::vector<float> h_merged(v_elems);
  XPU_CHECK(xpu_memcpy(h_merged.data(), df_merged, v_elems * sizeof(float), XPU_DEVICE_TO_HOST));

  // va dominates: merged ≈ 1.0
  for (int64_t i = 0; i < v_elems; ++i) {
    EXPECT_NEAR(h_merged[i], 1.0f, 0.01f) << "dominant merge mismatch at " << i;
  }

  XPU_CHECK(xpu_free(df_va));
  XPU_CHECK(xpu_free(df_vb));
  XPU_CHECK(xpu_free(d_sa));
  XPU_CHECK(xpu_free(d_sb));
  XPU_CHECK(xpu_free(df_merged));
  XPU_CHECK(xpu_free(d_va));
  XPU_CHECK(xpu_free(d_vb));
  XPU_CHECK(xpu_free(d_merged));
}

// ── XpuDpskPrefillRotaryEmbedding (smoke test) ──────────────────────────────
// Verifies the kernel runs without error and produces finite outputs.
// Uses DeepSeek V2 MLA dimensions with position=0 (identity RoPE).

TEST_F(KunlunAttentionKernelTest, DpskPrefillRotaryEmbeddingSmoke) {
  constexpr int64_t num_tokens = 2;
  constexpr int64_t q_head_num = 4;
  constexpr int64_t kv_head_num = 1;
  constexpr int64_t qk_nope_head_dim = 128;
  constexpr int64_t qk_rope_head_dim = 64;
  constexpr int64_t v_head_dim = 128;
  constexpr int64_t q_head_dim = qk_nope_head_dim + qk_rope_head_dim;  // 192
  constexpr int64_t kv_head_dim = v_head_dim;                          // 128
  constexpr int64_t latent_cache_dim = 512;                            // kv_lora_rank
  constexpr int64_t rot_dim = qk_rope_head_dim;                        // 64
  constexpr int64_t max_seq_len = 4;

  // kv input: [num_tokens, kv_head_num, kv_head_dim * 2]
  constexpr int64_t kv_elems = num_tokens * kv_head_num * kv_head_dim * 2;
  // q: [num_tokens, q_head_num, q_head_dim]
  constexpr int64_t q_elems = num_tokens * q_head_num * q_head_dim;
  // latent_cache: [num_tokens, 1, latent_cache_dim + rot_dim]
  constexpr int64_t lc_inner = latent_cache_dim + rot_dim;
  constexpr int64_t lc_elems = num_tokens * 1 * lc_inner;
  // k output: [num_tokens, kv_head_num, kv_head_dim + rot_dim]
  constexpr int64_t k_inner = kv_head_dim + rot_dim;
  constexpr int64_t k_elems = num_tokens * kv_head_num * k_inner;
  // v output: [num_tokens, kv_head_num, kv_head_dim]
  constexpr int64_t v_elems = num_tokens * kv_head_num * kv_head_dim;
  // rotary_pos_emb: [max_seq_len, rot_dim]
  constexpr int64_t emb_elems = max_seq_len * rot_dim;

  // Fill inputs with small values
  std::vector<float> h_kv(kv_elems, 0.1f);
  std::vector<float> h_q(q_elems, 0.2f);
  std::vector<float> h_lc(lc_elems, 0.3f);
  // pos=0: cos=1, sin=0 (identity rope)
  std::vector<float> h_emb(emb_elems, 0.0f);
  int64_t half_dim = rot_dim / 2;
  for (int64_t pos = 0; pos < max_seq_len; ++pos) {
    for (int64_t j = 0; j < half_dim; ++j) {
      float freq = 1.0f / std::pow(10000.0f, 2.0f * j / rot_dim);
      h_emb[pos * rot_dim + j] = std::cos(pos * freq);
      h_emb[pos * rot_dim + half_dim + j] = std::sin(pos * freq);
    }
  }
  std::vector<int64_t> h_pos = {0, 0};

  float *df_kv, *df_q, *df_lc, *d_emb, *df_k, *df_v;
  float16 *d_kv, *d_q, *d_lc, *d_k, *d_v;
  int64_t* d_pos;
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&df_kv), kv_elems * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&df_q), q_elems * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&df_lc), lc_elems * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&d_emb), emb_elems * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&df_k), k_elems * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&df_v), v_elems * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&d_kv), kv_elems * sizeof(float16), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&d_q), q_elems * sizeof(float16), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&d_lc), lc_elems * sizeof(float16), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&d_k), k_elems * sizeof(float16), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&d_v), v_elems * sizeof(float16), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&d_pos), num_tokens * sizeof(int64_t), XPU_MEM_MAIN));

  XPU_CHECK(xpu_memcpy(df_kv, h_kv.data(), kv_elems * sizeof(float), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(df_q, h_q.data(), q_elems * sizeof(float), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(df_lc, h_lc.data(), lc_elems * sizeof(float), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(d_emb, h_emb.data(), emb_elems * sizeof(float), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(d_pos, h_pos.data(), num_tokens * sizeof(int64_t), XPU_HOST_TO_DEVICE));
  ASSERT_EQ(0, (XpuCast<float, float16>(xpuctx, df_kv, d_kv, kv_elems)));
  ASSERT_EQ(0, (XpuCast<float, float16>(xpuctx, df_q, d_q, q_elems)));
  ASSERT_EQ(0, (XpuCast<float, float16>(xpuctx, df_lc, d_lc, lc_elems)));
  XPU_CHECK(xpu_wait(stream));

  int ret = XpuDpskPrefillRotaryEmbedding<float16, float, int64_t>(ctx, d_kv, d_emb, d_q, d_lc, d_k, d_v, d_pos,
                                                                   q_head_num, kv_head_num, q_head_dim, kv_head_dim,
                                                                   latent_cache_dim, rot_dim, num_tokens);
  ASSERT_EQ(0, ret) << "XpuDpskPrefillRotaryEmbedding failed with ret=" << ret;
  XPU_CHECK(xpu_wait(stream));

  // Verify outputs are finite
  ASSERT_EQ(0, (XpuCast<float16, float>(xpuctx, d_k, df_k, k_elems)));
  ASSERT_EQ(0, (XpuCast<float16, float>(xpuctx, d_v, df_v, v_elems)));
  XPU_CHECK(xpu_wait(stream));

  std::vector<float> h_k(k_elems), h_v(v_elems);
  XPU_CHECK(xpu_memcpy(h_k.data(), df_k, k_elems * sizeof(float), XPU_DEVICE_TO_HOST));
  XPU_CHECK(xpu_memcpy(h_v.data(), df_v, v_elems * sizeof(float), XPU_DEVICE_TO_HOST));

  int finite_k = 0, finite_v = 0;
  for (int64_t i = 0; i < k_elems; ++i) {
    ASSERT_TRUE(std::isfinite(h_k[i])) << "k NaN/Inf at " << i;
    if (std::fabs(h_k[i]) > 1e-7f) ++finite_k;
  }
  for (int64_t i = 0; i < v_elems; ++i) {
    ASSERT_TRUE(std::isfinite(h_v[i])) << "v NaN/Inf at " << i;
    if (std::fabs(h_v[i]) > 1e-7f) ++finite_v;
  }
  EXPECT_GT(finite_k, 0) << "k output all zeros";
  EXPECT_GT(finite_v, 0) << "v output all zeros";

  // With position=0 (identity rope), v should equal the second half of kv (split)
  // v = kv[:, :, kv_head_dim:2*kv_head_dim] → all 0.1
  ASSERT_EQ(0, (XpuCast<float16, float>(xpuctx, d_v, df_v, v_elems)));
  XPU_CHECK(xpu_wait(stream));
  XPU_CHECK(xpu_memcpy(h_v.data(), df_v, v_elems * sizeof(float), XPU_DEVICE_TO_HOST));
  for (int64_t i = 0; i < v_elems; ++i) {
    EXPECT_NEAR(h_v[i], 0.1f, 3e-3f) << "v mismatch at " << i;
  }

  XPU_CHECK(xpu_free(df_kv));
  XPU_CHECK(xpu_free(df_q));
  XPU_CHECK(xpu_free(df_lc));
  XPU_CHECK(xpu_free(d_emb));
  XPU_CHECK(xpu_free(df_k));
  XPU_CHECK(xpu_free(df_v));
  XPU_CHECK(xpu_free(d_kv));
  XPU_CHECK(xpu_free(d_q));
  XPU_CHECK(xpu_free(d_lc));
  XPU_CHECK(xpu_free(d_k));
  XPU_CHECK(xpu_free(d_v));
  XPU_CHECK(xpu_free(d_pos));
}

// ── XpuMlaReshapeCachedKV bfloat16 ──────────────────────────────────────────

TEST_F(KunlunAttentionKernelTest, MlaReshapeCachedKVBf16) {
  constexpr int64_t kv_lora_rank = 512;
  constexpr int64_t qk_rope_head_dim = 64;
  constexpr int64_t kv_cache_dim = kv_lora_rank + qk_rope_head_dim;
  constexpr int64_t num_tokens = 2;
  constexpr int64_t num_blocks = 4;
  constexpr int64_t block_size = 1;

  std::vector<int32_t> h_slots = {3, 1};
  int64_t data_elems = num_tokens * kv_cache_dim;
  int64_t cache_elems = num_blocks * kv_cache_dim;

  std::vector<float> h_kv_data(data_elems);
  std::fill(h_kv_data.begin(), h_kv_data.begin() + kv_cache_dim, 2.0f);
  std::fill(h_kv_data.begin() + kv_cache_dim, h_kv_data.end(), -2.0f);

  float *df_data, *df_cache;
  bfloat16 *d_data, *d_cache;
  int32_t* d_slots;
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&df_data), data_elems * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&df_cache), cache_elems * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&d_data), data_elems * sizeof(bfloat16), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&d_cache), cache_elems * sizeof(bfloat16), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&d_slots), num_tokens * sizeof(int32_t), XPU_MEM_MAIN));

  std::vector<float> zeros(cache_elems, 0.0f);
  XPU_CHECK(xpu_memcpy(df_cache, zeros.data(), cache_elems * sizeof(float), XPU_HOST_TO_DEVICE));
  ASSERT_EQ(0, (XpuCast<float, bfloat16>(xpuctx, df_cache, d_cache, cache_elems)));
  XPU_CHECK(xpu_memcpy(df_data, h_kv_data.data(), data_elems * sizeof(float), XPU_HOST_TO_DEVICE));
  ASSERT_EQ(0, (XpuCast<float, bfloat16>(xpuctx, df_data, d_data, data_elems)));
  XPU_CHECK(xpu_memcpy(d_slots, h_slots.data(), num_tokens * sizeof(int32_t), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_wait(stream));

  ASSERT_EQ(0, (XpuMlaReshapeCachedKV<bfloat16>(ctx, d_data, d_cache, d_slots, num_tokens, num_blocks, kv_cache_dim,
                                                block_size)));
  XPU_CHECK(xpu_wait(stream));

  ASSERT_EQ(0, (XpuCast<bfloat16, float>(xpuctx, d_cache, df_cache, cache_elems)));
  XPU_CHECK(xpu_wait(stream));
  std::vector<float> h_cache(cache_elems);
  XPU_CHECK(xpu_memcpy(h_cache.data(), df_cache, cache_elems * sizeof(float), XPU_DEVICE_TO_HOST));

  // slot 3 → token0 (2.0), slot 1 → token1 (-2.0)
  for (int64_t d = 0; d < kv_cache_dim; ++d) {
    EXPECT_NEAR(h_cache[3 * kv_cache_dim + d], 2.0f, 1e-2f) << "slot3 d=" << d;
    EXPECT_NEAR(h_cache[1 * kv_cache_dim + d], -2.0f, 1e-2f) << "slot1 d=" << d;
  }

  XPU_CHECK(xpu_free(df_data));
  XPU_CHECK(xpu_free(df_cache));
  XPU_CHECK(xpu_free(d_data));
  XPU_CHECK(xpu_free(d_cache));
  XPU_CHECK(xpu_free(d_slots));
}

// ── XpuFlashMlaAttention (fp16 numeric correctness) ────────────────────────
// Loads input tensors + PyTorch baseline output from mla_baselines/flash_mla_attention/
//   q: [total_q_tokens, head_num, qk_head_dim]
//   k: [total_kv_tokens, kv_head_num, qk_head_dim]
//   v: [total_kv_tokens, kv_head_num, v_head_dim]
//   output: [total_q_tokens, head_num, v_head_dim]
// Compares kernel output against torch reference (atol=2e-2 for fp16).

TEST_F(KunlunAttentionKernelTest, FlashMlaAttentionFp16Numeric) {
  using namespace test_npy;
  const std::string base = join(BaselinesDir(), "flash_mla_attention");

  npy_data_t nq = load_npy(join(base, "q.npy"));
  if (nq.raw.empty()) {
    GTEST_SKIP() << "Baselines not found at " << base << " (baseline generation failed or python3 unavailable)";
  }
  npy_data_t nk = load_npy(join(base, "k.npy"));
  npy_data_t nv = load_npy(join(base, "v.npy"));
  npy_data_t nqlod = load_npy(join(base, "qlod.npy"));
  npy_data_t nkvlod = load_npy(join(base, "kvlod.npy"));
  npy_data_t nout_ref = load_npy(join(base, "output.npy"));
  npy_data_t nscale = load_npy(join(base, "scale.npy"));
  ASSERT_FALSE(nout_ref.raw.empty()) << "Failed to load output.npy";

  const int64_t q_elems = nq.numel;
  const int64_t k_elems = nk.numel;
  const int64_t v_elems = nv.numel;
  const int64_t out_elems = nout_ref.numel;
  const int64_t qlod_len = nqlod.numel;
  const int64_t kvlod_len = nkvlod.numel;
  const int64_t batch = nqlod.as_i32()[qlod_len - 1] / nqlod.as_i32()[1];  // infer batch
  const int64_t max_seq_q = nqlod.as_i32()[1];
  const int64_t max_seq_kv = nkvlod.as_i32()[1];
  const int64_t head_num = nq.shape[1];
  const int64_t kv_head_num = nk.shape[1];
  const int64_t qk_head_dim = nq.shape[2];
  const int64_t v_head_dim = nv.shape[2];
  const int64_t block_size = 1;
  const float scale = nscale.scalar_f32();
  // xfa::flash_attention_context_vllm internally divides by sqrt(head_dim),
  // so the alpha parameter must be pre-multiplied to compensate.
  const float alpha = scale * std::sqrt(static_cast<float>(qk_head_dim));

  // Copy host → device as fp16
  float *df_q, *df_k, *df_v, *df_out;
  float16 *dq, *dk, *dv, *dout;
  int32_t *d_qlod, *d_kvlod;
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&df_q), q_elems * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&df_k), k_elems * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&df_v), v_elems * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&df_out), out_elems * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dq), q_elems * sizeof(float16), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dk), k_elems * sizeof(float16), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dv), v_elems * sizeof(float16), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dout), out_elems * sizeof(float16), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&d_qlod), qlod_len * sizeof(int32_t), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&d_kvlod), kvlod_len * sizeof(int32_t), XPU_MEM_MAIN));

  XPU_CHECK(xpu_memcpy(df_q, nq.as_f32(), q_elems * sizeof(float), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(df_k, nk.as_f32(), k_elems * sizeof(float), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(df_v, nv.as_f32(), v_elems * sizeof(float), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(d_qlod, nqlod.as_i32(), qlod_len * sizeof(int32_t), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(d_kvlod, nkvlod.as_i32(), kvlod_len * sizeof(int32_t), XPU_HOST_TO_DEVICE));
  ASSERT_EQ(0, (XpuCast<float, float16>(xpuctx, df_q, dq, q_elems)));
  ASSERT_EQ(0, (XpuCast<float, float16>(xpuctx, df_k, dk, k_elems)));
  ASSERT_EQ(0, (XpuCast<float, float16>(xpuctx, df_v, dv, v_elems)));
  XPU_CHECK(xpu_wait(stream));

  {
    XpuTimer timer("XpuFlashMlaAttention<float16>", stream);
    ASSERT_EQ(0, (XpuFlashMlaAttention<float16, int32_t>(
                     ctx, dq, dk, dv, dout, batch, max_seq_q, max_seq_kv, head_num, kv_head_num, qk_head_dim,
                     v_head_dim, /*is_causal=*/true, alpha, nqlod.as_i32(), static_cast<int>(qlod_len), nkvlod.as_i32(),
                     static_cast<int>(kvlod_len), block_size, d_qlod, d_kvlod)));
  }
  XPU_CHECK(xpu_wait(stream));

  // Compare with torch baseline
  ASSERT_EQ(0, (XpuCast<float16, float>(xpuctx, dout, df_out, out_elems)));
  XPU_CHECK(xpu_wait(stream));
  std::vector<float> h_out(out_elems);
  XPU_CHECK(xpu_memcpy(h_out.data(), df_out, out_elems * sizeof(float), XPU_DEVICE_TO_HOST));

  const float* ref = nout_ref.as_f32();
  int mismatches = 0;
  for (int64_t i = 0; i < out_elems; ++i) {
    ASSERT_TRUE(std::isfinite(h_out[i])) << "NaN/Inf at output[" << i << "]";
    if (!std::isfinite(ref[i])) continue;  // skip NaN ref (shouldn't happen)
    if (std::fabs(h_out[i] - ref[i]) > 2e-2f) {
      if (mismatches < 5) {  // print first 5 mismatches
        KLLM_LOG_INFO << fmt::format("FlashMla mismatch at [{}]: got={:.5f} ref={:.5f}", i, h_out[i], ref[i]);
      }
      ++mismatches;
    }
  }
  // Allow up to 5% mismatches (fp16 + different softmax implementations)
  EXPECT_LT(mismatches, out_elems / 20) << "Too many mismatches: " << mismatches << "/" << out_elems;

  XPU_CHECK(xpu_free(df_q));
  XPU_CHECK(xpu_free(df_k));
  XPU_CHECK(xpu_free(df_v));
  XPU_CHECK(xpu_free(df_out));
  XPU_CHECK(xpu_free(dq));
  XPU_CHECK(xpu_free(dk));
  XPU_CHECK(xpu_free(dv));
  XPU_CHECK(xpu_free(dout));
  XPU_CHECK(xpu_free(d_qlod));
  XPU_CHECK(xpu_free(d_kvlod));
}

// ── XpuPagedMlaAttention (fp16 numeric correctness) ─────────────────────────
// Loads input tensors + PyTorch baseline output from mla_baselines/paged_mla_attention/
//   query:    [batch * q_seq_len, num_heads, kv_lora_rank + qk_rope_dim]
//   kv_cache: [num_blocks, block_size, kv_lora_rank + qk_rope_dim]
//   output:   [batch * q_seq_len, num_heads, kv_lora_rank]
// Compares kernel output against torch reference (atol=2e-2 for fp16).

TEST_F(KunlunAttentionKernelTest, PagedMlaAttentionFp16Numeric) {
  using namespace test_npy;
  const std::string base = join(BaselinesDir(), "paged_mla_attention");

  npy_data_t nq = load_npy(join(base, "query.npy"));
  if (nq.raw.empty()) {
    GTEST_SKIP() << "Baselines not found at " << base << " (baseline generation failed or python3 unavailable)";
  }
  npy_data_t ncache = load_npy(join(base, "kv_cache.npy"));
  npy_data_t nbtable = load_npy(join(base, "block_table.npy"));
  npy_data_t nctxlen = load_npy(join(base, "context_len.npy"));
  npy_data_t nout_ref = load_npy(join(base, "output.npy"));
  npy_data_t nscale = load_npy(join(base, "scale.npy"));
  ASSERT_FALSE(nout_ref.raw.empty()) << "Failed to load output.npy";

  const int64_t q_elems = nq.numel;
  const int64_t cache_elems = ncache.numel;
  const int64_t out_elems = nout_ref.numel;
  const int64_t batch = nq.shape[0];
  const int64_t num_heads = nq.shape[1];
  const int64_t q_head_dim = nq.shape[2];
  // kv_lora_rank = output's last dim (MLA decode outputs the latent rank)
  const int64_t kv_lora_rank = nout_ref.shape[2];
  const int64_t qk_rope_dim = q_head_dim - kv_lora_rank;
  const int64_t block_size = 1;
  const int64_t ctx_len = nctxlen.as_i32()[0];
  const int64_t max_ctx_len = ctx_len;
  const int64_t max_blocks_per_seq = nbtable.shape[1];
  const float scale = nscale.scalar_f32();

  // Copy host → device as fp16
  float *df_q, *df_cache, *df_out;
  float16 *dq, *d_cache, *dout;
  int32_t *d_block_table, *d_ctxlen;
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&df_q), q_elems * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&df_cache), cache_elems * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&df_out), out_elems * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dq), q_elems * sizeof(float16), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&d_cache), cache_elems * sizeof(float16), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dout), out_elems * sizeof(float16), XPU_MEM_MAIN));
  XPU_CHECK(
      xpu_malloc(reinterpret_cast<void**>(&d_block_table), batch * max_blocks_per_seq * sizeof(int32_t), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&d_ctxlen), batch * sizeof(int32_t), XPU_MEM_MAIN));

  XPU_CHECK(xpu_memcpy(df_q, nq.as_f32(), q_elems * sizeof(float), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(df_cache, ncache.as_f32(), cache_elems * sizeof(float), XPU_HOST_TO_DEVICE));
  XPU_CHECK(
      xpu_memcpy(d_block_table, nbtable.as_i32(), batch * max_blocks_per_seq * sizeof(int32_t), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(d_ctxlen, nctxlen.as_i32(), batch * sizeof(int32_t), XPU_HOST_TO_DEVICE));
  ASSERT_EQ(0, (XpuCast<float, float16>(xpuctx, df_q, dq, q_elems)));
  ASSERT_EQ(0, (XpuCast<float, float16>(xpuctx, df_cache, d_cache, cache_elems)));
  XPU_CHECK(xpu_wait(stream));

  {
    XpuTimer timer("XpuPagedMlaAttention<float16>", stream);
    ASSERT_EQ(0, (XpuPagedMlaAttention<float16, int32_t>(
                     ctx, dout, dq, d_cache, d_block_table, d_ctxlen, nctxlen.as_i32(), batch, /*q_seq_len=*/1,
                     num_heads, kv_lora_rank, qk_rope_dim, scale, block_size, max_ctx_len, max_blocks_per_seq)));
  }
  XPU_CHECK(xpu_wait(stream));

  // Compare with torch baseline
  ASSERT_EQ(0, (XpuCast<float16, float>(xpuctx, dout, df_out, out_elems)));
  XPU_CHECK(xpu_wait(stream));
  std::vector<float> h_out(out_elems);
  XPU_CHECK(xpu_memcpy(h_out.data(), df_out, out_elems * sizeof(float), XPU_DEVICE_TO_HOST));

  const float* ref = nout_ref.as_f32();
  int mismatches = 0;
  for (int64_t i = 0; i < out_elems; ++i) {
    ASSERT_TRUE(std::isfinite(h_out[i])) << "NaN/Inf at output[" << i << "]";
    if (!std::isfinite(ref[i])) continue;
    if (std::fabs(h_out[i] - ref[i]) > 2e-2f) {
      if (mismatches < 5) {
        KLLM_LOG_INFO << fmt::format("PagedMla mismatch at [{}]: got={:.5f} ref={:.5f}", i, h_out[i], ref[i]);
      }
      ++mismatches;
    }
  }
  EXPECT_LT(mismatches, out_elems / 20) << "Too many mismatches: " << mismatches << "/" << out_elems;

  XPU_CHECK(xpu_free(df_q));
  XPU_CHECK(xpu_free(df_cache));
  XPU_CHECK(xpu_free(df_out));
  XPU_CHECK(xpu_free(dq));
  XPU_CHECK(xpu_free(d_cache));
  XPU_CHECK(xpu_free(dout));
  XPU_CHECK(xpu_free(d_block_table));
  XPU_CHECK(xpu_free(d_ctxlen));
}

// ── XpuFlashMlaAttentionWithLse (is_causal=true, fp16 numeric correctness) ──
// Loads input tensors + PyTorch baseline from mla_baselines/flash_mla_with_lse_causal/
// Self-attention: Q × K/V from same tokens, causal mask.
//   softmax_lse: [total_q_tokens, head_num] (unpadded)
//   output: [total_q_tokens, head_num, v_head_dim]
// Compares kernel output AND LSE against torch reference.

TEST_F(KunlunAttentionKernelTest, FlashMlaAttentionWithLseCausalFp16Numeric) {
  using namespace test_npy;
  const std::string base = join(BaselinesDir(), "flash_mla_with_lse_causal");

  npy_data_t nq = load_npy(join(base, "q.npy"));
  if (nq.raw.empty()) {
    GTEST_SKIP() << "Baselines not found at " << base << " (baseline generation failed or python3 unavailable)";
  }
  npy_data_t nk = load_npy(join(base, "k.npy"));
  npy_data_t nv = load_npy(join(base, "v.npy"));
  npy_data_t nqlod = load_npy(join(base, "qlod.npy"));
  npy_data_t nkvlod = load_npy(join(base, "kvlod.npy"));
  npy_data_t nout_ref = load_npy(join(base, "output.npy"));
  npy_data_t nlse_ref = load_npy(join(base, "lse.npy"));
  npy_data_t nscale = load_npy(join(base, "scale.npy"));
  ASSERT_FALSE(nout_ref.raw.empty()) << "Failed to load output.npy";

  const int64_t q_elems = nq.numel;
  const int64_t k_elems = nk.numel;
  const int64_t v_elems = nv.numel;
  const int64_t out_elems = nout_ref.numel;
  const int64_t lse_elems = nlse_ref.numel;
  const int64_t qlod_len = nqlod.numel;
  const int64_t kvlod_len = nkvlod.numel;
  const int64_t batch = nqlod.as_i32()[qlod_len - 1] / nqlod.as_i32()[1];
  const int64_t max_seq_q = nqlod.as_i32()[1];
  const int64_t max_seq_kv = nkvlod.as_i32()[1];
  const int64_t head_num = nq.shape[1];
  const int64_t kv_head_num = nk.shape[1];
  const int64_t qk_head_dim = nq.shape[2];
  const int64_t v_head_dim = nv.shape[2];
  const int64_t block_size = 1;
  const float scale = nscale.scalar_f32();
  // xfa::flash_attention_context_vllm internally divides by sqrt(head_dim),
  // so the alpha parameter must be pre-multiplied to compensate.
  const float alpha = scale * std::sqrt(static_cast<float>(qk_head_dim));

  float *df_q, *df_k, *df_v, *df_out, *d_lse;
  float16 *dq, *dk, *dv, *dout;
  int32_t *d_qlod, *d_kvlod;
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&df_q), q_elems * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&df_k), k_elems * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&df_v), v_elems * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&df_out), out_elems * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&d_lse), lse_elems * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dq), q_elems * sizeof(float16), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dk), k_elems * sizeof(float16), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dv), v_elems * sizeof(float16), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dout), out_elems * sizeof(float16), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&d_qlod), qlod_len * sizeof(int32_t), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&d_kvlod), kvlod_len * sizeof(int32_t), XPU_MEM_MAIN));

  XPU_CHECK(xpu_memcpy(df_q, nq.as_f32(), q_elems * sizeof(float), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(df_k, nk.as_f32(), k_elems * sizeof(float), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(df_v, nv.as_f32(), v_elems * sizeof(float), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(d_qlod, nqlod.as_i32(), qlod_len * sizeof(int32_t), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(d_kvlod, nkvlod.as_i32(), kvlod_len * sizeof(int32_t), XPU_HOST_TO_DEVICE));
  ASSERT_EQ(0, (XpuCast<float, float16>(xpuctx, df_q, dq, q_elems)));
  ASSERT_EQ(0, (XpuCast<float, float16>(xpuctx, df_k, dk, k_elems)));
  ASSERT_EQ(0, (XpuCast<float, float16>(xpuctx, df_v, dv, v_elems)));
  XPU_CHECK(xpu_wait(stream));

  {
    XpuTimer timer("XpuFlashMlaAttentionWithLse<causal=true>", stream);
    ASSERT_EQ(
        0, (XpuFlashMlaAttentionWithLse<float16, int32_t>(
               ctx, dq, dk, dv, dout, d_lse, /*unpadded_lse=*/true, batch, max_seq_q, max_seq_kv, head_num, kv_head_num,
               qk_head_dim, v_head_dim, /*is_causal=*/true, alpha, nqlod.as_i32(), static_cast<int>(qlod_len),
               nkvlod.as_i32(), static_cast<int>(kvlod_len), block_size, d_qlod, d_kvlod)));
  }
  XPU_CHECK(xpu_wait(stream));

  // Compare output
  ASSERT_EQ(0, (XpuCast<float16, float>(xpuctx, dout, df_out, out_elems)));
  XPU_CHECK(xpu_wait(stream));
  std::vector<float> h_out(out_elems);
  XPU_CHECK(xpu_memcpy(h_out.data(), df_out, out_elems * sizeof(float), XPU_DEVICE_TO_HOST));

  const float* ref_out = nout_ref.as_f32();
  int out_mismatches = 0;
  for (int64_t i = 0; i < out_elems; ++i) {
    ASSERT_TRUE(std::isfinite(h_out[i])) << "output NaN/Inf at [" << i << "]";
    if (!std::isfinite(ref_out[i])) continue;
    if (std::fabs(h_out[i] - ref_out[i]) > 2e-2f) {
      if (out_mismatches < 5) {
        KLLM_LOG_INFO << fmt::format("CausalLSE output mismatch [{}]: got={:.5f} ref={:.5f}", i, h_out[i], ref_out[i]);
      }
      ++out_mismatches;
    }
  }
  EXPECT_LT(out_mismatches, out_elems / 20) << "Too many output mismatches: " << out_mismatches << "/" << out_elems;

  // Compare LSE
  // xfa outputs LSE in [head_num, batch, max_seq_q] layout;
  // reference is in [total_tokens, head_num] layout. Reorder for comparison.
  std::vector<float> h_lse(lse_elems);
  XPU_CHECK(xpu_memcpy(h_lse.data(), d_lse, lse_elems * sizeof(float), XPU_DEVICE_TO_HOST));
  const float* ref_lse = nlse_ref.as_f32();
  int lse_mismatches = 0;
  for (int64_t h = 0; h < head_num; ++h) {
    for (int64_t b = 0; b < batch; ++b) {
      for (int64_t s = 0; s < max_seq_q; ++s) {
        int64_t kernel_idx = h * batch * max_seq_q + b * max_seq_q + s;
        int64_t ref_idx = (b * max_seq_q + s) * head_num + h;
        ASSERT_TRUE(std::isfinite(h_lse[kernel_idx])) << "LSE NaN/Inf at [" << kernel_idx << "]";
        if (!std::isfinite(ref_lse[ref_idx])) continue;
        if (std::fabs(h_lse[kernel_idx] - ref_lse[ref_idx]) > 5e-2f) {
          if (lse_mismatches < 5) {
            KLLM_LOG_INFO << fmt::format("CausalLSE lse mismatch h={} b={} s={}: got={:.5f} ref={:.5f}", h, b, s,
                                         h_lse[kernel_idx], ref_lse[ref_idx]);
          }
          ++lse_mismatches;
        }
      }
    }
  }
  EXPECT_LT(lse_mismatches, lse_elems / 20) << "Too many LSE mismatches: " << lse_mismatches << "/" << lse_elems;

  XPU_CHECK(xpu_free(df_q));
  XPU_CHECK(xpu_free(df_k));
  XPU_CHECK(xpu_free(df_v));
  XPU_CHECK(xpu_free(df_out));
  XPU_CHECK(xpu_free(d_lse));
  XPU_CHECK(xpu_free(dq));
  XPU_CHECK(xpu_free(dk));
  XPU_CHECK(xpu_free(dv));
  XPU_CHECK(xpu_free(dout));
  XPU_CHECK(xpu_free(d_qlod));
  XPU_CHECK(xpu_free(d_kvlod));
}

// ── XpuFlashMlaAttentionWithLse (is_causal=false, fp16 numeric correctness) ─
// Loads input tensors + PyTorch baseline from mla_baselines/flash_mla_with_lse_noncausal/
// Cross-attention: Q × prefix K/V, no causal mask.
//   qlod ≠ kvlod: Q=4 tokens/batch, KV=6 tokens/batch (prefix cache scenario)
// Compares kernel output AND LSE against torch reference.

TEST_F(KunlunAttentionKernelTest, FlashMlaAttentionWithLseNonCausalFp16Numeric) {
  using namespace test_npy;
  const std::string base = join(BaselinesDir(), "flash_mla_with_lse_noncausal");

  npy_data_t nq = load_npy(join(base, "q.npy"));
  if (nq.raw.empty()) {
    GTEST_SKIP() << "Baselines not found at " << base << " (baseline generation failed or python3 unavailable)";
  }
  npy_data_t nk = load_npy(join(base, "k.npy"));
  npy_data_t nv = load_npy(join(base, "v.npy"));
  npy_data_t nqlod = load_npy(join(base, "qlod.npy"));
  npy_data_t nkvlod = load_npy(join(base, "kvlod.npy"));
  npy_data_t nout_ref = load_npy(join(base, "output.npy"));
  npy_data_t nlse_ref = load_npy(join(base, "lse.npy"));
  npy_data_t nscale = load_npy(join(base, "scale.npy"));
  ASSERT_FALSE(nout_ref.raw.empty()) << "Failed to load output.npy";

  const int64_t q_elems = nq.numel;
  const int64_t k_elems = nk.numel;
  const int64_t v_elems = nv.numel;
  const int64_t out_elems = nout_ref.numel;
  const int64_t lse_elems = nlse_ref.numel;
  const int64_t qlod_len = nqlod.numel;
  const int64_t kvlod_len = nkvlod.numel;
  const int64_t batch = nqlod.as_i32()[qlod_len - 1] / nqlod.as_i32()[1];
  const int64_t max_seq_q = nqlod.as_i32()[1];
  const int64_t max_seq_kv = nkvlod.as_i32()[1];
  const int64_t head_num = nq.shape[1];
  const int64_t kv_head_num = nk.shape[1];
  const int64_t qk_head_dim = nq.shape[2];
  const int64_t v_head_dim = nv.shape[2];
  const int64_t block_size = 1;
  const float scale = nscale.scalar_f32();
  // xfa::flash_attention_context_vllm internally divides by sqrt(head_dim),
  // so the alpha parameter must be pre-multiplied to compensate.
  const float alpha = scale * std::sqrt(static_cast<float>(qk_head_dim));

  float *df_q, *df_k, *df_v, *df_out, *d_lse;
  float16 *dq, *dk, *dv, *dout;
  int32_t *d_qlod, *d_kvlod;
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&df_q), q_elems * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&df_k), k_elems * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&df_v), v_elems * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&df_out), out_elems * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&d_lse), lse_elems * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dq), q_elems * sizeof(float16), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dk), k_elems * sizeof(float16), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dv), v_elems * sizeof(float16), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dout), out_elems * sizeof(float16), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&d_qlod), qlod_len * sizeof(int32_t), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&d_kvlod), kvlod_len * sizeof(int32_t), XPU_MEM_MAIN));

  XPU_CHECK(xpu_memcpy(df_q, nq.as_f32(), q_elems * sizeof(float), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(df_k, nk.as_f32(), k_elems * sizeof(float), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(df_v, nv.as_f32(), v_elems * sizeof(float), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(d_qlod, nqlod.as_i32(), qlod_len * sizeof(int32_t), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_memcpy(d_kvlod, nkvlod.as_i32(), kvlod_len * sizeof(int32_t), XPU_HOST_TO_DEVICE));
  ASSERT_EQ(0, (XpuCast<float, float16>(xpuctx, df_q, dq, q_elems)));
  ASSERT_EQ(0, (XpuCast<float, float16>(xpuctx, df_k, dk, k_elems)));
  ASSERT_EQ(0, (XpuCast<float, float16>(xpuctx, df_v, dv, v_elems)));
  XPU_CHECK(xpu_wait(stream));

  {
    XpuTimer timer("XpuFlashMlaAttentionWithLse<causal=false>", stream);
    ASSERT_EQ(
        0, (XpuFlashMlaAttentionWithLse<float16, int32_t>(
               ctx, dq, dk, dv, dout, d_lse, /*unpadded_lse=*/true, batch, max_seq_q, max_seq_kv, head_num, kv_head_num,
               qk_head_dim, v_head_dim, /*is_causal=*/false, alpha, nqlod.as_i32(), static_cast<int>(qlod_len),
               nkvlod.as_i32(), static_cast<int>(kvlod_len), block_size, d_qlod, d_kvlod)));
  }
  XPU_CHECK(xpu_wait(stream));

  // Compare output
  ASSERT_EQ(0, (XpuCast<float16, float>(xpuctx, dout, df_out, out_elems)));
  XPU_CHECK(xpu_wait(stream));
  std::vector<float> h_out(out_elems);
  XPU_CHECK(xpu_memcpy(h_out.data(), df_out, out_elems * sizeof(float), XPU_DEVICE_TO_HOST));

  const float* ref_out = nout_ref.as_f32();
  int out_mismatches = 0;
  for (int64_t i = 0; i < out_elems; ++i) {
    ASSERT_TRUE(std::isfinite(h_out[i])) << "output NaN/Inf at [" << i << "]";
    if (!std::isfinite(ref_out[i])) continue;
    if (std::fabs(h_out[i] - ref_out[i]) > 2e-2f) {
      if (out_mismatches < 5) {
        KLLM_LOG_INFO << fmt::format("NonCausalLSE output mismatch [{}]: got={:.5f} ref={:.5f}", i, h_out[i],
                                     ref_out[i]);
      }
      ++out_mismatches;
    }
  }
  EXPECT_LT(out_mismatches, out_elems / 20) << "Too many output mismatches: " << out_mismatches << "/" << out_elems;

  // Compare LSE
  // xfa outputs LSE in [head_num, batch, max_seq_q] layout;
  // reference is in [total_tokens, head_num] layout. Reorder for comparison.
  std::vector<float> h_lse(lse_elems);
  XPU_CHECK(xpu_memcpy(h_lse.data(), d_lse, lse_elems * sizeof(float), XPU_DEVICE_TO_HOST));
  const float* ref_lse = nlse_ref.as_f32();
  int lse_mismatches = 0;
  for (int64_t h = 0; h < head_num; ++h) {
    for (int64_t b = 0; b < batch; ++b) {
      for (int64_t s = 0; s < max_seq_q; ++s) {
        int64_t kernel_idx = h * batch * max_seq_q + b * max_seq_q + s;
        int64_t ref_idx = (b * max_seq_q + s) * head_num + h;
        ASSERT_TRUE(std::isfinite(h_lse[kernel_idx])) << "LSE NaN/Inf at [" << kernel_idx << "]";
        if (!std::isfinite(ref_lse[ref_idx])) continue;
        if (std::fabs(h_lse[kernel_idx] - ref_lse[ref_idx]) > 5e-2f) {
          if (lse_mismatches < 5) {
            KLLM_LOG_INFO << fmt::format("NonCausalLSE lse mismatch h={} b={} s={}: got={:.5f} ref={:.5f}", h, b, s,
                                         h_lse[kernel_idx], ref_lse[ref_idx]);
          }
          ++lse_mismatches;
        }
      }
    }
  }
  EXPECT_LT(lse_mismatches, lse_elems / 20) << "Too many LSE mismatches: " << lse_mismatches << "/" << lse_elems;

  XPU_CHECK(xpu_free(df_q));
  XPU_CHECK(xpu_free(df_k));
  XPU_CHECK(xpu_free(df_v));
  XPU_CHECK(xpu_free(df_out));
  XPU_CHECK(xpu_free(d_lse));
  XPU_CHECK(xpu_free(dq));
  XPU_CHECK(xpu_free(dk));
  XPU_CHECK(xpu_free(dv));
  XPU_CHECK(xpu_free(dout));
  XPU_CHECK(xpu_free(d_qlod));
  XPU_CHECK(xpu_free(d_kvlod));
}

// ── XpuPagedMlaAttention: bf16 vs fp16 vs xbf16 latency + accuracy sweep ─────
// Benchmarks THREE instances of the SAME xfa kernel
// (xfa::speculative_multi_latent_attention, from the deployed torch_xmlir
// libxpu_flash_attention.so — i.e. the exact kernel this environment runs):
//
//   bf16  : <bfloat16 , bfloat16 , bfloat16 , float     , float     , float, int, 0>
//           (KsanaLLM XpuPagedMlaAttention<bfloat16>)
//   fp16  : <float16  , float16  , float16  , float16   , float16   , float, int, 0>
//           (KsanaLLM XpuPagedMlaAttention<float16>, TEW=float)
//   xbf16 : <xbfloat16, xbfloat16, xbfloat16, xbfloat16 , xbfloat16 , float, int, 0>
//           (Kunlunxin "fast bf16 / InITTE"; xsglang path — not wrapped by
//            KsanaLLM, so called directly here)
//
// Paged single-token decode (max_qlen = 1, i.e. NON-MTP forward), sweeping the
// KV sequence length. For each instance it reports the averaged XPU latency
// (warmup + timed loop) and the max / mean absolute error against an fp32 CPU
// reference (the xfa <float,...,0> instance run on a kCPU api::Context over the
// SAME pre-quant float inputs). Inputs are drawn from a fixed per-seqlen seed,
// so the fp32 golden is identical across the three instances and the error
// columns are directly comparable.
//
// The xbf16-vs-bf16 error columns answer the "storage vs compute" question:
// xbfloat16 has the SAME 2-byte storage footprint as bfloat16 but a different
// value encoding + compute path (its own host xbfloat16(float) ctor /
// operator float()). If it were a mis-storage / bit-reinterpret of bf16 the
// error vs fp32 would be O(1)/garbage; a legit 16-bit float shows error on the
// same order as bf16.

TEST_F(KunlunAttentionKernelTest, PagedMlaDecodePerfSweep) {
  namespace api = baidu::xpu::api;

  // ---- fixed MLA decode shape (DeepSeek-V3 style) --------------------------
  // Swept over batch: batch=1 => m (query rows = batch*q_seq_len) = 1, i.e. a
  // single request with a single new token; batch=8 is the multi-request case.
  const std::vector<int64_t> batches = {1, 8};
  const int64_t num_heads = 128;
  const int64_t kv_lora_rank = 512;
  const int64_t qk_rope_dim = 64;
  const int64_t q_head_dim = kv_lora_rank + qk_rope_dim;  // 576
  const int64_t block_size = 64;
  const int64_t q_seq_len = 1;  // single-token / non-MTP decode
  const float scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(q_head_dim)));

  // Iteration counts (overridable via env for quick smoke runs).
  auto env_int = [](const char* name, int fallback) -> int {
    const char* v = std::getenv(name);
    return (v && *v) ? std::atoi(v) : fallback;
  };
  const int iters = std::max(1, env_int("MLA_PERF_ITERS", 50));
  const int warmup = std::max(0, env_int("MLA_PERF_WARMUP", 10));

  const std::vector<int64_t> seqlens = {2048, 4096, 8192};

  struct Row {
    int64_t batch;
    int64_t seqlen;
    const char* inst;
    double avg_us;
    double max_err;
    double mean_err;
    bool has_acc;
  };
  std::vector<Row> table;

  // fp32 CPU reference context (same kernel family, float instance).
  api::Context cpu_ctx(api::kCPU);

  for (int64_t batch : batches) {
    for (int64_t S : seqlens) {
      const int64_t max_blocks_per_seq = (S + block_size - 1) / block_size;
      const int64_t num_blocks = batch * max_blocks_per_seq;  // each seq its own blocks
      const int64_t q_elems = batch * q_seq_len * num_heads * q_head_dim;
      const int64_t cache_elems = num_blocks * block_size * q_head_dim;
      const int64_t out_elems = batch * q_seq_len * num_heads * kv_lora_rank;

      // ---- deterministic host inputs (identical across all instances) --------
      std::mt19937 gen(static_cast<uint32_t>(1234 + S));
      std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
      std::vector<float> hq(q_elems), hcache(cache_elems);
      for (auto& x : hq) x = dist(gen);
      for (auto& x : hcache) x = dist(gen);

      std::vector<int32_t> h_block_table(batch * max_blocks_per_seq);
      for (int64_t b = 0; b < batch; ++b) {
        for (int64_t j = 0; j < max_blocks_per_seq; ++j) {
          h_block_table[b * max_blocks_per_seq + j] = static_cast<int32_t>(b * max_blocks_per_seq + j);
        }
      }
      std::vector<int32_t> h_ctxlen(batch, static_cast<int32_t>(S));
      std::vector<int32_t> h_valid(batch);
      std::iota(h_valid.begin(), h_valid.end(), 0);

      // ---- fp32 CPU golden ----------------------------------------------------
      // VectorParam on a CPU context: host ptr doubles as the "device" ptr.
      std::vector<float> h_ref(out_elems, 0.0f);
      api::VectorParam<int32_t> seq_lengths_cpu = {h_ctxlen.data(), batch, h_ctxlen.data()};
      api::VectorParam<int32_t> valid_batch_cpu = {h_valid.data(), batch, h_valid.data()};
      int ref_ret = xfa::speculative_multi_latent_attention<float, float, float, float, float, float, int32_t, 0>(
          &cpu_ctx, h_ref.data(), hq.data(), /*kv=*/nullptr, hcache.data(), h_block_table.data(), seq_lengths_cpu,
          valid_batch_cpu, batch, q_seq_len, /*max_seq_len=*/S, num_heads, kv_lora_rank, qk_rope_dim,
          /*attn_mask=*/nullptr, scale, block_size, max_blocks_per_seq, /*max_window_size=*/-1,
          /*q_maxptr=*/nullptr, /*kv_cache_maxptr=*/nullptr, /*o_maxptr=*/nullptr);
      const bool have_ref = (ref_ret == 0);
      if (!have_ref) {
        KLLM_LOG_WARNING << "fp32 CPU reference failed (ret=" << ref_ret << ") at seqlen=" << S
                         << "; accuracy columns will be n/a.";
      }

      // ---- device float inputs (shared by bf16/fp16 casts) -------------------
      float *df_q, *df_cache;
      int32_t *d_block_table, *d_ctxlen;
      XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&df_q), q_elems * sizeof(float), XPU_MEM_MAIN));
      XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&df_cache), cache_elems * sizeof(float), XPU_MEM_MAIN));
      XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&d_block_table), batch * max_blocks_per_seq * sizeof(int32_t),
                           XPU_MEM_MAIN));
      XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&d_ctxlen), batch * sizeof(int32_t), XPU_MEM_MAIN));
      XPU_CHECK(xpu_memcpy(df_q, hq.data(), q_elems * sizeof(float), XPU_HOST_TO_DEVICE));
      XPU_CHECK(xpu_memcpy(df_cache, hcache.data(), cache_elems * sizeof(float), XPU_HOST_TO_DEVICE));
      XPU_CHECK(xpu_memcpy(d_block_table, h_block_table.data(), batch * max_blocks_per_seq * sizeof(int32_t),
                           XPU_HOST_TO_DEVICE));
      XPU_CHECK(xpu_memcpy(d_ctxlen, h_ctxlen.data(), batch * sizeof(int32_t), XPU_HOST_TO_DEVICE));
      XPU_CHECK(xpu_wait(stream));

      // averaged wall-clock (µs) over `iters` runs, after `warmup` runs.
      auto bench = [&](auto&& run) -> double {
        for (int i = 0; i < warmup; ++i) run();
        XPU_CHECK(xpu_wait(stream));
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) run();
        XPU_CHECK(xpu_wait(stream));
        auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
      };

      // max / mean abs error of a host-fp32 output vs the golden.
      auto acc = [&](const std::vector<float>& out) -> std::pair<double, double> {
        double mx = 0.0, sum = 0.0;
        for (int64_t i = 0; i < out_elems; ++i) {
          double d = std::fabs(static_cast<double>(out[i]) - static_cast<double>(h_ref[i]));
          mx = std::max(mx, d);
          sum += d;
        }
        return {mx, out_elems ? sum / out_elems : 0.0};
      };

      // ---- bf16 (KsanaLLM wrapper) -------------------------------------------
      {
        bfloat16 *dq, *dcache, *dout;
        float* df_out;
        XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dq), q_elems * sizeof(bfloat16), XPU_MEM_MAIN));
        XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dcache), cache_elems * sizeof(bfloat16), XPU_MEM_MAIN));
        XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dout), out_elems * sizeof(bfloat16), XPU_MEM_MAIN));
        XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&df_out), out_elems * sizeof(float), XPU_MEM_MAIN));
        ASSERT_EQ(0, (XpuCast<float, bfloat16>(xpuctx, df_q, dq, q_elems)));
        ASSERT_EQ(0, (XpuCast<float, bfloat16>(xpuctx, df_cache, dcache, cache_elems)));
        XPU_CHECK(xpu_wait(stream));
        double us = bench([&] {
          XpuPagedMlaAttention<bfloat16, int32_t>(ctx, dout, dq, dcache, d_block_table, d_ctxlen, h_ctxlen.data(),
                                                  batch, q_seq_len, num_heads, kv_lora_rank, qk_rope_dim, scale,
                                                  block_size, S, max_blocks_per_seq);
        });
        std::vector<float> h_out(out_elems);
        ASSERT_EQ(0, (XpuCast<bfloat16, float>(xpuctx, dout, df_out, out_elems)));
        XPU_CHECK(xpu_wait(stream));
        XPU_CHECK(xpu_memcpy(h_out.data(), df_out, out_elems * sizeof(float), XPU_DEVICE_TO_HOST));
        auto [mx, mean] = acc(h_out);
        table.push_back({batch, S, "bf16", us, mx, mean, have_ref});
        XPU_CHECK(xpu_free(dq));
        XPU_CHECK(xpu_free(dcache));
        XPU_CHECK(xpu_free(dout));
        XPU_CHECK(xpu_free(df_out));
      }

      // ---- fp16 (KsanaLLM wrapper) -------------------------------------------
      {
        float16 *dq, *dcache, *dout;
        float* df_out;
        XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dq), q_elems * sizeof(float16), XPU_MEM_MAIN));
        XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dcache), cache_elems * sizeof(float16), XPU_MEM_MAIN));
        XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dout), out_elems * sizeof(float16), XPU_MEM_MAIN));
        XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&df_out), out_elems * sizeof(float), XPU_MEM_MAIN));
        ASSERT_EQ(0, (XpuCast<float, float16>(xpuctx, df_q, dq, q_elems)));
        ASSERT_EQ(0, (XpuCast<float, float16>(xpuctx, df_cache, dcache, cache_elems)));
        XPU_CHECK(xpu_wait(stream));
        double us = bench([&] {
          XpuPagedMlaAttention<float16, int32_t>(ctx, dout, dq, dcache, d_block_table, d_ctxlen, h_ctxlen.data(), batch,
                                                 q_seq_len, num_heads, kv_lora_rank, qk_rope_dim, scale, block_size, S,
                                                 max_blocks_per_seq);
        });
        std::vector<float> h_out(out_elems);
        ASSERT_EQ(0, (XpuCast<float16, float>(xpuctx, dout, df_out, out_elems)));
        XPU_CHECK(xpu_wait(stream));
        XPU_CHECK(xpu_memcpy(h_out.data(), df_out, out_elems * sizeof(float), XPU_DEVICE_TO_HOST));
        auto [mx, mean] = acc(h_out);
        table.push_back({batch, S, "fp16", us, mx, mean, have_ref});
        XPU_CHECK(xpu_free(dq));
        XPU_CHECK(xpu_free(dcache));
        XPU_CHECK(xpu_free(dout));
        XPU_CHECK(xpu_free(df_out));
      }

      // ---- xbf16 (direct xfa call; on-device float<->xbf16 conversion) -------
      // Convert on-device with the deployed baidu::xpu::api::cus_cast<> kernels
      // (exported by libxpuapi.so) — the same encoding path the runtime uses —
      // rather than the host xbfloat16(float) ctor/operator float() (whose
      // xbf16_ieee_* helpers live only in the XTDK host lib, not linked here).
      {
        xbfloat16 *dq, *dcache, *dout;
        float* df_out;
        XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dq), q_elems * sizeof(xbfloat16), XPU_MEM_MAIN));
        XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dcache), cache_elems * sizeof(xbfloat16), XPU_MEM_MAIN));
        XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dout), out_elems * sizeof(xbfloat16), XPU_MEM_MAIN));
        XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&df_out), out_elems * sizeof(float), XPU_MEM_MAIN));
        ASSERT_EQ(0, (xdnn::cus_cast<float, xbfloat16>(ctx, df_q, dq, q_elems)));
        ASSERT_EQ(0, (xdnn::cus_cast<float, xbfloat16>(ctx, df_cache, dcache, cache_elems)));
        XPU_CHECK(xpu_wait(stream));

        api::VectorParam<int32_t> seq_lengths = {h_ctxlen.data(), batch, d_ctxlen};
        api::VectorParam<int32_t> valid_batch = {h_valid.data(), batch, nullptr};
        double us = bench([&] {
          xfa::speculative_multi_latent_attention<xbfloat16, xbfloat16, xbfloat16, xbfloat16, xbfloat16, float, int32_t,
                                                  0>(
              ctx, dout, dq, /*kv=*/nullptr, dcache, d_block_table, seq_lengths, valid_batch, batch, q_seq_len,
              /*max_seq_len=*/S, num_heads, kv_lora_rank, qk_rope_dim, /*attn_mask=*/nullptr, scale, block_size,
              max_blocks_per_seq, /*max_window_size=*/-1, /*q_maxptr=*/nullptr, /*kv_cache_maxptr=*/nullptr,
              /*o_maxptr=*/nullptr);
        });
        std::vector<float> h_out(out_elems);
        ASSERT_EQ(0, (xdnn::cus_cast<xbfloat16, float>(ctx, dout, df_out, out_elems)));
        XPU_CHECK(xpu_wait(stream));
        XPU_CHECK(xpu_memcpy(h_out.data(), df_out, out_elems * sizeof(float), XPU_DEVICE_TO_HOST));
        auto [mx, mean] = acc(h_out);
        table.push_back({batch, S, "xbf16", us, mx, mean, have_ref});
        XPU_CHECK(xpu_free(df_out));
        XPU_CHECK(xpu_free(dq));
        XPU_CHECK(xpu_free(dcache));
        XPU_CHECK(xpu_free(dout));
      }

      XPU_CHECK(xpu_free(df_q));
      XPU_CHECK(xpu_free(df_cache));
      XPU_CHECK(xpu_free(d_block_table));
      XPU_CHECK(xpu_free(d_ctxlen));
    }
  }

  // ---- results table -------------------------------------------------------
  KLLM_LOG_INFO << "===== XpuPagedMlaAttention paged single-token decode (max_qlen=1) — "
                << "heads=" << num_heads << " kv_lora=" << kv_lora_rank << " rope=" << qk_rope_dim
                << " block=" << block_size << " iters=" << iters << " =====";
  KLLM_LOG_INFO << fmt::format("{:>6}  {:>8}  {:<6}  {:>12}  {:>9}  {:>14}  {:>14}", "batch", "seqlen", "inst",
                               "avg_us", "vs_bf16", "max_abs_err", "mean_abs_err");
  double base_us = 0.0;
  for (const Row& r : table) {
    if (std::string(r.inst) == "bf16") base_us = r.avg_us;
    double speedup = (r.avg_us > 0.0 && base_us > 0.0) ? base_us / r.avg_us : 0.0;
    std::string emax = r.has_acc ? fmt::format("{:.6f}", r.max_err) : "n/a";
    std::string emean = r.has_acc ? fmt::format("{:.6f}", r.mean_err) : "n/a";
    KLLM_LOG_INFO << fmt::format("{:>6}  {:>8}  {:<6}  {:>12.3f}  {:>8.2f}x  {:>14}  {:>14}", r.batch, r.seqlen, r.inst,
                                 r.avg_us, speedup, emax, emean);
  }
  KLLM_LOG_INFO << "note: xbf16 shares bf16's 2-byte storage but a different encoding/compute path; "
                   "max_abs_err(xbf16) on the same order as bf16 (both << O(1)) => a real 16-bit float, "
                   "not a re-storage of bf16.";

  // Sanity: no instance should diverge to garbage vs the fp32 reference.
  for (const Row& r : table) {
    if (r.has_acc) {
      EXPECT_LT(r.max_err, 0.5) << r.inst << " (seqlen=" << r.seqlen << ") diverges from fp32 reference";
    }
  }
}

// ── xbfloat16 bit-layout probe (exponent / mantissa width) ──────────────────
// Determines the on-wire format of xbfloat16 by round-tripping probe values
// through the DEPLOYED device kernels baidu::xpu::api::cus_cast<float,xbfloat16>
// and reading back the raw 2-byte pattern. Decodes each pattern under the
// hypothesis "IEEE half (1 sign / 5 exp / 10 mantissa, bias 15)" and prints the
// reconstructed value + the exact float you get back through cus_cast.
//
// The decisive discriminator is dynamic range:
//   * e5m10 (fp16 layout): max finite = 65504; anything larger -> +Inf (0x7c00).
//   * e8m7  (bf16 layout): range ~ fp32; 1e30 stays finite.
// If 65504 stays finite and 131072 / 1e30 become Inf, the layout is fp16's.
TEST_F(KunlunAttentionKernelTest, Xbf16LayoutProbe) {
  const std::vector<float> probes = {
      1.0f,
      2.0f,
      -2.0f,
      3.0f,
      0.5f,
      1.0f + 1.0f / 1024.0f,  // 1 + 2^-10 : representable iff >=10 mantissa bits
      1.0f + 1.0f / 256.0f,   // 1 + 2^-8  : representable by both fp16 and bf16
      65504.0f,               // fp16 max finite
      131072.0f,              // 2^17 : > fp16 max -> Inf iff 5 exp bits
      1e30f                   // huge : finite iff 8 exp bits (bf16), Inf iff 5 (fp16)
  };
  const int n = static_cast<int>(probes.size());

  float *df, *dback;
  xbfloat16* dx;
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&df), n * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dx), n * sizeof(xbfloat16), XPU_MEM_MAIN));
  XPU_CHECK(xpu_malloc(reinterpret_cast<void**>(&dback), n * sizeof(float), XPU_MEM_MAIN));
  XPU_CHECK(xpu_memcpy(df, probes.data(), n * sizeof(float), XPU_HOST_TO_DEVICE));
  XPU_CHECK(xpu_wait(stream));

  ASSERT_EQ(0, (xdnn::cus_cast<float, xbfloat16>(ctx, df, dx, n)));
  ASSERT_EQ(0, (xdnn::cus_cast<xbfloat16, float>(ctx, dx, dback, n)));
  XPU_CHECK(xpu_wait(stream));

  std::vector<uint16_t> raw(n);
  std::vector<float> back(n);
  XPU_CHECK(xpu_memcpy(raw.data(), dx, n * sizeof(xbfloat16), XPU_DEVICE_TO_HOST));
  XPU_CHECK(xpu_memcpy(back.data(), dback, n * sizeof(float), XPU_DEVICE_TO_HOST));

  KLLM_LOG_INFO << "===== xbfloat16 bit-layout probe (device cus_cast) — decoding as e5m10 (IEEE half) =====";
  KLLM_LOG_INFO << fmt::format("{:>14}  {:>8}  {:>4} {:>4} {:>6}  {:>16}  {:>16}", "input", "raw", "s", "exp5", "man10",
                               "e5m10_value", "cast_back");
  bool inf_65504 = false, inf_131072 = false, inf_1e30 = false, has_2m10 = false;
  for (int i = 0; i < n; ++i) {
    uint16_t h = raw[i];
    int sign = (h >> 15) & 0x1;
    int exp5 = (h >> 10) & 0x1f;
    int man10 = h & 0x3ff;
    // reconstruct assuming IEEE half (bias 15)
    double val;
    const char* tag = "";
    if (exp5 == 0x1f) {
      val = man10 ? std::nan("") : std::numeric_limits<double>::infinity();
      tag = man10 ? " NaN" : " Inf";
    } else if (exp5 == 0) {
      val = std::ldexp(static_cast<double>(man10) / 1024.0, -14);  // subnormal
    } else {
      val = std::ldexp(1.0 + static_cast<double>(man10) / 1024.0, exp5 - 15);
    }
    if (sign) val = -val;
    KLLM_LOG_INFO << fmt::format("{:>14.6g}  {:#06x}  {:>4} {:>4} {:>6}  {:>16.6g}  {:>16.6g}{}", probes[i], h, sign,
                                 exp5, man10, val, back[i], tag);
    if (probes[i] == 65504.0f && std::isinf(back[i])) inf_65504 = true;
    if (probes[i] == 131072.0f && std::isinf(back[i])) inf_131072 = true;
    if (probes[i] == 1e30f && std::isinf(back[i])) inf_1e30 = true;
    if (probes[i] == 1.0f + 1.0f / 1024.0f && back[i] != 1.0f) has_2m10 = true;
  }
  KLLM_LOG_INFO << "verdict: exp bits distinguished by range — fp16(e5m10) saturates >65504 to Inf; "
                   "bf16(e8m7) keeps ~fp32 range.";
  KLLM_LOG_INFO << fmt::format("  65504->Inf: {}  |  131072->Inf: {}  |  1e30->Inf: {}  |  1+2^-10 representable: {}",
                               inf_65504, inf_131072, inf_1e30, has_2m10);

  // Assertions encoding the e5m10 (IEEE-half) hypothesis:
  //   - 65504 is the largest finite half -> must survive.
  //   - 131072 and 1e30 exceed it -> must overflow to Inf (proves 5 exponent bits).
  //   - 1 + 2^-10 must be representable -> proves >=10 mantissa bits (bf16's 7 could not).
  EXPECT_FALSE(inf_65504) << "65504 should be finite in fp16-layout xbf16";
  EXPECT_TRUE(inf_131072) << "2^17 should overflow to Inf if xbf16 has only 5 exponent bits";
  EXPECT_TRUE(inf_1e30) << "1e30 should overflow to Inf if xbf16 has only 5 exponent bits (bf16 would keep it finite)";
  EXPECT_TRUE(has_2m10) << "1 + 2^-10 should be representable if xbf16 has >=10 mantissa bits";

  XPU_CHECK(xpu_free(df));
  XPU_CHECK(xpu_free(dx));
  XPU_CHECK(xpu_free(dback));
}

#endif  // ENABLE_KUNLUN

}  // namespace ksana_llm
