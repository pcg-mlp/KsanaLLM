/* Copyright 2026 Tencent Inc.  All rights reserved.

   W4A8 model layer unit tests for Kunlun XPU backend.
   Tests: SplitLayer, RmsnormSplitColLayer, BatchedMatMulLayer,
          KunlunInt8MatMulLayer, KunlunMoeLayer (W4A8 kernel-level),
          FlashMlaAttentionLayer, PagedMlaAttentionLayer.

   Build:
     cmake .. -DWITH_KUNLUN=ON -DWITH_TESTING=ON
     make kunlun_layer_test

   Run:
     ./bin/kunlun_layer_test --gtest_filter=KunlunW4A8LayerTest.*

==============================================================================*/

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

#include "device/common_device.h"
#include "device/device_types.h"
#include "device/kunlun/kunlun_device_context_impl.h"
#include "kernels/kunlun/moe_kernel_wrapper.h"
#include "kernels/kunlun/quant_gemm_wrapper.h"
#include "layers/batched_matmul_layer.h"
#include "layers/flash_mla_attention_layer.h"
#include "layers/kunlun/host_fp_helpers.h"
#include "layers/kunlun_int8_matmul_layer.h"
#include "layers/kunlun_moe_layer.h"
#include "layers/kunlun_rmsnorm_split_col_layer.h"
#include "layers/paged_mla_attention_layer.h"
#include "layers/split_layer.h"
#include "test.h"

namespace ksana_llm {

using kunlun_host::Fp32ToHalf;
using kunlun_host::HalfToFp32;

// ══ Toy DeepSeek-V3 MLA Config ══════════════════════════════════════════════
// Use real DeepSeek-V3 MLA dimensions for attention tests (kernel has strict param constraints)
static constexpr int kNumHeads = 128;
static constexpr int kQLoraRank = 1536;
static constexpr int kKvLoraRank = 512;
static constexpr int kQkNopeHeadDim = 128;
static constexpr int kQkRopeHeadDim = 64;
static constexpr int kVHeadDim = 128;
static constexpr int kBlockTokenNum = 16;
static constexpr int kLayerNum = 2;
static constexpr int kLayerIndex = 0;
static constexpr int kMaxBlockNum = 32;
static constexpr int kMaxPositionEmbeddings = 256;

// ══ BF16 Helpers ═════════════════════════════════════════════════════════════
static std::vector<uint16_t> ToBf16(const std::vector<float>& f) {
  std::vector<uint16_t> h(f.size());
  for (size_t i = 0; i < f.size(); ++i) h[i] = Fp32ToHalf<bfloat16>(f[i]);
  return h;
}

static std::vector<float> FromBf16(const std::vector<uint16_t>& h) {
  std::vector<float> f(h.size());
  for (size_t i = 0; i < h.size(); ++i) f[i] = HalfToFp32<bfloat16>(h[i]);
  return f;
}

// ══ CPU Reference Implementations ═══════════════════════════════════════════
static void CpuRmsNorm(const float* x, float* out, int cols, float eps) {
  float ss = 0.0f;
  for (int j = 0; j < cols; ++j) ss += x[j] * x[j];
  float rms = 1.0f / std::sqrt(ss / static_cast<float>(cols) + eps);
  for (int j = 0; j < cols; ++j) out[j] = x[j] * rms;
}

static void CpuMatmulTransB(const float* A, const float* B, float* C, int m, int n, int k) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      float sum = 0.0f;
      for (int l = 0; l < k; ++l) sum += A[i * k + l] * B[j * k + l];
      C[i * n + j] = sum;
    }
  }
}

static float CpuRelError(const std::vector<float>& ref, const std::vector<float>& act) {
  float max_abs_ref = 0.0f;
  for (size_t i = 0; i < ref.size(); ++i) max_abs_ref = std::max(max_abs_ref, std::fabs(ref[i]));
  // Skip elements whose |ref| is below 1% of max — they contribute noise to relative error
  float skip_threshold = max_abs_ref * 0.01f;

  float max_rel = 0.0f;
  for (size_t i = 0; i < ref.size(); ++i) {
    if (std::fabs(ref[i]) < skip_threshold) continue;
    float denom = std::fabs(ref[i]);
    float rel = std::fabs(ref[i] - act[i]) / denom;
    if (rel > max_rel) max_rel = rel;
  }
  return max_rel;
}

// ══ Testable Subclasses ═════════════════════════════════════════════════════
class TestableSplitLayer : public SplitLayer {
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

class TestableRmsnormSplitColLayer : public KunlunRmsnormSplitColLayer {
 public:
  void SetupForTest(std::shared_ptr<Context> ctx, DataType dtype, float eps, int64_t q_dim, int64_t kv_dim,
                    int64_t rot_dim) {
    context_ = ctx;
    rank_ = 0;
    inter_data_type_ = dtype;
    tp_size_ = 1;
    dp_size_ = 1;
    attn_dp_atp_size_ = 1;
    attn_dp_group_id_ = 0;
    attn_dp_rank_id_ = 0;
    rms_norm_eps_ = eps;
    q_dim_ = q_dim;
    kv_dim_ = kv_dim;
    rot_dim_ = rot_dim;
  }
};

class TestableBatchedMatMulLayer : public BatchedMatMulLayer {
 public:
  void SetupForTest(std::shared_ptr<Context> ctx, DataType dtype) {
    context_ = ctx;
    rank_ = 0;
    inter_data_type_ = dtype;
  }
};

class TestableKunlunInt8MatMulLayer : public KunlunInt8MatMulLayer {
 public:
  void SetupForTest(std::shared_ptr<Context> ctx, DataType dtype) {
    context_ = ctx;
    rank_ = 0;
    inter_data_type_ = dtype;
  }
};

class TestableFlashMlaAttentionLayer : public FlashMlaAttentionLayer {
 public:
  void SetupForTest(std::shared_ptr<Context> ctx, DataType dtype, size_t max_tokens) {
    context_ = ctx;
    rank_ = 0;
    inter_data_type_ = dtype;
    kv_cache_dtype_ = dtype;
    tp_size_ = 1;
    dp_size_ = 1;
    attn_dp_atp_size_ = 1;
    attn_dp_group_id_ = 0;
    attn_dp_rank_id_ = 0;

    layer_num_ = kLayerNum;
    layer_index_ = kLayerIndex;
    block_token_num_ = kBlockTokenNum;
    num_heads_ = kNumHeads;
    num_kv_heads_ = 1;
    head_size_ = kQkNopeHeadDim + kQkRopeHeadDim;
    stride_size_ = kNumHeads * (kQkNopeHeadDim + kQkRopeHeadDim);
    tensor_para_size_ = 1;
    max_position_embeddings_ = kMaxPositionEmbeddings;
    base_ = 10000.0f;
    is_causal_ = true;
    no_rope_ = false;

    q_lora_rank_ = kQLoraRank;
    kv_lora_rank_ = kKvLoraRank;
    qk_nope_head_dim_ = kQkNopeHeadDim;
    qk_rope_head_dim_ = kQkRopeHeadDim;
    v_head_dim_ = kVHeadDim;
    attn_scale_ = 1.0f / std::sqrt(static_cast<float>(kQkNopeHeadDim + kQkRopeHeadDim));

    // Refactored FlashMlaAttentionLayer no longer has max_token_num_; workspace is sized from
    // max_chunked_kv_tokens_ (per-chunk prefix tokens). Set both to the test's token budget.
    max_step_token_num_ = max_tokens;
    max_chunked_kv_tokens_ = max_tokens;
    block_size_ = kBlockTokenNum * (kKvLoraRank + kQkRopeHeadDim) * sizeof(uint16_t);

    size_t ws_bytes = GetWorkspaceSize();
    workspace_buffer_ =
        std::make_shared<Tensor>(MemoryLocation::LOCATION_DEVICE, TYPE_INT8, std::vector<size_t>{ws_bytes});
  }

  void SetCosSinCache(void* ptr) { cos_sin_cache_ptr_ = ptr; }
};

class TestablePagedMlaAttentionLayer : public PagedMlaAttentionLayer {
 public:
  void SetupForTest(std::shared_ptr<Context> ctx, DataType dtype) {
    context_ = ctx;
    rank_ = 0;
    inter_data_type_ = dtype;
    kv_cache_dtype_ = dtype;
    tp_size_ = 1;
    dp_size_ = 1;
    attn_dp_atp_size_ = 1;
    attn_dp_group_id_ = 0;
    attn_dp_rank_id_ = 0;

    layer_num_ = kLayerNum;
    layer_index_ = kLayerIndex;
    block_token_num_ = kBlockTokenNum;
    num_heads_ = kNumHeads;
    num_kv_heads_ = 1;
    head_size_ = kQkNopeHeadDim + kQkRopeHeadDim;
    stride_size_ = kNumHeads * (kQkNopeHeadDim + kQkRopeHeadDim);
    tensor_para_size_ = 1;
    max_position_embeddings_ = kMaxPositionEmbeddings;
    base_ = 10000.0f;
    is_causal_ = true;
    no_rope_ = false;

    q_lora_rank_ = kQLoraRank;
    kv_lora_rank_ = kKvLoraRank;
    qk_nope_head_dim_ = kQkNopeHeadDim;
    qk_rope_head_dim_ = kQkRopeHeadDim;
    v_head_dim_ = kVHeadDim;
    attn_scale_ = 1.0f / std::sqrt(static_cast<float>(kQkNopeHeadDim + kQkRopeHeadDim));

    block_size_ = kBlockTokenNum * (kKvLoraRank + kQkRopeHeadDim) * sizeof(uint16_t);
  }

  void SetCosSinCache(void* ptr) { cos_sin_cache_ptr_ = ptr; }
};

// Testable subclass for KunlunMoeLayer — sets protected members directly to
// bypass Init() (which requires a full RuntimeConfig with parallel_basic_config).
class TestableKunlunMoeLayer : public KunlunMoeLayer {
 public:
  void SetupForTest(std::shared_ptr<Context> ctx, DataType dtype, size_t max_token_num, int layer_idx,
                    size_t expert_num_per_node, size_t hidden_size, size_t inter_size, size_t topk,
                    bool is_w4a8_quantized, size_t ep_size, size_t ep_rank, const std::string& scoring_func = "softmax",
                    bool norm_topk_prob = false, float routed_scaling_factor = 1.0f, uint32_t num_expert_group = 1,
                    uint32_t topk_group = 1, bool use_e_score_correction_bias = false) {
    context_ = ctx;
    rank_ = 0;
    inter_data_type_ = dtype;
    max_token_num_ = max_token_num;
    layer_idx_ = layer_idx;
    expert_num_per_node_ = expert_num_per_node;
    expert_hidden_size_ = hidden_size;
    expert_inter_size_ = inter_size;
    expert_topk_ = topk;
    norm_topk_prob_ = norm_topk_prob;
    scoring_func_ = scoring_func;
    routed_scaling_factor_ = routed_scaling_factor;
    num_expert_group_ = num_expert_group;
    topk_group_ = topk_group;
    use_e_score_correction_bias_ = use_e_score_correction_bias;
    is_w4a8_quantized_ = is_w4a8_quantized;
    ep_size_ = ep_size;
    ep_rank_ = ep_rank;
    expert_num_total_ = expert_num_per_node_ * ep_size_;
  }
};

// ══ Test Fixture ═════════════════════════════════════════════════════════════
class KunlunW4A8LayerTest : public testing::Test {
 protected:
  void SetUp() override {
    context_ = std::make_shared<Context>();
    context_->device = std::make_shared<DeviceContext>(1);
  }
  void TearDown() override {}

  std::shared_ptr<Context> context_{nullptr};
};

// ═══════════════════════════════════════════════════════════════════════════════
// TEST 1: SplitLayer BF16
// ═══════════════════════════════════════════════════════════════════════════════
TEST_F(KunlunW4A8LayerTest, SplitLayerBf16) {
  constexpr int kRows = 4;
  constexpr int kCol0 = 32, kCol1 = 32, kCol2 = 16;
  constexpr int kTotalCols = kCol0 + kCol1 + kCol2;

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  std::vector<float> h_input(kRows * kTotalCols);
  for (auto& v : h_input) v = dist(rng);
  auto h_input_bf16 = ToBf16(h_input);

  Tensor input(MemoryLocation::LOCATION_DEVICE, TYPE_BF16,
               {static_cast<size_t>(kRows), static_cast<size_t>(kTotalCols)});
  MemcpyAsync(input.GetPtr<void>(), h_input_bf16.data(), h_input_bf16.size() * sizeof(uint16_t), MEMCPY_HOST_TO_DEVICE,
              context_->device->GetMemoryManageStream());

  std::vector<Tensor> outputs(3);
  outputs[0] =
      Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16, {static_cast<size_t>(kRows), static_cast<size_t>(kCol0)});
  outputs[1] =
      Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16, {static_cast<size_t>(kRows), static_cast<size_t>(kCol1)});
  outputs[2] =
      Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16, {static_cast<size_t>(kRows), static_cast<size_t>(kCol2)});

  StreamSynchronize(context_->device->GetMemoryManageStream());

  TestableSplitLayer layer;
  layer.SetupForTest(context_, TYPE_BF16);
  Status status = layer.Forward(
      &input, {static_cast<size_t>(kCol0), static_cast<size_t>(kCol1), static_cast<size_t>(kCol2)},
      {&outputs[0], &outputs[1], &outputs[2]});
  StreamSynchronize(context_->device->GetPrimaryComputeStream());
  ASSERT_TRUE(status.OK()) << status.ToString();

  // Verify: bit-exact match (copy only, no computation)
  int cols[3] = {kCol0, kCol1, kCol2};
  int col_offset = 0;
  for (int s = 0; s < 3; ++s) {
    std::vector<uint16_t> h_out(kRows * cols[s]);
    Memcpy(h_out.data(), outputs[s].GetPtr<void>(), h_out.size() * sizeof(uint16_t), MEMCPY_DEVICE_TO_HOST);
    for (int r = 0; r < kRows; ++r) {
      for (int c = 0; c < cols[s]; ++c) {
        EXPECT_EQ(h_out[r * cols[s] + c], h_input_bf16[r * kTotalCols + col_offset + c])
            << "Mismatch at split=" << s << " row=" << r << " col=" << c;
      }
    }
    col_offset += cols[s];
  }
  printf("SplitLayer BF16: rows=%d cols=[%d,%d,%d] — bit-exact PASS\n", kRows, kCol0, kCol1, kCol2);
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST 2: RmsnormSplitColLayer BF16
// ═══════════════════════════════════════════════════════════════════════════════
TEST_F(KunlunW4A8LayerTest, RmsnormSplitColBf16) {
  constexpr int kRows = 4;
  constexpr int64_t kQDim = 1536;
  constexpr int64_t kKvDim = 512;
  constexpr int64_t kRotDim = 64;
  constexpr int64_t kInputCols = kQDim + kKvDim + kRotDim;  // 2112
  constexpr float kEps = 1e-6f;

  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

  // Input hidden states [rows, input_cols] BF16
  std::vector<float> h_input(kRows * kInputCols);
  for (auto& v : h_input) v = dist(rng);

  // Weights [q_dim] and [kv_dim]. Since commit 5416bef, norm weights keep their native
  // activation dtype (BF16 here) instead of being cast to FP32, and KunlunRmsnormSplitColLayer
  // reads them as T (BF16). Build the weight tensors as BF16 to match the layer's GetPtr<T>().
  std::vector<float> h_q_weight(kQDim), h_kv_weight(kKvDim);
  for (auto& v : h_q_weight) v = dist(rng) + 1.0f;
  for (auto& v : h_kv_weight) v = dist(rng) + 1.0f;

  // Upload to device
  auto h_input_bf16 = ToBf16(h_input);
  auto h_q_weight_bf16 = ToBf16(h_q_weight);
  auto h_kv_weight_bf16 = ToBf16(h_kv_weight);
  // BF16-rounded weights for the CPU reference, so it compares apples-to-apples with the
  // BF16 weights actually consumed by the kernel.
  auto h_q_weight_ref = FromBf16(h_q_weight_bf16);
  auto h_kv_weight_ref = FromBf16(h_kv_weight_bf16);
  Tensor input_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16,
                      {static_cast<size_t>(kRows), static_cast<size_t>(kInputCols)});
  Tensor q_weight_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16, {static_cast<size_t>(kQDim)});
  Tensor kv_weight_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16, {static_cast<size_t>(kKvDim)});

  MemcpyAsync(input_tensor.GetPtr<void>(), h_input_bf16.data(), h_input_bf16.size() * sizeof(uint16_t),
              MEMCPY_HOST_TO_DEVICE, context_->device->GetMemoryManageStream());
  MemcpyAsync(q_weight_tensor.GetPtr<void>(), h_q_weight_bf16.data(), kQDim * sizeof(uint16_t), MEMCPY_HOST_TO_DEVICE,
              context_->device->GetMemoryManageStream());
  MemcpyAsync(kv_weight_tensor.GetPtr<void>(), h_kv_weight_bf16.data(), kKvDim * sizeof(uint16_t),
              MEMCPY_HOST_TO_DEVICE, context_->device->GetMemoryManageStream());

  // Output tensors
  Tensor out_q(MemoryLocation::LOCATION_DEVICE, TYPE_BF16, {static_cast<size_t>(kRows), static_cast<size_t>(kQDim)});
  Tensor out_kv(MemoryLocation::LOCATION_DEVICE, TYPE_BF16, {static_cast<size_t>(kRows), static_cast<size_t>(kKvDim)});
  Tensor out_latent(MemoryLocation::LOCATION_DEVICE, TYPE_BF16,
                    {static_cast<size_t>(kRows), static_cast<size_t>(kKvDim + kRotDim)});

  StreamSynchronize(context_->device->GetMemoryManageStream());

  TestableRmsnormSplitColLayer layer;
  layer.SetupForTest(context_, TYPE_BF16, kEps, kQDim, kKvDim, kRotDim);

  std::vector<Tensor> inputs = {input_tensor, q_weight_tensor, kv_weight_tensor};
  std::vector<Tensor> outputs = {out_q, out_kv, out_latent};
  Status status = layer.Forward(inputs, outputs);
  StreamSynchronize(context_->device->GetPrimaryComputeStream());
  ASSERT_TRUE(status.OK()) << status.ToString();

  // D2H
  std::vector<uint16_t> h_out_q_bf16(kRows * kQDim);
  std::vector<uint16_t> h_out_kv_bf16(kRows * kKvDim);
  std::vector<uint16_t> h_out_latent_bf16(kRows * (kKvDim + kRotDim));
  Memcpy(h_out_q_bf16.data(), outputs[0].GetPtr<void>(), h_out_q_bf16.size() * sizeof(uint16_t), MEMCPY_DEVICE_TO_HOST);
  Memcpy(h_out_kv_bf16.data(), outputs[1].GetPtr<void>(), h_out_kv_bf16.size() * sizeof(uint16_t),
         MEMCPY_DEVICE_TO_HOST);
  Memcpy(h_out_latent_bf16.data(), outputs[2].GetPtr<void>(), h_out_latent_bf16.size() * sizeof(uint16_t),
         MEMCPY_DEVICE_TO_HOST);

  auto act_q = FromBf16(h_out_q_bf16);
  auto act_kv = FromBf16(h_out_kv_bf16);
  auto act_latent = FromBf16(h_out_latent_bf16);

  // CPU reference: RMSNorm each row of q-portion, apply q_weight; same for kv portion
  std::vector<float> ref_q(kRows * kQDim), ref_kv(kRows * kKvDim);
  std::vector<float> ref_latent(kRows * (kKvDim + kRotDim));
  for (int r = 0; r < kRows; ++r) {
    const float* row = h_input.data() + r * kInputCols;
    // Q portion: first q_dim elements
    std::vector<float> normed_q(kQDim);
    CpuRmsNorm(row, normed_q.data(), kQDim, kEps);
    for (int j = 0; j < kQDim; ++j) ref_q[r * kQDim + j] = normed_q[j] * h_q_weight_ref[j];

    // KV portion: elements [q_dim, q_dim+kv_dim)
    std::vector<float> normed_kv(kKvDim);
    CpuRmsNorm(row + kQDim, normed_kv.data(), kKvDim, kEps);
    for (int j = 0; j < kKvDim; ++j) ref_kv[r * kKvDim + j] = normed_kv[j] * h_kv_weight_ref[j];

    // Latent: normed_kv + rot portion (raw copy of last rot_dim elements)
    for (int j = 0; j < kKvDim; ++j) ref_latent[r * (kKvDim + kRotDim) + j] = ref_kv[r * kKvDim + j];
    for (int j = 0; j < kRotDim; ++j) ref_latent[r * (kKvDim + kRotDim) + kKvDim + j] = row[kQDim + kKvDim + j];
  }

  float rel_q = CpuRelError(ref_q, act_q);
  float rel_kv = CpuRelError(ref_kv, act_kv);
  float rel_latent = CpuRelError(ref_latent, act_latent);
  printf("RmsnormSplitCol BF16: rel_err q=%.4f kv=%.4f latent=%.4f\n", rel_q, rel_kv, rel_latent);
  EXPECT_LT(rel_q, 0.02f);
  EXPECT_LT(rel_kv, 0.02f);
  EXPECT_LT(rel_latent, 0.02f);
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST 3a: BatchedMatMulLayer BF16
// ═══════════════════════════════════════════════════════════════════════════════
TEST_F(KunlunW4A8LayerTest, BatchedMatMulBf16) {
  constexpr int M = 4, B = 8, N = 32, K = 32;

  std::mt19937 rng(7);
  std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

  // A [M, B, K]
  std::vector<float> h_a(M * B * K);
  for (auto& v : h_a) v = dist(rng);
  // B_weight [B, N, K] (trans_b=true)
  std::vector<float> h_b(B * N * K);
  for (auto& v : h_b) v = dist(rng);

  auto h_a_bf16 = ToBf16(h_a);
  auto h_b_bf16 = ToBf16(h_b);

  Tensor a_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16,
                  {static_cast<size_t>(M), static_cast<size_t>(B), static_cast<size_t>(K)});
  Tensor b_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16,
                  {static_cast<size_t>(B), static_cast<size_t>(N), static_cast<size_t>(K)});
  Tensor c_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16,
                  {static_cast<size_t>(M), static_cast<size_t>(B), static_cast<size_t>(N)});

  MemcpyAsync(a_tensor.GetPtr<void>(), h_a_bf16.data(), h_a_bf16.size() * sizeof(uint16_t), MEMCPY_HOST_TO_DEVICE,
              context_->device->GetMemoryManageStream());
  MemcpyAsync(b_tensor.GetPtr<void>(), h_b_bf16.data(), h_b_bf16.size() * sizeof(uint16_t), MEMCPY_HOST_TO_DEVICE,
              context_->device->GetMemoryManageStream());
  StreamSynchronize(context_->device->GetMemoryManageStream());

  TestableBatchedMatMulLayer layer;
  layer.SetupForTest(context_, TYPE_BF16);

  std::vector<Tensor> outputs = {c_tensor};
  Status status = layer.Forward(&a_tensor, &b_tensor, &outputs[0]);
  StreamSynchronize(context_->device->GetPrimaryComputeStream());
  ASSERT_TRUE(status.OK()) << status.ToString();

  // D2H
  std::vector<uint16_t> h_c_bf16(M * B * N);
  Memcpy(h_c_bf16.data(), outputs[0].GetPtr<void>(), h_c_bf16.size() * sizeof(uint16_t), MEMCPY_DEVICE_TO_HOST);
  auto act_c = FromBf16(h_c_bf16);

  // CPU reference: per-batch C[m,b,n] = sum_k A[m,b,k] * B[b,n,k]
  std::vector<float> ref_c(M * B * N, 0.0f);
  for (int b = 0; b < B; ++b) {
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
          sum += h_a[m * B * K + b * K + k] * h_b[b * N * K + n * K + k];
        }
        ref_c[m * B * N + b * N + n] = sum;
      }
    }
  }

  float rel = CpuRelError(ref_c, act_c);
  printf("BatchedMatMul BF16: M=%d B=%d N=%d K=%d rel_err=%.4f\n", M, B, N, K, rel);
  EXPECT_LT(rel, 0.15f);
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST 3b: BatchedMatMulLayer INT8
// ═══════════════════════════════════════════════════════════════════════════════
TEST_F(KunlunW4A8LayerTest, BatchedMatMulInt8) {
  constexpr int M = 4, B = 8, N = 32, K = 32;

  std::mt19937 rng(11);
  std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
  std::uniform_int_distribution<int> int_dist(-64, 63);

  // A [M, B, K] BF16
  std::vector<float> h_a(M * B * K);
  for (auto& v : h_a) v = dist(rng);
  auto h_a_bf16 = ToBf16(h_a);

  // B [B, N, K] INT8
  std::vector<int8_t> h_b(B * N * K);
  for (auto& v : h_b) v = static_cast<int8_t>(int_dist(rng));

  // weight_scales [B, N] FP32
  std::vector<float> h_scales(B * N);
  for (auto& v : h_scales) v = 0.01f + dist(rng) * 0.005f;

  Tensor a_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16,
                  {static_cast<size_t>(M), static_cast<size_t>(B), static_cast<size_t>(K)});
  Tensor b_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT8,
                  {static_cast<size_t>(B), static_cast<size_t>(N), static_cast<size_t>(K)});
  Tensor scale_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {static_cast<size_t>(B), static_cast<size_t>(N)});
  Tensor c_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16,
                  {static_cast<size_t>(M), static_cast<size_t>(B), static_cast<size_t>(N)});

  MemcpyAsync(a_tensor.GetPtr<void>(), h_a_bf16.data(), h_a_bf16.size() * sizeof(uint16_t), MEMCPY_HOST_TO_DEVICE,
              context_->device->GetMemoryManageStream());
  MemcpyAsync(b_tensor.GetPtr<void>(), h_b.data(), h_b.size() * sizeof(int8_t), MEMCPY_HOST_TO_DEVICE,
              context_->device->GetMemoryManageStream());
  MemcpyAsync(scale_tensor.GetPtr<void>(), h_scales.data(), h_scales.size() * sizeof(float), MEMCPY_HOST_TO_DEVICE,
              context_->device->GetMemoryManageStream());
  StreamSynchronize(context_->device->GetMemoryManageStream());

  // Bind weight_scales
  b_tensor.weight_scales = &scale_tensor;

  TestableBatchedMatMulLayer layer;
  layer.SetupForTest(context_, TYPE_BF16);

  std::vector<Tensor> outputs = {c_tensor};
  Status status = layer.Forward(&a_tensor, &b_tensor, &outputs[0]);
  StreamSynchronize(context_->device->GetPrimaryComputeStream());
  ASSERT_TRUE(status.OK()) << status.ToString();

  // D2H
  std::vector<uint16_t> h_c_bf16(M * B * N);
  Memcpy(h_c_bf16.data(), outputs[0].GetPtr<void>(), h_c_bf16.size() * sizeof(uint16_t), MEMCPY_DEVICE_TO_HOST);
  auto act_c = FromBf16(h_c_bf16);

  // CPU reference: C[m,b,n] = sum_k(A_fp[m,b,k] * B_int8[b,n,k]) * scale[b,n] / 127
  // weight_scale is pre-multiplied by 127 (Kunlun convention)
  std::vector<float> ref_c(M * B * N, 0.0f);
  for (int b = 0; b < B; ++b) {
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n) {
        float sum = 0.0f;
        float s = h_scales[b * N + n] / 127.0f;
        for (int k = 0; k < K; ++k) {
          sum += h_a[m * B * K + b * K + k] * static_cast<float>(h_b[b * N * K + n * K + k]) * s;
        }
        ref_c[m * B * N + b * N + n] = sum;
      }
    }
  }

  float rel = CpuRelError(ref_c, act_c);
  printf("BatchedMatMul INT8: M=%d B=%d N=%d K=%d rel_err=%.4f\n", M, B, N, K, rel);
  EXPECT_LT(rel, 0.20f);
}
// ═══════════════════════════════════════════════════════════════════════════════
TEST_F(KunlunW4A8LayerTest, Int8MatMulBf16) {
  constexpr int kM = 8, kN = 32, kK = 64;

  std::mt19937 rng(77);
  std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
  std::uniform_int_distribution<int> int_dist(-64, 63);

  // x [m, k] BF16
  std::vector<float> h_x(kM * kK);
  for (auto& v : h_x) v = dist(rng);
  auto h_x_bf16 = ToBf16(h_x);

  // w [n, k] INT8
  std::vector<int8_t> h_w(kN * kK);
  for (auto& v : h_w) v = static_cast<int8_t>(int_dist(rng));

  // weight_scales [n] FP32
  std::vector<float> h_scales(kN);
  for (auto& v : h_scales) v = 0.01f + std::fabs(dist(rng)) * 0.01f;

  Tensor x_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16, {static_cast<size_t>(kM), static_cast<size_t>(kK)});
  Tensor w_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT8, {static_cast<size_t>(kN), static_cast<size_t>(kK)});
  Tensor scale_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {static_cast<size_t>(kN)});
  Tensor y_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16, {static_cast<size_t>(kM), static_cast<size_t>(kN)});

  MemcpyAsync(x_tensor.GetPtr<void>(), h_x_bf16.data(), h_x_bf16.size() * sizeof(uint16_t), MEMCPY_HOST_TO_DEVICE,
              context_->device->GetMemoryManageStream());
  MemcpyAsync(w_tensor.GetPtr<void>(), h_w.data(), h_w.size() * sizeof(int8_t), MEMCPY_HOST_TO_DEVICE,
              context_->device->GetMemoryManageStream());
  MemcpyAsync(scale_tensor.GetPtr<void>(), h_scales.data(), h_scales.size() * sizeof(float), MEMCPY_HOST_TO_DEVICE,
              context_->device->GetMemoryManageStream());
  StreamSynchronize(context_->device->GetMemoryManageStream());

  w_tensor.weight_scales = &scale_tensor;

  TestableKunlunInt8MatMulLayer layer;
  layer.SetupForTest(context_, TYPE_BF16);

  std::vector<Tensor> outputs = {y_tensor};
  Status status = layer.Forward(&x_tensor, &w_tensor, &outputs[0]);
  StreamSynchronize(context_->device->GetPrimaryComputeStream());
  ASSERT_TRUE(status.OK()) << status.ToString();

  // D2H
  std::vector<uint16_t> h_y_bf16(kM * kN);
  Memcpy(h_y_bf16.data(), outputs[0].GetPtr<void>(), h_y_bf16.size() * sizeof(uint16_t), MEMCPY_DEVICE_TO_HOST);
  auto act_y = FromBf16(h_y_bf16);

  // CPU reference: y[m,n] = sum_k(x_fp[m,k] * w_int8[n,k]) * scale_w[n] / 127
  // Note: kernel does dynamic per-token quant on x, then int8 gemm, then dequant with scale_x * scale_w.
  // The scale_w passed to kernel is pre-multiplied by quantization range (127 for INT8),
  // so the effective dequant factor per output element is scale_w / 127.
  std::vector<float> ref_y(kM * kN, 0.0f);
  for (int m = 0; m < kM; ++m) {
    for (int n = 0; n < kN; ++n) {
      float sum = 0.0f;
      for (int k = 0; k < kK; ++k) {
        sum += h_x[m * kK + k] * static_cast<float>(h_w[n * kK + k]);
      }
      ref_y[m * kN + n] = sum * h_scales[n] / 127.0f;
    }
  }

  float rel = CpuRelError(ref_y, act_y);
  printf("KunlunInt8MatMul: m=%d n=%d k=%d rel_err=%.4f\n", kM, kN, kK, rel);
  EXPECT_LT(rel, 0.15f);

  // Verify outputs are finite and non-zero
  int nonzero = 0;
  for (auto v : act_y) {
    ASSERT_TRUE(std::isfinite(v));
    if (std::fabs(v) > 1e-7f) ++nonzero;
  }
  EXPECT_GT(nonzero, 0);
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST 5: KunlunMoeLayer W4A8 Kernel-Level Test
// ═══════════════════════════════════════════════════════════════════════════════
TEST_F(KunlunW4A8LayerTest, MoeW4A8KernelFcFusion) {
  // Directly test XpuMoeW4A8FcFusion kernel without going through KunlunMoeLayer
  // (which requires ep_size==8).
  constexpr int kNumTokens = 4;
  constexpr int kTopK = 2;
  constexpr int kMTotal = kNumTokens * kTopK;  // 8
  constexpr int kNumExperts = 4;
  constexpr int kHidden = 64;
  constexpr int kInter = 128;

  std::mt19937 rng(99);
  std::uniform_int_distribution<int> int_dist(-60, 60);
  std::uniform_real_distribution<float> scale_dist(0.005f, 0.02f);

  // Input activation [m_total, hidden] INT8
  std::vector<int8_t> h_act(kMTotal * kHidden);
  for (auto& v : h_act) v = static_cast<int8_t>(int_dist(rng));

  // Per-token activation scale [m_total] FP32
  std::vector<float> h_act_scale(kMTotal);
  for (auto& v : h_act_scale) v = scale_dist(rng);

  // Weight [num_experts, inter, hidden/2] INT4x2 packed
  // Each byte stores 2 INT4 values: low_nibble (bits 0-3) | high_nibble (bits 4-7)
  const int kPackedK = kHidden / 2;
  std::vector<int8_t> h_weight(kNumExperts * kInter * kPackedK);
  std::uniform_int_distribution<int> nibble_dist(-7, 7);
  for (auto& v : h_weight) {
    int low = nibble_dist(rng) & 0xF;
    int high = nibble_dist(rng) & 0xF;
    v = static_cast<int8_t>(low | (high << 4));
  }

  // Per-channel weight scale [num_experts * inter] FP32
  std::vector<float> h_weight_scale(kNumExperts * kInter);
  for (auto& v : h_weight_scale) v = scale_dist(rng);

  // Routing info: sorted_tokens_ids [m_total], sorted_tokens_num_lod [num_experts+1]
  // Simple: tokens 0-1 → expert 0, 2-3 → expert 1, 4-5 → expert 2, 6-7 → expert 3
  std::vector<int> h_sorted_ids(kMTotal);
  std::iota(h_sorted_ids.begin(), h_sorted_ids.end(), 0);
  std::vector<int> h_lod = {0, 2, 4, 6, 8};

  // Allocate device tensors
  Tensor act_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT8,
                    {static_cast<size_t>(kMTotal), static_cast<size_t>(kHidden)});
  Tensor act_scale_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {static_cast<size_t>(kMTotal)});
  Tensor weight_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT8,
                       {static_cast<size_t>(kNumExperts), static_cast<size_t>(kInter), static_cast<size_t>(kPackedK)});
  Tensor weight_scale_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {static_cast<size_t>(kNumExperts * kInter)});
  Tensor output_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16,
                       {static_cast<size_t>(kMTotal), static_cast<size_t>(kInter)});
  Tensor sorted_ids_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {static_cast<size_t>(kMTotal)});
  Tensor lod_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {static_cast<size_t>(kNumExperts + 1)});

  MemcpyAsync(act_tensor.GetPtr<void>(), h_act.data(), h_act.size() * sizeof(int8_t), MEMCPY_HOST_TO_DEVICE,
              context_->device->GetMemoryManageStream());
  MemcpyAsync(act_scale_tensor.GetPtr<void>(), h_act_scale.data(), h_act_scale.size() * sizeof(float),
              MEMCPY_HOST_TO_DEVICE, context_->device->GetMemoryManageStream());
  MemcpyAsync(weight_tensor.GetPtr<void>(), h_weight.data(), h_weight.size() * sizeof(int8_t), MEMCPY_HOST_TO_DEVICE,
              context_->device->GetMemoryManageStream());
  MemcpyAsync(weight_scale_tensor.GetPtr<void>(), h_weight_scale.data(), h_weight_scale.size() * sizeof(float),
              MEMCPY_HOST_TO_DEVICE, context_->device->GetMemoryManageStream());
  MemcpyAsync(sorted_ids_tensor.GetPtr<void>(), h_sorted_ids.data(), h_sorted_ids.size() * sizeof(int),
              MEMCPY_HOST_TO_DEVICE, context_->device->GetMemoryManageStream());
  MemcpyAsync(lod_tensor.GetPtr<void>(), h_lod.data(), h_lod.size() * sizeof(int), MEMCPY_HOST_TO_DEVICE,
              context_->device->GetMemoryManageStream());
  StreamSynchronize(context_->device->GetMemoryManageStream());

  // Get xdnn context and call kernel directly
  xdnn::Context* xctx = static_cast<xdnn::Context*>(
      context_->device->ext->GetXdnnContext(context_->device->GetPrimaryComputeStream().Get()));

  int ret = XpuMoeW4A8FcFusion<bfloat16, int>(
      xctx, act_tensor.GetPtr<int8_t>(), weight_tensor.GetPtr<void>(), output_tensor.GetPtr<bfloat16>(),
      act_scale_tensor.GetPtr<float>(), weight_scale_tensor.GetPtr<float>(),
      /*tokens=*/kMTotal, /*topk=*/1, /*m_total=*/kMTotal, /*n=*/kInter, /*k=*/kHidden, /*num_experts=*/kNumExperts,
      sorted_ids_tensor.GetPtr<int>(), lod_tensor.GetPtr<int>());
  StreamSynchronize(context_->device->GetPrimaryComputeStream());

  ASSERT_EQ(ret, 0) << "XpuMoeW4A8FcFusion failed with ret=" << ret;

  // D2H and verify finite + non-zero
  std::vector<uint16_t> h_out_bf16(kMTotal * kInter);
  Memcpy(h_out_bf16.data(), output_tensor.GetPtr<void>(), h_out_bf16.size() * sizeof(uint16_t), MEMCPY_DEVICE_TO_HOST);
  auto act_out = FromBf16(h_out_bf16);

  int finite_count = 0, nonzero_count = 0;
  for (size_t i = 0; i < act_out.size(); ++i) {
    ASSERT_TRUE(std::isfinite(act_out[i])) << "NaN/Inf at output[" << i << "]";
    ++finite_count;
    if (std::fabs(act_out[i]) > 1e-7f) ++nonzero_count;
  }
  EXPECT_GT(nonzero_count, 0) << "All output elements are zero";

  // CPU approximate reference: y[t,n] = act_scale[t] * sum_k(act_int8[t,k] * unpack(w[expert,n,k/2]) *
  // w_scale[expert*inter+n]) For simplicity, verify a subset of outputs against loose tolerance
  std::vector<float> ref_out(kMTotal * kInter, 0.0f);
  for (int t = 0; t < kMTotal; ++t) {
    int expert_id = -1;
    for (int e = 0; e < kNumExperts; ++e) {
      if (t >= h_lod[e] && t < h_lod[e + 1]) {
        expert_id = e;
        break;
      }
    }
    for (int n = 0; n < kInter; ++n) {
      float sum = 0.0f;
      for (int k = 0; k < kHidden; ++k) {
        int packed_idx = expert_id * kInter * kPackedK + n * kPackedK + k / 2;
        int8_t packed = h_weight[packed_idx];
        int w_val;
        if (k % 2 == 0) {
          w_val = (packed & 0xF);
          if (w_val >= 8) w_val -= 16;
        } else {
          w_val = ((packed >> 4) & 0xF);
          if (w_val >= 8) w_val -= 16;
        }
        sum += static_cast<float>(h_act[t * kHidden + k]) * static_cast<float>(w_val);
      }
      ref_out[t * kInter + n] = sum * h_act_scale[t] * h_weight_scale[expert_id * kInter + n] / (127.0f * 7.0f);
    }
  }

  float rel = CpuRelError(ref_out, act_out);
  printf("MoeW4A8 kernel: tokens=%d experts=%d hidden=%d inter=%d rel_err=%.4f finite=%d nonzero=%d\n", kMTotal,
         kNumExperts, kHidden, kInter, rel, finite_count, nonzero_count);
  EXPECT_LT(rel, 0.15f);
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST 6: FlashMlaAttentionLayer BF16 (functional verification)
// ═══════════════════════════════════════════════════════════════════════════════
TEST_F(KunlunW4A8LayerTest, FlashMlaAttentionBf16) {
  constexpr int kBatchSize = 2;
  constexpr int kSeqLen0 = 3, kSeqLen1 = 5;
  constexpr int kTotalTokens = kSeqLen0 + kSeqLen1;
  constexpr int kSlotDim = kTotalTokens;
  constexpr int kQDim = kNumHeads * (kQkNopeHeadDim + kQkRopeHeadDim);  // 8 * 48 = 384
  constexpr int kKvDim = kKvLoraRank * 2;                               // 64 (kv projection output)
  constexpr int kLatentDim = kKvLoraRank + kQkRopeHeadDim;              // 48
  constexpr int kOutDim = kNumHeads * kVHeadDim;                        // 256

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-0.01f, 0.01f);

  // Prepare cos_sin_cache on device: [max_pos, rope_dim] float, layout: [cos_half | sin_half]
  const int rope_dim = kQkRopeHeadDim;
  std::vector<float> h_cos_sin(kMaxPositionEmbeddings * rope_dim);
  for (int pos = 0; pos < kMaxPositionEmbeddings; ++pos) {
    for (int d = 0; d < rope_dim / 2; ++d) {
      float freq = 1.0f / std::pow(10000.0f, 2.0f * d / static_cast<float>(rope_dim));
      h_cos_sin[pos * rope_dim + d] = std::cos(pos * freq);
      h_cos_sin[pos * rope_dim + rope_dim / 2 + d] = std::sin(pos * freq);
    }
  }
  Tensor cos_sin_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32,
                        {static_cast<size_t>(kMaxPositionEmbeddings), static_cast<size_t>(rope_dim)});
  MemcpyAsync(cos_sin_tensor.GetPtr<void>(), h_cos_sin.data(), h_cos_sin.size() * sizeof(float), MEMCPY_HOST_TO_DEVICE,
              context_->device->GetMemoryManageStream());

  // KV cache: [max_block_num, block_token_num, latent_dim] BF16
  Tensor kv_cache(
      MemoryLocation::LOCATION_DEVICE, TYPE_BF16,
      {static_cast<size_t>(kMaxBlockNum), static_cast<size_t>(kBlockTokenNum), static_cast<size_t>(kLatentDim)});

  // [0] dp_input_offset HOST: [batch+1] int64
  Tensor dp_input_offset(MemoryLocation::LOCATION_HOST, TYPE_INT64, {static_cast<size_t>(kBatchSize + 1)});
  {
    int64_t* p = dp_input_offset.GetPtr<int64_t>();
    p[0] = 0;
    p[1] = kSeqLen0;
    p[2] = kTotalTokens;
  }

  // [1] dp_input_offset_int32 DEVICE: [batch+1] int32
  Tensor dp_input_offset_int32(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {static_cast<size_t>(kBatchSize + 1)});
  {
    std::vector<int32_t> d = {0, kSeqLen0, kTotalTokens};
    MemcpyAsync(dp_input_offset_int32.GetPtr<void>(), d.data(), d.size() * sizeof(int32_t), MEMCPY_HOST_TO_DEVICE,
                context_->device->GetMemoryManageStream());
  }

  // [2] kv_list (placeholder)
  Tensor kv_list(MemoryLocation::LOCATION_DEVICE, TYPE_INT64, {1});

  // [3] dp_input_prefix HOST: [batch] int64 (all zeros, no prefix)
  Tensor dp_input_prefix(MemoryLocation::LOCATION_HOST, TYPE_INT64, {static_cast<size_t>(kBatchSize)});
  memset(dp_input_prefix.GetPtr<void>(), 0, kBatchSize * sizeof(int64_t));

  // [4] dp_prefill_q_offset HOST: [batch+1] int64
  Tensor dp_prefill_q_offset(MemoryLocation::LOCATION_HOST, TYPE_INT64, {static_cast<size_t>(kBatchSize + 1)});
  {
    int64_t* p = dp_prefill_q_offset.GetPtr<int64_t>();
    p[0] = 0;
    p[1] = kSeqLen0;
    p[2] = kTotalTokens;
  }

  // [5] dp_prefill_q_offset_int32 DEVICE
  Tensor dp_prefill_q_offset_int32(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {static_cast<size_t>(kBatchSize + 1)});
  {
    std::vector<int32_t> d = {0, kSeqLen0, kTotalTokens};
    MemcpyAsync(dp_prefill_q_offset_int32.GetPtr<void>(), d.data(), d.size() * sizeof(int32_t), MEMCPY_HOST_TO_DEVICE,
                context_->device->GetMemoryManageStream());
  }

  // [6] kv_cache_offset (placeholder)
  Tensor kv_cache_offset(MemoryLocation::LOCATION_DEVICE, TYPE_INT64, {1});

  // [7] rotary_embedding_pos DEVICE: [total_tokens] int64
  Tensor rotary_pos(MemoryLocation::LOCATION_DEVICE, TYPE_INT64, {static_cast<size_t>(kTotalTokens)});
  {
    std::vector<int64_t> positions = {0, 1, 2, 0, 1, 2, 3, 4};
    MemcpyAsync(rotary_pos.GetPtr<void>(), positions.data(), positions.size() * sizeof(int64_t), MEMCPY_HOST_TO_DEVICE,
                context_->device->GetMemoryManageStream());
  }

  // [8] rotary_embedding_mask (placeholder)
  Tensor rotary_mask(MemoryLocation::LOCATION_DEVICE, TYPE_INT64, {1});

  // [9-16] 8 placeholder tensors (prefix cache, unused)
  Tensor placeholder(MemoryLocation::LOCATION_DEVICE, TYPE_INT64, {1});

  // [17] forward_shape HOST: CUDA 13D layout + Kunlun [13].
  // Test scenario: pure flash (no decode, no prefix), kBatchSize flash requests.
  //   [0] multi_token_request_num      = kBatchSize
  //   [1] multi_token_request_max_tokens = kSeqLen1 (max of kSeqLen0/kSeqLen1)
  //   [2] context_kv_cache_block_num   = 0 (test doesn't allocate real blocks)
  //   [3] single_token_request_num     = 0 (no decode)
  //   [4] single_token_request_max_tokens = 0
  //   [5] decode_kv_cache_block_num    = 0
  //   [6] dp_max_forwarding_tokens     = kSeqLen1
  //   [7] dp_multi_token_request_num   = kBatchSize (DP-group flash count = len(qlod)-1)
  //   [8] dp_multi_token_request_max_tokens = kSeqLen1
  //   [9] dp_single_token_request_num  = 0
  //   [10] dp_single_token_request_max_tokens = 0
  //   [11] dp_total_prefix_len         = 0 (no prefix in this test)
  //   [12] dp_decode_tokens            = 0
  //   [13] dp_max_prefix_per_req       = 0 (Kunlun-specific, no prefix)
  Tensor forward_shape(
      MemoryLocation::LOCATION_HOST, TYPE_INT32,
      {static_cast<size_t>(kBatchSize), static_cast<size_t>(kSeqLen1), 0, 0, 0, 0, static_cast<size_t>(kSeqLen1),
       static_cast<size_t>(kBatchSize), static_cast<size_t>(kSeqLen1), 0, 0, 0, 0, 0});

  // [18] layer_kv_cache: metadata containing [int64_t block_num, void* kv_cache_ptr]
  Tensor layer_kv_cache(MemoryLocation::LOCATION_HOST, TYPE_INT64, {4});
  {
    int64_t* meta = layer_kv_cache.GetPtr<int64_t>();
    meta[0] = kMaxBlockNum;
    void** ptr_slot = reinterpret_cast<void**>(&meta[1]);
    *ptr_slot = kv_cache.GetPtr<void>();
  }

  // [19] block_table DEVICE: [batch, max_blocks_per_seq] int32
  constexpr int kMaxBlocksPerSeq = 4;
  Tensor block_table(MemoryLocation::LOCATION_DEVICE, TYPE_INT32,
                     {static_cast<size_t>(kBatchSize), static_cast<size_t>(kMaxBlocksPerSeq)});
  {
    std::vector<int32_t> bt(kBatchSize * kMaxBlocksPerSeq, 0);
    bt[0] = 0;
    bt[1] = 1;  // seq0: blocks 0,1
    bt[kMaxBlocksPerSeq] = 2;
    bt[kMaxBlocksPerSeq + 1] = 3;  // seq1: blocks 2,3
    MemcpyAsync(block_table.GetPtr<void>(), bt.data(), bt.size() * sizeof(int32_t), MEMCPY_HOST_TO_DEVICE,
                context_->device->GetMemoryManageStream());
  }

  // [20] q_nope_rope_tensor DEVICE: [total_tokens, num_heads, qk_nope+qk_rope] BF16
  constexpr int kQHeadDim = kQkNopeHeadDim + kQkRopeHeadDim;
  Tensor q_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16,
                  {static_cast<size_t>(kTotalTokens), static_cast<size_t>(kNumHeads), static_cast<size_t>(kQHeadDim)});
  {
    std::vector<float> f(kTotalTokens * kNumHeads * kQHeadDim);
    for (auto& v : f) v = dist(rng);
    auto bf = ToBf16(f);
    MemcpyAsync(q_tensor.GetPtr<void>(), bf.data(), bf.size() * sizeof(uint16_t), MEMCPY_HOST_TO_DEVICE,
                context_->device->GetMemoryManageStream());
  }

  // [21] kv_buffer DEVICE: [total_tokens, num_heads, qk_nope_head_dim + v_head_dim] BF16
  // This is the output of kv_b_proj_combined (a single matmul that produces K_nope and V combined)
  constexpr int kMlaKvHeadDim = kQkNopeHeadDim + kVHeadDim;  // 256
  Tensor kv_buffer(
      MemoryLocation::LOCATION_DEVICE, TYPE_BF16,
      {static_cast<size_t>(kTotalTokens), static_cast<size_t>(kNumHeads), static_cast<size_t>(kMlaKvHeadDim)});
  {
    size_t n = kTotalTokens * kNumHeads * kMlaKvHeadDim;
    std::vector<float> f(n);
    for (auto& v : f) v = dist(rng);
    auto bf = ToBf16(f);
    MemcpyAsync(kv_buffer.GetPtr<void>(), bf.data(), bf.size() * sizeof(uint16_t), MEMCPY_HOST_TO_DEVICE,
                context_->device->GetMemoryManageStream());
  }

  // [22] k_rope_buffer (latent_cache): [total_tokens, 1, kv_lora_rank + qk_rope] BF16
  Tensor latent_cache(MemoryLocation::LOCATION_DEVICE, TYPE_BF16,
                      {static_cast<size_t>(kTotalTokens), 1, static_cast<size_t>(kLatentDim)});
  {
    std::vector<float> f(kTotalTokens * kLatentDim);
    for (auto& v : f) v = dist(rng);
    auto bf = ToBf16(f);
    MemcpyAsync(latent_cache.GetPtr<void>(), bf.data(), bf.size() * sizeof(uint16_t), MEMCPY_HOST_TO_DEVICE,
                context_->device->GetMemoryManageStream());
  }

  // [23-25] weight placeholders (unused on Kunlun)
  Tensor w_placeholder(MemoryLocation::LOCATION_DEVICE, TYPE_BF16, {1});

  // [26] k_buffer workspace
  size_t k_ws_size = kTotalTokens * kNumHeads * kQHeadDim;
  Tensor k_buffer(MemoryLocation::LOCATION_DEVICE, TYPE_BF16, {k_ws_size});

  // [27] v_buffer workspace
  size_t v_ws_size = kTotalTokens * kNumHeads * kVHeadDim;
  Tensor v_buffer(MemoryLocation::LOCATION_DEVICE, TYPE_BF16, {v_ws_size});

  // [28] qlod HOST: [batch+1] int32
  Tensor qlod(MemoryLocation::LOCATION_HOST, TYPE_INT32, {static_cast<size_t>(kBatchSize + 1)});
  {
    int32_t* p = qlod.GetPtr<int32_t>();
    p[0] = 0;
    p[1] = kSeqLen0;
    p[2] = kTotalTokens;
  }

  // [29] kvlod HOST: [batch+1] int32
  Tensor kvlod(MemoryLocation::LOCATION_HOST, TYPE_INT32, {static_cast<size_t>(kBatchSize + 1)});
  {
    int32_t* p = kvlod.GetPtr<int32_t>();
    p[0] = 0;
    p[1] = kSeqLen0;
    p[2] = kTotalTokens;
  }

  // [30] layers_slot_mapping DEVICE: [layer_num * slot_dim] int32
  Tensor slot_mapping(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {static_cast<size_t>(kLayerNum * kSlotDim)});
  {
    std::vector<int32_t> slots(kLayerNum * kSlotDim);
    for (int l = 0; l < kLayerNum; ++l)
      for (int t = 0; t < kSlotDim; ++t) slots[l * kSlotDim + t] = t;
    MemcpyAsync(slot_mapping.GetPtr<void>(), slots.data(), slots.size() * sizeof(int32_t), MEMCPY_HOST_TO_DEVICE,
                context_->device->GetMemoryManageStream());
  }

  // [31] atb_attention_attr HOST: [2] uint64
  Tensor atb_attr(MemoryLocation::LOCATION_HOST, TYPE_UINT64, {2});
  {
    uint64_t* p = atb_attr.GetPtr<uint64_t>();
    p[0] = static_cast<uint64_t>(kSlotDim);
    p[1] = static_cast<uint64_t>(kMaxBlocksPerSeq);
  }

  // Output buffer (hidden_buffer_0). In the Kunlun no-prefix MLA path the combined kv_b_proj
  // output is read back from this buffer, so it must hold total_tokens * num_heads *
  // (qk_nope_head_dim + v_head_dim) elements — larger than the final attention output
  // (total_tokens * num_heads * v_head_dim). Pre-fill it with the combined-kv activations since
  // this test skips the kv_b_proj module that would normally write here.
  Tensor output(MemoryLocation::LOCATION_DEVICE, TYPE_BF16,
                {static_cast<size_t>(kTotalTokens), static_cast<size_t>(kNumHeads * kMlaKvHeadDim)});
  {
    std::vector<float> f(static_cast<size_t>(kTotalTokens) * kNumHeads * kMlaKvHeadDim);
    for (auto& v : f) v = dist(rng);
    auto bf = ToBf16(f);
    MemcpyAsync(output.GetPtr<void>(), bf.data(), bf.size() * sizeof(uint16_t), MEMCPY_HOST_TO_DEVICE,
                context_->device->GetMemoryManageStream());
  }

  StreamSynchronize(context_->device->GetMemoryManageStream());

  // Setup layer
  TestableFlashMlaAttentionLayer layer;
  layer.SetupForTest(context_, TYPE_BF16, kTotalTokens);
  layer.SetCosSinCache(cos_sin_tensor.GetPtr<void>());

  // The refactored FlashMlaAttentionLayer takes a typed FlashMlaAttentionInputs struct.
  // Only the fields the Kunlun no-prefix path dereferences are populated; all other pointers
  // remain nullptr (value-initialized), which is safe because dp_total_prefix_len == 0 here.
  FlashMlaAttentionInputs mla_inputs{};
  mla_inputs.dp_input_offset = &dp_input_offset;
  mla_inputs.dp_input_offset_int32 = &dp_input_offset_int32;
  mla_inputs.kv_list = &kv_list;
  mla_inputs.dp_input_prefix = &dp_input_prefix;
  mla_inputs.dp_prefill_q_offset = &dp_prefill_q_offset;
  mla_inputs.dp_prefill_q_offset_int32 = &dp_prefill_q_offset_int32;
  mla_inputs.kv_cache_offset = &kv_cache_offset;
  mla_inputs.rotary_embedding_pos = &rotary_pos;
  mla_inputs.rotary_embedding_mask = &rotary_mask;
  mla_inputs.forward_shape = &forward_shape;
  mla_inputs.layer_kv_cache = &layer_kv_cache;
  mla_inputs.block_table = &block_table;
  mla_inputs.q_nope_rope = &q_tensor;
  mla_inputs.kv_buffer = &kv_buffer;
  mla_inputs.k_rope_buffer = &latent_cache;
  mla_inputs.k_buffer = &k_buffer;
  mla_inputs.v_buffer = &v_buffer;
  mla_inputs.qlod_host_tensor = &qlod;
  mla_inputs.kvlod_host_tensor = &kvlod;
  mla_inputs.layers_slot_mapping = &slot_mapping;
  mla_inputs.atb_attention_attr = &atb_attr;

  Status status = layer.Forward(mla_inputs, &output);
  StreamSynchronize(context_->device->GetPrimaryComputeStream());
  ASSERT_TRUE(status.OK()) << "FlashMlaAttentionLayer::Forward() failed: " << status.ToString();

  // Verify shape
  EXPECT_EQ(output.shape[0], static_cast<size_t>(kTotalTokens));

  // Verify outputs: with random data at large scale, some NaN/Inf may occur due to softmax overflow.
  // Primary assertion is that Forward() succeeded (kernel executed without error).
  std::vector<uint16_t> h_out(kTotalTokens * kOutDim);
  Memcpy(h_out.data(), output.GetPtr<void>(), h_out.size() * sizeof(uint16_t), MEMCPY_DEVICE_TO_HOST);

  int finite_count = 0, nonzero_count = 0;
  for (size_t i = 0; i < h_out.size(); ++i) {
    float v = HalfToFp32<bfloat16>(h_out[i]);
    if (std::isfinite(v)) ++finite_count;
    if (std::isfinite(v) && std::fabs(v) > 1e-7f) ++nonzero_count;
  }
  // At least batch0's tokens (kSeqLen0 * kOutDim elements) should be finite
  int batch0_finite = 0;
  for (int i = 0; i < kSeqLen0 * kOutDim; ++i) {
    float v = HalfToFp32<bfloat16>(h_out[i]);
    if (std::isfinite(v)) ++batch0_finite;
  }
  EXPECT_EQ(batch0_finite, kSeqLen0 * kOutDim) << "Batch0 outputs contain NaN/Inf";

  // ── CPU naive causal attention reference for batch0 ──
  // After Forward, k_buffer=[tokens, heads, qk_head_dim], v_buffer=[tokens, heads, v_head_dim]
  // q_tensor has been modified in-place (RoPE applied to rope portion).
  // Read them back and compute naive attention for batch0 (seq_len=3).
  const int kQkHeadDim = kQkNopeHeadDim + kQkRopeHeadDim;  // 192
  std::vector<uint16_t> h_q_after(kTotalTokens * kNumHeads * kQkHeadDim);
  std::vector<uint16_t> h_k(kTotalTokens * kNumHeads * kQkHeadDim);
  std::vector<uint16_t> h_v(kTotalTokens * kNumHeads * kVHeadDim);
  Memcpy(h_q_after.data(), q_tensor.GetPtr<void>(), h_q_after.size() * sizeof(uint16_t), MEMCPY_DEVICE_TO_HOST);
  Memcpy(h_k.data(), k_buffer.GetPtr<void>(), h_k.size() * sizeof(uint16_t), MEMCPY_DEVICE_TO_HOST);
  Memcpy(h_v.data(), v_buffer.GetPtr<void>(), h_v.size() * sizeof(uint16_t), MEMCPY_DEVICE_TO_HOST);

  // Naive causal attention for batch0 (tokens 0..kSeqLen0-1), first head only (spot-check)
  float prefill_alpha = (1.0f / std::sqrt(static_cast<float>(kQkHeadDim))) *
                        std::sqrt(static_cast<float>(kQkHeadDim));  // attn_scale * sqrt(head_dim) = 1.0
  // Actually: attn_scale_ = 1/sqrt(qk_nope+qk_rope), prefill_alpha = attn_scale_ * sqrt(qk_head_dim) = 1.0
  // xfa internally divides by sqrt(qk_head_dim), net effect = attn_scale_
  float softmax_scale = 1.0f / std::sqrt(static_cast<float>(kQkHeadDim));

  // Check head 0, token kSeqLen0-1 (last token of batch0, attends to all 3 kv tokens)
  int check_token = kSeqLen0 - 1;  // token index 2
  int head = 0;
  // Q[check_token, head, :] dot K[t, head, :] for t=0..check_token
  std::vector<float> scores(kSeqLen0);
  for (int t = 0; t <= check_token; ++t) {
    float dot = 0.0f;
    for (int d = 0; d < kQkHeadDim; ++d) {
      float q_val = HalfToFp32<bfloat16>(h_q_after[(check_token * kNumHeads + head) * kQkHeadDim + d]);
      float k_val = HalfToFp32<bfloat16>(h_k[(t * kNumHeads + head) * kQkHeadDim + d]);
      dot += q_val * k_val;
    }
    scores[t] = dot * softmax_scale;
  }
  // Softmax
  float max_score = *std::max_element(scores.begin(), scores.begin() + kSeqLen0);
  float sum_exp = 0.0f;
  for (int t = 0; t < kSeqLen0; ++t) {
    scores[t] = std::exp(scores[t] - max_score);
    sum_exp += scores[t];
  }
  for (int t = 0; t < kSeqLen0; ++t) scores[t] /= sum_exp;

  // Weighted sum of V
  std::vector<float> ref_out(kVHeadDim, 0.0f);
  for (int t = 0; t < kSeqLen0; ++t) {
    for (int d = 0; d < kVHeadDim; ++d) {
      float v_val = HalfToFp32<bfloat16>(h_v[(t * kNumHeads + head) * kVHeadDim + d]);
      ref_out[d] += scores[t] * v_val;
    }
  }

  // Compare with GPU output for this token/head
  // GPU output layout: [total_tokens, num_heads * v_head_dim]
  std::vector<float> gpu_out(kVHeadDim);
  for (int d = 0; d < kVHeadDim; ++d) {
    gpu_out[d] = HalfToFp32<bfloat16>(h_out[check_token * kOutDim + head * kVHeadDim + d]);
  }

  float rel = CpuRelError(ref_out, gpu_out);
  printf("FlashMlaAttention BF16 CPU-ref: token=%d head=%d rel_err=%.4f (ref[0]=%.6f gpu[0]=%.6f)\n", check_token, head,
         rel, ref_out[0], gpu_out[0]);
  EXPECT_LT(rel, 0.05f) << "Flash attention output diverges from naive CPU attention";

  printf("FlashMlaAttention BF16: batch=%d tokens=%d out_dim=%d finite=%d/%zu — PASS\n", kBatchSize, kTotalTokens,
         kOutDim, finite_count, h_out.size());
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST 7: PagedMlaAttentionLayer BF16 (functional verification)
// ═══════════════════════════════════════════════════════════════════════════════
TEST_F(KunlunW4A8LayerTest, PagedMlaAttentionBf16) {
  constexpr int kBatchSize = 2;
  constexpr int kContextLen0 = 4, kContextLen1 = 6;
  constexpr int kTotalTokens = kBatchSize;  // decode: 1 token per seq
  constexpr int kSlotDim = kTotalTokens;
  constexpr int kLatentDim = kKvLoraRank + kQkRopeHeadDim;  // 48
  constexpr int kOutDim = kNumHeads * kVHeadDim;            // 256
  constexpr int kMaxBlocksPerSeq = 4;

  std::mt19937 rng(55);
  std::uniform_real_distribution<float> dist(-0.01f, 0.01f);

  // Prepare cos_sin_cache on device
  const int rope_dim = kQkRopeHeadDim;
  std::vector<float> h_cos_sin(kMaxPositionEmbeddings * rope_dim);
  for (int pos = 0; pos < kMaxPositionEmbeddings; ++pos) {
    for (int d = 0; d < rope_dim / 2; ++d) {
      float freq = 1.0f / std::pow(10000.0f, 2.0f * d / static_cast<float>(rope_dim));
      h_cos_sin[pos * rope_dim + d] = std::cos(pos * freq);
      h_cos_sin[pos * rope_dim + rope_dim / 2 + d] = std::sin(pos * freq);
    }
  }
  Tensor cos_sin_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32,
                        {static_cast<size_t>(kMaxPositionEmbeddings), static_cast<size_t>(rope_dim)});
  MemcpyAsync(cos_sin_tensor.GetPtr<void>(), h_cos_sin.data(), h_cos_sin.size() * sizeof(float), MEMCPY_HOST_TO_DEVICE,
              context_->device->GetMemoryManageStream());

  // KV cache blocks: [max_block_num, block_token_num, latent_dim] BF16
  // Pre-fill with random data to simulate prefill phase having written cache
  Tensor kv_cache_blocks(
      MemoryLocation::LOCATION_DEVICE, TYPE_BF16,
      {static_cast<size_t>(kMaxBlockNum), static_cast<size_t>(kBlockTokenNum), static_cast<size_t>(kLatentDim)});
  {
    size_t n = kMaxBlockNum * kBlockTokenNum * kLatentDim;
    std::vector<float> f(n);
    for (auto& v : f) v = dist(rng);
    auto bf = ToBf16(f);
    MemcpyAsync(kv_cache_blocks.GetPtr<void>(), bf.data(), bf.size() * sizeof(uint16_t), MEMCPY_HOST_TO_DEVICE,
                context_->device->GetMemoryManageStream());
  }

  // [0] hidden_buffer_1 (workspace): needs q_concat + k_concat space
  size_t ws_elements = kTotalTokens * kNumHeads * kLatentDim + kTotalTokens * kLatentDim;
  Tensor workspace(MemoryLocation::LOCATION_DEVICE, TYPE_BF16, {ws_elements + 1024});

  // [1] seq_len_host HOST: [batch] int32 — context lengths
  Tensor seq_len_host(MemoryLocation::LOCATION_HOST, TYPE_INT32, {static_cast<size_t>(kBatchSize)});
  {
    int32_t* p = seq_len_host.GetPtr<int32_t>();
    p[0] = kContextLen0;
    p[1] = kContextLen1;
  }

  // context_len_device DEVICE: [batch] int32 — same context lengths, on device.
  // The refactored PagedMlaAttentionInputs requires this as a distinct field (the kernel reads
  // context lengths from device memory in addition to the HOST copy above).
  Tensor context_len_device(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {static_cast<size_t>(kBatchSize)});
  {
    std::vector<int32_t> d = {static_cast<int32_t>(kContextLen0), static_cast<int32_t>(kContextLen1)};
    MemcpyAsync(context_len_device.GetPtr<void>(), d.data(), d.size() * sizeof(int32_t), MEMCPY_HOST_TO_DEVICE,
                context_->device->GetMemoryManageStream());
  }

  // [2] k_cache_blocks_base DEVICE
  // Just use the kv_cache_blocks tensor directly as the base
  Tensor& k_cache_base = kv_cache_blocks;

  // [3] layers_slot_mapping DEVICE: [layer_num * slot_dim] int32
  Tensor slot_mapping(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {static_cast<size_t>(kLayerNum * kSlotDim)});
  {
    // Decode tokens get slots after prefill tokens
    std::vector<int32_t> slots(kLayerNum * kSlotDim);
    for (int l = 0; l < kLayerNum; ++l) {
      slots[l * kSlotDim + 0] = kContextLen0;       // seq0 decode token → slot after its context
      slots[l * kSlotDim + 1] = kContextLen1 + 16;  // seq1 decode token → slot in its block
    }
    MemcpyAsync(slot_mapping.GetPtr<void>(), slots.data(), slots.size() * sizeof(int32_t), MEMCPY_HOST_TO_DEVICE,
                context_->device->GetMemoryManageStream());
  }

  // [4] rotary_embedding_pos DEVICE: [batch] int64
  Tensor rotary_pos(MemoryLocation::LOCATION_DEVICE, TYPE_INT64, {static_cast<size_t>(kTotalTokens)});
  {
    std::vector<int64_t> pos = {static_cast<int64_t>(kContextLen0), static_cast<int64_t>(kContextLen1)};
    MemcpyAsync(rotary_pos.GetPtr<void>(), pos.data(), pos.size() * sizeof(int64_t), MEMCPY_HOST_TO_DEVICE,
                context_->device->GetMemoryManageStream());
  }

  // [5] rotary_embedding_mask (placeholder)
  Tensor rotary_mask(MemoryLocation::LOCATION_DEVICE, TYPE_INT64, {1});

  // [6] layer_kv_cache metadata HOST
  Tensor layer_kv_cache(MemoryLocation::LOCATION_HOST, TYPE_INT64, {4});
  {
    int64_t* meta = layer_kv_cache.GetPtr<int64_t>();
    meta[0] = kMaxBlockNum;
    // Not used in decode (kv_cache_base is used directly)
  }

  // [7] layers_block_table DEVICE: [layer_num * batch * max_blocks_per_seq] int32
  const int bt_size = kLayerNum * kBatchSize * kMaxBlocksPerSeq;
  Tensor block_table(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {static_cast<size_t>(bt_size)});
  {
    std::vector<int32_t> bt(bt_size, 0);
    for (int l = 0; l < kLayerNum; ++l) {
      int base = l * kBatchSize * kMaxBlocksPerSeq;
      bt[base + 0] = 0;
      bt[base + 1] = 1;  // seq0: blocks 0,1
      bt[base + kMaxBlocksPerSeq + 0] = 2;
      bt[base + kMaxBlocksPerSeq + 1] = 3;  // seq1: blocks 2,3
    }
    MemcpyAsync(block_table.GetPtr<void>(), bt.data(), bt.size() * sizeof(int32_t), MEMCPY_HOST_TO_DEVICE,
                context_->device->GetMemoryManageStream());
  }

  // [8] q_nope_tensor: [batch, num_heads, kv_lora_rank] BF16
  Tensor q_nope(MemoryLocation::LOCATION_DEVICE, TYPE_BF16,
                {static_cast<size_t>(kTotalTokens), static_cast<size_t>(kNumHeads), static_cast<size_t>(kKvLoraRank)});
  {
    std::vector<float> f(kTotalTokens * kNumHeads * kKvLoraRank);
    for (auto& v : f) v = dist(rng);
    auto bf = ToBf16(f);
    MemcpyAsync(q_nope.GetPtr<void>(), bf.data(), bf.size() * sizeof(uint16_t), MEMCPY_HOST_TO_DEVICE,
                context_->device->GetMemoryManageStream());
  }

  // [9] q_pe_tensor: [batch, num_heads, qk_rope_head_dim] BF16
  Tensor q_pe(MemoryLocation::LOCATION_DEVICE, TYPE_BF16,
              {static_cast<size_t>(kTotalTokens), static_cast<size_t>(kNumHeads), static_cast<size_t>(kQkRopeHeadDim)});
  {
    std::vector<float> f(kTotalTokens * kNumHeads * kQkRopeHeadDim);
    for (auto& v : f) v = dist(rng);
    auto bf = ToBf16(f);
    MemcpyAsync(q_pe.GetPtr<void>(), bf.data(), bf.size() * sizeof(uint16_t), MEMCPY_HOST_TO_DEVICE,
                context_->device->GetMemoryManageStream());
  }

  // [10] compressed_kv_tensor: [batch, kv_lora_rank] BF16
  Tensor compressed_kv(MemoryLocation::LOCATION_DEVICE, TYPE_BF16,
                       {static_cast<size_t>(kTotalTokens), static_cast<size_t>(kKvLoraRank)});
  {
    std::vector<float> f(kTotalTokens * kKvLoraRank);
    for (auto& v : f) v = dist(rng);
    auto bf = ToBf16(f);
    MemcpyAsync(compressed_kv.GetPtr<void>(), bf.data(), bf.size() * sizeof(uint16_t), MEMCPY_HOST_TO_DEVICE,
                context_->device->GetMemoryManageStream());
  }

  // [11] k_pe_tensor: [batch, qk_rope_head_dim] BF16
  Tensor k_pe(MemoryLocation::LOCATION_DEVICE, TYPE_BF16,
              {static_cast<size_t>(kTotalTokens), static_cast<size_t>(kQkRopeHeadDim)});
  {
    std::vector<float> f(kTotalTokens * kQkRopeHeadDim);
    for (auto& v : f) v = dist(rng);
    auto bf = ToBf16(f);
    MemcpyAsync(k_pe.GetPtr<void>(), bf.data(), bf.size() * sizeof(uint16_t), MEMCPY_HOST_TO_DEVICE,
                context_->device->GetMemoryManageStream());
  }

  // [12-13] placeholder (tile_scheduler, num_splits)
  Tensor ph12(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {1});
  Tensor ph13(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {1});

  // [14] atb_attention_attr HOST: [2] uint64
  Tensor atb_attr(MemoryLocation::LOCATION_HOST, TYPE_UINT64, {2});
  {
    uint64_t* p = atb_attr.GetPtr<uint64_t>();
    p[0] = static_cast<uint64_t>(kSlotDim);
    p[1] = static_cast<uint64_t>(kMaxBlocksPerSeq);
  }

  // Output buffer. XpuPagedMlaAttention writes the absorbed MLA decode result
  // [total_tokens, num_heads, kv_lora_rank], which is larger than num_heads * v_head_dim,
  // so size the buffer for the wider layout to avoid an out-of-bounds write.
  Tensor output(MemoryLocation::LOCATION_DEVICE, TYPE_BF16,
                {static_cast<size_t>(kTotalTokens), static_cast<size_t>(kNumHeads * kKvLoraRank)});

  StreamSynchronize(context_->device->GetMemoryManageStream());

  // Setup layer
  TestablePagedMlaAttentionLayer layer;
  layer.SetupForTest(context_, TYPE_BF16);
  layer.SetCosSinCache(cos_sin_tensor.GetPtr<void>());

  // The refactored PagedMlaAttentionLayer takes a typed PagedMlaAttentionInputs struct.
  PagedMlaAttentionInputs mla_inputs{};
  mla_inputs.hidden_buffer_1 = &workspace;
  mla_inputs.seq_len_host = &seq_len_host;
  mla_inputs.k_cache_blocks_base = &k_cache_base;
  mla_inputs.layers_slot_mapping = &slot_mapping;
  mla_inputs.rotary_embedding_pos = &rotary_pos;
  mla_inputs.rotary_embedding_mask = &rotary_mask;
  mla_inputs.layer_kv_cache = &layer_kv_cache;
  mla_inputs.layers_block_table = &block_table;
  mla_inputs.q_nope = &q_nope;
  mla_inputs.q_pe = &q_pe;
  mla_inputs.compressed_kv = &compressed_kv;
  mla_inputs.k_pe = &k_pe;
  mla_inputs.tile_scheduler_metadata = &ph12;
  mla_inputs.num_splits = &ph13;
  mla_inputs.atb_attention_attr = &atb_attr;
  mla_inputs.context_len_device = &context_len_device;

  Status status = layer.Forward(mla_inputs, &output);
  StreamSynchronize(context_->device->GetPrimaryComputeStream());
  ASSERT_TRUE(status.OK()) << "PagedMlaAttentionLayer::Forward() failed: " << status.ToString();

  // Verify outputs: with random data, some NaN/Inf may occur. Primary check is status.OK().
  std::vector<uint16_t> h_out(kTotalTokens * kOutDim);
  Memcpy(h_out.data(), output.GetPtr<void>(), h_out.size() * sizeof(uint16_t), MEMCPY_DEVICE_TO_HOST);

  int finite_count = 0, nonzero_count = 0;
  for (size_t i = 0; i < h_out.size(); ++i) {
    float v = HalfToFp32<bfloat16>(h_out[i]);
    if (std::isfinite(v)) ++finite_count;
    if (std::isfinite(v) && std::fabs(v) > 1e-7f) ++nonzero_count;
  }
  EXPECT_EQ(finite_count, static_cast<int>(h_out.size())) << "Some output elements are NaN/Inf";
  EXPECT_GT(nonzero_count, 0) << "All output elements are zero";
  printf("PagedMlaAttention BF16: batch=%d context=[%d,%d] out_dim=%d finite=%d/%zu nonzero=%d — PASS\n", kBatchSize,
         kContextLen0, kContextLen1, kOutDim, finite_count, h_out.size(), nonzero_count);
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST 8: KunlunMoeLayer W4A8 Layer-Level Forward (Kimi-K2.6 config, ep_size=64)
//
// Exercises the full KunlunMoeLayer::Forward() W4A8 code path using
// Kimi-K2.6 model parameters. The EP kernel (moe_ep_pre_sorted_quant2d)
// requires ep_size>1 and production-scale dimensions to satisfy its
// internal constraints.
//
// 参数选择说明：
//   - hidden_size/inter_size/num_experts_total/topk/scoring_func 必须使用真实模型
//     参数，infer_ops EP quant kernel 对这些维度有未文档化的最低要求，缩小会返回
//     错误码 1（invalid param）。
//   - ep_size 设为 64（2 的幂）以减少 local expert 数（384/64=6），从而控制权重
//     内存（~130MB vs ep_size=8 时 ~1GB）和测试耗时（~2s vs ~16s）。ep_size 只
//     影响权重 batch 维度，不影响 kernel 链路的正确性验证。
//
// Flow: gating(sigmoid+bias) → topk → EpPreSortedQuant2d → W4A8 FC1 →
//       SiluAndMulQuantUnt → W4A8 FC2 → EpFfnPostFusion → output
//
// Verifies: status OK, output finite, output non-zero.
// ═══════════════════════════════════════════════════════════════════════════════
TEST_F(KunlunW4A8LayerTest, MoeW4A8LayerForwardEp8) {
  // ── Config (Kimi-K2.6 W4A8 model parameters) ───────────────────────────────
  constexpr int kNumTokens = 64;
  constexpr int kTopK = 8;
  constexpr int kNumExpertsTotal = 384;  // n_routed_experts (Kimi-K2.6)
  constexpr int kEpSize = 64;            // 2的幂，减少local expert数以控制耗时
  constexpr int kEpRank = 0;
  constexpr int kNumExpertsPerNode = kNumExpertsTotal / kEpSize;  // 6 local experts
  constexpr int kHidden = 7168;
  constexpr int kInter = 2048;  // moe_intermediate_size
  constexpr int kPackedK = kHidden / 2;

  std::mt19937 rng(2024);
  std::uniform_real_distribution<float> fp_dist(-0.5f, 0.5f);
  std::uniform_int_distribution<int> int_dist(-60, 60);
  std::uniform_real_distribution<float> scale_dist(0.005f, 0.02f);
  std::uniform_int_distribution<int> nibble_dist(-7, 7);

  // ── Prepare host data ────────────────────────────────────────────────────────
  // hidden_states [m, H] BF16
  std::vector<float> h_hidden(kNumTokens * kHidden);
  for (auto& v : h_hidden) v = fp_dist(rng);
  auto h_hidden_bf16 = ToBf16(h_hidden);

  // gating_output [m, num_experts_total] BF16 — random logits
  std::vector<float> h_gating(kNumTokens * kNumExpertsTotal);
  for (auto& v : h_gating) v = fp_dist(rng);
  auto h_gating_bf16 = ToBf16(h_gating);

  // up_gate_proj_weight [E_local, 2*inter, hidden/2] INT4x2 packed
  std::vector<int8_t> h_w1(kNumExpertsPerNode * 2 * kInter * kPackedK);
  for (auto& v : h_w1) {
    int low = nibble_dist(rng) & 0xF;
    int high = nibble_dist(rng) & 0xF;
    v = static_cast<int8_t>(low | (high << 4));
  }
  // w1 scale [E_local * 2*inter] FP32
  std::vector<float> h_w1_scale(kNumExpertsPerNode * 2 * kInter);
  for (auto& v : h_w1_scale) v = scale_dist(rng);

  // down_proj_weight [E_local, hidden, inter/2] INT4x2 packed
  std::vector<int8_t> h_w2(kNumExpertsPerNode * kHidden * (kInter / 2));
  for (auto& v : h_w2) {
    int low = nibble_dist(rng) & 0xF;
    int high = nibble_dist(rng) & 0xF;
    v = static_cast<int8_t>(low | (high << 4));
  }
  // w2 scale [E_local * hidden] FP32
  std::vector<float> h_w2_scale(kNumExpertsPerNode * kHidden);
  for (auto& v : h_w2_scale) v = scale_dist(rng);

  // ── Upload to device ─────────────────────────────────────────────────────────
  Tensor hidden_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16,
                       {static_cast<size_t>(kNumTokens), static_cast<size_t>(kHidden)});
  Tensor gating_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16,
                       {static_cast<size_t>(kNumTokens), static_cast<size_t>(kNumExpertsTotal)});
  Tensor w1_tensor(
      MemoryLocation::LOCATION_DEVICE, TYPE_INT8,
      {static_cast<size_t>(kNumExpertsPerNode), static_cast<size_t>(2 * kInter), static_cast<size_t>(kPackedK)});
  Tensor w1_scale_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32,
                         {static_cast<size_t>(kNumExpertsPerNode * 2 * kInter)});
  Tensor w2_tensor(
      MemoryLocation::LOCATION_DEVICE, TYPE_INT8,
      {static_cast<size_t>(kNumExpertsPerNode), static_cast<size_t>(kHidden), static_cast<size_t>(kInter / 2)});
  Tensor w2_scale_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32,
                         {static_cast<size_t>(kNumExpertsPerNode * kHidden)});
  Tensor output_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16,
                       {static_cast<size_t>(kNumTokens), static_cast<size_t>(kHidden)});

  MemcpyAsync(hidden_tensor.GetPtr<void>(), h_hidden_bf16.data(), h_hidden_bf16.size() * sizeof(uint16_t),
              MEMCPY_HOST_TO_DEVICE, context_->device->GetMemoryManageStream());
  MemcpyAsync(gating_tensor.GetPtr<void>(), h_gating_bf16.data(), h_gating_bf16.size() * sizeof(uint16_t),
              MEMCPY_HOST_TO_DEVICE, context_->device->GetMemoryManageStream());
  MemcpyAsync(w1_tensor.GetPtr<void>(), h_w1.data(), h_w1.size() * sizeof(int8_t), MEMCPY_HOST_TO_DEVICE,
              context_->device->GetMemoryManageStream());
  MemcpyAsync(w1_scale_tensor.GetPtr<void>(), h_w1_scale.data(), h_w1_scale.size() * sizeof(float),
              MEMCPY_HOST_TO_DEVICE, context_->device->GetMemoryManageStream());
  MemcpyAsync(w2_tensor.GetPtr<void>(), h_w2.data(), h_w2.size() * sizeof(int8_t), MEMCPY_HOST_TO_DEVICE,
              context_->device->GetMemoryManageStream());
  MemcpyAsync(w2_scale_tensor.GetPtr<void>(), h_w2_scale.data(), h_w2_scale.size() * sizeof(float),
              MEMCPY_HOST_TO_DEVICE, context_->device->GetMemoryManageStream());
  StreamSynchronize(context_->device->GetMemoryManageStream());

  // Bind weight_scales (W4A8 path reads input_tensors[2].weight_scales / [3].weight_scales)
  w1_tensor.weight_scales = &w1_scale_tensor;
  w2_tensor.weight_scales = &w2_scale_tensor;

  // ── Setup layer ───────────────────────────────────────────────────────────────
  TestableKunlunMoeLayer layer;
  layer.SetupForTest(/*ctx=*/context_, /*dtype=*/TYPE_BF16, /*max_token_num=*/kNumTokens, /*layer_idx=*/0,
                     /*expert_num_per_node=*/kNumExpertsPerNode, /*hidden_size=*/kHidden, /*inter_size=*/kInter,
                     /*topk=*/kTopK, /*is_w4a8_quantized=*/true, /*ep_size=*/kEpSize, /*ep_rank=*/kEpRank,
                     /*scoring_func=*/"sigmoid", /*norm_topk_prob=*/true, /*routed_scaling_factor=*/2.827f,
                     /*num_expert_group=*/1, /*topk_group=*/1, /*use_e_score_correction_bias=*/true);

  // Allocate workspace
  size_t ws_bytes = layer.GetWorkspaceSize();
  auto ws_buffer = std::make_shared<Tensor>(MemoryLocation::LOCATION_DEVICE, TYPE_INT8, std::vector<size_t>{ws_bytes});
  layer.SetWorkspaceBuffer(ws_buffer);

  // ── Forward ───────────────────────────────────────────────────────────────────
  //   [0] hidden_states [m, H]
  //   [1] gating_output [m, num_experts_total]
  //   [2] up_gate_proj_weight [E_local, 2*I, H/2] (INT4x2 packed, weight_scales bound)
  //   [3] down_proj_weight [E_local, H, I/2] (INT4x2 packed, weight_scales bound)
  //   [4] expert_map (ignored for kunlun)
  //   [5] e_score_correction_bias [num_experts_total] FP32
  Tensor bias_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP32, {static_cast<size_t>(kNumExpertsTotal)});
  {
    std::vector<float> h_bias(kNumExpertsTotal);
    for (auto& v : h_bias) v = fp_dist(rng) * 0.1f;
    MemcpyAsync(bias_tensor.GetPtr<void>(), h_bias.data(), h_bias.size() * sizeof(float), MEMCPY_HOST_TO_DEVICE,
                context_->device->GetMemoryManageStream());
    StreamSynchronize(context_->device->GetMemoryManageStream());
  }
  // eplb_map is ignored on Kunlun; workspace param is unused (buffer carved from workspace_buffer_
  // set via SetWorkspaceBuffer above). e_score_correction_bias is consumed (use_e_score_..._bias=true).
  Tensor eplb_map(MemoryLocation::LOCATION_DEVICE, TYPE_INT64, {1});
  Tensor dummy_ws(MemoryLocation::LOCATION_DEVICE, TYPE_INT8, {1});
  std::vector<Tensor> outputs = {output_tensor};

  Status status = layer.Forward(&hidden_tensor, &gating_tensor, &w1_tensor, &w2_tensor, &eplb_map, &outputs[0],
                                &dummy_ws, &bias_tensor);
  StreamSynchronize(context_->device->GetPrimaryComputeStream());
  ASSERT_TRUE(status.OK()) << "KunlunMoeLayer W4A8 ep=8 Forward failed: " << status.ToString();

  // ── Verify output shape ───────────────────────────────────────────────────────
  EXPECT_EQ(outputs[0].shape[0], static_cast<size_t>(kNumTokens));
  EXPECT_EQ(outputs[0].shape[1], static_cast<size_t>(kHidden));

  // ── D2H and verify finite + non-zero ──────────────────────────────────────────
  std::vector<uint16_t> h_out_bf16(kNumTokens * kHidden);
  Memcpy(h_out_bf16.data(), outputs[0].GetPtr<void>(), h_out_bf16.size() * sizeof(uint16_t), MEMCPY_DEVICE_TO_HOST);
  auto act_out = FromBf16(h_out_bf16);

  int finite_count = 0, nonzero_count = 0;
  for (size_t i = 0; i < act_out.size(); ++i) {
    ASSERT_TRUE(std::isfinite(act_out[i])) << "NaN/Inf at output[" << i << "]";
    ++finite_count;
    if (std::fabs(act_out[i]) > 1e-7f) ++nonzero_count;
  }
  EXPECT_EQ(finite_count, kNumTokens * kHidden) << "Output contains NaN/Inf";
  EXPECT_GT(nonzero_count, 0) << "All output elements are zero";

  printf(
      "MoeW4A8LayerForward ep=8: tokens=%d experts_per_node=%d experts_total=%d topk=%d hidden=%d inter=%d finite=%d "
      "nonzero=%d — PASS\n",
      kNumTokens, kNumExpertsPerNode, kNumExpertsTotal, kTopK, kHidden, kInter, finite_count, nonzero_count);
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST 9: KunlunMoeLayer BF16 Layer-Level Forward (ep_size=1, sanity baseline)
//
// Exercises the BF16 (non-quantized) MoE code path as a baseline sanity check
// using the same test harness. This validates the TestableKunlunMoeLayer setup
// is correct by running a simpler (non-W4A8) path.
// ═══════════════════════════════════════════════════════════════════════════════
TEST_F(KunlunW4A8LayerTest, MoeBf16LayerForwardEp1) {
  constexpr int kNumTokens = 4;
  constexpr int kTopK = 2;
  constexpr int kNumExperts = 4;
  constexpr int kHidden = 64;
  constexpr int kInter = 128;

  std::mt19937 rng(314);
  std::uniform_real_distribution<float> fp_dist(-0.5f, 0.5f);

  std::vector<float> h_hidden(kNumTokens * kHidden);
  for (auto& v : h_hidden) v = fp_dist(rng);
  auto h_hidden_bf16 = ToBf16(h_hidden);

  std::vector<float> h_gating(kNumTokens * kNumExperts);
  for (auto& v : h_gating) v = fp_dist(rng);
  auto h_gating_bf16 = ToBf16(h_gating);

  // BF16 weights [E, 2*inter, hidden], [E, hidden, inter]
  std::vector<float> h_w1(kNumExperts * 2 * kInter * kHidden);
  for (auto& v : h_w1) v = fp_dist(rng) * 0.1f;
  auto h_w1_bf16 = ToBf16(h_w1);

  std::vector<float> h_w2(kNumExperts * kHidden * kInter);
  for (auto& v : h_w2) v = fp_dist(rng) * 0.1f;
  auto h_w2_bf16 = ToBf16(h_w2);

  Tensor hidden_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16,
                       {static_cast<size_t>(kNumTokens), static_cast<size_t>(kHidden)});
  Tensor gating_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16,
                       {static_cast<size_t>(kNumTokens), static_cast<size_t>(kNumExperts)});
  Tensor w1_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16,
                   {static_cast<size_t>(kNumExperts), static_cast<size_t>(2 * kInter), static_cast<size_t>(kHidden)});
  Tensor w2_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16,
                   {static_cast<size_t>(kNumExperts), static_cast<size_t>(kHidden), static_cast<size_t>(kInter)});
  Tensor output_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BF16,
                       {static_cast<size_t>(kNumTokens), static_cast<size_t>(kHidden)});

  MemcpyAsync(hidden_tensor.GetPtr<void>(), h_hidden_bf16.data(), h_hidden_bf16.size() * sizeof(uint16_t),
              MEMCPY_HOST_TO_DEVICE, context_->device->GetMemoryManageStream());
  MemcpyAsync(gating_tensor.GetPtr<void>(), h_gating_bf16.data(), h_gating_bf16.size() * sizeof(uint16_t),
              MEMCPY_HOST_TO_DEVICE, context_->device->GetMemoryManageStream());
  MemcpyAsync(w1_tensor.GetPtr<void>(), h_w1_bf16.data(), h_w1_bf16.size() * sizeof(uint16_t), MEMCPY_HOST_TO_DEVICE,
              context_->device->GetMemoryManageStream());
  MemcpyAsync(w2_tensor.GetPtr<void>(), h_w2_bf16.data(), h_w2_bf16.size() * sizeof(uint16_t), MEMCPY_HOST_TO_DEVICE,
              context_->device->GetMemoryManageStream());
  StreamSynchronize(context_->device->GetMemoryManageStream());

  TestableKunlunMoeLayer layer;
  layer.SetupForTest(/*ctx=*/context_, /*dtype=*/TYPE_BF16, /*max_token_num=*/kNumTokens, /*layer_idx=*/0,
                     /*expert_num_per_node=*/kNumExperts, /*hidden_size=*/kHidden, /*inter_size=*/kInter,
                     /*topk=*/kTopK, /*is_w4a8_quantized=*/false, /*ep_size=*/1, /*ep_rank=*/0,
                     /*scoring_func=*/"softmax", /*norm_topk_prob=*/false);

  size_t ws_bytes = layer.GetWorkspaceSize();
  auto ws_buffer = std::make_shared<Tensor>(MemoryLocation::LOCATION_DEVICE, TYPE_INT8, std::vector<size_t>{ws_bytes});
  layer.SetWorkspaceBuffer(ws_buffer);

  Tensor eplb_map(MemoryLocation::LOCATION_DEVICE, TYPE_INT64, {1});
  Tensor dummy_ws(MemoryLocation::LOCATION_DEVICE, TYPE_INT8, {1});
  std::vector<Tensor> outputs = {output_tensor};

  // No e_score bias in the BF16 baseline path (use_e_score_correction_bias=false).
  Status status =
      layer.Forward(&hidden_tensor, &gating_tensor, &w1_tensor, &w2_tensor, &eplb_map, &outputs[0], &dummy_ws);
  StreamSynchronize(context_->device->GetPrimaryComputeStream());
  ASSERT_TRUE(status.OK()) << "KunlunMoeLayer BF16 ep=1 Forward failed: " << status.ToString();

  EXPECT_EQ(outputs[0].shape[0], static_cast<size_t>(kNumTokens));
  EXPECT_EQ(outputs[0].shape[1], static_cast<size_t>(kHidden));

  std::vector<uint16_t> h_out_bf16(kNumTokens * kHidden);
  Memcpy(h_out_bf16.data(), outputs[0].GetPtr<void>(), h_out_bf16.size() * sizeof(uint16_t), MEMCPY_DEVICE_TO_HOST);
  auto act_out = FromBf16(h_out_bf16);

  int finite_count = 0, nonzero_count = 0;
  for (size_t i = 0; i < act_out.size(); ++i) {
    ASSERT_TRUE(std::isfinite(act_out[i])) << "NaN/Inf at output[" << i << "]";
    ++finite_count;
    if (std::fabs(act_out[i]) > 1e-7f) ++nonzero_count;
  }
  EXPECT_EQ(finite_count, kNumTokens * kHidden);
  EXPECT_GT(nonzero_count, 0);

  printf("MoeBf16LayerForward ep=1: tokens=%d experts=%d topk=%d hidden=%d inter=%d finite=%d nonzero=%d — PASS\n",
         kNumTokens, kNumExperts, kTopK, kHidden, kInter, finite_count, nonzero_count);
}

}  // namespace ksana_llm
