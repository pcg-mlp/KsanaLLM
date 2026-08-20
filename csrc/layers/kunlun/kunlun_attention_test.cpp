/* Copyright 2026 Tencent Inc.  All rights reserved.

   FlashAttentionLayer::Forward() functional test on Kunlun XPU.
   Uses TYPE_FP16 (the only supported path) with random float16 QKV data.
   Verifies the full pipeline: QKV-split → RoPE-skip → XpuFlashAttentionContext → KV-scatter.

   Build:
     cmake .. -DWITH_KUNLUN=ON -DWITH_TESTING=ON
     make kunlun_layer_test

   Run:
     ./bin/kunlun_layer_test --gtest_filter=*FlashAttention*
==============================================================================*/

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include "device/common_device.h"
#include "device/device_types.h"
#include "layers/flash_attention_layer.h"
#include "layers/kunlun/host_fp_helpers.h"
#include "layers/paged_attention_layer.h"
#include "test.h"
#include "utils/logger.h"

namespace ksana_llm {

using kunlun_host::Fp32ToHalf;
using kunlun_host::HalfToFp32;

// ── Qwen3-0.6B config ────────────────────────────────────────────────────────
static constexpr int kNumHeads = 16;
static constexpr int kNumKvHeads = 8;
static constexpr int kHeadDim = 128;
static constexpr int kLayerNum = 28;
static constexpr int kLayerIndex = 0;
static constexpr int kBlockTokenNum = 16;

static constexpr int kBatchSize = 2;
static constexpr int kSeqLen0 = 3;
static constexpr int kSeqLen1 = 5;
static constexpr int kTotalTokenNum = kSeqLen0 + kSeqLen1;  // 8
static constexpr int kMaxBlockNum = 64;
static constexpr int kSlotMappingDim = kTotalTokenNum;

// ── Testable subclass ─────────────────────────────────────────────────────────
class TestableFlashAttentionLayer : public FlashAttentionLayer {
 public:
  void SetupForTest(std::shared_ptr<Context> ctx, DataType dtype, bool use_qk_norm = false) {
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
    block_size_ = kBlockTokenNum * kNumKvHeads * kHeadDim * sizeof(uint16_t);
    num_heads_ = kNumHeads;
    num_kv_heads_ = kNumKvHeads;
    head_size_ = kHeadDim;
    stride_size_ = (kNumHeads + 2 * kNumKvHeads) * kHeadDim;
    tensor_para_size_ = 1;
    max_position_embeddings_ = 32768;
    base_ = 1000000.0f;
    use_qk_norm_ = use_qk_norm;
    layernorm_eps_ = 1e-6f;
    enable_qk_pre_norm_before_rotary_pos_ = false;
    k_scale_ = 1.0f;
    v_scale_ = 1.0f;
    is_causal_ = true;
    no_rope_ = false;
    cos_sin_cache_ptr_ = nullptr;  // RoPE skipped (cos_sin_cache nullptr branch)

    mm_quant_mode_ = QUANT_NONE;
    q_lora_rank_ = 0;
    kv_lora_rank_ = 0;
    qk_nope_head_dim_ = 0;
    qk_rope_head_dim_ = 0;
    v_head_dim_ = 0;
    attn_scale_ = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
    attn_temperature_tuning_ = 0;
    floor_scale_ = 0;

    // workspace: q_cont + k_cont + v_cont for all tokens.
    // When QK-norm is enabled, the layer casts norm weights (fp16) to fp32 into
    // the region immediately after the QKV split buffers, so we need extra space.
    int q_dim = kNumHeads * kHeadDim;
    int kv_dim = kNumKvHeads * kHeadDim;
    size_t ws_bytes = static_cast<size_t>(kTotalTokenNum) * (q_dim + 2 * kv_dim) * sizeof(uint16_t);
    if (use_qk_norm) ws_bytes += 2 * static_cast<size_t>(kHeadDim) * sizeof(float);
    workspace_buffer_ =
        std::make_shared<Tensor>(MemoryLocation::LOCATION_DEVICE, TYPE_INT8, std::vector<size_t>{ws_bytes});
  }
};

// ── Test fixture ──────────────────────────────────────────────────────────────
class KunlunFlashAttentionTest : public testing::Test {
 protected:
  void SetUp() override {
    context_ = std::make_shared<Context>();
    context_->device = std::make_shared<DeviceContext>(1);
  }
  void TearDown() override {}

  std::shared_ptr<Context> context_{nullptr};
};

// ── Helper: build float16 host vector from float data ────────────────────────
static std::vector<uint16_t> ToFloat16(const std::vector<float>& f) {
  std::vector<uint16_t> h(f.size());
  for (size_t i = 0; i < f.size(); ++i) h[i] = Fp32ToHalf<float16>(f[i]);
  return h;
}

// ── Test ──────────────────────────────────────────────────────────────────────
TEST_F(KunlunFlashAttentionTest, ForwardFloat16) {
  constexpr int kQkvDim = (kNumHeads + 2 * kNumKvHeads) * kHeadDim;  // 4096
  constexpr int kOutDim = kNumHeads * kHeadDim;                      // 2048

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-0.3f, 0.3f);

  // ── input[0]: qkv_tensor [total_token_num, qkv_dim], FP16 ────────────────
  Tensor qkv_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16,
                    {static_cast<size_t>(kTotalTokenNum), static_cast<size_t>(kQkvDim)});
  {
    std::vector<float> f(kTotalTokenNum * kQkvDim);
    for (auto& v : f) v = dist(rng);
    auto h = ToFloat16(f);
    MemcpyAsync(qkv_tensor.GetPtr<void>(), h.data(), h.size() * sizeof(uint16_t), MEMCPY_HOST_TO_DEVICE,
                context_->device->GetMemoryManageStream());
  }

  // ── input[1]: rotary_embedding_pos [total_token_num], int64 ──────────────
  Tensor rotary_pos(MemoryLocation::LOCATION_DEVICE, TYPE_INT64, {static_cast<size_t>(kTotalTokenNum)});
  {
    std::vector<int64_t> positions = {0, 1, 2, 0, 1, 2, 3, 4};
    MemcpyAsync(rotary_pos.GetPtr<void>(), positions.data(), positions.size() * sizeof(int64_t), MEMCPY_HOST_TO_DEVICE,
                context_->device->GetMemoryManageStream());
  }

  // ── input[2]: slot_mapping [layer_num * slot_mapping_dim], int32 ─────────
  Tensor slot_mapping(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {static_cast<size_t>(kLayerNum * kSlotMappingDim)});
  {
    std::vector<int32_t> slots(kLayerNum * kSlotMappingDim);
    for (int layer = 0; layer < kLayerNum; ++layer) {
      for (int t = 0; t < kSlotMappingDim; ++t) slots[layer * kSlotMappingDim + t] = t;
    }
    MemcpyAsync(slot_mapping.GetPtr<void>(), slots.data(), slots.size() * sizeof(int32_t), MEMCPY_HOST_TO_DEVICE,
                context_->device->GetMemoryManageStream());
  }

  // ── input[3,4]: k_cache / v_cache [max_block_num, block_token_num, num_kv_heads, head_size], FP16
  Tensor k_cache(MemoryLocation::LOCATION_DEVICE, TYPE_FP16,
                 {static_cast<size_t>(kMaxBlockNum), static_cast<size_t>(kBlockTokenNum),
                  static_cast<size_t>(kNumKvHeads), static_cast<size_t>(kHeadDim)});
  Tensor v_cache(MemoryLocation::LOCATION_DEVICE, TYPE_FP16,
                 {static_cast<size_t>(kMaxBlockNum), static_cast<size_t>(kBlockTokenNum),
                  static_cast<size_t>(kNumKvHeads), static_cast<size_t>(kHeadDim)});

  // ── input[5]: seq_len_host [batch_size], int32, HOST ─────────────────────
  Tensor seq_len_host(MemoryLocation::LOCATION_HOST, TYPE_INT32, {static_cast<size_t>(kBatchSize)});
  {
    int32_t* p = seq_len_host.GetPtr<int32_t>();
    p[0] = kSeqLen0;
    p[1] = kSeqLen1;
  }

  // ── input[6]: forward_shape (CUDA 13D + Kunlun [13]) ────────────────────
  // Flash test: 2 flash requests, no decode, no prefix.
  // shape[7] = dp_multi_token_request_num = kBatchSize (read by FlashAttentionLayer).
  Tensor forward_shape(MemoryLocation::LOCATION_HOST, TYPE_INT32,
                       {static_cast<size_t>(kBatchSize), static_cast<size_t>(kTotalTokenNum), 0, 0, 0, 0, 0,
                        static_cast<size_t>(kBatchSize), static_cast<size_t>(kTotalTokenNum), 0, 0, 0, 0, 0});

  // ── input[7]: atb_attention_attr [2], uint64, HOST ────────────────────────
  Tensor atb_attr(MemoryLocation::LOCATION_HOST, TYPE_UINT64, {2});
  {
    uint64_t* p = atb_attr.GetPtr<uint64_t>();
    p[0] = static_cast<uint64_t>(kSlotMappingDim);  // layers_slot_mapping_dim_1
    p[1] = static_cast<uint64_t>(4);                // max_num_blocks_per_query
  }

  // ── output[0]: flash_attention_output [total_token_num, num_heads * head_size], FP16
  Tensor output_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16,
                       {static_cast<size_t>(kTotalTokenNum), static_cast<size_t>(kOutDim)});

  StreamSynchronize(context_->device->GetMemoryManageStream());

  // input[8], input[9]: placeholders (MLA qk_norm weights, unused here)
  Tensor placeholder8(MemoryLocation::LOCATION_DEVICE, TYPE_FP16, {0});
  Tensor placeholder9(MemoryLocation::LOCATION_DEVICE, TYPE_FP16, {0});

  // input[10]: qlod_host_tensor [batch_size+1], int32, HOST — cumulative token offsets
  Tensor qlod_host(MemoryLocation::LOCATION_HOST, TYPE_INT32, {static_cast<size_t>(kBatchSize + 1)});
  {
    int32_t* p = qlod_host.GetPtr<int32_t>();
    p[0] = 0;
    p[1] = kSeqLen0;
    p[2] = kSeqLen0 + kSeqLen1;
  }

  TestableFlashAttentionLayer layer;
  layer.SetupForTest(context_, TYPE_FP16);

  // FlashAttentionLayer now takes a typed KunlunForwardParams struct instead of a Tensor vector.
  // Scalars come from the Kunlun 13-D forward_shape layout: shape[7]=batch_size,
  // shape[8]=max_tokens, shape[2]=layer_block_num.
  FlashAttentionLayer::KunlunForwardParams params;
  params.qkv = &qkv_tensor;
  params.rotary_embedding_pos = &rotary_pos;
  params.layers_slot_mapping = &slot_mapping;
  params.k_cache_blocks_base = &k_cache;
  params.v_cache_blocks_base = &v_cache;
  params.seq_len_host = &seq_len_host;
  params.batch_size = kBatchSize;
  params.max_tokens = kTotalTokenNum;
  params.layer_block_num = 0;
  params.atb_attention_attr = &atb_attr;
  params.query_layernorm_weight = &placeholder8;
  params.key_layernorm_weight = &placeholder9;
  params.qlod = &qlod_host;

  Status status = layer.Forward(params, &output_tensor);
  StreamSynchronize(context_->device->GetPrimaryComputeStream());

  ASSERT_TRUE(status.OK()) << "FlashAttentionLayer::Forward() failed: " << status.ToString();

  EXPECT_EQ(output_tensor.shape[0], static_cast<size_t>(kTotalTokenNum));
  EXPECT_EQ(output_tensor.shape[1], static_cast<size_t>(kOutDim));

  // Read output back to host as float16, convert to float, verify finite + non-zero
  std::vector<uint16_t> out_f16(kTotalTokenNum * kOutDim);
  MemcpyAsync(out_f16.data(), output_tensor.GetPtr<void>(), out_f16.size() * sizeof(uint16_t), MEMCPY_DEVICE_TO_HOST,
              context_->device->GetPrimaryComputeStream());
  StreamSynchronize(context_->device->GetPrimaryComputeStream());

  int finite_count = 0, nonzero_count = 0;
  for (size_t i = 0; i < out_f16.size(); ++i) {
    float v = HalfToFp32<float16>(out_f16[i]);
    ASSERT_TRUE(std::isfinite(v)) << "NaN/Inf at output[" << i << "]";
    if (std::fabs(v) > 1e-7f) ++nonzero_count;
    ++finite_count;
  }
  EXPECT_GT(nonzero_count, 0) << "All output elements are zero — FA likely not running";

  KLLM_LOG_INFO << "FlashAttentionLayer::Forward() FP16: batch=" << kBatchSize << " tokens=" << kTotalTokenNum
                << " out_dim=" << kOutDim << " finite=" << finite_count << " nonzero=" << nonzero_count;
}

// ── FlashAttentionLayer QK-norm (Qwen3-specific) ─────────────────────────────
// Same config as ForwardFloat16 but with use_qk_norm=true and extra inputs[8/9]
// for q/k norm weights. Verifies the per-head RMSNorm path before attention.
TEST_F(KunlunFlashAttentionTest, ForwardFloat16WithQkNorm) {
  constexpr int kQkvDim = (kNumHeads + 2 * kNumKvHeads) * kHeadDim;
  constexpr int kOutDim = kNumHeads * kHeadDim;

  std::mt19937 rng(99);
  std::uniform_real_distribution<float> dist(-0.3f, 0.3f);

  // input[0]: qkv_tensor
  Tensor qkv_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16,
                    {static_cast<size_t>(kTotalTokenNum), static_cast<size_t>(kQkvDim)});
  {
    std::vector<float> f(kTotalTokenNum * kQkvDim);
    for (auto& v : f) v = dist(rng);
    auto h = ToFloat16(f);
    MemcpyAsync(qkv_tensor.GetPtr<void>(), h.data(), h.size() * sizeof(uint16_t), MEMCPY_HOST_TO_DEVICE,
                context_->device->GetMemoryManageStream());
  }

  // input[1]: rotary positions
  Tensor rotary_pos(MemoryLocation::LOCATION_DEVICE, TYPE_INT64, {static_cast<size_t>(kTotalTokenNum)});
  {
    std::vector<int64_t> positions = {0, 1, 2, 0, 1, 2, 3, 4};
    MemcpyAsync(rotary_pos.GetPtr<void>(), positions.data(), positions.size() * sizeof(int64_t), MEMCPY_HOST_TO_DEVICE,
                context_->device->GetMemoryManageStream());
  }

  // input[2]: slot_mapping
  Tensor slot_mapping(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {static_cast<size_t>(kLayerNum * kSlotMappingDim)});
  {
    std::vector<int32_t> slots(kLayerNum * kSlotMappingDim);
    for (int l = 0; l < kLayerNum; ++l)
      for (int t = 0; t < kSlotMappingDim; ++t) slots[l * kSlotMappingDim + t] = t;
    MemcpyAsync(slot_mapping.GetPtr<void>(), slots.data(), slots.size() * sizeof(int32_t), MEMCPY_HOST_TO_DEVICE,
                context_->device->GetMemoryManageStream());
  }

  // input[3,4]: k/v cache
  Tensor k_cache(MemoryLocation::LOCATION_DEVICE, TYPE_FP16,
                 {static_cast<size_t>(kMaxBlockNum), static_cast<size_t>(kBlockTokenNum),
                  static_cast<size_t>(kNumKvHeads), static_cast<size_t>(kHeadDim)});
  Tensor v_cache(MemoryLocation::LOCATION_DEVICE, TYPE_FP16,
                 {static_cast<size_t>(kMaxBlockNum), static_cast<size_t>(kBlockTokenNum),
                  static_cast<size_t>(kNumKvHeads), static_cast<size_t>(kHeadDim)});

  // input[5]: seq_len_host
  Tensor seq_len_host(MemoryLocation::LOCATION_HOST, TYPE_INT32, {static_cast<size_t>(kBatchSize)});
  {
    int32_t* p = seq_len_host.GetPtr<int32_t>();
    p[0] = kSeqLen0;
    p[1] = kSeqLen1;
  }

  // input[6]: forward_shape (CUDA 13D + Kunlun [13])
  // Flash test with qk_norm: 2 flash requests, no decode, no prefix.
  Tensor forward_shape(MemoryLocation::LOCATION_HOST, TYPE_INT32,
                       {static_cast<size_t>(kBatchSize), static_cast<size_t>(kTotalTokenNum), 0, 0, 0, 0, 0,
                        static_cast<size_t>(kBatchSize), static_cast<size_t>(kTotalTokenNum), 0, 0, 0, 0, 0});

  // input[7]: atb_attention_attr
  Tensor atb_attr(MemoryLocation::LOCATION_HOST, TYPE_UINT64, {2});
  {
    uint64_t* p = atb_attr.GetPtr<uint64_t>();
    p[0] = kSlotMappingDim;
    p[1] = 4;
  }

  // input[8]: q_norm_weight [head_size], fp16 — non-unit weights to make the norm non-trivial
  Tensor q_norm_w(MemoryLocation::LOCATION_DEVICE, TYPE_FP16, {static_cast<size_t>(kHeadDim)});
  {
    std::vector<float> fw(kHeadDim);
    for (int i = 0; i < kHeadDim; ++i) fw[i] = 0.5f + static_cast<float>(i) * 0.001f;
    auto h = ToFloat16(fw);
    MemcpyAsync(q_norm_w.GetPtr<void>(), h.data(), h.size() * sizeof(uint16_t), MEMCPY_HOST_TO_DEVICE,
                context_->device->GetMemoryManageStream());
  }

  // input[9]: k_norm_weight [head_size], fp16
  Tensor k_norm_w(MemoryLocation::LOCATION_DEVICE, TYPE_FP16, {static_cast<size_t>(kHeadDim)});
  {
    std::vector<float> fw(kHeadDim);
    for (int i = 0; i < kHeadDim; ++i) fw[i] = 1.0f - static_cast<float>(i) * 0.001f;
    auto h = ToFloat16(fw);
    MemcpyAsync(k_norm_w.GetPtr<void>(), h.data(), h.size() * sizeof(uint16_t), MEMCPY_HOST_TO_DEVICE,
                context_->device->GetMemoryManageStream());
  }

  // output
  Tensor output_tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16,
                       {static_cast<size_t>(kTotalTokenNum), static_cast<size_t>(kOutDim)});

  StreamSynchronize(context_->device->GetMemoryManageStream());

  // input[10]: qlod_host_tensor [batch_size+1], int32, HOST — cumulative token offsets
  Tensor qlod_host(MemoryLocation::LOCATION_HOST, TYPE_INT32, {static_cast<size_t>(kBatchSize + 1)});
  {
    int32_t* p = qlod_host.GetPtr<int32_t>();
    p[0] = 0;
    p[1] = kSeqLen0;
    p[2] = kSeqLen0 + kSeqLen1;
  }

  TestableFlashAttentionLayer layer;
  layer.SetupForTest(context_, TYPE_FP16, /*use_qk_norm=*/true);

  FlashAttentionLayer::KunlunForwardParams params;
  params.qkv = &qkv_tensor;
  params.rotary_embedding_pos = &rotary_pos;
  params.layers_slot_mapping = &slot_mapping;
  params.k_cache_blocks_base = &k_cache;
  params.v_cache_blocks_base = &v_cache;
  params.seq_len_host = &seq_len_host;
  params.batch_size = kBatchSize;
  params.max_tokens = kTotalTokenNum;
  params.layer_block_num = 0;
  params.atb_attention_attr = &atb_attr;
  params.query_layernorm_weight = &q_norm_w;
  params.key_layernorm_weight = &k_norm_w;
  params.qlod = &qlod_host;

  Status status = layer.Forward(params, &output_tensor);
  StreamSynchronize(context_->device->GetPrimaryComputeStream());

  ASSERT_TRUE(status.OK()) << "FlashAttentionLayer::Forward(qk_norm=true) failed: " << status.ToString();
  EXPECT_EQ(output_tensor.shape[0], static_cast<size_t>(kTotalTokenNum));
  EXPECT_EQ(output_tensor.shape[1], static_cast<size_t>(kOutDim));

  std::vector<uint16_t> out_f16(kTotalTokenNum * kOutDim);
  MemcpyAsync(out_f16.data(), output_tensor.GetPtr<void>(), out_f16.size() * sizeof(uint16_t), MEMCPY_DEVICE_TO_HOST,
              context_->device->GetPrimaryComputeStream());
  StreamSynchronize(context_->device->GetPrimaryComputeStream());

  int nonzero_count = 0;
  for (size_t i = 0; i < out_f16.size(); ++i) {
    float v = HalfToFp32<float16>(out_f16[i]);
    ASSERT_TRUE(std::isfinite(v)) << "NaN/Inf at output[" << i << "]";
    if (std::fabs(v) > 1e-7f) ++nonzero_count;
  }
  EXPECT_GT(nonzero_count, 0) << "All outputs zero — QK-norm FA path likely not running";
  KLLM_LOG_INFO << "FlashAttentionLayer::Forward() FP16 qk_norm=true: tokens=" << kTotalTokenNum
                << " nonzero=" << nonzero_count;
}

// ─────────────────────────────────────────────────────────────────────────────
// PagedAttentionLayer tests (decode path, 1 token per sequence)
// Config: Qwen3-0.6B, batch=2, seq_len=[4,4], block_size=4, 4 blocks in cache
// ─────────────────────────────────────────────────────────────────────────────

static constexpr int kDecodeBatch = 2;
static constexpr int kDecodeSeqLen = 4;     // tokens per sequence after this decode step
static constexpr int kDecodeBlockSize = 4;  // block_token_num
static constexpr int kDecodeNumBlocks = 4;
static constexpr int kDecodeMaxBlocks = 2;  // max blocks per sequence

class TestablePagedAttentionLayer : public PagedAttentionLayer {
 public:
  void SetupForTest(std::shared_ptr<Context> ctx, DataType dtype, bool use_qk_norm = false) {
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
    block_token_num_ = kDecodeBlockSize;
    block_size_ = kDecodeBlockSize * kNumKvHeads * kHeadDim * sizeof(uint16_t);
    num_heads_ = kNumHeads;
    num_kv_heads_ = kNumKvHeads;
    head_size_ = kHeadDim;
    stride_size_ = (kNumHeads + 2 * kNumKvHeads) * kHeadDim;
    tensor_para_size_ = 1;
    max_position_embeddings_ = 32768;
    base_ = 1000000.0f;
    use_qk_norm_ = use_qk_norm;
    layernorm_eps_ = 1e-6f;
    enable_qk_pre_norm_before_rotary_pos_ = false;
    k_scale_ = 1.0f;
    v_scale_ = 1.0f;
    is_causal_ = true;
    no_rope_ = false;
    cos_sin_cache_ptr_ = nullptr;

    mm_quant_mode_ = QUANT_NONE;
    q_lora_rank_ = 0;
    kv_lora_rank_ = 0;
    qk_nope_head_dim_ = 0;
    qk_rope_head_dim_ = 0;
    v_head_dim_ = 0;
    attn_scale_ = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
    attn_temperature_tuning_ = 0;
    floor_scale_ = 0;

    // workspace layout (decode, batch_size tokens):
    //   [q_cont | k_cont | v_cont | context_len_device (or qk_norm fp32)]
    // context_len_device shares the same base offset as the qk_norm scratch area;
    // the two uses are temporally non-overlapping, so we just need the larger.
    int q_dim = kNumHeads * kHeadDim;
    int kv_dim = kNumKvHeads * kHeadDim;
    size_t qkv_ws = static_cast<size_t>(kDecodeBatch) * (q_dim + 2 * kv_dim) * sizeof(uint16_t);
    size_t extra = use_qk_norm ? 2 * static_cast<size_t>(kHeadDim) * sizeof(float)
                               : static_cast<size_t>(kDecodeBatch) * sizeof(int32_t);
    workspace_buffer_ =
        std::make_shared<Tensor>(MemoryLocation::LOCATION_DEVICE, TYPE_INT8, std::vector<size_t>{qkv_ws + extra});
  }
};

class KunlunPagedAttentionTest : public testing::Test {
 protected:
  void SetUp() override {
    context_ = std::make_shared<Context>();
    context_->device = std::make_shared<DeviceContext>(1);
  }
  void TearDown() override {}

  std::shared_ptr<Context> context_{nullptr};
};

// Build the 8 mandatory PagedAttentionLayer input tensors.
// Optionally returns a pre-filled k/v cache (random fp16 data).
// slot_mapping: seq0 → slot (kDecodeSeqLen-1), seq1 → slot (kDecodeBlockSize + kDecodeSeqLen-1)
// block_table:  seq0 → block 0, seq1 → block 1
static void BuildPagedAttnInputs(std::shared_ptr<Context> ctx, std::vector<Tensor>& inputs, Tensor& output_tensor,
                                 bool pre_fill_cache = true) {
  std::mt19937 rng(77);
  std::uniform_real_distribution<float> dist(-0.3f, 0.3f);

  const int kQkvDim = (kNumHeads + 2 * kNumKvHeads) * kHeadDim;
  const int kOutDim = kNumHeads * kHeadDim;

  // input[0]: qkv [batch_size, qkv_dim]
  Tensor qkv(MemoryLocation::LOCATION_DEVICE, TYPE_FP16,
             {static_cast<size_t>(kDecodeBatch), static_cast<size_t>(kQkvDim)});
  {
    std::vector<float> f(kDecodeBatch * kQkvDim);
    for (auto& v : f) v = dist(rng);
    auto h = ToFloat16(f);
    MemcpyAsync(qkv.GetPtr<void>(), h.data(), h.size() * sizeof(uint16_t), MEMCPY_HOST_TO_DEVICE,
                ctx->device->GetMemoryManageStream());
  }

  // input[1]: positions [batch_size], each = kDecodeSeqLen-1 (last position)
  Tensor pos(MemoryLocation::LOCATION_DEVICE, TYPE_INT64, {static_cast<size_t>(kDecodeBatch)});
  {
    std::vector<int64_t> p(kDecodeBatch, static_cast<int64_t>(kDecodeSeqLen - 1));
    MemcpyAsync(pos.GetPtr<void>(), p.data(), p.size() * sizeof(int64_t), MEMCPY_HOST_TO_DEVICE,
                ctx->device->GetMemoryManageStream());
  }

  // input[2]: slot_mapping [layer_num * batch_size]
  // layer_index=0 selects [0 .. batch_size)
  Tensor slot_mapping(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {static_cast<size_t>(kLayerNum * kDecodeBatch)});
  {
    std::vector<int32_t> s(kLayerNum * kDecodeBatch, 0);
    // seq0 → slot (kDecodeSeqLen-1) in block 0; seq1 → slot in block 1
    s[kLayerIndex * kDecodeBatch + 0] = kDecodeSeqLen - 1;
    s[kLayerIndex * kDecodeBatch + 1] = kDecodeBlockSize + kDecodeSeqLen - 1;
    MemcpyAsync(slot_mapping.GetPtr<void>(), s.data(), s.size() * sizeof(int32_t), MEMCPY_HOST_TO_DEVICE,
                ctx->device->GetMemoryManageStream());
  }

  // input[3]: block_table [layer_num * batch_size * max_blocks_per_seq]
  Tensor block_table(MemoryLocation::LOCATION_DEVICE, TYPE_INT32,
                     {static_cast<size_t>(kLayerNum * kDecodeBatch * kDecodeMaxBlocks)});
  {
    std::vector<int32_t> bt(kLayerNum * kDecodeBatch * kDecodeMaxBlocks, -1);
    int base = kLayerIndex * kDecodeBatch * kDecodeMaxBlocks;
    bt[base + 0 * kDecodeMaxBlocks + 0] = 0;  // seq0 → block 0
    bt[base + 1 * kDecodeMaxBlocks + 0] = 1;  // seq1 → block 1
    MemcpyAsync(block_table.GetPtr<void>(), bt.data(), bt.size() * sizeof(int32_t), MEMCPY_HOST_TO_DEVICE,
                ctx->device->GetMemoryManageStream());
  }

  // input[4,5]: k_cache, v_cache [num_blocks, block_token_num, num_kv_heads, head_dim]
  size_t cache_elems = static_cast<size_t>(kDecodeNumBlocks) * kDecodeBlockSize * kNumKvHeads * kHeadDim;
  Tensor k_cache(MemoryLocation::LOCATION_DEVICE, TYPE_FP16,
                 {static_cast<size_t>(kDecodeNumBlocks), static_cast<size_t>(kDecodeBlockSize),
                  static_cast<size_t>(kNumKvHeads), static_cast<size_t>(kHeadDim)});
  Tensor v_cache(MemoryLocation::LOCATION_DEVICE, TYPE_FP16,
                 {static_cast<size_t>(kDecodeNumBlocks), static_cast<size_t>(kDecodeBlockSize),
                  static_cast<size_t>(kNumKvHeads), static_cast<size_t>(kHeadDim)});
  if (pre_fill_cache) {
    std::vector<float> f(cache_elems);
    for (auto& v : f) v = dist(rng);
    auto hk = ToFloat16(f);
    for (auto& v : f) v = dist(rng);
    auto hv = ToFloat16(f);
    MemcpyAsync(k_cache.GetPtr<void>(), hk.data(), hk.size() * sizeof(uint16_t), MEMCPY_HOST_TO_DEVICE,
                ctx->device->GetMemoryManageStream());
    MemcpyAsync(v_cache.GetPtr<void>(), hv.data(), hv.size() * sizeof(uint16_t), MEMCPY_HOST_TO_DEVICE,
                ctx->device->GetMemoryManageStream());
  }

  // input[6]: seq_len_host [batch_size], HOST — total tokens per seq after decode
  Tensor seq_len_host(MemoryLocation::LOCATION_HOST, TYPE_INT32, {static_cast<size_t>(kDecodeBatch)});
  {
    int32_t* p = seq_len_host.GetPtr<int32_t>();
    for (int i = 0; i < kDecodeBatch; ++i) p[i] = kDecodeSeqLen;
  }

  // input[7]: forward_shape (CUDA 13D + Kunlun [13]), HOST
  // Decode test: 2 decode requests, no flash.
  // shape[9] = dp_single_token_request_num = kDecodeBatch (read by PagedAttentionLayer).
  Tensor forward_shape(
      MemoryLocation::LOCATION_HOST, TYPE_INT32,
      {0, 0, 0, static_cast<size_t>(kDecodeBatch), 1, 0, 0, 0, 0, static_cast<size_t>(kDecodeBatch), 1, 0, 0, 0});

  // input[8]: atb_attention_attr [2], HOST: {slot_mapping_dim_1, max_blocks_per_query}
  Tensor atb_attr(MemoryLocation::LOCATION_HOST, TYPE_UINT64, {2});
  {
    uint64_t* p = atb_attr.GetPtr<uint64_t>();
    p[0] = static_cast<uint64_t>(kDecodeBatch);  // slot_mapping_dim_1
    p[1] = static_cast<uint64_t>(kDecodeMaxBlocks);
  }

  StreamSynchronize(ctx->device->GetMemoryManageStream());

  inputs = {qkv, pos, slot_mapping, block_table, k_cache, v_cache, seq_len_host, forward_shape, atb_attr};

  output_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16,
                         {static_cast<size_t>(kDecodeBatch), static_cast<size_t>(kOutDim)});
}

TEST_F(KunlunPagedAttentionTest, ForwardFloat16) {
  std::vector<Tensor> inputs;
  Tensor output_tensor;
  BuildPagedAttnInputs(context_, inputs, output_tensor);

  TestablePagedAttentionLayer layer;
  layer.SetupForTest(context_, TYPE_FP16);

  // PagedAttentionLayer now takes a typed PagedAttentionInputs struct; output stays a vector.
  PagedAttentionInputs pa_inputs;
  pa_inputs.qkv_tensor = &inputs[0];
  pa_inputs.rotary_embedding_pos = &inputs[1];
  pa_inputs.layers_slot_mapping = &inputs[2];
  pa_inputs.layers_block_table = &inputs[3];
  pa_inputs.k_cache_blocks_base = &inputs[4];
  pa_inputs.v_cache_blocks_base = &inputs[5];
  pa_inputs.seq_len_host = &inputs[6];
  pa_inputs.forward_shape = &inputs[7];
  pa_inputs.atb_attention_attr = &inputs[8];

  std::vector<Tensor> outputs = {output_tensor};
  Status status = layer.Forward(pa_inputs, outputs);
  StreamSynchronize(context_->device->GetPrimaryComputeStream());

  ASSERT_TRUE(status.OK()) << "PagedAttentionLayer::Forward() failed: " << status.ToString();
  EXPECT_EQ(outputs[0].shape[0], static_cast<size_t>(kDecodeBatch));
  EXPECT_EQ(outputs[0].shape[1], static_cast<size_t>(kNumHeads * kHeadDim));

  const int kOutDim = kNumHeads * kHeadDim;
  std::vector<uint16_t> out_f16(kDecodeBatch * kOutDim);
  MemcpyAsync(out_f16.data(), outputs[0].GetPtr<void>(), out_f16.size() * sizeof(uint16_t), MEMCPY_DEVICE_TO_HOST,
              context_->device->GetPrimaryComputeStream());
  StreamSynchronize(context_->device->GetPrimaryComputeStream());

  int nonzero = 0;
  for (size_t i = 0; i < out_f16.size(); ++i) {
    float v = HalfToFp32<float16>(out_f16[i]);
    ASSERT_TRUE(std::isfinite(v)) << "NaN/Inf at output[" << i << "]";
    if (std::fabs(v) > 1e-7f) ++nonzero;
  }
  EXPECT_GT(nonzero, 0) << "All outputs zero — paged attention layer likely not running";
  KLLM_LOG_INFO << "PagedAttentionLayer::Forward() FP16: batch=" << kDecodeBatch << " nonzero=" << nonzero;
}

// Qwen3 decode path: PagedAttentionLayer with use_qk_norm=true.
// Adds q_norm_weight (input[9]) and k_norm_weight (input[10]) — fp16 per-head scale tensors.
TEST_F(KunlunPagedAttentionTest, ForwardFloat16WithQkNorm) {
  std::vector<Tensor> inputs;
  Tensor output_tensor;
  BuildPagedAttnInputs(context_, inputs, output_tensor);

  // input[9]: q_norm_weight [head_dim], fp16
  Tensor q_norm_w(MemoryLocation::LOCATION_DEVICE, TYPE_FP16, {static_cast<size_t>(kHeadDim)});
  {
    std::vector<float> fw(kHeadDim);
    for (int i = 0; i < kHeadDim; ++i) fw[i] = 0.5f + static_cast<float>(i) * 0.001f;
    auto h = ToFloat16(fw);
    MemcpyAsync(q_norm_w.GetPtr<void>(), h.data(), h.size() * sizeof(uint16_t), MEMCPY_HOST_TO_DEVICE,
                context_->device->GetMemoryManageStream());
  }

  // input[10]: k_norm_weight [head_dim], fp16
  Tensor k_norm_w(MemoryLocation::LOCATION_DEVICE, TYPE_FP16, {static_cast<size_t>(kHeadDim)});
  {
    std::vector<float> fw(kHeadDim);
    for (int i = 0; i < kHeadDim; ++i) fw[i] = 1.0f - static_cast<float>(i) * 0.001f;
    auto h = ToFloat16(fw);
    MemcpyAsync(k_norm_w.GetPtr<void>(), h.data(), h.size() * sizeof(uint16_t), MEMCPY_HOST_TO_DEVICE,
                context_->device->GetMemoryManageStream());
  }

  StreamSynchronize(context_->device->GetMemoryManageStream());
  inputs.push_back(q_norm_w);
  inputs.push_back(k_norm_w);

  TestablePagedAttentionLayer layer;
  layer.SetupForTest(context_, TYPE_FP16, /*use_qk_norm=*/true);

  PagedAttentionInputs pa_inputs;
  pa_inputs.qkv_tensor = &inputs[0];
  pa_inputs.rotary_embedding_pos = &inputs[1];
  pa_inputs.layers_slot_mapping = &inputs[2];
  pa_inputs.layers_block_table = &inputs[3];
  pa_inputs.k_cache_blocks_base = &inputs[4];
  pa_inputs.v_cache_blocks_base = &inputs[5];
  pa_inputs.seq_len_host = &inputs[6];
  pa_inputs.forward_shape = &inputs[7];
  pa_inputs.atb_attention_attr = &inputs[8];
  pa_inputs.query_layernorm_weight = &inputs[9];
  pa_inputs.key_layernorm_weight = &inputs[10];

  std::vector<Tensor> outputs = {output_tensor};
  Status status = layer.Forward(pa_inputs, outputs);
  StreamSynchronize(context_->device->GetPrimaryComputeStream());

  ASSERT_TRUE(status.OK()) << "PagedAttentionLayer::Forward(qk_norm=true) failed: " << status.ToString();

  const int kOutDim = kNumHeads * kHeadDim;
  std::vector<uint16_t> out_f16(kDecodeBatch * kOutDim);
  MemcpyAsync(out_f16.data(), outputs[0].GetPtr<void>(), out_f16.size() * sizeof(uint16_t), MEMCPY_DEVICE_TO_HOST,
              context_->device->GetPrimaryComputeStream());
  StreamSynchronize(context_->device->GetPrimaryComputeStream());

  int nonzero = 0;
  for (size_t i = 0; i < out_f16.size(); ++i) {
    float v = HalfToFp32<float16>(out_f16[i]);
    ASSERT_TRUE(std::isfinite(v)) << "NaN/Inf at output[" << i << "]";
    if (std::fabs(v) > 1e-7f) ++nonzero;
  }
  EXPECT_GT(nonzero, 0) << "All outputs zero — QK-norm paged attention path likely not running";
  KLLM_LOG_INFO << "PagedAttentionLayer::Forward() FP16 qk_norm=true: batch=" << kDecodeBatch << " nonzero=" << nonzero;
}

// ═════════════════════════════════════════════════════════════════════════════
// cos_sin_cache_ptr construction tests
//
// Exercises AttentionLayer::Init()'s cos_sin_cache / neox_cache build logic
// (attention_layer.cpp lines 143–306) without running the full Forward path.
// We call FlashAttentionLayer::Init (which delegates to AttentionLayer::Init)
// with a hand-crafted parameter vector, then read back the device-side cache
// and verify:
//   • cos/sin values match the expected math (default + YARN)
//   • standalone cache layout [max_pos, rotary_dim] = [cos_half | sin_half]
//   • neox cache layout [2, max_pos, rotary_dim] = [cos_plane | sin_plane] with
//     each frequency duplicated across both halves
//   • DeepSeek YARN uses FP32 and skips neox cache
//   • NO_ROPE / nullptr cos_sin_cache_ptr skip cache construction
// ═════════════════════════════════════════════════════════════════════════════

// ── Testable subclass exposing protected cos_sin_cache members ──────────────
class TestableAttentionLayerForCache : public FlashAttentionLayer {
 public:
  void* GetCosSinCachePtr() { return cos_sin_cache_ptr_; }
  void* GetCosSinCacheNeoxPtr() { return cos_sin_cache_neox_ptr_; }
  bool GetNoRope() { return no_rope_; }
};

// ── Helper: build the std::any parameter vector for AttentionLayer::Init ────
//
// Parameter order must match attention_layer.cpp Init() exactly.
// We use small max_position_embeddings to keep the test fast.
static std::vector<std::any> BuildCacheInitParams(int max_pos, int rotary_dim, float theta, bool use_qk_norm,
                                                  bool is_neox, PositionEncoding pos_enc, void* cos_sin_cache_ptr,
                                                  RoPEScalingFactor rope_scaling, bool enable_qk_pre_norm = false) {
  std::vector<std::any> params;
  params.push_back(std::any(QUANT_NONE));                                // mm_quant_mode_
  params.push_back(std::any(1e-6f));                                     // layernorm_eps_
  params.push_back(std::any(use_qk_norm));                               // use_qk_norm_
  params.push_back(std::any(0));                                         // layer_index_
  params.push_back(std::any(kLayerNum));                                 // layer_num_
  params.push_back(std::any(max_pos));                                   // max_position_embeddings_
  params.push_back(std::any(kNumHeads));                                 // num_heads_
  params.push_back(std::any(kNumKvHeads));                               // num_kv_heads_
  params.push_back(std::any(kHeadDim));                                  // head_size_
  params.push_back(std::any((kNumHeads + 2 * kNumKvHeads) * kHeadDim));  // stride_size_
  params.push_back(std::any(size_t{1}));                                 // tensor_para_size_
  params.push_back(std::any(DataType::TYPE_FP16));                       // kv_cache_dtype_
  params.push_back(std::any(1.0f));                                      // k_scale_
  params.push_back(std::any(1.0f));                                      // v_scale_
  params.push_back(std::any(rotary_dim));                                // rotary_dim (Init extracts as const int)
  params.push_back(std::any(theta));                                     // base_
  params.push_back(std::any(static_cast<uint32_t>(0)));                  // qk_rope_head_dim_
  params.push_back(std::any(static_cast<uint32_t>(0)));                  // qk_nope_head_dim_
  params.push_back(std::any(static_cast<uint32_t>(0)));                  // q_lora_rank_
  params.push_back(std::any(static_cast<uint32_t>(0)));                  // kv_lora_rank_
  params.push_back(std::any(static_cast<uint32_t>(0)));                  // v_head_dim_
  params.push_back(std::any(is_neox));                                   // is_neox
  params.push_back(std::any(pos_enc));                                   // position_encoding
  params.push_back(std::any(cos_sin_cache_ptr));                         // cos_sin_cache_ptr (void*)
  params.push_back(std::any(rope_scaling));                              // rope_scaling_factor_config
  params.push_back(std::any(size_t{1}));                                 // max_batch_size
  params.push_back(std::any(size_t{0}));                                 // attn_temperature_tuning
  params.push_back(std::any(0.0f));                                      // attn_scale_ (overwritten for yarn)
  params.push_back(std::any(size_t{0}));                                 // floor_scale_
  params.push_back(std::any(false));                                     // is_multi_token_forward
  // mrope/xdrope section ptr (consumed but not any_cast by Init)
  params.push_back(std::any(nullptr));
  // enable_qk_pre_norm_before_rotary_pos
  params.push_back(std::any(enable_qk_pre_norm));
  return params;
}

// ── Helper: compute reference cos/sin on host ───────────────────────────────
static void ComputeRefCosSin(int max_pos, int rotary_dim, float theta, std::vector<float>& cos_vals,
                             std::vector<float>& sin_vals, const RoPEScalingFactor& rope_scaling = {}) {
  int half_dim = rotary_dim / 2;
  std::vector<float> freqs(half_dim);
  bool is_yarn = (rope_scaling.type == "yarn");

  if (is_yarn) {
    float scaling_factor = rope_scaling.factor;
    float beta_fast_val = rope_scaling.beta_fast;
    float beta_slow_val = rope_scaling.beta_slow;
    int orig_max_pos = rope_scaling.original_max_position_embeddings;

    float low_dim = (rotary_dim * std::log(static_cast<float>(orig_max_pos) / (beta_fast_val * 2.0f * M_PI))) /
                    (2.0f * std::log(theta));
    float low = std::max(std::floor(low_dim), 0.0f);
    float high_dim = (rotary_dim * std::log(static_cast<float>(orig_max_pos) / (beta_slow_val * 2.0f * M_PI))) /
                     (2.0f * std::log(theta));
    float high = std::min(std::ceil(high_dim), static_cast<float>(rotary_dim) - 1.0f);

    for (int i = 0; i < half_dim; ++i) {
      float pos_freqs = std::pow(theta, static_cast<float>(2 * i) / rotary_dim);
      float inv_freq_extrapolation = 1.0f / pos_freqs;
      float inv_freq_interpolation = 1.0f / (scaling_factor * pos_freqs);
      float ramp_mask = (low == high) ? (static_cast<float>(i) - low) / (high + 0.001f - low)
                                      : (static_cast<float>(i) - low) / (high - low);
      ramp_mask = std::max(0.0f, std::min(1.0f, ramp_mask));
      float inv_freq_mask = (1.0f - ramp_mask) * 1.0f;
      freqs[i] = inv_freq_interpolation * (1.0f - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask;
    }
  } else {
    for (int i = 0; i < half_dim; ++i) {
      freqs[i] = 1.0f / std::pow(theta, static_cast<float>(2 * i) / rotary_dim);
    }
  }

  // mscale
  float cache_mscale = 1.0f;
  if (is_yarn) {
    bool is_deepseek_yarn = rope_scaling.use_deepseek_rope;
    if (is_deepseek_yarn) {
      auto yarn_get_mscale = [](float scale, float m) -> float {
        return (scale <= 1.0f) ? 1.0f : 0.1f * m * std::log(scale) + 1.0f;
      };
      cache_mscale = yarn_get_mscale(rope_scaling.factor, rope_scaling.mscale) /
                     yarn_get_mscale(rope_scaling.factor, rope_scaling.mscale_all_dim);
    } else {
      cache_mscale = (rope_scaling.factor <= 1.0f) ? 1.0f : 0.1f * std::log(rope_scaling.factor) + 1.0f;
    }
  }

  cos_vals.resize(static_cast<size_t>(max_pos) * half_dim);
  sin_vals.resize(static_cast<size_t>(max_pos) * half_dim);
  for (int pos = 0; pos < max_pos; ++pos) {
    for (int i = 0; i < half_dim; ++i) {
      float angle = pos * freqs[i];
      cos_vals[pos * half_dim + i] = std::cos(angle) * cache_mscale;
      sin_vals[pos * half_dim + i] = std::sin(angle) * cache_mscale;
    }
  }
}

// ── Test fixture for cos_sin_cache tests ─────────────────────────────────────
class KunlunCosSinCacheTest : public testing::Test {
 protected:
  void SetUp() override {
    context_ = std::make_shared<Context>();
    context_->device = std::make_shared<DeviceContext>(1);
    // BaseLayer::Init accesses context_->global->GetTensorParallelSize() and
    // runtime_config.parallel_basic_config; both must be initialized to avoid segfault.
    context_->global = std::make_shared<GlobalContext>(/*tp=*/1, /*attn_dp=*/1, /*max_batch=*/1, /*env=*/nullptr);
    ReleaseNeoxCosSinCache();  // ensure clean state between tests
  }
  void TearDown() override { ReleaseNeoxCosSinCache(); }

  std::shared_ptr<Context> context_{nullptr};
};

// ── Test: default RoPE fills standalone cos_sin_cache (FP16) ─────────────────
TEST_F(KunlunCosSinCacheTest, DefaultRopeFillsStandaloneCacheFp16) {
  constexpr int kMaxPos = 64;
  constexpr int kRotaryDim = 128;
  constexpr float kTheta = 10000.0f;

  // Allocate device buffer for cos_sin_cache: [max_pos, rotary_dim] * fp16
  size_t cache_bytes = static_cast<size_t>(kMaxPos) * kRotaryDim * sizeof(uint16_t);
  void* cos_sin_cache_dev = nullptr;
  ASSERT_EQ(xpu_malloc(&cos_sin_cache_dev, cache_bytes), 0) << "Failed to allocate cos_sin_cache device buffer";
  // Zero-fill to detect unfilled entries
  std::vector<uint16_t> zeros(kMaxPos * kRotaryDim, 0);
  MemcpyAsync(cos_sin_cache_dev, zeros.data(), cache_bytes, MEMCPY_HOST_TO_DEVICE,
              context_->device->GetMemoryManageStream());
  StreamSynchronize(context_->device->GetMemoryManageStream());

  RoPEScalingFactor rope_scaling;  // default type
  auto params = BuildCacheInitParams(kMaxPos, kRotaryDim, kTheta, /*use_qk_norm=*/false, /*is_neox=*/true,
                                     PositionEncoding::ROPE, cos_sin_cache_dev, rope_scaling);

  RuntimeConfig runtime_config;
  runtime_config.attn_backend_config.block_size = kBlockTokenNum * kNumKvHeads * kHeadDim * sizeof(uint16_t);
  runtime_config.attn_backend_config.block_token_num = kBlockTokenNum;
  runtime_config.inter_data_type = DataType::TYPE_FP16;

  TestableAttentionLayerForCache layer;
  Status status = layer.Init(params, runtime_config, context_, /*rank=*/0);
  ASSERT_TRUE(status.OK()) << "Init failed: " << status.ToString();

  // cos_sin_cache_ptr_ should be set (not fused path)
  EXPECT_EQ(layer.GetCosSinCachePtr(), cos_sin_cache_dev);
  // neox cache should NOT be built (no qk_norm)
  EXPECT_EQ(layer.GetCosSinCacheNeoxPtr(), nullptr);

  // Read back and verify values
  std::vector<uint16_t> cache_host(kMaxPos * kRotaryDim);
  MemcpyAsync(cache_host.data(), cos_sin_cache_dev, cache_bytes, MEMCPY_DEVICE_TO_HOST,
              context_->device->GetMemoryManageStream());
  StreamSynchronize(context_->device->GetMemoryManageStream());

  // Reference
  std::vector<float> ref_cos, ref_sin;
  ComputeRefCosSin(kMaxPos, kRotaryDim, kTheta, ref_cos, ref_sin);

  int half_dim = kRotaryDim / 2;
  for (int pos = 0; pos < kMaxPos; ++pos) {
    for (int i = 0; i < half_dim; ++i) {
      float expected_cos = ref_cos[pos * half_dim + i];
      float expected_sin = ref_sin[pos * half_dim + i];
      float actual_cos = HalfToFp32<float16>(cache_host[pos * kRotaryDim + i]);
      float actual_sin = HalfToFp32<float16>(cache_host[pos * kRotaryDim + half_dim + i]);
      EXPECT_NEAR(actual_cos, expected_cos, 0.02f) << "cos mismatch at pos=" << pos << " i=" << i;
      EXPECT_NEAR(actual_sin, expected_sin, 0.02f) << "sin mismatch at pos=" << pos << " i=" << i;
    }
  }

  xpu_free(cos_sin_cache_dev);
}

// ── Test: YARN scaling fills cos_sin_cache with interpolated frequencies ─────
TEST_F(KunlunCosSinCacheTest, YarnScalingFillsStandaloneCache) {
  constexpr int kMaxPos = 64;
  constexpr int kRotaryDim = 128;
  constexpr float kTheta = 10000.0f;

  size_t cache_bytes = static_cast<size_t>(kMaxPos) * kRotaryDim * sizeof(uint16_t);
  void* cos_sin_cache_dev = nullptr;
  ASSERT_EQ(xpu_malloc(&cos_sin_cache_dev, cache_bytes), 0);
  std::vector<uint16_t> zeros(kMaxPos * kRotaryDim, 0);
  MemcpyAsync(cos_sin_cache_dev, zeros.data(), cache_bytes, MEMCPY_HOST_TO_DEVICE,
              context_->device->GetMemoryManageStream());
  StreamSynchronize(context_->device->GetMemoryManageStream());

  RoPEScalingFactor rope_scaling;
  rope_scaling.type = "yarn";
  rope_scaling.factor = 4.0f;
  rope_scaling.beta_fast = 32.0f;
  rope_scaling.beta_slow = 1.0f;
  rope_scaling.original_max_position_embeddings = 4096;
  rope_scaling.use_deepseek_rope = false;

  auto params = BuildCacheInitParams(kMaxPos, kRotaryDim, kTheta, /*use_qk_norm=*/false, /*is_neox=*/true,
                                     PositionEncoding::ROPE, cos_sin_cache_dev, rope_scaling);

  RuntimeConfig runtime_config;
  runtime_config.attn_backend_config.block_size = kBlockTokenNum * kNumKvHeads * kHeadDim * sizeof(uint16_t);
  runtime_config.attn_backend_config.block_token_num = kBlockTokenNum;
  runtime_config.inter_data_type = DataType::TYPE_FP16;

  TestableAttentionLayerForCache layer;
  Status status = layer.Init(params, runtime_config, context_, 0);
  ASSERT_TRUE(status.OK()) << "Init failed: " << status.ToString();

  EXPECT_EQ(layer.GetCosSinCachePtr(), cos_sin_cache_dev);
  EXPECT_EQ(layer.GetCosSinCacheNeoxPtr(), nullptr);

  // Read back
  std::vector<uint16_t> cache_host(kMaxPos * kRotaryDim);
  MemcpyAsync(cache_host.data(), cos_sin_cache_dev, cache_bytes, MEMCPY_DEVICE_TO_HOST,
              context_->device->GetMemoryManageStream());
  StreamSynchronize(context_->device->GetMemoryManageStream());

  // Reference with YARN
  std::vector<float> ref_cos, ref_sin;
  ComputeRefCosSin(kMaxPos, kRotaryDim, kTheta, ref_cos, ref_sin, rope_scaling);

  int half_dim = kRotaryDim / 2;
  // With YARN factor=4, the mscale should be > 1
  float expected_mscale = 0.1f * std::log(rope_scaling.factor) + 1.0f;
  EXPECT_GT(expected_mscale, 1.0f);

  for (int pos = 0; pos < kMaxPos; ++pos) {
    for (int i = 0; i < half_dim; ++i) {
      float expected_cos = ref_cos[pos * half_dim + i];
      float expected_sin = ref_sin[pos * half_dim + i];
      float actual_cos = HalfToFp32<float16>(cache_host[pos * kRotaryDim + i]);
      float actual_sin = HalfToFp32<float16>(cache_host[pos * kRotaryDim + half_dim + i]);
      EXPECT_NEAR(actual_cos, expected_cos, 0.02f) << "YARN cos mismatch at pos=" << pos << " i=" << i;
      EXPECT_NEAR(actual_sin, expected_sin, 0.02f) << "YARN sin mismatch at pos=" << pos << " i=" << i;
    }
  }

  xpu_free(cos_sin_cache_dev);
}

// ── Test: QK-norm builds neox-format cos_sin_cache with duplicated freqs ─────
TEST_F(KunlunCosSinCacheTest, QkNormBuildsNeoxCache) {
  constexpr int kMaxPos = 64;
  constexpr int kRotaryDim = 128;
  constexpr float kTheta = 1000000.0f;

  // standalone cache buffer (will be nulled by Init since fused path is active)
  size_t standalone_bytes = static_cast<size_t>(kMaxPos) * kRotaryDim * sizeof(uint16_t);
  void* cos_sin_cache_dev = nullptr;
  ASSERT_EQ(xpu_malloc(&cos_sin_cache_dev, standalone_bytes), 0);
  std::vector<uint16_t> zeros(kMaxPos * kRotaryDim, 0);
  MemcpyAsync(cos_sin_cache_dev, zeros.data(), standalone_bytes, MEMCPY_HOST_TO_DEVICE,
              context_->device->GetMemoryManageStream());
  StreamSynchronize(context_->device->GetMemoryManageStream());

  RoPEScalingFactor rope_scaling;  // default
  auto params = BuildCacheInitParams(kMaxPos, kRotaryDim, kTheta, /*use_qk_norm=*/true, /*is_neox=*/true,
                                     PositionEncoding::ROPE, cos_sin_cache_dev, rope_scaling);

  RuntimeConfig runtime_config;
  runtime_config.attn_backend_config.block_size = kBlockTokenNum * kNumKvHeads * kHeadDim * sizeof(uint16_t);
  runtime_config.attn_backend_config.block_token_num = kBlockTokenNum;
  runtime_config.inter_data_type = DataType::TYPE_FP16;

  TestableAttentionLayerForCache layer;
  Status status = layer.Init(params, runtime_config, context_, 0);
  ASSERT_TRUE(status.OK()) << "Init failed: " << status.ToString();

  // With QK-norm + default rope, fused path is active:
  //   cos_sin_cache_ptr_ should be nullptr (standalone not used)
  //   cos_sin_cache_neox_ptr_ should be allocated
  EXPECT_EQ(layer.GetCosSinCachePtr(), nullptr);
  void* neox_ptr = layer.GetCosSinCacheNeoxPtr();
  ASSERT_NE(neox_ptr, nullptr) << "neox cache should be allocated when use_qk_norm=true";

  // Read back neox cache: layout [2, max_pos, rotary_dim] = [cos_plane | sin_plane]
  size_t neox_bytes = static_cast<size_t>(2) * kMaxPos * kRotaryDim * sizeof(uint16_t);
  std::vector<uint16_t> neox_host(2 * kMaxPos * kRotaryDim);
  MemcpyAsync(neox_host.data(), neox_ptr, neox_bytes, MEMCPY_DEVICE_TO_HOST, context_->device->GetMemoryManageStream());
  StreamSynchronize(context_->device->GetMemoryManageStream());

  // Reference
  std::vector<float> ref_cos, ref_sin;
  ComputeRefCosSin(kMaxPos, kRotaryDim, kTheta, ref_cos, ref_sin);

  int half_dim = kRotaryDim / 2;
  uint16_t* cos_plane = neox_host.data();
  uint16_t* sin_plane = neox_host.data() + kMaxPos * kRotaryDim;

  for (int pos = 0; pos < kMaxPos; ++pos) {
    for (int i = 0; i < half_dim; ++i) {
      float expected_cos = ref_cos[pos * half_dim + i];
      float expected_sin = ref_sin[pos * half_dim + i];

      // cos_plane: freq i duplicated to both [i] and [half_dim + i]
      float cos_first = HalfToFp32<float16>(cos_plane[pos * kRotaryDim + i]);
      float cos_second = HalfToFp32<float16>(cos_plane[pos * kRotaryDim + half_dim + i]);
      EXPECT_NEAR(cos_first, expected_cos, 0.02f) << "neox cos[i] mismatch at pos=" << pos << " i=" << i;
      EXPECT_NEAR(cos_second, expected_cos, 0.02f) << "neox cos[half+i] mismatch at pos=" << pos << " i=" << i;

      // sin_plane: freq i duplicated to both [i] and [half_dim + i]
      float sin_first = HalfToFp32<float16>(sin_plane[pos * kRotaryDim + i]);
      float sin_second = HalfToFp32<float16>(sin_plane[pos * kRotaryDim + half_dim + i]);
      EXPECT_NEAR(sin_first, expected_sin, 0.02f) << "neox sin[i] mismatch at pos=" << pos << " i=" << i;
      EXPECT_NEAR(sin_second, expected_sin, 0.02f) << "neox sin[half+i] mismatch at pos=" << pos << " i=" << i;
    }
  }

  xpu_free(cos_sin_cache_dev);
}

// ── Test: DeepSeek YARN uses FP32 cache and skips neox cache ─────────────────
TEST_F(KunlunCosSinCacheTest, DeepSeekYarnUsesFp32AndSkipsNeox) {
  constexpr int kMaxPos = 64;
  constexpr int kRotaryDim = 64;  // deepseek uses qk_rope_head_dim as rotary_dim
  constexpr float kTheta = 10000.0f;

  // FP32 cache buffer
  size_t cache_bytes = static_cast<size_t>(kMaxPos) * kRotaryDim * sizeof(float);
  void* cos_sin_cache_dev = nullptr;
  ASSERT_EQ(xpu_malloc(&cos_sin_cache_dev, cache_bytes), 0);
  std::vector<float> zeros(kMaxPos * kRotaryDim, 0.0f);
  MemcpyAsync(cos_sin_cache_dev, zeros.data(), cache_bytes, MEMCPY_HOST_TO_DEVICE,
              context_->device->GetMemoryManageStream());
  StreamSynchronize(context_->device->GetMemoryManageStream());

  RoPEScalingFactor rope_scaling;
  rope_scaling.type = "yarn";
  rope_scaling.factor = 40.0f;
  rope_scaling.beta_fast = 32.0f;
  rope_scaling.beta_slow = 1.0f;
  rope_scaling.original_max_position_embeddings = 4096;
  rope_scaling.use_deepseek_rope = true;
  rope_scaling.mscale = 0.707f;
  rope_scaling.mscale_all_dim = 0.707f;

  // DeepSeek YARN overrides rotary_dim to qk_rope_head_dim_.
  // We need qk_rope_head_dim_ == kRotaryDim and qk_nope_head_dim_ > 0 for attn_scale.
  auto params = BuildCacheInitParams(kMaxPos, kRotaryDim, kTheta, /*use_qk_norm=*/false, /*is_neox=*/true,
                                     PositionEncoding::ROPE, cos_sin_cache_dev, rope_scaling);
  // Override qk_rope_head_dim_ and qk_nope_head_dim_ for DeepSeek
  params[16] = std::any(static_cast<uint32_t>(kRotaryDim));  // qk_rope_head_dim_
  params[17] = std::any(static_cast<uint32_t>(128));         // qk_nope_head_dim_

  RuntimeConfig runtime_config;
  runtime_config.attn_backend_config.block_size = kBlockTokenNum * kNumKvHeads * kHeadDim * sizeof(uint16_t);
  runtime_config.attn_backend_config.block_token_num = kBlockTokenNum;
  runtime_config.inter_data_type = DataType::TYPE_FP16;

  TestableAttentionLayerForCache layer;
  Status status = layer.Init(params, runtime_config, context_, 0);
  ASSERT_TRUE(status.OK()) << "Init failed: " << status.ToString();

  // DeepSeek YARN: standalone cache IS used (not fused), neox cache is SKIPPED
  EXPECT_EQ(layer.GetCosSinCachePtr(), cos_sin_cache_dev);
  EXPECT_EQ(layer.GetCosSinCacheNeoxPtr(), nullptr)
      << "neox cache should be skipped for DeepSeek YARN (fp32 incompatible with fp16/bf16 kernel)";

  // Read back FP32 cache
  std::vector<float> cache_host(kMaxPos * kRotaryDim);
  MemcpyAsync(cache_host.data(), cos_sin_cache_dev, cache_bytes, MEMCPY_DEVICE_TO_HOST,
              context_->device->GetMemoryManageStream());
  StreamSynchronize(context_->device->GetMemoryManageStream());

  // Reference with DeepSeek YARN
  std::vector<float> ref_cos, ref_sin;
  ComputeRefCosSin(kMaxPos, kRotaryDim, kTheta, ref_cos, ref_sin, rope_scaling);

  int half_dim = kRotaryDim / 2;
  for (int pos = 0; pos < kMaxPos; ++pos) {
    for (int i = 0; i < half_dim; ++i) {
      float expected_cos = ref_cos[pos * half_dim + i];
      float expected_sin = ref_sin[pos * half_dim + i];
      float actual_cos = cache_host[pos * kRotaryDim + i];
      float actual_sin = cache_host[pos * kRotaryDim + half_dim + i];
      // FP32 should be very precise
      EXPECT_NEAR(actual_cos, expected_cos, 1e-4f) << "DeepSeek YARN FP32 cos mismatch at pos=" << pos << " i=" << i;
      EXPECT_NEAR(actual_sin, expected_sin, 1e-4f) << "DeepSeek YARN FP32 sin mismatch at pos=" << pos << " i=" << i;
    }
  }

  xpu_free(cos_sin_cache_dev);
}

// ── Test: NO_ROPE skips cache construction ───────────────────────────────────
TEST_F(KunlunCosSinCacheTest, NoRopeSkipsCacheConstruction) {
  constexpr int kMaxPos = 32;
  constexpr int kRotaryDim = 128;
  constexpr float kTheta = 10000.0f;

  // Allocate a dummy buffer — it should remain untouched
  size_t cache_bytes = static_cast<size_t>(kMaxPos) * kRotaryDim * sizeof(uint16_t);
  void* cos_sin_cache_dev = nullptr;
  ASSERT_EQ(xpu_malloc(&cos_sin_cache_dev, cache_bytes), 0);
  // Fill with a sentinel pattern
  std::vector<uint16_t> sentinel(kMaxPos * kRotaryDim, 0xABCD);
  MemcpyAsync(cos_sin_cache_dev, sentinel.data(), cache_bytes, MEMCPY_HOST_TO_DEVICE,
              context_->device->GetMemoryManageStream());
  StreamSynchronize(context_->device->GetMemoryManageStream());

  RoPEScalingFactor rope_scaling;
  auto params = BuildCacheInitParams(kMaxPos, kRotaryDim, kTheta, /*use_qk_norm=*/false, /*is_neox=*/true,
                                     PositionEncoding::NO_ROPE, cos_sin_cache_dev, rope_scaling);

  RuntimeConfig runtime_config;
  runtime_config.attn_backend_config.block_size = kBlockTokenNum * kNumKvHeads * kHeadDim * sizeof(uint16_t);
  runtime_config.attn_backend_config.block_token_num = kBlockTokenNum;
  runtime_config.inter_data_type = DataType::TYPE_FP16;

  TestableAttentionLayerForCache layer;
  Status status = layer.Init(params, runtime_config, context_, 0);
  ASSERT_TRUE(status.OK()) << "Init failed: " << status.ToString();

  EXPECT_TRUE(layer.GetNoRope());
  // cos_sin_cache_ptr_ should still be set (not fused path), but cache content should NOT be overwritten
  EXPECT_EQ(layer.GetCosSinCachePtr(), cos_sin_cache_dev);
  EXPECT_EQ(layer.GetCosSinCacheNeoxPtr(), nullptr);

  // Read back and verify sentinel is intact (cache was NOT filled)
  std::vector<uint16_t> cache_host(kMaxPos * kRotaryDim);
  MemcpyAsync(cache_host.data(), cos_sin_cache_dev, cache_bytes, MEMCPY_DEVICE_TO_HOST,
              context_->device->GetMemoryManageStream());
  StreamSynchronize(context_->device->GetMemoryManageStream());

  for (size_t i = 0; i < cache_host.size(); ++i) {
    EXPECT_EQ(cache_host[i], 0xABCD) << "cache was overwritten at index " << i << " despite NO_ROPE";
  }

  xpu_free(cos_sin_cache_dev);
}

// ── Test: nullptr cos_sin_cache_ptr skips all cache construction ─────────────
TEST_F(KunlunCosSinCacheTest, NullptrCosSinCachePtrSkipsConstruction) {
  constexpr int kMaxPos = 32;
  constexpr int kRotaryDim = 128;
  constexpr float kTheta = 10000.0f;

  RoPEScalingFactor rope_scaling;
  auto params = BuildCacheInitParams(kMaxPos, kRotaryDim, kTheta, /*use_qk_norm=*/false, /*is_neox=*/true,
                                     PositionEncoding::ROPE, /*cos_sin_cache_ptr=*/nullptr, rope_scaling);

  RuntimeConfig runtime_config;
  runtime_config.attn_backend_config.block_size = kBlockTokenNum * kNumKvHeads * kHeadDim * sizeof(uint16_t);
  runtime_config.attn_backend_config.block_token_num = kBlockTokenNum;
  runtime_config.inter_data_type = DataType::TYPE_FP16;

  TestableAttentionLayerForCache layer;
  Status status = layer.Init(params, runtime_config, context_, 0);
  ASSERT_TRUE(status.OK()) << "Init failed: " << status.ToString();

  EXPECT_EQ(layer.GetCosSinCachePtr(), nullptr);
  EXPECT_EQ(layer.GetCosSinCacheNeoxPtr(), nullptr);
}

// ── Test: neox cache is shared across layers (g_neox_cache_map dedup) ───────
TEST_F(KunlunCosSinCacheTest, NeoxCacheSharedAcrossLayers) {
  constexpr int kMaxPos = 32;
  constexpr int kRotaryDim = 128;
  constexpr float kTheta = 1000000.0f;

  size_t standalone_bytes = static_cast<size_t>(kMaxPos) * kRotaryDim * sizeof(uint16_t);
  void* cos_sin_cache_dev = nullptr;
  ASSERT_EQ(xpu_malloc(&cos_sin_cache_dev, standalone_bytes), 0);

  RoPEScalingFactor rope_scaling;
  auto params1 = BuildCacheInitParams(kMaxPos, kRotaryDim, kTheta, /*use_qk_norm=*/true, /*is_neox=*/true,
                                      PositionEncoding::ROPE, cos_sin_cache_dev, rope_scaling);

  RuntimeConfig runtime_config;
  runtime_config.attn_backend_config.block_size = kBlockTokenNum * kNumKvHeads * kHeadDim * sizeof(uint16_t);
  runtime_config.attn_backend_config.block_token_num = kBlockTokenNum;
  runtime_config.inter_data_type = DataType::TYPE_FP16;

  // First layer: should allocate neox cache
  TestableAttentionLayerForCache layer1;
  Status s1 = layer1.Init(params1, runtime_config, context_, 0);
  ASSERT_TRUE(s1.OK()) << "layer1 Init failed: " << s1.ToString();
  void* neox_ptr1 = layer1.GetCosSinCacheNeoxPtr();
  ASSERT_NE(neox_ptr1, nullptr);

  // Second layer with same cos_sin_cache_ptr: should reuse the same neox cache
  auto params2 = BuildCacheInitParams(kMaxPos, kRotaryDim, kTheta, /*use_qk_norm=*/true, /*is_neox=*/true,
                                      PositionEncoding::ROPE, cos_sin_cache_dev, rope_scaling);
  TestableAttentionLayerForCache layer2;
  Status s2 = layer2.Init(params2, runtime_config, context_, 0);
  ASSERT_TRUE(s2.OK()) << "layer2 Init failed: " << s2.ToString();
  void* neox_ptr2 = layer2.GetCosSinCacheNeoxPtr();

  EXPECT_EQ(neox_ptr1, neox_ptr2)
      << "neox cache should be shared (deduplicated) across layers with same cos_sin_cache_ptr";

  xpu_free(cos_sin_cache_dev);
}

}  // namespace ksana_llm
