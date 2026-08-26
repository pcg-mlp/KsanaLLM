/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "models/kimi_k3/attention_meta/kimi_k3_attention_meta_builder.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "cache_manager/cache_layout/base_cache_layout.h"
#include "configure/context.h"
#include "configure/environment.h"
#include "configure/global_context.h"
#include "device/device_utils.h"
#include "device/nvidia/nccl_utils.h"
#include "models/deepseek_v4/attention_meta/deepseek_v4_compressor_slot_resource.h"
#include "models/kimi_k3/attention_meta/kimi_k3_attention_meta.h"
#include "test.h"
#include "utils/singleton.h"

namespace ksana_llm {

// Call the real private methods through a test subclass and verify host
// vectors + device tensor shapes. kKdaChunkSize = 64.
class TestKimiK3FlashBuilder : public KimiK3FlashAttentionMetaBuilder {
 public:
  using KimiK3FlashAttentionMetaBuilder::KimiK3FlashAttentionMetaBuilder;
  using KimiK3FlashAttentionMetaBuilder::PrepareChunkKDA;
  using KimiK3FlashAttentionMetaBuilder::PrepareConv1D;
  using KimiK3FlashAttentionMetaBuilder::PrepareStateIndices;
};

class TestKimiK3PagedBuilder : public KimiK3PagedAttentionMetaBuilder {
 public:
  using KimiK3PagedAttentionMetaBuilder::KimiK3PagedAttentionMetaBuilder;
  using KimiK3PagedAttentionMetaBuilder::PrepareStateIndices;
};

namespace {

// Minimal fakes of the cache-layout interfaces so
// KimiK3SharedAttentionMeta::Init can be exercised without standing up the full
// cache_manager. GetKvCacheLayerOffsets returns per-layer conv (k==v) and
// recurrent (k==v) offsets by default; flip the mismatch flags to drive the
// k!=v throw branches in Init.
class FakeBlockCacheLayout : public BaseBlockCacheLayout {
 public:
  size_t GetBlockCacheSize() override { return block_cache_size; }

  KvCacheLayerOffsets GetKvCacheLayerOffsets(int32_t layer_index, const std::string& kv_cache_type) override {
    if (kv_cache_type == "conv") {
      const size_t base = 1000u + static_cast<size_t>(layer_index);
      return {base, conv_mismatch ? base + 1 : base};
    }
    const size_t base = 2000u + static_cast<size_t>(layer_index);
    return {base, recurrent_mismatch ? base + 1 : base};
  }

  size_t block_cache_size = 16;
  bool conv_mismatch = false;
  bool recurrent_mismatch = false;
};

class FakeModelCacheLayout : public BaseModelCacheLayout {
 public:
  explicit FakeModelCacheLayout(std::shared_ptr<BaseBlockCacheLayout> block) {
    RegisterBlockCacheLayout("mamba-state", std::move(block));
  }
};

}  // namespace

class KimiK3AttentionMetaBuilderTest : public testing::Test {
 protected:
  void SetUp() override {
    InitLoguru();
    const auto& env = Singleton<Environment>::GetInstance();
    context_ = std::make_shared<Context>();
    context_->global.reset(new GlobalContext(1, 1, 1, env));
    context_->global->InitDistributedConfig();
    ncclUniqueId nccl_uid = GenerateNCCLUniqueID();
    context_->device = std::make_shared<DeviceContext>(1);
    context_->device->ext->InitNcclParam(nccl_uid);
    EventCreateWithFlags(&kvcache_offset_event_, EVENT_DISABLE_TIMING);
    EventCreateWithFlags(&rotary_embedding_event_, EVENT_DISABLE_TIMING);
    runtime_config_.max_batch_size = 8;
  }

  AttentionBuilderConfig MakeConfig() {
    return AttentionBuilderConfig{model_config_,
                                  runtime_config_,
                                  /*enable_blocked_multi_token_forwarding_kv=*/false,
                                  /*use_flashinfer_for_decode=*/false,
                                  /*rank=*/0,
                                  /*layer_num_on_node=*/0,
                                  /*block_size=*/0,
                                  kv_cache_types_,
                                  kv_cache_layer_offsets_,
                                  /*attn_dp_group_id=*/0,
                                  /*attn_dp_rank_id=*/0,
                                  context_,
                                  kvcache_offset_event_,
                                  rotary_embedding_event_};
  }

  // Build a ForwardRequest with the given extend/prefix lengths and mamba block id.
  // GetInputIdsLength() == extend_len.
  static ForwardRequest MakeReq(int extend_len, int prefix_len, int block_id, int64_t req_id = 0) {
    ForwardRequest req;
    req.req_id = req_id;
    req.forwarding_tokens = std::make_shared<std::vector<int>>(extend_len + prefix_len, 0);
    req.kv_cached_token_num = prefix_len;
    req.prefix_cache_len = prefix_len;
    req.mamba_kv_cache_block_id = block_id;
    return req;
  }

  void TearDown() override {
    EventDestroy(kvcache_offset_event_);
    EventDestroy(rotary_embedding_event_);
  }

  ModelConfig model_config_;
  RuntimeConfig runtime_config_;
  std::shared_ptr<Context> context_;
  std::vector<std::string> kv_cache_types_;
  std::unordered_map<std::string, std::vector<KvCacheLayerOffsets>> kv_cache_layer_offsets_;
  Event kvcache_offset_event_;
  Event rotary_embedding_event_;
  std::vector<ForwardRequest> reqs_;
};

// ── Flash PrepareStateIndices ───────────────────────────────────────────────

TEST_F(KimiK3AttentionMetaBuilderTest, Flash_PrepareStateIndices_Basic) {
  reqs_ = {MakeReq(5, 0, 3, 0), MakeReq(3, 8, 7, 1), MakeReq(10, 2, 11, 2)};

  CompressorSlotResource slot_resource;
  slot_resource.Init(runtime_config_);
  slot_resource.Alloc(0);
  slot_resource.Alloc(1);
  slot_resource.Alloc(2);

  KimiK3SharedAttentionMeta kimi_shared;
  kimi_shared.state_indices_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {8});
  kimi_shared.state_indices_tensor.shape[0] = 0;  // Reset cursor (normally done by BuildAttentionMetas).
  kimi_shared.mamba_state_buffer = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_UINT8, {1, 1, 1});
  kimi_shared.slot_resource_ = &slot_resource;

  KimiK3FlashAttentionMeta meta;
  for (auto& r : reqs_) {
    meta.dp_reqs.push_back(&r);
  }

  auto config = MakeConfig();
  TestKimiK3FlashBuilder builder(config);
  builder.PrepareStateIndices(meta, kimi_shared);
  StreamSynchronize(context_->device->GetH2DStream());

  ASSERT_EQ(meta.state_indices_host.size(), 3u);
  EXPECT_EQ(meta.state_indices_host[0], slot_resource.Get(0) * 1);
  EXPECT_EQ(meta.state_indices_host[1], slot_resource.Get(1) * 1);
  EXPECT_EQ(meta.state_indices_host[2], slot_resource.Get(2) * 1);
  EXPECT_EQ(meta.state_indices_tensor.shape, (std::vector<size_t>{3}));
}

TEST_F(KimiK3AttentionMetaBuilderTest, Flash_PrepareStateIndices_NullSlotResourceThrows) {
  reqs_ = {MakeReq(5, 0, -1, 0)};

  KimiK3SharedAttentionMeta kimi_shared;
  kimi_shared.state_indices_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {4});
  kimi_shared.state_indices_tensor.shape[0] = 0;  // Reset cursor (normally done by BuildAttentionMetas).
  // slot_resource_ left null.

  KimiK3FlashAttentionMeta meta;
  meta.dp_reqs.push_back(&reqs_[0]);

  auto config = MakeConfig();
  TestKimiK3FlashBuilder builder(config);
  EXPECT_THROW(builder.PrepareStateIndices(meta, kimi_shared), std::runtime_error);
}

// state_indices_host[i] = slot_id * decode_token_num (shape[1]). Use shape[1]=4
// so the multiplier is actually exercised (Basic uses 1).
TEST_F(KimiK3AttentionMetaBuilderTest, Flash_PrepareStateIndices_DecodeTokenNumMultiplier) {
  reqs_ = {MakeReq(5, 0, 3, 0), MakeReq(3, 8, 7, 1)};

  CompressorSlotResource slot_resource;
  slot_resource.Init(runtime_config_);
  slot_resource.Alloc(0);
  slot_resource.Alloc(1);

  constexpr int32_t kDecodeTokenNum = 4;
  KimiK3SharedAttentionMeta kimi_shared;
  kimi_shared.state_indices_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {8});
  kimi_shared.state_indices_tensor.shape[0] = 0;  // Reset cursor (normally done by BuildAttentionMetas).
  kimi_shared.mamba_state_buffer = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_UINT8, {8, kDecodeTokenNum, 16});
  kimi_shared.slot_resource_ = &slot_resource;

  KimiK3FlashAttentionMeta meta;
  for (auto& r : reqs_) {
    meta.dp_reqs.push_back(&r);
  }

  auto config = MakeConfig();
  TestKimiK3FlashBuilder builder(config);
  builder.PrepareStateIndices(meta, kimi_shared);
  StreamSynchronize(context_->device->GetH2DStream());

  ASSERT_EQ(meta.state_indices_host.size(), 2u);
  EXPECT_EQ(meta.state_indices_host[0], slot_resource.Get(0) * kDecodeTokenNum);
  EXPECT_EQ(meta.state_indices_host[1], slot_resource.Get(1) * kDecodeTokenNum);
  EXPECT_EQ(meta.state_indices_tensor.shape, (std::vector<size_t>{2}));
}

// ── Flash PrepareConv1D ──────────────────────────────────────────────────────

TEST_F(KimiK3AttentionMetaBuilderTest, Flash_PrepareConv1D_Mixed) {
  // req0: extend=5, prefix=0 → 0; req1: extend=3, prefix=8 → 1; req2: extend=10, prefix=2 → 1.
  reqs_ = {MakeReq(5, 0, 0, 0), MakeReq(3, 8, 0, 1), MakeReq(10, 2, 0, 2)};

  KimiK3FlashAttentionMeta meta;
  meta.has_initial_state_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_BOOL, {8});
  for (auto& r : reqs_) {
    meta.dp_reqs.push_back(&r);
  }

  auto config = MakeConfig();
  TestKimiK3FlashBuilder builder(config);
  builder.PrepareConv1D(meta);
  StreamSynchronize(context_->device->GetH2DStream());

  EXPECT_EQ(meta.max_extend_len_host, 10);
  ASSERT_EQ(meta.has_initial_state_host.size(), 3u);
  EXPECT_EQ(meta.has_initial_state_host[0], 0);
  EXPECT_EQ(meta.has_initial_state_host[1], 1);
  EXPECT_EQ(meta.has_initial_state_host[2], 1);
  EXPECT_EQ(meta.has_initial_state_tensor.shape, (std::vector<size_t>{3}));
}

// ── Flash PrepareChunkKDA ────────────────────────────────────────────────────

TEST_F(KimiK3AttentionMetaBuilderTest, Flash_PrepareChunkKDA_MultiReq) {
  // req0: 64 → 1 chunk, req1: 100 → 2 chunks (ceil), req2: 30 → 1 chunk. total = 4.
  reqs_ = {MakeReq(64, 0, 0, 0), MakeReq(100, 0, 0, 1), MakeReq(30, 0, 0, 2)};

  KimiK3FlashAttentionMeta meta;
  const size_t max_batch = 8;
  meta.chunk_offsets_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {max_batch + 1});
  meta.chunk_indices_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {64, 2});
  for (auto& r : reqs_) {
    meta.dp_reqs.push_back(&r);
  }

  auto config = MakeConfig();
  TestKimiK3FlashBuilder builder(config);
  builder.PrepareChunkKDA(meta);
  StreamSynchronize(context_->device->GetH2DStream());

  EXPECT_EQ(meta.chunk_offsets_host, (std::vector<int32_t>{0, 1, 3, 4}));
  EXPECT_EQ(meta.chunk_indices_host, (std::vector<int32_t>{0, 0, 1, 0, 1, 1, 2, 0}));
  EXPECT_EQ(meta.num_chunks_host, 4);
  EXPECT_EQ(meta.chunk_offsets_tensor.shape, (std::vector<size_t>{4}));
  EXPECT_EQ(meta.chunk_indices_tensor.shape, (std::vector<size_t>{4, 2}));
}

TEST_F(KimiK3AttentionMetaBuilderTest, Flash_PrepareChunkKDA_ZeroExtend) {
  // req0: 0 → 0 chunks, req1: 64 → 1 chunk, req2: 0 → 0 chunks. total = 1.
  // Only req1 contributes chunk_indices.
  reqs_ = {MakeReq(0, 0, 0, 0), MakeReq(64, 0, 0, 1), MakeReq(0, 0, 0, 2)};

  KimiK3FlashAttentionMeta meta;
  meta.chunk_offsets_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {8});
  meta.chunk_indices_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {64, 2});
  for (auto& r : reqs_) {
    meta.dp_reqs.push_back(&r);
  }

  auto config = MakeConfig();
  TestKimiK3FlashBuilder builder(config);
  builder.PrepareChunkKDA(meta);
  StreamSynchronize(context_->device->GetH2DStream());

  EXPECT_EQ(meta.chunk_offsets_host, (std::vector<int32_t>{0, 0, 1, 1}));
  EXPECT_EQ(meta.chunk_indices_host, (std::vector<int32_t>{1, 0}));
  EXPECT_EQ(meta.num_chunks_host, 1);
}

TEST_F(KimiK3AttentionMetaBuilderTest, Flash_PrepareChunkKDA_EmptyBatch) {
  KimiK3FlashAttentionMeta meta;
  meta.chunk_offsets_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {8});
  meta.chunk_indices_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {64, 2});

  auto config = MakeConfig();
  TestKimiK3FlashBuilder builder(config);
  builder.PrepareChunkKDA(meta);
  StreamSynchronize(context_->device->GetH2DStream());

  EXPECT_EQ(meta.chunk_offsets_host, (std::vector<int32_t>{0}));
  EXPECT_TRUE(meta.chunk_indices_host.empty());
  EXPECT_EQ(meta.num_chunks_host, 0);
}

// ── Paged PrepareStateIndices ────────────────────────────────────────────────

TEST_F(KimiK3AttentionMetaBuilderTest, Paged_PrepareStateIndices_Basic) {
  reqs_ = {MakeReq(1, 5, 3, 0), MakeReq(1, 9, 7, 1), MakeReq(1, 3, 11, 2)};

  CompressorSlotResource slot_resource;
  slot_resource.Init(runtime_config_);
  slot_resource.Alloc(0);
  slot_resource.Alloc(1);
  slot_resource.Alloc(2);

  KimiK3SharedAttentionMeta kimi_shared;
  kimi_shared.state_indices_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_INT32, {8});
  kimi_shared.state_indices_tensor.shape[0] = 0;  // Reset cursor (normally done by BuildAttentionMetas).
  kimi_shared.mamba_state_buffer = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_UINT8, {1, 1, 1});
  kimi_shared.slot_resource_ = &slot_resource;

  KimiK3PagedAttentionMeta meta;
  for (auto& r : reqs_) {
    meta.dp_reqs.push_back(&r);
  }

  auto config = MakeConfig();
  TestKimiK3PagedBuilder builder(config);
  builder.PrepareStateIndices(meta, kimi_shared);
  StreamSynchronize(context_->device->GetH2DStream());

  ASSERT_EQ(meta.state_indices_host.size(), 3u);
  EXPECT_EQ(meta.state_indices_host[0], slot_resource.Get(0) * 1);
  EXPECT_EQ(meta.state_indices_host[1], slot_resource.Get(1) * 1);
  EXPECT_EQ(meta.state_indices_host[2], slot_resource.Get(2) * 1);
  EXPECT_EQ(meta.state_indices_tensor.shape, (std::vector<size_t>{3}));
}

// ── KimiK3SharedAttentionMeta::Init ──────────────────────────────────────────
// Init allocates the slicing tensor + mamba pointer table + mamba_state_buffer,
// and seeds per-layer conv/recurrent offsets into the pointer table.

// Happy path: 2 layers, max_batch_size=8, max_pp_batch_num=2 -> slot_pool=16.
TEST_F(KimiK3AttentionMetaBuilderTest, SharedInit_PopulatesTensorsAndOffsets) {
  constexpr int kLayerNum = 2;
  auto block = std::make_shared<FakeBlockCacheLayout>();  // block_cache_size = 16
  auto model_layout = std::make_shared<FakeModelCacheLayout>(block);

  runtime_config_.max_batch_size = 8;
  runtime_config_.max_pp_batch_num = 2;  // slot_pool_capacity = max_batch_size * max_pp_batch_num = 16

  KimiK3SharedAttentionMeta kimi_shared;
  kimi_shared.Init(model_config_, runtime_config_, kLayerNum, /*rank=*/0, model_layout);

  // state_indices_tensor: DEVICE INT32 [max_batch_size].
  EXPECT_EQ(kimi_shared.state_indices_tensor.shape, (std::vector<size_t>{8}));
  EXPECT_EQ(kimi_shared.state_indices_tensor.dtype, TYPE_INT32);
  EXPECT_EQ(kimi_shared.state_indices_tensor.location, MemoryLocation::LOCATION_DEVICE);
  EXPECT_EQ(kimi_shared.layer_num_on_node, kLayerNum);

  // layer_mamba_kv_cache_ptr_tensor: HOST INT64 [1 + 2 * layer_num_on_node].
  // Slot 0 is lazy (filled later by InitMambaBasePointersIfNeeded); slots 1..2N hold offsets.
  ASSERT_EQ(kimi_shared.layer_mamba_kv_cache_ptr_tensor.shape, (std::vector<size_t>{1u + 2u * kLayerNum}));
  EXPECT_EQ(kimi_shared.layer_mamba_kv_cache_ptr_tensor.dtype, TYPE_INT64);
  EXPECT_EQ(kimi_shared.layer_mamba_kv_cache_ptr_tensor.location, MemoryLocation::LOCATION_HOST);
  const int64_t* ptr = kimi_shared.layer_mamba_kv_cache_ptr_tensor.GetPtr<int64_t>();
  EXPECT_EQ(ptr[1], 1000);  // conv layer 0
  EXPECT_EQ(ptr[2], 2000);  // recurrent layer 0
  EXPECT_EQ(ptr[3], 1001);  // conv layer 1
  EXPECT_EQ(ptr[4], 2001);  // recurrent layer 1

  // mamba_state_buffer: DEVICE BYTES [slot_pool_capacity, decode_token_num, state_block_size].
  EXPECT_EQ(kimi_shared.mamba_state_buffer.shape,
            (std::vector<size_t>{16, GetDecodeTokenNumThreshold(), block->block_cache_size}));
  EXPECT_EQ(kimi_shared.mamba_state_buffer.dtype, TYPE_BYTES);
  EXPECT_EQ(kimi_shared.mamba_state_buffer.location, MemoryLocation::LOCATION_DEVICE);
}

// conv k_offset != v_offset -> Init must throw on the first layer's conv check.
TEST_F(KimiK3AttentionMetaBuilderTest, SharedInit_ConvOffsetMismatchThrows) {
  auto block = std::make_shared<FakeBlockCacheLayout>();
  block->conv_mismatch = true;
  auto model_layout = std::make_shared<FakeModelCacheLayout>(block);

  KimiK3SharedAttentionMeta kimi_shared;
  EXPECT_THROW(kimi_shared.Init(model_config_, runtime_config_, /*layer_num_on_node=*/1, /*rank=*/0, model_layout),
               std::runtime_error);
}

// recurrent k_offset != v_offset (conv is valid) -> Init must throw on recurrent check.
TEST_F(KimiK3AttentionMetaBuilderTest, SharedInit_RecurrentOffsetMismatchThrows) {
  auto block = std::make_shared<FakeBlockCacheLayout>();
  block->recurrent_mismatch = true;
  auto model_layout = std::make_shared<FakeModelCacheLayout>(block);

  KimiK3SharedAttentionMeta kimi_shared;
  EXPECT_THROW(kimi_shared.Init(model_config_, runtime_config_, /*layer_num_on_node=*/1, /*rank=*/0, model_layout),
               std::runtime_error);
}

}  // namespace ksana_llm
