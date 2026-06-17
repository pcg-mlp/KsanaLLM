/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <gtest/gtest.h>

#include <memory>

#include "configure/schedule_config_parser.h"
#include "executor/inflight_resource.h"
#include "models/deepseek_v4/attention_meta/deepseek_v4_compressor_slot_resource.h"
#include "runtime/executor_infer_request.h"
#include "runtime/schedule_output.h"

namespace ksana_llm {
namespace {

// ---------------------------------------------------------------------------
// CompressorSlotResource
// ---------------------------------------------------------------------------

class CompressorSlotResourceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // max_batch_size=32, max_pp_batch_num=1 → slot_pool_capacity=32
    config_.max_batch_size = 32;
    config_.max_pp_batch_num = 1;
    resource_ = std::make_shared<CompressorSlotResource>();
    resource_->Init(config_);
  }

  RuntimeConfig config_{};
  std::shared_ptr<CompressorSlotResource> resource_;
};

TEST_F(CompressorSlotResourceTest, InitCreatesFullPool) {
  // All 32 slots should be free after Init.
  EXPECT_FALSE(resource_->Has(0));
  EXPECT_FALSE(resource_->Has(100));
}

TEST_F(CompressorSlotResourceTest, AllocAssignsSlot) {
  resource_->Alloc(42);
  EXPECT_TRUE(resource_->Has(42));
  // The pool uses LIFO (vector stack); the specific slot index is implementation-defined.
  // Assert only that a valid slot in [0, pool_capacity) is assigned.
  int slot = resource_->Get(42);
  EXPECT_GE(slot, 0);
  EXPECT_LT(slot, 32);  // pool capacity = max_batch_size * max_pp_batch_num = 32 * 1
}

TEST_F(CompressorSlotResourceTest, AllocIsIdempotent) {
  resource_->Alloc(1);
  int first = resource_->Get(1);
  resource_->Alloc(1);  // second call: no-op
  EXPECT_EQ(first, resource_->Get(1));
}

TEST_F(CompressorSlotResourceTest, AllocReturnsDistinctSlots) {
  resource_->Alloc(10);
  resource_->Alloc(20);
  resource_->Alloc(30);
  // The pool now uses LIFO (vector stack), so specific slot indices depend on
  // pool initialisation order. Only assert that each request gets a distinct slot.
  EXPECT_NE(resource_->Get(10), resource_->Get(20));
  EXPECT_NE(resource_->Get(20), resource_->Get(30));
  EXPECT_NE(resource_->Get(10), resource_->Get(30));
}

TEST_F(CompressorSlotResourceTest, FreeRecyclesSlot) {
  resource_->Alloc(7);
  resource_->Free(7);
  EXPECT_FALSE(resource_->Has(7));
  // The freed slot must be recycled back into the pool so a new alloc can succeed.
  resource_->Alloc(99);
  EXPECT_TRUE(resource_->Has(99));
}

TEST_F(CompressorSlotResourceTest, FreeUnknownIdIsNoop) { EXPECT_NO_FATAL_FAILURE(resource_->Free(999)); }

TEST_F(CompressorSlotResourceTest, ExhaustPoolTriggersCheck) {
  for (int i = 0; i < 32; ++i) {
    resource_->Alloc(1000 + i);
  }
  EXPECT_THROW(resource_->Alloc(2000), std::runtime_error);
}

TEST_F(CompressorSlotResourceTest, GetAndHasAreConsistent) {
  resource_->Alloc(123);
  EXPECT_TRUE(resource_->Has(123));
  resource_->Free(123);
  EXPECT_FALSE(resource_->Has(123));
}

TEST_F(CompressorSlotResourceTest, TailPreparedMovedToCompressorShared) {
  // tail_prepared_req_ids_ has been moved out of CompressorSlotResource.
  // Verify that Alloc/Free/Get work independently of tail-bookkeeping.
  resource_->Alloc(1);
  resource_->Free(1);
  resource_->Alloc(1);  // re-use the freed slot
  EXPECT_TRUE(resource_->Has(1));
}

// ---------------------------------------------------------------------------
// InflightResourceManager
// ---------------------------------------------------------------------------

class InflightResourceManagerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    mgr_ = std::make_shared<InflightResourceManager>();
    // Use the same small pool as the slot test so exhaustion tests make sense.
    RuntimeConfig cfg;
    cfg.max_batch_size = 4;
    cfg.max_pp_batch_num = 1;
    auto res = std::make_shared<CompressorSlotResource>();
    res->Init(cfg);
    slot_res_ = res.get();
    mgr_->RegisterResource("compressor_slot", std::move(res));
  }

  // Helper: create one ExecutorInferRequest with the given req_id.
  static std::shared_ptr<ExecutorInferRequest> MakeReq(int64_t req_id) {
    auto req = std::make_shared<ExecutorInferRequest>();
    req->req_id = req_id;
    return req;
  }

  // Helper: build a simple ScheduleOutput with running / finish / recompute lists.
  static ScheduleOutput MakeSO(std::vector<int64_t> running_ids, std::vector<int64_t> finish_ids = {},
                               std::vector<int64_t> recompute_ids = {}) {
    ScheduleOutput so;
    for (auto id : running_ids) so.executor_running_reqs.push_back(MakeReq(id));
    if (!finish_ids.empty()) {
      so.finish_req_ids.push_back(finish_ids);
      so.recompute_req_ids.push_back({});
    } else if (!recompute_ids.empty()) {
      so.finish_req_ids.push_back({});
      so.recompute_req_ids.push_back(recompute_ids);
    } else {
      so.finish_req_ids.push_back({});
      so.recompute_req_ids.push_back({});
    }
    return so;
  }

  std::shared_ptr<InflightResourceManager> mgr_;
  CompressorSlotResource* slot_res_ = nullptr;  // non-owning, for direct assertions
};

TEST_F(InflightResourceManagerTest, ProcessAllocAssignsSlots) {
  auto so = MakeSO({10, 20});
  mgr_->ProcessScheduleOutput(so);
  EXPECT_TRUE(slot_res_->Has(10));
  EXPECT_TRUE(slot_res_->Has(20));
  EXPECT_NE(slot_res_->Get(10), slot_res_->Get(20));
}

TEST_F(InflightResourceManagerTest, ProcessAllocIdempotentAcrossCalls) {
  auto so1 = MakeSO({1});
  mgr_->ProcessScheduleOutput(so1);
  int slot = slot_res_->Get(1);

  auto so2 = MakeSO({1});  // same req
  mgr_->ProcessScheduleOutput(so2);
  EXPECT_EQ(slot, slot_res_->Get(1));  // slot unchanged
}

TEST_F(InflightResourceManagerTest, ProcessFreeReturnsSlot) {
  auto so_alloc = MakeSO({5});
  mgr_->ProcessScheduleOutput(so_alloc);
  EXPECT_TRUE(slot_res_->Has(5));

  auto so_free = MakeSO({}, {5});
  mgr_->ProcessScheduleOutput(so_free);
  EXPECT_FALSE(slot_res_->Has(5));
}

TEST_F(InflightResourceManagerTest, ProcessFreeUnknownIsNoop) {
  auto so = MakeSO({}, {999});
  EXPECT_NO_FATAL_FAILURE(mgr_->ProcessScheduleOutput(so));
}

TEST_F(InflightResourceManagerTest, ProcessRecomputeReturnsSlot) {
  auto so_alloc = MakeSO({7});
  mgr_->ProcessScheduleOutput(so_alloc);
  EXPECT_TRUE(slot_res_->Has(7));

  auto so_rec = MakeSO({}, {}, {7});
  mgr_->ProcessScheduleOutput(so_rec);
  EXPECT_FALSE(slot_res_->Has(7));
}

TEST_F(InflightResourceManagerTest, ExhaustThenFreeAllowsReuse) {
  // Fill 4 slots.
  auto so_fill = MakeSO({0, 1, 2, 3});
  mgr_->ProcessScheduleOutput(so_fill);
  EXPECT_THROW(mgr_->ProcessScheduleOutput(MakeSO({4})), std::runtime_error);

  // Free one and allocate again.
  mgr_->ProcessScheduleOutput(MakeSO({}, {0}));
  EXPECT_NO_FATAL_FAILURE(mgr_->ProcessScheduleOutput(MakeSO({4})));
}

TEST_F(InflightResourceManagerTest, FreeAndAllocInSameSO) {
  // Fill all 4 slots (mirrors max_pp_batch_num=2 with full batches).
  mgr_->ProcessScheduleOutput(MakeSO({0, 1, 2, 3}));

  // A single SO that simultaneously finishes {0, 1} and adds new requests {4, 5}
  // while keeping {2, 3} running.  The slot pool (capacity=4) would be exhausted
  // if Alloc ran before Free; Free-first must allow this to succeed.
  EXPECT_NO_THROW(mgr_->ProcessScheduleOutput(MakeSO({2, 3, 4, 5}, {0, 1})));

  EXPECT_FALSE(slot_res_->Has(0));
  EXPECT_FALSE(slot_res_->Has(1));
  EXPECT_TRUE(slot_res_->Has(2));
  EXPECT_TRUE(slot_res_->Has(3));
  EXPECT_TRUE(slot_res_->Has(4));
  EXPECT_TRUE(slot_res_->Has(5));
}

TEST_F(InflightResourceManagerTest, GetTypedResourceReturnsCorrectPointer) {
  auto* p = mgr_->GetTypedResource<CompressorSlotResource>("compressor_slot");
  EXPECT_EQ(slot_res_, p);
}

TEST_F(InflightResourceManagerTest, GetTypedResourceMissingNameReturnsNull) {
  EXPECT_EQ(nullptr, mgr_->GetTypedResource<CompressorSlotResource>("nonexistent"));
}

TEST_F(InflightResourceManagerTest, ProcessWithNoResourcesIsNoop) {
  auto empty_mgr = std::make_shared<InflightResourceManager>();
  auto so = MakeSO({1, 2});
  EXPECT_NO_FATAL_FAILURE(empty_mgr->ProcessScheduleOutput(so));
}

}  // namespace
}  // namespace ksana_llm
