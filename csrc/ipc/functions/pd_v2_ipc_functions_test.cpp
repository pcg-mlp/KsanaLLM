/* Copyright 2026 Tencent Inc.  All rights reserved.

 Unit tests for the pd_v2 IPC function callables / (de)serialization:
   - create_block_manager.cpp   (request/response + callable + callback)
   - get_kv_pool_base_va.cpp    (response serialize + callable query path)
   - pd_v2_barrier_done.cpp     (request serialize + callable dispatch)
 ==============================================================================*/

#include <gtest/gtest.h>

#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "cache_manager/block_allocator/block_allocator_interface.h"
#include "cache_manager/block_allocator/block_allocator_manager.h"
#include "configure/environment.h"
#include "ipc/functions/create_block_manager.h"
#include "ipc/functions/get_kv_pool_base_va.h"
#include "ipc/functions/pd_v2_barrier_done.h"
#include "loguru.hpp"
#include "pd_v2/public/pd_v2_connector.h"

namespace ksana_llm {
namespace {

class MaxLogVerbosityEnv : public ::testing::Environment {
 public:
  void SetUp() override { loguru::g_stderr_verbosity = loguru::Verbosity_MAX; }
};
static auto* unused_log_env = ::testing::AddGlobalTestEnvironment(new MaxLogVerbosityEnv());

// ----- Minimal fakes for the create_block_manager connector path -----

class FakeBlockAllocator : public BlockAllocatorInterface {
 public:
  void PreAllocateBlocks() override {}
  void Clear() override {}
  Status AllocateBlocks(size_t, std::vector<int>&) override { return Status(); }
  Status FreeBlocks(const std::vector<int>&) override { return Status(); }
  Status GetBlockPtrs(const std::vector<int>&, std::vector<void*>&) override { return Status(); }
  void* GetBlocksBasePtr() override { return base_; }
  int GetBlocksBaseId() override { return 0; }
  size_t GetFreeBlockNumber() override { return 0; }
  size_t GetUsedBlockNumber() override { return 0; }
  void* base_ = nullptr;
};

class FakeBlockAllocatorGroup : public BlockAllocatorGroupInterface {
 public:
  std::vector<int> GetBlockAllocatorDevices() const override { return {0}; }
  std::shared_ptr<BlockAllocatorInterface> GetHostBlockAllocator() const override { return nullptr; }
  std::shared_ptr<BlockAllocatorInterface> GetDeviceBlockAllocator(int) const override { return dev_; }
  const BlockAllocatorGroupConfig& GetBlockAllocatorGroupConfig() const override { return cfg_; }
  std::shared_ptr<BlockAllocatorInterface> dev_;
  BlockAllocatorGroupConfig cfg_;
};

class FakeBlockAllocatorManager : public BlockAllocatorManagerInterface {
 public:
  FakeBlockAllocatorManager() = default;
  FakeBlockAllocatorManager(std::string name, std::vector<std::string>* init_order)
      : name_(std::move(name)), init_order_(init_order) {}

  void Initialize(const BlockAllocatorManagerConfig& config) override {
    ++init_calls;
    last_config = config;
    if (init_order_ != nullptr) {
      init_order_->push_back(name_);
    }
  }
  std::shared_ptr<BlockAllocatorGroupInterface> GetBlockAllocatorGroup(int) override { return group_; }
  int init_calls = 0;
  BlockAllocatorManagerConfig last_config;
  std::shared_ptr<BlockAllocatorGroupInterface> group_;

 private:
  std::string name_;
  std::vector<std::string>* init_order_ = nullptr;
};

class FakePdV2Connector : public pd_v2::PdV2ConnectorInterface {
 public:
  Status Initialize(const pd_v2::PdV2ConnectorConfig&) override { return Status(); }
  Status Shutdown() override { return Status(); }
  Status RegisterKvPool(void* base, size_t total_bytes, size_t block_size, size_t num_blocks,
                        size_t num_layers) override {
    ++register_calls;
    last_base = base;
    last_total_bytes = total_bytes;
    last_num_layers = num_layers;
    return register_status;
  }
  Status GetKvBufferVa(int, uint64_t*, uint64_t*) override { return Status(); }
  bool CanAccept(uint64_t, uint32_t) override { return true; }
  Status SubmitPrefillRequest(uint64_t, const std::vector<int>&, const std::vector<int>&,
                              const pd_v2::SamplingConfigWire&, uint32_t, uint32_t, uint32_t,
                              pd_v2::PdRequestCompleteCb, void*) override {
    return Status();
  }
  Status Release(uint64_t) override { return Status(); }
  void RegisterIncomingHook(pd_v2::PdIncomingReqCb, void*) override {}
  Status SendPrefillComplete(const std::string&, uint64_t, pd_v2::PdCompleteStatus, const std::vector<int>&,
                             const pd_v2::PeerHealth&, const pd_v2::PdPrefillMetrics&) override {
    return Status();
  }

  int register_calls = 0;
  void* last_base = nullptr;
  size_t last_total_bytes = 0;
  size_t last_num_layers = 0;
  Status register_status = Status();
};

}  // namespace

// ================= CreateBlockManager (de)serialization =================

class CreateBlockManagerSerdeTest : public ::testing::Test {
 protected:
  void SetUp() override { buf_ = new uint8_t[64 * 1024]; std::memset(buf_, 0, 64 * 1024); }
  void TearDown() override { delete[] buf_; }
  uint8_t* buf_;
};

TEST_F(CreateBlockManagerSerdeTest, RequestRoundTrip) {
  CreateBlockManagerRequest req;
  BlockAllocatorGroupConfig g;
  g.devices = {0, 1};
  g.device_block_num = 10;
  g.host_block_num = 5;
  g.block_size = 4096;
  req.block_allocator_manager_config[0] = g;
  req.swa_block_allocator_manager_config[1] = g;
  req.mamba_block_allocator_manager_config[2] = g;

  size_t bytes = 0;
  ASSERT_TRUE(req.Serialize(buf_, &bytes).OK());

  CreateBlockManagerRequest restored;
  ASSERT_TRUE(restored.Deserialize(buf_).OK());
  ASSERT_EQ(restored.block_allocator_manager_config.count(0), 1u);
  EXPECT_EQ(restored.block_allocator_manager_config[0].device_block_num, 10u);
  EXPECT_EQ(restored.block_allocator_manager_config[0].block_size, 4096u);
  EXPECT_EQ(restored.block_allocator_manager_config[0].devices, std::vector<int>({0, 1}));
  EXPECT_EQ(restored.swa_block_allocator_manager_config.count(1), 1u);
  EXPECT_EQ(restored.mamba_block_allocator_manager_config.count(2), 1u);
}

TEST_F(CreateBlockManagerSerdeTest, ResponseRoundTrip) {
  CreateBlockManagerResponse rsp;
  rsp.result_status = Status(RET_RUNTIME_FAILED, "boom");
  size_t bytes = 0;
  ASSERT_TRUE(rsp.Serialize(buf_, &bytes).OK());

  CreateBlockManagerResponse restored;
  ASSERT_TRUE(restored.Deserialize(buf_).OK());
  EXPECT_EQ(restored.result_status.GetCode(), RET_RUNTIME_FAILED);
}

// ================= CreateBlockManagerCallable / Callback =================

class CreateBlockManagerCallableTest : public ::testing::Test {
 protected:
  CreateBlockManagerCallable callable_;
  CreateBlockManagerRequest req_;
};

TEST_F(CreateBlockManagerCallableTest, NoConnectorInitializesManagers) {
  auto mgr = std::make_shared<FakeBlockAllocatorManager>();
  auto swa = std::make_shared<FakeBlockAllocatorManager>();
  auto mamba = std::make_shared<FakeBlockAllocatorManager>();
  callable_.SetBlockAllocatorManager(mgr);
  callable_.SetSwaBlockAllocatorManager(swa);
  callable_.SetMambaBlockAllocatorManager(mamba);
  // SWA config non-empty so the swa Initialize branch runs.
  req_.swa_block_allocator_manager_config[0] = BlockAllocatorGroupConfig{};
  req_.mamba_block_allocator_manager_config[0] = BlockAllocatorGroupConfig{};

  CreateBlockManagerResponse rsp;
  EXPECT_TRUE(callable_.Call(&req_, &rsp, nullptr).OK());
  EXPECT_EQ(mgr->init_calls, 1);
  EXPECT_EQ(swa->init_calls, 1);
  EXPECT_EQ(mamba->init_calls, 1);
}

TEST_F(CreateBlockManagerCallableTest, InitializesMambaPoolAfterKvAndSwaPools) {
  std::vector<std::string> init_order;
  auto kv = std::make_shared<FakeBlockAllocatorManager>("kv", &init_order);
  auto swa = std::make_shared<FakeBlockAllocatorManager>("swa", &init_order);
  auto mamba = std::make_shared<FakeBlockAllocatorManager>("mamba", &init_order);
  callable_.SetBlockAllocatorManager(kv);
  callable_.SetSwaBlockAllocatorManager(swa);
  callable_.SetMambaBlockAllocatorManager(mamba);
  req_.swa_block_allocator_manager_config[0] = BlockAllocatorGroupConfig{};
  req_.mamba_block_allocator_manager_config[0] = BlockAllocatorGroupConfig{};

  CreateBlockManagerResponse rsp;
  ASSERT_TRUE(callable_.Call(&req_, &rsp, nullptr).OK());
  EXPECT_EQ(init_order, std::vector<std::string>({"kv", "swa", "mamba"}));
}

TEST_F(CreateBlockManagerCallableTest, MambaExecutorConfigKeepsUsableBlockCount) {
  auto kv = std::make_shared<FakeBlockAllocatorManager>();
  auto mamba = std::make_shared<FakeBlockAllocatorManager>();
  callable_.SetBlockAllocatorManager(kv);
  callable_.SetMambaBlockAllocatorManager(mamba);
  BlockAllocatorGroupConfig config{};
  config.device_block_num = 4;
  req_.mamba_block_allocator_manager_config[0] = config;

  CreateBlockManagerResponse rsp;
  ASSERT_TRUE(callable_.Call(&req_, &rsp, nullptr).OK());
  ASSERT_EQ(mamba->last_config.count(0), 1u);
  EXPECT_EQ(mamba->last_config.at(0).device_block_num, 4u);
}

TEST_F(CreateBlockManagerCallableTest, ConnectorGroupNullSkips) {
  auto mgr = std::make_shared<FakeBlockAllocatorManager>();
  mgr->group_ = nullptr;  // lookup returns null -> warning, early return
  callable_.SetBlockAllocatorManager(mgr);
  callable_.SetPdV2Connector(std::make_shared<FakePdV2Connector>(), std::make_shared<Environment>(), 0, 0);

  CreateBlockManagerResponse rsp;
  EXPECT_TRUE(callable_.Call(&req_, &rsp, nullptr).OK());
}

TEST_F(CreateBlockManagerCallableTest, ConnectorDeviceAllocatorNullSkips) {
  auto mgr = std::make_shared<FakeBlockAllocatorManager>();
  auto group = std::make_shared<FakeBlockAllocatorGroup>();
  group->dev_ = nullptr;  // device allocator null
  mgr->group_ = group;
  callable_.SetBlockAllocatorManager(mgr);
  callable_.SetPdV2Connector(std::make_shared<FakePdV2Connector>(), std::make_shared<Environment>(), 0, 0);

  CreateBlockManagerResponse rsp;
  EXPECT_TRUE(callable_.Call(&req_, &rsp, nullptr).OK());
}

TEST_F(CreateBlockManagerCallableTest, ConnectorBasePtrNullSkips) {
  auto mgr = std::make_shared<FakeBlockAllocatorManager>();
  auto group = std::make_shared<FakeBlockAllocatorGroup>();
  auto dev = std::make_shared<FakeBlockAllocator>();
  dev->base_ = nullptr;  // base ptr null
  group->dev_ = dev;
  mgr->group_ = group;
  callable_.SetBlockAllocatorManager(mgr);
  callable_.SetPdV2Connector(std::make_shared<FakePdV2Connector>(), std::make_shared<Environment>(), 0, 0);

  CreateBlockManagerResponse rsp;
  EXPECT_TRUE(callable_.Call(&req_, &rsp, nullptr).OK());
}

TEST_F(CreateBlockManagerCallableTest, ConnectorRegisterKvPoolSuccess) {
  auto mgr = std::make_shared<FakeBlockAllocatorManager>();
  auto group = std::make_shared<FakeBlockAllocatorGroup>();
  auto dev = std::make_shared<FakeBlockAllocator>();
  int dummy = 0;
  dev->base_ = &dummy;
  group->cfg_.device_block_num = 8;
  group->cfg_.block_size = 1024;
  group->dev_ = dev;
  mgr->group_ = group;
  auto conn = std::make_shared<FakePdV2Connector>();
  callable_.SetBlockAllocatorManager(mgr);
  callable_.SetPdV2Connector(conn, std::make_shared<Environment>(), 0, 0);

  CreateBlockManagerResponse rsp;
  EXPECT_TRUE(callable_.Call(&req_, &rsp, nullptr).OK());
  EXPECT_EQ(conn->register_calls, 1);
  EXPECT_EQ(conn->last_base, &dummy);
  EXPECT_EQ(conn->last_total_bytes, (8u + 1u) * 1024u);  // (num_blocks+1)*block_size
}

TEST_F(CreateBlockManagerCallableTest, ConnectorRegisterKvPoolFailureSwallowed) {
  auto mgr = std::make_shared<FakeBlockAllocatorManager>();
  auto group = std::make_shared<FakeBlockAllocatorGroup>();
  auto dev = std::make_shared<FakeBlockAllocator>();
  int dummy = 0;
  dev->base_ = &dummy;
  group->cfg_.device_block_num = 1;
  group->cfg_.block_size = 256;
  group->dev_ = dev;
  mgr->group_ = group;
  auto conn = std::make_shared<FakePdV2Connector>();
  conn->register_status = Status(RET_RUNTIME_FAILED, "register failed");
  callable_.SetBlockAllocatorManager(mgr);
  callable_.SetPdV2Connector(conn, std::make_shared<Environment>(), 0, 0);

  CreateBlockManagerResponse rsp;
  // Failure is logged but NOT propagated as a CreateBlockManager error.
  EXPECT_TRUE(callable_.Call(&req_, &rsp, nullptr).OK());
  EXPECT_EQ(conn->register_calls, 1);
}

TEST(CreateBlockManagerCallbackTest, OkResponseNoError) {
  CreateBlockManagerCallback cb;
  CreateBlockManagerResponse rsp;  // default OK
  EXPECT_TRUE(cb.Call(&rsp, nullptr).OK());
  EXPECT_TRUE(cb.load_status.OK());
}

TEST(CreateBlockManagerCallbackTest, FailedResponseRecordsStatus) {
  CreateBlockManagerCallback cb;
  CreateBlockManagerResponse rsp;
  rsp.result_status = Status(RET_RUNTIME_FAILED, "create failed");
  cb.Call(&rsp, nullptr);
  EXPECT_FALSE(cb.load_status.OK());
  EXPECT_EQ(cb.load_status.GetCode(), RET_RUNTIME_FAILED);
}

// ================= GetKvPoolBaseVa =================

TEST(GetKvPoolBaseVaSerdeTest, ResponseRoundTrip) {
  uint8_t buf[1024];
  std::memset(buf, 0, sizeof(buf));
  GetKvPoolBaseVaResponse rsp;
  rsp.registered = true;
  rsp.kv_pool_base_va = 0xdeadbeef;
  rsp.total_bytes = 4096;
  rsp.block_size = 256;
  rsp.num_blocks = 16;
  rsp.num_layers = 62;

  size_t bytes = 0;
  ASSERT_TRUE(rsp.Serialize(buf, &bytes).OK());

  GetKvPoolBaseVaResponse restored;
  ASSERT_TRUE(restored.Deserialize(buf).OK());
  EXPECT_TRUE(restored.registered);
  EXPECT_EQ(restored.kv_pool_base_va, 0xdeadbeefu);
  EXPECT_EQ(restored.total_bytes, 4096u);
  EXPECT_EQ(restored.num_layers, 62u);
}

TEST(GetKvPoolBaseVaCallableTest, NoQueryFnReturnsUnregistered) {
  GetKvPoolBaseVaCallable callable;
  GetKvPoolBaseVaRequest req;
  GetKvPoolBaseVaResponse rsp;
  rsp.registered = true;  // ensure the callable resets it
  EXPECT_TRUE(callable.Call(&req, &rsp, nullptr).OK());
  EXPECT_FALSE(rsp.registered);
}

TEST(GetKvPoolBaseVaCallableTest, QueryFnPopulatesFields) {
  GetKvPoolBaseVaCallable callable;
  callable.SetQueryFn([](uint64_t* base, uint64_t* total, uint64_t* bs, uint64_t* nb, uint64_t* nl) {
    *base = 0x1000;
    *total = 8192;
    *bs = 512;
    *nb = 16;
    *nl = 60;
    return true;
  });
  GetKvPoolBaseVaRequest req;
  GetKvPoolBaseVaResponse rsp;
  EXPECT_TRUE(callable.Call(&req, &rsp, nullptr).OK());
  EXPECT_TRUE(rsp.registered);
  EXPECT_EQ(rsp.kv_pool_base_va, 0x1000u);
  EXPECT_EQ(rsp.total_bytes, 8192u);
  EXPECT_EQ(rsp.num_layers, 60u);
}

// ================= PdV2BarrierDone =================

TEST(PdV2BarrierDoneSerdeTest, RequestRoundTrip) {
  uint8_t buf[256];
  std::memset(buf, 0, sizeof(buf));
  PdV2BarrierDoneRequest req;
  req.req_id = 777;
  req.status = 1;

  size_t bytes = 0;
  ASSERT_TRUE(req.Serialize(buf, &bytes).OK());

  PdV2BarrierDoneRequest restored;
  ASSERT_TRUE(restored.Deserialize(buf).OK());
  EXPECT_EQ(restored.req_id, 777);
  EXPECT_EQ(restored.status, 1);
}

TEST(PdV2BarrierDoneCallableTest, DispatchesToCallback) {
  PdV2BarrierDoneCallable callable;
  int64_t seen_id = -1;
  uint8_t seen_status = 0;
  callable.SetCallback([&](int64_t id, uint8_t status) {
    seen_id = id;
    seen_status = status;
  });

  PdV2BarrierDoneRequest req;
  req.req_id = 555;
  req.status = 1;
  PdV2BarrierDoneResponse rsp;
  EXPECT_TRUE(callable.Call(&req, &rsp, nullptr).OK());
  EXPECT_EQ(seen_id, 555);
  EXPECT_EQ(seen_status, 1);
}

TEST(PdV2BarrierDoneCallableTest, NoCallbackDropsAndWarns) {
  PdV2BarrierDoneCallable callable;  // no callback set
  PdV2BarrierDoneRequest req;
  req.req_id = 42;
  PdV2BarrierDoneResponse rsp;
  EXPECT_TRUE(callable.Call(&req, &rsp, nullptr).OK());  // warning path, no crash
}

}  // namespace ksana_llm
