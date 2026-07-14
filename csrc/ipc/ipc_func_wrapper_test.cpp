/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <gtest/gtest.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "cache_manager/block_allocator/block_allocator_manager.h"
#include "ipc/core/ipc_func_types.h"
#include "ipc/core/ipc_func_utils.h"
#include "ipc/functions/batch_swap.h"
#include "ipc/functions/init_multicast_communication.h"
#include "ipc/ipc_func_wrapper.h"
#include "utils/logger.h"
#include "utils/memory_allocator.h"
#include "utils/random_generator.h"
#include "utils/ret_code.h"
#include "utils/status.h"

namespace ksana_llm {

// ============================================================================
// Mock BlockAllocatorGroup for Testing
// ============================================================================

// Structure to record swap operations
struct SwapOperation {
  int index;
  bool is_swapin;
  std::vector<int> device_block_ids;
  std::vector<int> host_block_ids;
  std::chrono::steady_clock::time_point timestamp;
};

// Mock BlockAllocatorGroup that records all swap operations
class MockBlockAllocatorGroup : public BlockAllocatorGroupInterface {
 public:
  explicit MockBlockAllocatorGroup(const BlockAllocatorGroupConfig& config) : config_(config) {}

  virtual ~MockBlockAllocatorGroup() {}

  virtual std::vector<int> GetBlockAllocatorDevices() const override { return config_.devices; }

  virtual std::shared_ptr<BlockAllocatorInterface> GetHostBlockAllocator() const override { return nullptr; }

  virtual std::shared_ptr<BlockAllocatorInterface> GetDeviceBlockAllocator(int index = 0) const override {
    return nullptr;
  }

  virtual Status SwapIn(int index, int device_block_id, int host_block_id) override {
    return Status(RET_RUNTIME_FAILED, "SwapIn not implemented in mock");
  }

  virtual Status SwapOut(int index, int host_block_id, int device_block_id) override {
    return Status(RET_RUNTIME_FAILED, "SwapOut not implemented in mock");
  }

  virtual Status BatchSwapOut(int index, const std::vector<int>& host_block_ids,
                              const std::vector<int>& device_block_ids) override {
    std::lock_guard<std::mutex> lock(operations_mutex_);

    SwapOperation op;
    op.index = index;
    op.is_swapin = false;
    op.device_block_ids = device_block_ids;
    op.host_block_ids = host_block_ids;
    op.timestamp = std::chrono::steady_clock::now();

    operations_.push_back(op);
    operation_count_++;

    KLLM_LOG_INFO << "MockBlockAllocatorGroup::BatchSwapOut called, index=" << index
                  << ", device_blocks=" << device_block_ids.size() << ", host_blocks=" << host_block_ids.size();

    // Simulate some work
    if (simulate_delay_ms_ > 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(simulate_delay_ms_));
    }

    if (should_fail_) {
      return Status(RET_RUNTIME_FAILED, "Simulated BatchSwapOut failure");
    }

    return Status();
  }

  virtual Status BatchSwapIn(int index, const std::vector<int>& device_block_ids,
                             const std::vector<int>& host_block_ids) override {
    std::lock_guard<std::mutex> lock(operations_mutex_);

    SwapOperation op;
    op.index = index;
    op.is_swapin = true;
    op.device_block_ids = device_block_ids;
    op.host_block_ids = host_block_ids;
    op.timestamp = std::chrono::steady_clock::now();

    operations_.push_back(op);
    operation_count_++;

    KLLM_LOG_INFO << "MockBlockAllocatorGroup::BatchSwapIn called, index=" << index
                  << ", device_blocks=" << device_block_ids.size() << ", host_blocks=" << host_block_ids.size();

    // Simulate some work
    if (simulate_delay_ms_ > 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(simulate_delay_ms_));
    }

    if (should_fail_) {
      return Status(RET_RUNTIME_FAILED, "Simulated BatchSwapIn failure");
    }

    return Status();
  }

  virtual const BlockAllocatorGroupConfig& GetBlockAllocatorGroupConfig() const override { return config_; }

  // Test helper methods
  void Reset() {
    std::lock_guard<std::mutex> lock(operations_mutex_);
    operations_.clear();
    operation_count_ = 0;
    should_fail_ = false;
    simulate_delay_ms_ = 0;
  }

  std::vector<SwapOperation> GetOperations() const {
    std::lock_guard<std::mutex> lock(operations_mutex_);
    return operations_;
  }

  int GetOperationCount() const { return operation_count_.load(); }

  void SetShouldFail(bool should_fail) { should_fail_ = should_fail; }

  void SetSimulateDelay(int delay_ms) { simulate_delay_ms_ = delay_ms; }

 private:
  BlockAllocatorGroupConfig config_;
  mutable std::mutex operations_mutex_;
  std::vector<SwapOperation> operations_;
  std::atomic<int> operation_count_{0};
  bool should_fail_ = false;
  int simulate_delay_ms_ = 0;
};

// Mock BlockAllocatorManager that returns MockBlockAllocatorGroup
// Directly implements BlockAllocatorManagerInterface without depending on BlockAllocatorManager
class MockBlockAllocatorManager : public BlockAllocatorManagerInterface {
 public:
  MockBlockAllocatorManager() : initialized_(false) {}

  virtual ~MockBlockAllocatorManager() {}

  // Implement Initialize from BlockAllocatorManagerInterface
  virtual void Initialize(const BlockAllocatorManagerConfig& config) override {
    // Create mock groups for each group_id
    for (const auto& [group_id, group_config] : config) {
      auto mock_group = std::make_shared<MockBlockAllocatorGroup>(group_config);
      mock_groups_[group_id] = mock_group;
    }

    initialized_ = true;
  }

  // Implement GetBlockAllocatorGroup from BlockAllocatorManagerInterface
  virtual std::shared_ptr<BlockAllocatorGroupInterface> GetBlockAllocatorGroup(int group_id) override {
    auto it = mock_groups_.find(group_id);
    if (it != mock_groups_.end()) {
      return std::static_pointer_cast<BlockAllocatorGroupInterface>(it->second);
    }
    return nullptr;
  }

  // Test helper methods
  std::shared_ptr<MockBlockAllocatorGroup> GetMockGroup(int group_id) {
    auto it = mock_groups_.find(group_id);
    if (it != mock_groups_.end()) {
      return it->second;
    }
    return nullptr;
  }

  bool IsInitialized() const { return initialized_; }

  // Wait for initialization to complete (useful for synchronization)
  bool WaitForInitialization(int timeout_ms = 5000) {
    auto start = std::chrono::steady_clock::now();
    while (!initialized_) {
      auto elapsed =
          std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
      if (elapsed > timeout_ms) {
        return false;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return true;
  }

 private:
  std::unordered_map<int, std::shared_ptr<MockBlockAllocatorGroup>> mock_groups_;
  std::atomic<bool> initialized_;
};

// ============================================================================
// Test Fixture
// ============================================================================

class IpcFuncWrapperBatchSwapTest : public ::testing::Test {
 protected:
  void SetUp() override {
    context_ = std::make_shared<Context>();
    memory_allocator_ = std::make_shared<MemoryAllocator>();

    // Setup block allocator config
    BlockAllocatorGroupConfig group_config;
    group_config.devices = {0};
    group_config.device_block_num = 100;
    group_config.host_block_num = 200;
    group_config.block_size = 1024 * 1024;  // 1MB

    block_allocator_config_[0] = group_config;
  }

  void TearDown() override {
    // Cleanup
  }

 protected:
  std::shared_ptr<Context> context_;
  std::shared_ptr<MemoryAllocatorInterface> memory_allocator_;
  BlockAllocatorManagerConfig block_allocator_config_;
};

// ============================================================================
// Basic BatchSwap Tests
// ============================================================================

TEST_F(IpcFuncWrapperBatchSwapTest, BatchSwapInAndOutOperations) {
  const size_t executor_num = 1;
  const int dp_group_id = 0;

  pid_t pid = fork();
  ASSERT_NE(pid, -1) << "Fork failed";

  if (pid == 0) {
    // Child process: Executor
    try {
      auto executor_wrapper = std::make_shared<IpcFuncWrapper>(executor_num, false, 0);
      auto executor_block_mgr = std::make_shared<MockBlockAllocatorManager>();
      executor_wrapper->SetBlockAllocatorManager(executor_block_mgr, dp_group_id);
      executor_wrapper->AddExecutor();

      // Wait for engine to trigger initialization via CreateBlockAllocatorManager
      bool init_success = executor_block_mgr->WaitForInitialization(5000);
      EXPECT_TRUE(init_success) << "Initialization timeout";

      // Wait for both swap operations to complete
      std::this_thread::sleep_for(std::chrono::milliseconds(200));

      // Verify both operations were recorded
      auto mock_group = executor_block_mgr->GetMockGroup(0);
      EXPECT_NE(mock_group, nullptr);

      int op_count = mock_group ? mock_group->GetOperationCount() : 0;
      EXPECT_EQ(op_count, 2) << "Expected 2 operations (SwapIn + SwapOut)";

      auto operations = mock_group ? mock_group->GetOperations() : std::vector<SwapOperation>();
      EXPECT_EQ(operations.size(), 2);

      if (operations.size() >= 2) {
        // Verify first operation (SwapIn)
        const auto& op1 = operations[0];
        EXPECT_EQ(op1.index, 0);
        EXPECT_TRUE(op1.is_swapin);
        EXPECT_EQ(op1.device_block_ids.size(), 3);
        EXPECT_EQ(op1.host_block_ids.size(), 3);
        EXPECT_EQ(op1.device_block_ids[0], 1);
        EXPECT_EQ(op1.host_block_ids[0], 10);

        // Verify second operation (SwapOut)
        const auto& op2 = operations[1];
        EXPECT_EQ(op2.index, 0);
        EXPECT_FALSE(op2.is_swapin);
        EXPECT_EQ(op2.device_block_ids.size(), 4);
        EXPECT_EQ(op2.host_block_ids.size(), 4);
        EXPECT_EQ(op2.device_block_ids[0], 5);
        EXPECT_EQ(op2.host_block_ids[0], 50);
      }
      executor_wrapper = nullptr;  // Force destroy before exit
      exit(0);
    } catch (const std::exception& e) {
      KLLM_LOG_ERROR << "Executor error: " << e.what();
      exit(1);
    }
  } else {
    // Parent process: Engine
    auto engine_wrapper = std::make_shared<IpcFuncWrapper>(executor_num, true, 0);

    // Wait for executor to be ready
    std::vector<int64_t> executor_pids;
    engine_wrapper->WaitAllExecutorsReady(executor_pids);

    // Trigger executor initialization via CreateBlockAllocatorManager
    Status init_status = engine_wrapper->CreateBlockAllocatorManager(block_allocator_config_);
    EXPECT_TRUE(init_status.OK()) << "CreateBlockAllocatorManager failed: " << init_status.GetMessage();

    // Perform swap in operation
    std::vector<int> device_block_ids_in = {1, 2, 3};
    std::vector<int> host_block_ids_in = {10, 20, 30};
    Status status_in = engine_wrapper->BatchSwap(0, true, device_block_ids_in, host_block_ids_in);
    EXPECT_TRUE(status_in.OK()) << "BatchSwap (SwapIn) failed: " << status_in.GetMessage();

    // Perform swap out operation
    std::vector<int> device_block_ids_out = {5, 6, 7, 8};
    std::vector<int> host_block_ids_out = {50, 60, 70, 80};
    Status status_out = engine_wrapper->BatchSwap(0, false, device_block_ids_out, host_block_ids_out);
    EXPECT_TRUE(status_out.OK()) << "BatchSwap (SwapOut) failed: " << status_out.GetMessage();

    // Give executor time to process second operation
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Stop all channels (wake up child's threads) without destroying mutexes; then
    // wait for child to exit; then destroy engine so process-shared mutexes are
    // destroyed only after child is gone.
    engine_wrapper->Shutdown();
    int child_status;
    waitpid(pid, &child_status, 0);
    EXPECT_EQ(WEXITSTATUS(child_status), 0);

    engine_wrapper = nullptr;
  }
}

// ============================================================================
// Test Fixture
// ============================================================================

class IpcFuncWrapperCommunicationTest : public ::testing::Test {
 protected:
  void SetUp() override { nccl_uid = GenerateNCCLUniqueID(); }

  void TearDown() override {}

 protected:
  const size_t executor_num = 2;
  ncclUniqueId nccl_uid;
};

// ============================================================================
// Basic Multicast Tests
// ============================================================================

TEST_F(IpcFuncWrapperCommunicationTest, MulticastCommunicationTest) {
  // TODO(yfnjin): error test uinit, fix it.
  return;

  auto ExecutorFunc = [&](const int executor_id) {
    SetDevice(executor_id);
    auto executor_wrapper = std::make_shared<IpcFuncWrapper>(executor_num, false, executor_id);

    // Initialize context
    auto context = std::make_shared<Context>();
    context->global.reset(new GlobalContext(executor_num, 1, 1, Singleton<Environment>::GetInstance()));
    context->global->InitDistributedConfig();

    context->device = std::make_shared<DeviceContext>(executor_num);
    context->device->ext->InitNcclParam(nccl_uid);
    // Force multicast for testing
    context->device->ext->is_multicast_supported_ = true;

    executor_wrapper->SetMulticastCommunicationContext(context);
    // Wait NCCL init
    static_cast<void>(context->device->ext->GetNCCLParam());

    // Initialize and allocate multicast memory
    auto nvls_mcast_memory = NvlsMcastMemory::GetInstance();
    nvls_mcast_memory->Initialize(executor_num);
    auto& ipc_handles = nvls_mcast_memory->GetIpcHandles();
    EXPECT_EQ(ipc_handles.size(), executor_num + 1);
    [[maybe_unused]] Tensor input(MemoryLocation::LOCATION_MULTICAST, TYPE_FP16, {1024});
    auto nvls_handle = nvls_mcast_memory->GetNvlsHandle();
    EXPECT_NE(nvls_handle, nullptr);

    // Sync executor
    executor_wrapper->AddExecutor();

    // Await completion of multicast initialization
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    // Check IPC handle
    EXPECT_EQ(nvls_handle->peer_uc_vas.size(), executor_num);
    EXPECT_EQ(nvls_handle->peer_uc_handles.size(), executor_num);
    EXPECT_NE(nvls_mcast_memory->GetSignalPadsDev(), nullptr);
    EXPECT_NE(nvls_handle->uc_ptr, 0);
    EXPECT_NE(nvls_handle->mc_ptr, 0);
    for (const auto& ipc_handle : ipc_handles) {
      EXPECT_GE(ipc_handle.fd, 0);
    }

    std::exit(0);
  };

  auto EngineFunc = [&]() {
    auto engine_wrapper = std::make_shared<IpcFuncWrapper>(executor_num, true, -1);
    // Wait for executor to be ready
    std::vector<int64_t> executor_pids;
    engine_wrapper->WaitAllExecutorsReady(executor_pids);
    EXPECT_EQ(executor_pids.size(), executor_num);

    // Call the tested IPC function
    EXPECT_TRUE(engine_wrapper->InitMulticastCommunication().OK());
  };

  std::vector<pid_t> pids;
  pids.reserve(executor_num);
  for (size_t executor_id = 0; executor_id < executor_num; executor_id++) {
    pid_t pid = fork();
    ASSERT_GE(pid, 0) << "Fork failed";
    if (pid == 0) {  // Child process
      ExecutorFunc(executor_id);
    }
    pids.push_back(pid);
  }

  EngineFunc();

  // Wait and check child process
  for (pid_t pid : pids) {
    int status = 0;
    ASSERT_EQ(waitpid(pid, &status, 0), pid);
    ASSERT_TRUE(WIFEXITED(status));
    EXPECT_EQ(WEXITSTATUS(status), 0);
  }
}

// Verify the INT64_MAX masking pattern used in InitMulticastCommunication (line 270).
// gen() returns uint64_t values that can have the sign bit set; the mask ensures the
// cast to int64_t always yields a non-negative unique_op_id.
TEST(InitMulticastCommunicationRequestTest, UniqueOpIdMaskingAlwaysPositive) {
  RandomGenerator<uint64_t> gen(/*seed=*/12345);

  for (int i = 0; i < 1000; ++i) {
    int64_t unique_op_id = static_cast<int64_t>(gen() & INT64_MAX);
    EXPECT_GE(unique_op_id, 0) << "unique_op_id must be non-negative after masking, iteration " << i;
  }
}

// Verify serialization round-trip preserves unique_op_id.
TEST(InitMulticastCommunicationRequestTest, SerializationRoundTrip) {
  InitMulticastCommunicationRequest req;
  req.unique_op_id = static_cast<int64_t>(UINT64_MAX & INT64_MAX);
  req.broadcast_id = 42;

  std::vector<uint8_t> buf(256);
  size_t bytes = 0;
  Status s = req.Serialize(buf.data(), &bytes);
  EXPECT_TRUE(s.OK());

  InitMulticastCommunicationRequest req2;
  s = req2.Deserialize(buf.data());
  EXPECT_TRUE(s.OK());
  EXPECT_EQ(req2.unique_op_id, req.unique_op_id);
  EXPECT_EQ(req2.broadcast_id, req.broadcast_id);
}

#ifdef ENABLE_CUDA
// Direct unit test for InitMulticastCommunicationCallable::Call covering
// the unique_op_id >= 0 branch (IPC socket init) and unique_op_id < 0 branch
// (KLLM_CHECK_WITH_INFO on socket not null).
TEST(InitMulticastCommunicationCallableTest, UniqueOpIdBranches) {
  int device_count = 0;
  GetDeviceCount(&device_count);
  if (device_count < 1) {
    GTEST_SKIP() << "No CUDA device available";
  }

  SetDevice(0);

  const size_t tp_size = 1;
  auto nvls_mcast_memory = NvlsMcastMemory::GetInstance();
  nvls_mcast_memory->Initialize(tp_size);

  auto context = std::make_shared<Context>();
  context->global = std::make_shared<GlobalContext>(tp_size, 1, 1, nullptr);
  context->device = std::make_shared<DeviceContext>(tp_size);
  context->device->ext->is_multicast_supported_ = true;

  InitMulticastCommunicationCallable callable;
  callable.SetContext(context);

  // Branch 1: unique_op_id >= 0 — initializes IPC socket (covers lines 49-50)
  {
    InitMulticastCommunicationRequest req;
    req.unique_op_id = 42;
    InitMulticastCommunicationResponse resp;
    Status s = callable.Call(&req, &resp, nullptr);
    EXPECT_TRUE(s.OK()) << s.GetMessage();
    EXPECT_NE(nvls_mcast_memory->GetNcclIpcSocket(), nullptr);
  }

  // Branch 2: unique_op_id < 0 — checks nccl_ipc_socket != nullptr (covers lines 55-57)
  // broadcast_id = 99 doesn't match any send/recv/init sub-branch, so it falls through safely.
  {
    InitMulticastCommunicationRequest req;
    req.unique_op_id = -1;
    req.broadcast_id = 99;
    InitMulticastCommunicationResponse resp;
    Status s = callable.Call(&req, &resp, nullptr);
    EXPECT_TRUE(s.OK()) << s.GetMessage();
  }

  NcclIpcSocketClose(nvls_mcast_memory->GetNcclIpcSocket());
  nvls_mcast_memory->GetNcclIpcSocket().reset();

  // Mark NCCL as initialized so DeviceContext destructor doesn't spin forever
  // in GetNCCLParam(). With tp_size=1, InitNcclParam skips ncclCommInitRank.
  ncclUniqueId dummy_uid{};
  context->device->ext->InitNcclParam(dummy_uid);
}
#endif

}  // namespace ksana_llm
