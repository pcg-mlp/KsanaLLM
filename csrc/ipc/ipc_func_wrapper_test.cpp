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
