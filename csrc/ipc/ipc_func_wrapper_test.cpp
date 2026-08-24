/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <gtest/gtest.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "batch_scheduler/global_cache_fetcher.h"
#include "batch_scheduler/global_cache_put_tracker.h"
#include "cache_manager/block_allocator/block_allocator_manager.h"
#include "cache_manager/cache_manager_interface.h"
#include "device/device_utils.h"
#include "ipc/core/ipc_func_types.h"
#include "ipc/core/ipc_func_utils.h"
#ifdef ENABLE_MULTICAST
#  include "ipc/functions/init_multicast_communication.h"
#endif
#include "ipc/ipc_func_wrapper.h"
#include "profiler/timer.h"
#include "runtime/infer_request.h"
#include "utils/logger.h"
#include "utils/memory_allocator.h"
#include "utils/random_generator.h"
#include "utils/request.h"
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
  void SetUp() override {
#ifdef ENABLE_CUDA
    nccl_uid = GenerateNCCLUniqueID();
#endif
  }

  void TearDown() override {}

 protected:
  const size_t executor_num = 2;
#ifdef ENABLE_CUDA
  ncclUniqueId nccl_uid;
#endif
};

// ============================================================================
// Basic Multicast Tests
// ============================================================================

#ifdef ENABLE_MULTICAST
TEST_F(IpcFuncWrapperCommunicationTest, MulticastCommunicationTest) {
  auto ExecutorFunc = [&](const int executor_id) {
    SetDevice(executor_id);
    auto executor_wrapper = std::make_shared<IpcFuncWrapper>(executor_num, false, executor_id);

    // Initialize context
    auto context = std::make_shared<Context>();
    context->global.reset(new GlobalContext(executor_num, 1, 1, Singleton<Environment>::GetInstance()));
    context->global->InitDistributedConfig();

    context->device = std::make_shared<DeviceContext>(executor_num);
#ifdef ENABLE_CUDA
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
#endif

    // Sync executor
    executor_wrapper->AddExecutor();

    // Await completion of multicast initialization
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

#ifdef ENABLE_CUDA
    // Check IPC handle
    EXPECT_EQ(nvls_handle->peer_uc_vas.size(), executor_num);
    EXPECT_EQ(nvls_handle->peer_uc_handles.size(), executor_num);
    EXPECT_NE(nvls_mcast_memory->GetSignalPadsDev(), nullptr);
    EXPECT_NE(nvls_handle->uc_ptr, 0);
    EXPECT_NE(nvls_handle->mc_ptr, 0);
    for (const auto& ipc_handle : ipc_handles) {
      EXPECT_GE(ipc_handle.fd, 0);
    }
#endif

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

// 为 fork IPC 用例准备独立 shm 名，避免同进程多测复用默认 /ksana_llm_shm_<pid> 互相干扰。
static void PrepareUniqueIpcShmName(const char* tag) {
  static std::atomic<int> counter{0};
  const std::string name =
      std::string("ipc_wrapper_") + tag + "_" + std::to_string(getpid()) + "_" + std::to_string(counter.fetch_add(1));
  setenv("KSANA_IPC_NAME", name.c_str(), /*overwrite=*/1);
  shm_unlink(("/" + name).c_str());
}

// 子进程退出前必须 Destroy：_exit 会跳过析构，进程共享 mutex 被遗弃后父进程 Destroy/join 会偶发卡死。
static void DestroyIpcAndExit(IpcFuncWrapper& wrapper, int code) {
  wrapper.GetIpcFuncProxy()->Destroy();
  _exit(code);
}

static bool WaitPidWithTimeout(pid_t pid, int* status, int timeout_ms) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  while (std::chrono::steady_clock::now() < deadline) {
    const pid_t got = waitpid(pid, status, WNOHANG);
    if (got == pid) {
      return true;
    }
    if (got < 0) {
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  kill(pid, SIGKILL);
  waitpid(pid, status, 0);
  return false;
}

// ReportGlobalCacheQueryDone：executor→engine oneway IPC。
TEST(ReportGlobalCacheQueryDoneIpcTest, ExecutorReportsToEngine) {
  constexpr int64_t kReqId = 42;
  constexpr size_t kExecutorNum = 1;

  PrepareUniqueIpcShmName("gcd");
  pid_t pid = fork();
  ASSERT_NE(pid, -1);

  if (pid == 0) {
    try {
      std::this_thread::sleep_for(std::chrono::milliseconds(400));
      IpcFuncWrapper executor(kExecutorNum, /*is_engine=*/false, /*executor_id=*/0, /*initialize_device=*/false);
      Status status = executor.ReportGlobalCacheQueryDone(
          kReqId, /*device_id=*/0, /*matched_ranges=*/{0, 9}, /*success=*/true, /*cached_tokens=*/{7, 8},
          /*has_cached_tokens=*/true, /*wait_us=*/11, /*qf_us=*/22, /*token_us=*/33, /*report_us=*/0, /*call_us=*/44,
          /*report_wall_us=*/55, /*swa_aligned_hit_len=*/0, /*swa_filled_block_idxs=*/{1, 2});
      if (!status.OK()) {
        DestroyIpcAndExit(executor, 1);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
      DestroyIpcAndExit(executor, 0);
    } catch (...) {
      _exit(2);
    }
  }

  IpcFuncWrapper engine(kExecutorNum, /*is_engine=*/true, /*executor_id=*/0, /*initialize_device=*/false);
  auto fetcher = std::make_shared<GlobalCacheFetcher>(/*device_count=*/1, /*ipc_func_wrapper=*/nullptr);
  engine.SetGlobalCacheQueryDoneCallableFetcher(fetcher);

  auto input = std::make_shared<KsanaPythonInput>();
  auto req_ctx = std::make_shared<std::unordered_map<std::string, std::string>>();
  auto request = std::make_shared<Request>(input, req_ctx);
  auto infer_req = std::make_shared<InferRequest>(request, 0);
  infer_req->req_id = kReqId;
  infer_req->infer_stage = InferStage::kContext;
  infer_req->block_token_num = 16;
  for (int i = 0; i < 10; ++i) {
    infer_req->output_tokens.push_back(i);
  }
  infer_req->ResetPrefillingTokens();

  GlobalCacheFetcher::PendingRequest pending;
  pending.req = infer_req;
  pending.total_devices = 1;
  pending.devices_completed = 0;
  pending.device_done_us.assign(1, -1);
  pending.submit_time_us = ProfileTimer::GetCurrentTimeInUs();
  fetcher->pending_requests_[kReqId] = std::move(pending);

  std::vector<std::shared_ptr<InferRequest>> completed;
  for (int i = 0; i < 50; ++i) {
    completed = fetcher->TakeCompletedRequests();
    if (!completed.empty()) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  int status = 0;
  ASSERT_TRUE(WaitPidWithTimeout(pid, &status, /*timeout_ms=*/10000));
  ASSERT_TRUE(WIFEXITED(status));
  EXPECT_EQ(WEXITSTATUS(status), 0);

  ASSERT_EQ(completed.size(), 1u);
  EXPECT_EQ(completed[0]->req_id, kReqId);
  EXPECT_FALSE(fetcher->HasPendingRequests());

  engine.GetIpcFuncProxy()->Destroy();
  engine.GetIpcFuncProxy()->UnlinkIpcResource();
}

class RecordingCacheManagerForPutDone : public CacheManagerInterface {
 public:
  Status DropUnbackedSwaBlocks(int64_t, size_t, std::vector<int>&, std::vector<size_t>&) override { return Status(); }
  void BeginSwaPutHold(int64_t) override {}
  void EndSwaPutHold(int64_t req_id) override { end_swa_put_hold_ids.push_back(req_id); }
  void DestroyFinishedRequest(int64_t req_id) override { destroy_ids.push_back(req_id); }
  void InitializeCachedBlocks() override {}
  std::shared_ptr<BlockAllocatorGroupInterface> GetBlockAllocatorGroup() const override { return nullptr; }
  size_t GetUsableBlockNumber() override { return 0; }
  size_t GetHostFreeBlockNumber() override { return 0; }
  size_t GetUsedBlockNumber() override { return 0; }
  size_t GetRequestStepBlockNumber(int64_t, size_t, size_t) override { return 0; }
  size_t GetRequestStepBlockNumberForOneNextToken(int64_t) override { return 0; }
  size_t GetRequestUsableBlockNumber(int64_t) override { return 0; }
  Status GetRequestPrefixBlockNumber(int64_t, const std::vector<int>&, size_t, size_t&, size_t&, size_t&) override {
    return Status();
  }
  Status GetRequestPrefixBlockNumberBeforeRemoteCacheLoading(int64_t, const std::vector<int>&, size_t, size_t&, size_t&,
                                                             size_t&, size_t&, size_t&, size_t&) override {
    return Status();
  }
  Status AllocateBlocksForRemoteCache(int64_t, std::vector<int>&, size_t, size_t, std::vector<int>&,
                                      std::vector<int>&) override {
    return Status();
  }
  Status ConfirmRemoteCacheLoadingResult(int64_t, const std::vector<int>&, size_t,
                                         const std::vector<size_t>&) override {
    return Status();
  }
  Status AllocateRequestBlocks(int64_t, size_t, std::vector<int>&, std::vector<int>&, std::vector<size_t>&,
                               int&) override {
    return Status();
  }
  void UpdateFlexibleCache(int64_t, const std::vector<int>&, int, std::vector<FlexibleCachedCopyTask>&,
                           size_t&) override {}
  Status UpdateRequestTokens(int64_t, const std::vector<int>&, size_t, std::vector<int>&, bool&, std::vector<int>&,
                             std::vector<size_t>&) override {
    return Status();
  }
  bool IsPrefixCachingEnabled() override { return false; }

  std::vector<int64_t> destroy_ids;
  std::vector<int64_t> end_swa_put_hold_ids;
};

// ReportGlobalCachePutDone：executor→engine oneway IPC。
TEST(ReportGlobalCachePutDoneIpcTest, ExecutorReportsToEngine) {
  constexpr int64_t kReqId = 55;
  constexpr size_t kExecutorNum = 1;

  PrepareUniqueIpcShmName("gpd");
  pid_t pid = fork();
  ASSERT_NE(pid, -1);

  if (pid == 0) {
    try {
      std::this_thread::sleep_for(std::chrono::milliseconds(400));
      IpcFuncWrapper executor(kExecutorNum, /*is_engine=*/false, /*executor_id=*/0, /*initialize_device=*/false);
      Status status =
          executor.ReportGlobalCachePutDone(/*device_id=*/0, kReqId, /*fa_put_blocks=*/3, /*swa_put_blocks=*/1);
      if (!status.OK()) {
        DestroyIpcAndExit(executor, 1);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
      DestroyIpcAndExit(executor, 0);
    } catch (...) {
      _exit(2);
    }
  }

  IpcFuncWrapper engine(kExecutorNum, /*is_engine=*/true, /*executor_id=*/0, /*initialize_device=*/false);
  auto cache_manager = std::make_shared<RecordingCacheManagerForPutDone>();
  auto tracker = std::make_shared<GlobalCachePutTracker>(/*device_count=*/1, cache_manager);
  engine.SetGlobalCachePutDoneCallableTracker(tracker);

  auto input = std::make_shared<KsanaPythonInput>();
  auto req_ctx = std::make_shared<std::unordered_map<std::string, std::string>>();
  auto request = std::make_shared<Request>(input, req_ctx);
  auto infer_req = std::make_shared<InferRequest>(request, 0);
  infer_req->req_id = kReqId;
  infer_req->infer_stage = InferStage::kContext;
  infer_req->cache_manager = cache_manager;
  tracker->OnBatchDispatched({infer_req});
  ASSERT_TRUE(tracker->DeferDestructionIfNeeded(kReqId));

  bool drained = false;
  for (int i = 0; i < 50; ++i) {
    if (tracker->HasPendingCompletions()) {
      EXPECT_EQ(tracker->DrainPutCompletions(), 1u);
      drained = true;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  int status = 0;
  ASSERT_TRUE(WaitPidWithTimeout(pid, &status, /*timeout_ms=*/10000));
  ASSERT_TRUE(WIFEXITED(status));
  EXPECT_EQ(WEXITSTATUS(status), 0);

  ASSERT_TRUE(drained);
  ASSERT_EQ(cache_manager->destroy_ids.size(), 1u);
  EXPECT_EQ(cache_manager->destroy_ids[0], kReqId);
  ASSERT_EQ(cache_manager->end_swa_put_hold_ids.size(), 1u);
  EXPECT_EQ(cache_manager->end_swa_put_hold_ids[0], kReqId);
  EXPECT_FALSE(tracker->HasDeferredRequests());

  engine.GetIpcFuncProxy()->Destroy();
  engine.GetIpcFuncProxy()->UnlinkIpcResource();
}

}  // namespace ksana_llm
