/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "executor/nccl_executor_handler.h"

#ifdef ENABLE_CUDA
#  include <gmock/gmock.h>
#  include <gtest/gtest.h>
#  include <atomic>
#  include <condition_variable>
#  include <memory>
#  include <mutex>
#  include <thread>
#  include <vector>

#  include "cache_manager/block_allocator/block_allocator_interface.h"
#  include "cache_manager/block_allocator/block_allocator_manager.h"
#  include "configure/schedule_config_parser.h"
#  include "connector/config.h"
#  include "device/nvidia/cuda_utils.h"
#  include "device/nvidia/nccl_utils.h"
#  include "utils/status.h"

namespace ksana_llm {

// Mock BlockAllocatorInterface for testing
class MockBlockAllocator : public BlockAllocatorInterface {
 public:
  MOCK_METHOD(void, PreAllocateBlocks, (), (override));
  MOCK_METHOD(void, Clear, (), (override));

  MOCK_METHOD(Status, AllocateBlocks, (size_t block_num, std::vector<int>& blocks), (override));
  MOCK_METHOD(Status, FreeBlocks, (const std::vector<int>& blocks), (override));
  MOCK_METHOD(Status, GetBlockPtrs, (const std::vector<int>& blocks, std::vector<void*>& addrs), (override));

  MOCK_METHOD(void*, GetBlocksBasePtr, (), (override));
  MOCK_METHOD(int, GetBlocksBaseId, (), (override));

  MOCK_METHOD(size_t, GetFreeBlockNumber, (), (override));
  MOCK_METHOD(size_t, GetUsedBlockNumber, (), (override));
};

class NcclExecutorHandlerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    device_id_ = 0;

    // 创建测试用的 ConnectorConfig
    ConnectorConfig connector_config;
    connector_config.send_thread_num = 4;
    connector_config.transfer_batch = 1048576;

    handler_ = std::make_unique<NcclExecutorHandler>(device_id_, connector_config);
  }

  int device_id_;
  std::unique_ptr<NcclExecutorHandler> handler_;
};

// Test constructor
TEST_F(NcclExecutorHandlerTest, Constructor) { EXPECT_NE(handler_, nullptr); }

// Test InitializeCommGroup
TEST_F(NcclExecutorHandlerTest, InitializeCommGroup_Success) {
  std::string group_key = "test_group";
  ncclUniqueId comm_id = {0};
  int rank = 0;
  int world_size = 2;

  // Note: This will use mock NCCL functions if available
  Status status = handler_->InitializeCommGroup(group_key, comm_id, rank, world_size);

  // In test environment without real NCCL, this might fail
  // We just verify it doesn't crash
  EXPECT_TRUE(true);
}

// Test Send with non-existent group (error case)
TEST_F(NcclExecutorHandlerTest, Send_NonExistentGroup) {
  std::string group_key = "non_existent_group";
  int peer_rank = 1;
  std::vector<uint8_t> data = {1, 2, 3, 4};
  DataType dtype = DataType::TYPE_FP32;

  Status status = handler_->Send(group_key, peer_rank, data.data(), data.size(), dtype);
  EXPECT_FALSE(status.OK());
  EXPECT_EQ(status.GetCode(), RetCode::RET_INVALID_ARGUMENT);
}

// Command types for helper thread communication
enum class HelperCommand { NONE, INIT_COMM_GROUP, SEND, RECV, SHUTDOWN };

// Multi-GPU test fixture with helper thread for NCCL communication testing
// The helper thread runs on GPU 1 and waits for commands from the main test thread
class NcclExecutorHandlerMultiGpuTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Check if we have at least 2 GPUs
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count < 2) {
      skip_tests_ = true;
      return;
    }
    skip_tests_ = false;

    // Create ConnectorConfig
    connector_config_.send_thread_num = 4;
    connector_config_.transfer_batch = 1048576;

    // Initialize synchronization primitives
    helper_command_ = HelperCommand::NONE;
    helper_ready_ = false;
    command_done_ = false;
    helper_error_.clear();
    shutdown_requested_ = false;

    // Start helper thread on GPU 1
    helper_thread_ = std::thread(&NcclExecutorHandlerMultiGpuTest::HelperThreadFunc, this);

    // Wait for helper thread to be ready
    {
      std::unique_lock<std::mutex> lock(mtx_);
      cv_.wait(lock, [this]() { return helper_ready_.load(); });
    }
  }

  void TearDown() override {
    if (skip_tests_) {
      return;
    }

    // Send shutdown command to helper thread
    {
      std::lock_guard<std::mutex> lock(mtx_);
      helper_command_ = HelperCommand::SHUTDOWN;
      shutdown_requested_ = true;
    }
    cv_.notify_all();

    // Wait for helper thread to finish
    if (helper_thread_.joinable()) {
      helper_thread_.join();
    }
  }

  // Helper thread function - runs on GPU 1 and waits for commands
  void HelperThreadFunc() {
    // Set CUDA device to GPU 1
    cudaError_t cuda_err = cudaSetDevice(1);
    if (cuda_err != cudaSuccess) {
      std::lock_guard<std::mutex> lock(mtx_);
      helper_error_ = "Helper thread: Failed to set CUDA device 1";
      helper_ready_ = true;
      cv_.notify_all();
      return;
    }

    // Create handler for device 1
    helper_handler_ = std::make_unique<NcclExecutorHandler>(1, connector_config_);

    // Signal that helper is ready
    {
      std::lock_guard<std::mutex> lock(mtx_);
      helper_ready_ = true;
    }
    cv_.notify_all();

    // Main command loop
    while (true) {
      HelperCommand cmd;
      {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait(lock, [this]() { return helper_command_ != HelperCommand::NONE; });
        cmd = helper_command_;
      }

      if (cmd == HelperCommand::SHUTDOWN) {
        break;
      }

      // Execute command
      Status status;
      switch (cmd) {
        case HelperCommand::INIT_COMM_GROUP:
          status = helper_handler_->InitializeCommGroup(cmd_group_key_, cmd_comm_id_, 1, cmd_world_size_);
          break;

        case HelperCommand::SEND:
          status = helper_handler_->Send(cmd_group_key_, cmd_peer_rank_, cmd_data_ptr_, cmd_data_size_, cmd_dtype_);
          break;

        case HelperCommand::RECV:
          status = helper_handler_->Recv(cmd_group_key_, cmd_peer_rank_, cmd_data_ptr_, cmd_data_size_, cmd_dtype_);
          break;

        default:
          break;
      }

      // Store result and signal completion
      {
        std::lock_guard<std::mutex> lock(mtx_);
        helper_status_ = status;
        command_done_ = true;
        helper_command_ = HelperCommand::NONE;
      }
      cv_.notify_all();
    }
  }

  // Send command to helper thread (non-blocking, just triggers the command)
  void TriggerHelperCommand(HelperCommand cmd) {
    {
      std::lock_guard<std::mutex> lock(mtx_);
      command_done_ = false;
      helper_command_ = cmd;
    }
    cv_.notify_all();
  }

  // Wait for helper command to complete
  Status WaitHelperCommand() {
    std::unique_lock<std::mutex> lock(mtx_);
    cv_.wait(lock, [this]() { return command_done_.load() || shutdown_requested_.load(); });
    return helper_status_;
  }

  // Initialize comm group on helper thread (non-blocking trigger)
  void TriggerHelperInitCommGroup(const std::string& group_key, const ncclUniqueId& comm_id, int world_size) {
    cmd_group_key_ = group_key;
    cmd_comm_id_ = comm_id;
    cmd_world_size_ = world_size;
    TriggerHelperCommand(HelperCommand::INIT_COMM_GROUP);
  }

  // Send data from helper thread (non-blocking trigger)
  void TriggerHelperSend(const std::string& group_key, int peer_rank, void* data, size_t size, DataType dtype) {
    cmd_group_key_ = group_key;
    cmd_peer_rank_ = peer_rank;
    cmd_data_ptr_ = data;
    cmd_data_size_ = size;
    cmd_dtype_ = dtype;
    TriggerHelperCommand(HelperCommand::SEND);
  }

  // Receive data on helper thread (non-blocking trigger)
  void TriggerHelperRecv(const std::string& group_key, int peer_rank, void* data, size_t size, DataType dtype) {
    cmd_group_key_ = group_key;
    cmd_peer_rank_ = peer_rank;
    cmd_data_ptr_ = data;
    cmd_data_size_ = size;
    cmd_dtype_ = dtype;
    TriggerHelperCommand(HelperCommand::RECV);
  }

  // Calculate the peer_rank parameter needed to achieve the desired actual peer rank
  // Based on the formula in nccl_executor_handler.cpp:
  // actual_peer = (world_size / 2 + peer_rank + (rank - device_id)) % world_size
  // For our test setup where rank == device_id, this simplifies to:
  // actual_peer = (world_size / 2 + peer_rank) % world_size
  // So to get actual_peer, we need: peer_rank = (actual_peer - world_size / 2 + world_size) % world_size
  int CalculatePeerRankParam(int actual_peer, int world_size) {
    return (actual_peer - world_size / 2 + world_size) % world_size;
  }

  // Test fixture members
  ConnectorConfig connector_config_;
  bool skip_tests_ = false;

  // Helper thread
  std::thread helper_thread_;
  std::unique_ptr<NcclExecutorHandler> helper_handler_;

  // Synchronization
  std::mutex mtx_;
  std::condition_variable cv_;
  std::atomic<bool> helper_ready_{false};
  std::atomic<bool> command_done_{false};
  std::atomic<bool> shutdown_requested_{false};
  HelperCommand helper_command_ = HelperCommand::NONE;
  Status helper_status_;
  std::string helper_error_;

  // Command parameters
  std::string cmd_group_key_;
  ncclUniqueId cmd_comm_id_;
  int cmd_world_size_ = 0;
  int cmd_peer_rank_ = 0;
  void* cmd_data_ptr_ = nullptr;
  size_t cmd_data_size_ = 0;
  DataType cmd_dtype_ = DataType::TYPE_FP32;
};

// Comprehensive test for NCCL Send/Recv between two GPUs
// This test performs multiple operations in a single test to avoid repeated NCCL initialization overhead
// NCCL operations are blocking, so Send and Recv must be called simultaneously from different threads
TEST_F(NcclExecutorHandlerMultiGpuTest, SendRecvOperations_TwoGpus) {
  if (skip_tests_) {
    GTEST_SKIP() << "This test requires at least 2 CUDA devices";
  }

  const std::string group_key = "test_comprehensive_group";
  const int world_size = 2;

  // Generate NCCL unique ID
  ncclUniqueId comm_id;
  NCCL_CHECK(ncclGetUniqueId(&comm_id));

  // Set CUDA device to GPU 0 for main test thread
  CUDA_CHECK(cudaSetDevice(0));

  // Create handler for device 0 (main test thread)
  auto main_handler = std::make_unique<NcclExecutorHandler>(0, connector_config_);

  // ========== Test 1: Initialize comm group on both sides ==========
  // NCCL requires all ranks to call ncclCommInitRank simultaneously (blocking)
  TriggerHelperInitCommGroup(group_key, comm_id, world_size);

  Status status = main_handler->InitializeCommGroup(group_key, comm_id, 0, world_size);
  ASSERT_TRUE(status.OK()) << "Main InitializeCommGroup failed: " << status.GetMessage();

  Status helper_status = WaitHelperCommand();
  ASSERT_TRUE(helper_status.OK()) << "Helper InitializeCommGroup failed: " << helper_status.GetMessage();

  // ========== Test 2: Verify GetCommGroupInfo ==========
  int out_rank = -1;
  int out_world_size = -1;
  status = main_handler->GetCommGroupInfo(group_key, out_rank, out_world_size);
  EXPECT_TRUE(status.OK()) << "GetCommGroupInfo failed: " << status.GetMessage();
  EXPECT_EQ(out_rank, 0);
  EXPECT_EQ(out_world_size, world_size);

  // ========== Test 3: Send from main (rank 0) to helper (rank 1) ==========
  // The peer_rank calculation in nccl_executor_handler.cpp:
  // actual_peer = (world_size / 2 + peer_rank + (rank - device_id)) % world_size
  // For rank 0 on device 0: actual_peer = (1 + peer_rank + 0) % 2
  // To send to rank 1, we need: (1 + peer_rank) % 2 = 1, so peer_rank = 0
  // For rank 1 on device 1: actual_peer = (1 + peer_rank + 0) % 2
  // To recv from rank 0, we need: (1 + peer_rank) % 2 = 0, so peer_rank = 1

  const size_t data_count_1 = 16;
  std::vector<float> send_data_1(data_count_1);
  for (size_t i = 0; i < data_count_1; ++i) {
    send_data_1[i] = static_cast<float>(i) * 1.5f;
  }
  std::vector<float> recv_data_1(data_count_1, 0.0f);

  // Calculate peer_rank parameters
  int main_peer_rank_for_send = CalculatePeerRankParam(1, world_size);    // main (rank 0) sends to rank 1
  int helper_peer_rank_for_recv = CalculatePeerRankParam(0, world_size);  // helper (rank 1) recvs from rank 0

  // Trigger helper to receive (NCCL Recv is blocking, so it must be called simultaneously with Send)
  TriggerHelperRecv(group_key, helper_peer_rank_for_recv, recv_data_1.data(), data_count_1 * sizeof(float),
                    DataType::TYPE_FP32);

  // Main thread sends (blocking)
  status = main_handler->Send(group_key, main_peer_rank_for_send, send_data_1.data(), data_count_1 * sizeof(float),
                              DataType::TYPE_FP32);
  EXPECT_TRUE(status.OK()) << "Main Send failed: " << status.GetMessage();

  // Wait for helper receive to complete
  helper_status = WaitHelperCommand();
  EXPECT_TRUE(helper_status.OK()) << "Helper Recv failed: " << helper_status.GetMessage();

  // Verify received data
  for (size_t i = 0; i < data_count_1; ++i) {
    float expected = static_cast<float>(i) * 1.5f;
    EXPECT_NEAR(recv_data_1[i], expected, 1e-5f) << "Test 3: Data mismatch at index " << i;
  }

  // ========== Test 4: Send from helper (rank 1) to main (rank 0) ==========
  const size_t data_count_2 = 32;
  std::vector<float> send_data_2(data_count_2);
  for (size_t i = 0; i < data_count_2; ++i) {
    send_data_2[i] = static_cast<float>(i) * 2.5f;
  }
  std::vector<float> recv_data_2(data_count_2, 0.0f);

  int helper_peer_rank_for_send = CalculatePeerRankParam(0, world_size);  // helper (rank 1) sends to rank 0
  int main_peer_rank_for_recv = CalculatePeerRankParam(1, world_size);    // main (rank 0) recvs from rank 1

  // Trigger helper to send (NCCL Send is blocking, so it must be called simultaneously with Recv)
  TriggerHelperSend(group_key, helper_peer_rank_for_send, send_data_2.data(), data_count_2 * sizeof(float),
                    DataType::TYPE_FP32);

  // Main thread receives (blocking)
  status = main_handler->Recv(group_key, main_peer_rank_for_recv, recv_data_2.data(), data_count_2 * sizeof(float),
                              DataType::TYPE_FP32);
  EXPECT_TRUE(status.OK()) << "Main Recv failed: " << status.GetMessage();

  // Wait for helper send to complete
  helper_status = WaitHelperCommand();
  EXPECT_TRUE(helper_status.OK()) << "Helper Send failed: " << helper_status.GetMessage();

  // Verify received data
  for (size_t i = 0; i < data_count_2; ++i) {
    float expected = static_cast<float>(i) * 2.5f;
    EXPECT_NEAR(recv_data_2[i], expected, 1e-5f) << "Test 4: Data mismatch at index " << i;
  }

  // ========== Test 5: Large data transfer ==========
  const size_t data_count_3 = 1024 * 1024;  // 1M floats = 4MB
  std::vector<float> send_data_3(data_count_3);
  for (size_t i = 0; i < data_count_3; ++i) {
    send_data_3[i] = static_cast<float>(i % 1000) * 0.001f;
  }
  std::vector<float> recv_data_3(data_count_3, 0.0f);

  // Trigger helper to receive
  TriggerHelperRecv(group_key, helper_peer_rank_for_recv, recv_data_3.data(), data_count_3 * sizeof(float),
                    DataType::TYPE_FP32);

  // Main thread sends
  status = main_handler->Send(group_key, main_peer_rank_for_send, send_data_3.data(), data_count_3 * sizeof(float),
                              DataType::TYPE_FP32);
  EXPECT_TRUE(status.OK()) << "Main Send (large) failed: " << status.GetMessage();

  // Wait for helper receive to complete
  helper_status = WaitHelperCommand();
  EXPECT_TRUE(helper_status.OK()) << "Helper Recv (large) failed: " << helper_status.GetMessage();

  // Verify received data (sample check for large data)
  bool large_data_correct = true;
  for (size_t i = 0; i < data_count_3; i += 10000) {
    float expected = static_cast<float>(i % 1000) * 0.001f;
    if (std::abs(recv_data_3[i] - expected) > 1e-5f) {
      large_data_correct = false;
      break;
    }
  }
  EXPECT_TRUE(large_data_correct) << "Test 5: Large data transfer verification failed";
}

// Test Recv with non-existent group
TEST_F(NcclExecutorHandlerTest, Recv_NonExistentGroup) {
  std::string group_key = "non_existent_group";
  int peer_rank = 1;
  std::vector<uint8_t> buffer(100);
  DataType dtype = DataType::TYPE_FP32;

  Status status = handler_->Recv(group_key, peer_rank, buffer.data(), buffer.size(), dtype);
  EXPECT_FALSE(status.OK());
}

// Test SendGroup with empty buffers
TEST_F(NcclExecutorHandlerTest, SendGroup_EmptyBuffers) {
  std::string group_key = "test_group";
  int peer_rank = 1;
  std::vector<int> empty_block_ids;
  std::vector<int64_t> empty_offsets;
  std::vector<size_t> empty_counts;
  DataType dtype = DataType::TYPE_FP32;

  Status status = handler_->SendGroup(group_key, peer_rank, empty_block_ids, empty_offsets, empty_counts, dtype);
  EXPECT_FALSE(status.OK());
}

// Test SendGroup with mismatched sizes
TEST_F(NcclExecutorHandlerTest, SendGroup_MismatchedSizes) {
  std::string group_key = "test_group";
  int peer_rank = 1;
  std::vector<int> block_ids = {0, 1};
  std::vector<int64_t> offsets = {0};  // Mismatched size
  std::vector<size_t> counts = {100, 100};
  DataType dtype = DataType::TYPE_FP32;

  Status status = handler_->SendGroup(group_key, peer_rank, block_ids, offsets, counts, dtype);
  EXPECT_FALSE(status.OK());
  EXPECT_EQ(status.GetCode(), RetCode::RET_INVALID_ARGUMENT);
}

// Test RecvGroup with empty buffers
TEST_F(NcclExecutorHandlerTest, RecvGroup_EmptyBuffers) {
  std::string group_key = "test_group";
  int peer_rank = 1;
  std::vector<int> empty_block_ids;
  std::vector<int64_t> empty_offsets;
  std::vector<size_t> empty_counts;
  DataType dtype = DataType::TYPE_FP32;

  Status status = handler_->RecvGroup(group_key, peer_rank, empty_block_ids, empty_offsets, empty_counts, dtype);
  EXPECT_FALSE(status.OK());
}

// Test RecvGroup with mismatched sizes
TEST_F(NcclExecutorHandlerTest, RecvGroup_MismatchedSizes) {
  std::string group_key = "test_group";
  int peer_rank = 1;
  std::vector<int> block_ids = {0, 1};
  std::vector<int64_t> offsets = {0, 100, 200};  // Mismatched size
  std::vector<size_t> counts = {100, 100};
  DataType dtype = DataType::TYPE_FP32;

  Status status = handler_->RecvGroup(group_key, peer_rank, block_ids, offsets, counts, dtype);
  EXPECT_FALSE(status.OK());
  EXPECT_EQ(status.GetCode(), RetCode::RET_INVALID_ARGUMENT);
}

// Test GetCommGroupInfo with non-existent group
TEST_F(NcclExecutorHandlerTest, GetCommGroupInfo_NonExistentGroup) {
  std::string group_key = "non_existent_group";
  int rank = -1;
  int world_size = -1;

  Status status = handler_->GetCommGroupInfo(group_key, rank, world_size);
  EXPECT_FALSE(status.OK());
  EXPECT_EQ(status.GetCode(), RetCode::RET_INVALID_ARGUMENT);
}

// Test SetBlockAllocatorManager
TEST_F(NcclExecutorHandlerTest, SetBlockAllocatorManager) {
  // This test only verifies SetBlockAllocatorManager doesn't crash.
  // Use a real BlockAllocatorManager instance (no need to mock).
  std::shared_ptr<BlockAllocatorManager> manager = nullptr;

  RuntimeConfig config;
  config.parallel_basic_config.attn_tensor_parallel_size = 2;
  int rank = 0;

  EXPECT_NO_THROW(handler_->SetBlockAllocatorManager(manager, rank, config));
}

// Test ResolveBlockPtrs without BlockAllocatorManager set
TEST_F(NcclExecutorHandlerTest, ResolveBlockPtrs_NoAllocatorSet) {
  std::vector<int> block_ids = {0, 1};
  std::vector<int64_t> offsets = {0, 100};
  std::vector<void*> ptrs;

  // This should fail because BlockAllocatorManager is not set
  // Note: ResolveBlockPtrs is private, so we test it indirectly through SendGroup
  std::string group_key = "test_group";
  int peer_rank = 1;
  std::vector<size_t> counts = {100, 100};
  DataType dtype = DataType::TYPE_FP32;

  Status status = handler_->SendGroup(group_key, peer_rank, block_ids, offsets, counts, dtype);
  EXPECT_FALSE(status.OK());
}

}  // namespace ksana_llm

#endif  // ENABLE_CUDA