/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ipc/core/ipc_process_channel_manager.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstring>
#include <thread>

#include "gtest/gtest.h"
#include "ipc/core/ipc_func_types.h"
#include "tests/test.h"
#include "utils/logger.h"

namespace ksana_llm {

// ============================================================================
// Test Helper Classes and Utility Functions
// ============================================================================

// Test helper class: manages test environment
class IpcProcessChannelManagerTestHelper {
 public:
  // Generate unique shared memory name
  static std::string GenerateUniqueShmName() {
    static int counter = 0;
    return "/ksana_test_shm_" + std::to_string(getpid()) + "_" + std::to_string(counter++);
  }

  // Cleanup shared memory
  static void CleanupSharedMemory(const std::string& shm_name) { shm_unlink(shm_name.c_str()); }

  // Set environment variable
  static void SetShmNameEnv(const std::string& shm_name) {
    setenv("KSANA_IPC_NAME", shm_name.c_str() + 1, 1);  // Skip leading '/'
  }

  // Clear environment variable
  static void UnsetShmNameEnv() { unsetenv("KSANA_IPC_NAME"); }

  // Wait for child process to complete
  static bool WaitForChild(pid_t pid, int timeout_ms = 5000) {
    auto start = std::chrono::steady_clock::now();
    while (true) {
      int status;
      pid_t result = waitpid(pid, &status, WNOHANG);
      if (result == pid) {
        return WIFEXITED(status) && WEXITSTATUS(status) == 0;
      }

      auto now = std::chrono::steady_clock::now();
      auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
      if (elapsed > timeout_ms) {
        kill(pid, SIGKILL);
        waitpid(pid, &status, 0);
        return false;
      }

      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
};

// ============================================================================
// Basic Functionality Tests
// ============================================================================

class IpcProcessChannelManagerBasicTest : public ::testing::Test {
 protected:
  void SetUp() override {
    shm_name_ = IpcProcessChannelManagerTestHelper::GenerateUniqueShmName();
    IpcProcessChannelManagerTestHelper::SetShmNameEnv(shm_name_);
  }

  void TearDown() override {
    IpcProcessChannelManagerTestHelper::CleanupSharedMemory(shm_name_);
    IpcProcessChannelManagerTestHelper::UnsetShmNameEnv();
  }

  std::string shm_name_;
};

// Test: Constructor parameter validation - invalid executor_num
TEST_F(IpcProcessChannelManagerBasicTest, ConstructorInvalidExecutorNumZero) {
  EXPECT_THROW({ IpcProcessChannelManager manager(0, 0, true); }, std::exception);
}

// Test: Constructor parameter validation - executor_num exceeds maximum
TEST_F(IpcProcessChannelManagerBasicTest, ConstructorInvalidExecutorNumTooLarge) {
  EXPECT_THROW({ IpcProcessChannelManager manager(64, 0, true); }, std::exception);
}

// Test: Constructor parameter validation - invalid executor_id (negative)
TEST_F(IpcProcessChannelManagerBasicTest, ConstructorInvalidExecutorIdNegative) {
  EXPECT_THROW({ IpcProcessChannelManager manager(4, -1, false); }, std::exception);
}

// Test: Constructor parameter validation - executor_id out of range
TEST_F(IpcProcessChannelManagerBasicTest, ConstructorInvalidExecutorIdOutOfRange) {
  EXPECT_THROW({ IpcProcessChannelManager manager(4, 4, false); }, std::exception);
}

// Test: Normal construction and initialization in Engine mode
TEST_F(IpcProcessChannelManagerBasicTest, EngineConstructorAndInitialize) {
  EXPECT_NO_THROW({
    IpcProcessChannelManager manager(4, 0, true);
    manager.Initialize();
  });
}

// Test: Resource cleanup in Engine mode
TEST_F(IpcProcessChannelManagerBasicTest, EngineResourceCleanup) {
  {
    IpcProcessChannelManager manager(2, 0, true);
    manager.Initialize();
    // Destructor should properly clean up resources
  }

  // Verify shared memory has been cleaned up (attempt to open should fail)
  int fd = shm_open(shm_name_.c_str(), O_RDONLY, 0);
  EXPECT_EQ(fd, -1);
  if (fd >= 0) {
    close(fd);
  }
}

// Test: Get IPC buffer size calculation
TEST_F(IpcProcessChannelManagerBasicTest, GetIpcBufferSizeCalculation) {
  IpcProcessChannelManager manager(4, 0, true);

  // Verify buffer size through reflection or indirect method
  // Expected size = (executor_num + 1) * 2 * (CHANNEL_META_SIZE + CHANNEL_DATA_SIZE * CHANNEL_SLOT_NUM)
  size_t expected_size = (4 + 1) * 2 * (CHANNEL_META_SIZE + CHANNEL_DATA_SIZE * CHANNEL_SLOT_NUM);

  EXPECT_NO_THROW({ manager.Initialize(); });
}

// Test: UnlinkIpcResource method
TEST_F(IpcProcessChannelManagerBasicTest, UnlinkIpcResource) {
  IpcProcessChannelManager manager(2, 0, true);
  manager.Initialize();

  // First unlink should succeed
  EXPECT_NO_THROW({ manager.UnlinkIpcResource(); });

  // Second unlink should fail silently (no exception thrown)
  EXPECT_NO_THROW({ manager.UnlinkIpcResource(); });
}

// ============================================================================
// Channel Access Tests
// ============================================================================

class IpcProcessChannelManagerChannelAccessTest : public ::testing::Test {
 protected:
  void SetUp() override {
    shm_name_ = IpcProcessChannelManagerTestHelper::GenerateUniqueShmName();
    IpcProcessChannelManagerTestHelper::SetShmNameEnv(shm_name_);
  }

  void TearDown() override {
    IpcProcessChannelManagerTestHelper::CleanupSharedMemory(shm_name_);
    IpcProcessChannelManagerTestHelper::UnsetShmNameEnv();
  }

  std::string shm_name_;
};

// Test: Get all channels in Engine mode
TEST_F(IpcProcessChannelManagerChannelAccessTest, EngineGetAllChannels) {
  IpcProcessChannelManager manager(3, 0, true);
  manager.Initialize();

  // Get all engine2executor channels
  for (int i = 0; i < 3; ++i) {
    EXPECT_NO_THROW({
      auto& channel = manager.GetEngine2ExecutorChannel(i);
      EXPECT_NE(channel, nullptr);
    });
  }

  // Get all executor2engine channels
  for (int i = 0; i < 3; ++i) {
    EXPECT_NO_THROW({
      auto& channel = manager.GetExecutor2EngineChannel(i);
      EXPECT_NE(channel, nullptr);
    });
  }

  // Get broadcast channel
  EXPECT_NO_THROW({
    auto& channel = manager.GetBroadcastChannel();
    EXPECT_NE(channel, nullptr);
  });
}

// Test: Executor can only get its own channel
TEST_F(IpcProcessChannelManagerChannelAccessTest, ExecutorGetOwnChannelOnly) {
  // Create Engine first
  IpcProcessChannelManager engine_manager(3, 0, true);
  engine_manager.Initialize();

  // Create Executor
  IpcProcessChannelManager executor_manager(3, 1, false);
  executor_manager.Initialize();

  // Getting own channel should succeed
  EXPECT_NO_THROW({
    auto& channel = executor_manager.GetEngine2ExecutorChannel(1);
    EXPECT_NE(channel, nullptr);
  });

  EXPECT_NO_THROW({
    auto& channel = executor_manager.GetExecutor2EngineChannel(1);
    EXPECT_NE(channel, nullptr);
  });

  // Getting other executor's channel should fail
  EXPECT_THROW({ auto& channel = executor_manager.GetEngine2ExecutorChannel(0); }, std::exception);

  EXPECT_THROW({ auto& channel = executor_manager.GetExecutor2EngineChannel(2); }, std::exception);
}

// Test: Invalid device_id access
TEST_F(IpcProcessChannelManagerChannelAccessTest, InvalidDeviceIdAccess) {
  IpcProcessChannelManager manager(3, 0, true);
  manager.Initialize();

  // Negative device_id
  EXPECT_THROW({ auto& channel = manager.GetEngine2ExecutorChannel(-1); }, std::exception);

  // Out of range device_id
  EXPECT_THROW({ auto& channel = manager.GetEngine2ExecutorChannel(3); }, std::exception);

  EXPECT_THROW({ auto& channel = manager.GetExecutor2EngineChannel(10); }, std::exception);
}

// Test: Broadcast channel access
TEST_F(IpcProcessChannelManagerChannelAccessTest, BroadcastChannelAccess) {
  IpcProcessChannelManager manager(4, 0, true);
  manager.Initialize();

  auto& broadcast_channel = manager.GetBroadcastChannel();
  EXPECT_NE(broadcast_channel, nullptr);

  // Multiple gets should return the same channel
  auto& broadcast_channel2 = manager.GetBroadcastChannel();
  EXPECT_EQ(&broadcast_channel, &broadcast_channel2);
}

// ============================================================================
// Multi-Process Communication Tests
// ============================================================================

class IpcProcessChannelManagerMultiProcessTest : public ::testing::Test {
 protected:
  void SetUp() override {
    shm_name_ = IpcProcessChannelManagerTestHelper::GenerateUniqueShmName();
    IpcProcessChannelManagerTestHelper::SetShmNameEnv(shm_name_);
  }

  void TearDown() override {
    IpcProcessChannelManagerTestHelper::CleanupSharedMemory(shm_name_);
    IpcProcessChannelManagerTestHelper::UnsetShmNameEnv();
  }

  std::string shm_name_;
};

// Test: Communication between Engine and single Executor
TEST_F(IpcProcessChannelManagerMultiProcessTest, EngineExecutorCommunication) {
  const size_t executor_num = 1;
  const int executor_id = 0;

  pid_t pid = fork();
  ASSERT_NE(pid, -1) << "Fork failed";

  if (pid == 0) {
    // Child process: Executor
    try {
      // Wait for Engine to create shared memory
      std::this_thread::sleep_for(std::chrono::milliseconds(100));

      IpcProcessChannelManager executor_manager(executor_num, executor_id, false);
      executor_manager.Initialize();

      auto& recv_channel = executor_manager.GetEngine2ExecutorChannel(executor_id);
      auto& send_channel = executor_manager.GetExecutor2EngineChannel(executor_id);

      // Receive message from Engine
      if (recv_channel->WaitReadableAndLock()) {
        void* read_ptr = recv_channel->GetReadPtr();
        int received_value = *static_cast<int*>(read_ptr);
        EXPECT_EQ(received_value, 12345);
        recv_channel->ReadAndUnlock();
      }

      // Send response to Engine
      if (send_channel->WaitWritableAndLock()) {
        void* write_ptr = send_channel->GetWritePtr();
        *static_cast<int*>(write_ptr) = 54321;
        send_channel->WriteAndUnlock();
      }

      exit(0);
    } catch (const std::exception& e) {
      KLLM_LOG_ERROR << "Executor error: " << e.what();
      exit(1);
    }
  } else {
    // Parent process: Engine
    IpcProcessChannelManager engine_manager(executor_num, 0, true);
    engine_manager.Initialize();

    auto& send_channel = engine_manager.GetEngine2ExecutorChannel(executor_id);
    auto& recv_channel = engine_manager.GetExecutor2EngineChannel(executor_id);

    // Send message to Executor
    if (send_channel->WaitWritableAndLock()) {
      void* write_ptr = send_channel->GetWritePtr();
      *static_cast<int*>(write_ptr) = 12345;
      send_channel->WriteAndUnlock();
    }

    // Receive response from Executor
    if (recv_channel->WaitReadableAndLock()) {
      void* read_ptr = recv_channel->GetReadPtr();
      int received_value = *static_cast<int*>(read_ptr);
      EXPECT_EQ(received_value, 54321);
      recv_channel->ReadAndUnlock();
    }

    // Wait for child process to complete
    bool success = IpcProcessChannelManagerTestHelper::WaitForChild(pid);
    EXPECT_TRUE(success);
  }
}

// Test: Communication between Engine and multiple Executors
TEST_F(IpcProcessChannelManagerMultiProcessTest, EngineMultipleExecutorsCommunication) {
  const size_t executor_num = 3;
  std::vector<pid_t> child_pids;

  // Create Engine
  IpcProcessChannelManager engine_manager(executor_num, 0, true);
  engine_manager.Initialize();

  // Create multiple Executor child processes
  for (size_t i = 0; i < executor_num; ++i) {
    pid_t pid = fork();
    ASSERT_NE(pid, -1) << "Fork failed";

    if (pid == 0) {
      // Child process: Executor
      try {
        int executor_id = i;
        IpcProcessChannelManager executor_manager(executor_num, executor_id, false);
        executor_manager.Initialize();

        auto& recv_channel = executor_manager.GetEngine2ExecutorChannel(executor_id);
        auto& send_channel = executor_manager.GetExecutor2EngineChannel(executor_id);

        // Receive message from Engine
        if (recv_channel->WaitReadableAndLock()) {
          void* read_ptr = recv_channel->GetReadPtr();
          int received_value = *static_cast<int*>(read_ptr);
          EXPECT_EQ(received_value, 1000 + executor_id);
          recv_channel->ReadAndUnlock();
        }

        // Send response to Engine
        if (send_channel->WaitWritableAndLock()) {
          void* write_ptr = send_channel->GetWritePtr();
          *static_cast<int*>(write_ptr) = 2000 + executor_id;
          send_channel->WriteAndUnlock();
        }

        exit(0);
      } catch (const std::exception& e) {
        KLLM_LOG_ERROR << "Executor " << i << " error: " << e.what();
        exit(1);
      }
    } else {
      child_pids.push_back(pid);
    }
  }

  // Engine sends message to each Executor
  for (size_t i = 0; i < executor_num; ++i) {
    auto& send_channel = engine_manager.GetEngine2ExecutorChannel(i);
    if (send_channel->WaitWritableAndLock()) {
      void* write_ptr = send_channel->GetWritePtr();
      *static_cast<int*>(write_ptr) = 1000 + i;
      send_channel->WriteAndUnlock();
    }
  }

  // Engine receives response from each Executor
  for (size_t i = 0; i < executor_num; ++i) {
    auto& recv_channel = engine_manager.GetExecutor2EngineChannel(i);
    if (recv_channel->WaitReadableAndLock()) {
      void* read_ptr = recv_channel->GetReadPtr();
      int received_value = *static_cast<int*>(read_ptr);
      EXPECT_EQ(received_value, 2000 + i);
      recv_channel->ReadAndUnlock();
    }
  }

  // Wait for all child processes to complete
  for (pid_t pid : child_pids) {
    bool success = IpcProcessChannelManagerTestHelper::WaitForChild(pid);
    EXPECT_TRUE(success);
  }
}

// Test: Broadcast channel communication
TEST_F(IpcProcessChannelManagerMultiProcessTest, BroadcastCommunication) {
  const size_t executor_num = 2;
  std::vector<pid_t> child_pids;

  // Create Engine
  IpcProcessChannelManager engine_manager(executor_num, 0, true);
  engine_manager.Initialize();

  // Create multiple Executor child processes
  for (size_t i = 0; i < executor_num; ++i) {
    pid_t pid = fork();
    ASSERT_NE(pid, -1) << "Fork failed";

    if (pid == 0) {
      // Child process: Executor
      try {
        int executor_id = i;
        IpcProcessChannelManager executor_manager(executor_num, executor_id, false);
        executor_manager.Initialize();

        auto& broadcast_channel = executor_manager.GetBroadcastChannel();

        // Receive broadcast message
        if (broadcast_channel->WaitReadableAndLock()) {
          void* read_ptr = broadcast_channel->GetReadPtr();
          int received_value = *static_cast<int*>(read_ptr);
          EXPECT_EQ(received_value, 9999);
          broadcast_channel->ReadAndUnlock();
        }

        exit(0);
      } catch (const std::exception& e) {
        KLLM_LOG_ERROR << "Executor " << i << " error: " << e.what();
        exit(1);
      }
    } else {
      child_pids.push_back(pid);
    }
  }

  // Wait for all Executors to start
  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  // Engine sends broadcast message
  auto& broadcast_channel = engine_manager.GetBroadcastChannel();
  if (broadcast_channel->WaitWritableAndLock()) {
    void* write_ptr = broadcast_channel->GetWritePtr();
    *static_cast<int*>(write_ptr) = 9999;
    broadcast_channel->WriteAndUnlock();
  }

  // Wait for all child processes to complete
  for (pid_t pid : child_pids) {
    bool success = IpcProcessChannelManagerTestHelper::WaitForChild(pid);
    EXPECT_TRUE(success);
  }
}

// ============================================================================
// Exception Handling Tests
// ============================================================================

class IpcProcessChannelManagerExceptionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    shm_name_ = IpcProcessChannelManagerTestHelper::GenerateUniqueShmName();
    IpcProcessChannelManagerTestHelper::SetShmNameEnv(shm_name_);
  }

  void TearDown() override {
    IpcProcessChannelManagerTestHelper::CleanupSharedMemory(shm_name_);
    IpcProcessChannelManagerTestHelper::UnsetShmNameEnv();
  }

  std::string shm_name_;
};

// Test: Executor attempts to connect when Engine hasn't created shared memory
TEST_F(IpcProcessChannelManagerExceptionTest, ExecutorConnectWithoutEngine) {
  EXPECT_THROW(
      {
        IpcProcessChannelManager executor_manager(2, 0, false);
        executor_manager.Initialize();
      },
      std::exception);
}

// Test: Engine recreates shared memory
TEST_F(IpcProcessChannelManagerExceptionTest, EngineRecreateSharedMemory) {
  {
    IpcProcessChannelManager manager1(2, 0, true);
    manager1.Initialize();
  }

  // After first manager is destructed, should be able to create new one
  EXPECT_NO_THROW({
    IpcProcessChannelManager manager2(2, 0, true);
    manager2.Initialize();
  });
}

// ============================================================================
// Boundary Condition Tests
// ============================================================================

class IpcProcessChannelManagerBoundaryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    shm_name_ = IpcProcessChannelManagerTestHelper::GenerateUniqueShmName();
    IpcProcessChannelManagerTestHelper::SetShmNameEnv(shm_name_);
  }

  void TearDown() override {
    IpcProcessChannelManagerTestHelper::CleanupSharedMemory(shm_name_);
    IpcProcessChannelManagerTestHelper::UnsetShmNameEnv();
  }

  std::string shm_name_;
};

// Test: Minimum executor number (1)
TEST_F(IpcProcessChannelManagerBoundaryTest, MinimumExecutorNum) {
  EXPECT_NO_THROW({
    IpcProcessChannelManager manager(1, 0, true);
    manager.Initialize();

    auto& channel = manager.GetEngine2ExecutorChannel(0);
    EXPECT_NE(channel, nullptr);
  });
}

// Test: Maximum executor number (63)
TEST_F(IpcProcessChannelManagerBoundaryTest, MaximumExecutorNum) {
  EXPECT_NO_THROW({
    IpcProcessChannelManager manager(63, 0, true);
    manager.Initialize();

    // Verify all channels can be accessed
    for (int i = 0; i < 63; ++i) {
      auto& channel = manager.GetEngine2ExecutorChannel(i);
      EXPECT_NE(channel, nullptr);
    }
  });
}

// Test: Large data transfer
TEST_F(IpcProcessChannelManagerBoundaryTest, LargeDataTransfer) {
  const size_t executor_num = 1;
  const int executor_id = 0;

  pid_t pid = fork();
  ASSERT_NE(pid, -1) << "Fork failed";

  if (pid == 0) {
    // Child process: Executor
    try {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));

      IpcProcessChannelManager executor_manager(executor_num, executor_id, false);
      executor_manager.Initialize();

      auto& recv_channel = executor_manager.GetEngine2ExecutorChannel(executor_id);

      // Receive large amount of data
      if (recv_channel->WaitReadableAndLock()) {
        void* read_ptr = recv_channel->GetReadPtr();
        std::vector<uint8_t> received_data(CHANNEL_DATA_SIZE);
        std::memcpy(received_data.data(), read_ptr, CHANNEL_DATA_SIZE);

        // Verify data integrity
        for (size_t i = 0; i < CHANNEL_DATA_SIZE; ++i) {
          if (received_data[i] != static_cast<uint8_t>(i % 256)) {
            exit(1);
          }
        }

        recv_channel->ReadAndUnlock();
      }

      exit(0);
    } catch (const std::exception& e) {
      KLLM_LOG_ERROR << "Executor error: " << e.what();
      exit(1);
    }
  } else {
    // Parent process: Engine
    IpcProcessChannelManager engine_manager(executor_num, 0, true);
    engine_manager.Initialize();

    auto& send_channel = engine_manager.GetEngine2ExecutorChannel(executor_id);

    // Send large amount of data
    if (send_channel->WaitWritableAndLock()) {
      void* write_ptr = send_channel->GetWritePtr();
      std::vector<uint8_t> send_data(CHANNEL_DATA_SIZE);
      for (size_t i = 0; i < CHANNEL_DATA_SIZE; ++i) {
        send_data[i] = static_cast<uint8_t>(i % 256);
      }
      std::memcpy(write_ptr, send_data.data(), CHANNEL_DATA_SIZE);
      send_channel->WriteAndUnlock();
    }

    // Wait for child process to complete
    bool success = IpcProcessChannelManagerTestHelper::WaitForChild(pid);
    EXPECT_TRUE(success);
  }
}

// Test: Custom shared memory name
TEST_F(IpcProcessChannelManagerBoundaryTest, CustomSharedMemoryName) {
  std::string custom_name = "custom_test_shm_" + std::to_string(getpid());
  IpcProcessChannelManagerTestHelper::SetShmNameEnv(custom_name);

  EXPECT_NO_THROW({
    IpcProcessChannelManager manager(2, 0, true);
    manager.Initialize();
  });

  // Cleanup custom shared memory
  shm_unlink(("/" + custom_name).c_str());
}

// ============================================================================
// Performance Tests
// ============================================================================

class IpcProcessChannelManagerPerformanceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    shm_name_ = IpcProcessChannelManagerTestHelper::GenerateUniqueShmName();
    IpcProcessChannelManagerTestHelper::SetShmNameEnv(shm_name_);
  }

  void TearDown() override {
    IpcProcessChannelManagerTestHelper::CleanupSharedMemory(shm_name_);
    IpcProcessChannelManagerTestHelper::UnsetShmNameEnv();
  }

  std::string shm_name_;
};

// Test: Initialization performance
TEST_F(IpcProcessChannelManagerPerformanceTest, InitializationPerformance) {
  auto start = std::chrono::high_resolution_clock::now();

  IpcProcessChannelManager manager(8, 0, true);
  manager.Initialize();

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

  // Initialization should complete within reasonable time (e.g., < 1000ms)
  EXPECT_LT(duration, 1000);
}

// Test: Channel access performance
TEST_F(IpcProcessChannelManagerPerformanceTest, ChannelAccessPerformance) {
  IpcProcessChannelManager manager(16, 0, true);
  manager.Initialize();

  const int iterations = 10000;
  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < iterations; ++i) {
    for (int j = 0; j < 16; ++j) {
      auto& channel = manager.GetEngine2ExecutorChannel(j);
      (void)channel;  // Avoid unused variable warning
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  // Channel access should be very fast (e.g., < 1us)
  double avg_time = static_cast<double>(duration) / (iterations * 16);
  EXPECT_LT(avg_time, 1.0);
}

// Test: Multi-process concurrent communication throughput
TEST_F(IpcProcessChannelManagerPerformanceTest, ConcurrentCommunicationThroughput) {
  const size_t executor_num = 4;
  const int message_count = 100;
  std::vector<pid_t> child_pids;

  // Create Engine
  IpcProcessChannelManager engine_manager(executor_num, 0, true);
  engine_manager.Initialize();

  auto start = std::chrono::high_resolution_clock::now();

  // Create multiple Executor child processes
  for (size_t i = 0; i < executor_num; ++i) {
    pid_t pid = fork();
    ASSERT_NE(pid, -1) << "Fork failed";

    if (pid == 0) {
      // Child process: Executor
      try {
        int executor_id = i;
        IpcProcessChannelManager executor_manager(executor_num, executor_id, false);
        executor_manager.Initialize();

        auto& recv_channel = executor_manager.GetEngine2ExecutorChannel(executor_id);
        auto& send_channel = executor_manager.GetExecutor2EngineChannel(executor_id);

        // Receive and respond to multiple messages
        for (int j = 0; j < message_count; ++j) {
          if (recv_channel->WaitReadableAndLock()) {
            void* read_ptr = recv_channel->GetReadPtr();
            int received_value = *static_cast<int*>(read_ptr);
            recv_channel->ReadAndUnlock();

            if (send_channel->WaitWritableAndLock()) {
              void* write_ptr = send_channel->GetWritePtr();
              *static_cast<int*>(write_ptr) = received_value + 1;
              send_channel->WriteAndUnlock();
            }
          }
        }

        exit(0);
      } catch (const std::exception& e) {
        KLLM_LOG_ERROR << "Executor " << i << " error: " << e.what();
        exit(1);
      }
    } else {
      child_pids.push_back(pid);
    }
  }

  // Engine sends and receives messages
  for (int j = 0; j < message_count; ++j) {
    for (size_t i = 0; i < executor_num; ++i) {
      auto& send_channel = engine_manager.GetEngine2ExecutorChannel(i);
      if (send_channel->WaitWritableAndLock()) {
        void* write_ptr = send_channel->GetWritePtr();
        *static_cast<int*>(write_ptr) = j;
        send_channel->WriteAndUnlock();
      }
    }

    for (size_t i = 0; i < executor_num; ++i) {
      auto& recv_channel = engine_manager.GetExecutor2EngineChannel(i);
      if (recv_channel->WaitReadableAndLock()) {
        void* read_ptr = recv_channel->GetReadPtr();
        int received_value = *static_cast<int*>(read_ptr);
        EXPECT_EQ(received_value, j + 1);
        recv_channel->ReadAndUnlock();
      }
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

  int total_messages = message_count * executor_num * 2;  // Bidirectional communication
  double throughput = static_cast<double>(total_messages) / (duration / 1000.0);

  // Wait for all child processes to complete
  for (pid_t pid : child_pids) {
    bool success = IpcProcessChannelManagerTestHelper::WaitForChild(pid);
    EXPECT_TRUE(success);
  }
}

}  // namespace ksana_llm
