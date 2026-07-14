/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ipc/core/ipc_process_channel.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

#include "tests/test.h"
#include "utils/logger.h"

namespace ksana_llm {

// Helper class, manage shared memory.
class SharedMemoryHelper {
 public:
  SharedMemoryHelper(size_t meta_size, size_t data_size) : meta_size_(meta_size), data_size_(data_size) {
    total_size_ = meta_size_ + data_size_;

    // Create anonymous shared memory.
    meta_ptr_ =
        static_cast<uint8_t*>(mmap(nullptr, total_size_, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0));

    if (meta_ptr_ == MAP_FAILED) {
      throw std::runtime_error("Failed to create shared memory");
    }

    data_ptr_ = meta_ptr_ + meta_size_;

    // Reset memory to 0.
    memset(meta_ptr_, 0, total_size_);
  }

  ~SharedMemoryHelper() {
    if (meta_ptr_ != nullptr && meta_ptr_ != MAP_FAILED) {
      munmap(meta_ptr_, total_size_);
    }
  }

  uint8_t* GetMetaPtr() { return meta_ptr_; }
  uint8_t* GetDataPtr() { return data_ptr_; }

 private:
  size_t meta_size_;
  size_t data_size_;
  size_t total_size_;
  uint8_t* meta_ptr_ = nullptr;
  uint8_t* data_ptr_ = nullptr;
};

class IpcProcessChannelBasicTest : public ::testing::Test {
 protected:
  void SetUp() override {
    shm_helper_ = std::make_unique<SharedMemoryHelper>(CHANNEL_META_SIZE, CHANNEL_DATA_SIZE * CHANNEL_SLOT_NUM);
  }

  void TearDown() override { shm_helper_.reset(); }

  std::unique_ptr<SharedMemoryHelper> shm_helper_;
};

TEST_F(IpcProcessChannelBasicTest, ConstructorEngineMode) {
  ASSERT_NO_THROW(
      { IpcProcessChannel channel(shm_helper_->GetMetaPtr(), shm_helper_->GetDataPtr(), 2, 0, false, true); });
}

TEST_F(IpcProcessChannelBasicTest, ConstructorExecutorMode) {
  { IpcProcessChannel engine_channel(shm_helper_->GetMetaPtr(), shm_helper_->GetDataPtr(), 2, 0, false, true); }

  ASSERT_NO_THROW({
    IpcProcessChannel executor_channel(shm_helper_->GetMetaPtr(), shm_helper_->GetDataPtr(), 2, 1, false, false);
  });
}

TEST_F(IpcProcessChannelBasicTest, ConstructorTooManyExecutors) {
  EXPECT_THROW({ IpcProcessChannel channel(shm_helper_->GetMetaPtr(), shm_helper_->GetDataPtr(), 64, 0, false, true); },
               std::runtime_error);
}

TEST_F(IpcProcessChannelBasicTest, SharedMemoryInitialization) {
  IpcProcessChannel channel(shm_helper_->GetMetaPtr(), shm_helper_->GetDataPtr(), 2, 0, false, true);

  auto* shared_data = reinterpret_cast<IpcProcessChannel::SharedData*>(shm_helper_->GetMetaPtr());

  EXPECT_EQ(shared_data->used_slot_num, 0);
  EXPECT_FALSE(shared_data->stopped);
  EXPECT_EQ(shared_data->read_offset[0], 0);
  EXPECT_EQ(shared_data->write_offset, 0);
}

TEST_F(IpcProcessChannelBasicTest, GetPtrMethods) {
  IpcProcessChannel channel(shm_helper_->GetMetaPtr(), shm_helper_->GetDataPtr(), 2, 0, false, true);

  void* write_ptr = channel.GetWritePtr();
  void* read_ptr = channel.GetReadPtr();

  EXPECT_NE(write_ptr, nullptr);
  EXPECT_NE(read_ptr, nullptr);

  // Both write and read offset are 0.
  EXPECT_EQ(write_ptr, read_ptr);
}

class IpcProcessChannelSyncTest : public ::testing::Test {
 protected:
  void SetUp() override {
    shm_helper_ = std::make_unique<SharedMemoryHelper>(CHANNEL_META_SIZE, CHANNEL_DATA_SIZE * CHANNEL_SLOT_NUM);
  }

  void TearDown() override { shm_helper_.reset(); }

  std::unique_ptr<SharedMemoryHelper> shm_helper_;
};

TEST_F(IpcProcessChannelSyncTest, BasicWriteAndRead) {
  IpcProcessChannel channel(shm_helper_->GetMetaPtr(), shm_helper_->GetDataPtr(), 2, 0, false, true);

  EXPECT_TRUE(channel.WaitWritableAndLock());
  int* write_data = static_cast<int*>(channel.GetWritePtr());
  *write_data = 42;
  EXPECT_TRUE(channel.WriteAndUnlock());

  EXPECT_TRUE(channel.WaitReadableAndLock());
  int* read_data = static_cast<int*>(channel.GetReadPtr());
  EXPECT_EQ(*read_data, 42);
  EXPECT_TRUE(channel.ReadAndUnlock());
}

TEST_F(IpcProcessChannelSyncTest, MultipleWriteAndRead) {
  IpcProcessChannel channel(shm_helper_->GetMetaPtr(), shm_helper_->GetDataPtr(), 2, 0, false, true);

  const int test_count = CHANNEL_SLOT_NUM;

  for (int i = 0; i < test_count; ++i) {
    EXPECT_TRUE(channel.WaitWritableAndLock());
    int* write_data = static_cast<int*>(channel.GetWritePtr());
    *write_data = i * 10;
    EXPECT_TRUE(channel.WriteAndUnlock());
  }

  for (int i = 0; i < test_count; ++i) {
    EXPECT_TRUE(channel.WaitReadableAndLock());
    int* read_data = static_cast<int*>(channel.GetReadPtr());
    EXPECT_EQ(*read_data, i * 10);
    EXPECT_TRUE(channel.ReadAndUnlock());
  }
}

TEST_F(IpcProcessChannelSyncTest, StopChannel) {
  IpcProcessChannel channel(shm_helper_->GetMetaPtr(), shm_helper_->GetDataPtr(), 2, 0, false, true);

  std::atomic<bool> read_returned{false};
  std::thread reader([&]() {
    bool result = channel.WaitReadableAndLock();
    read_returned = true;
    EXPECT_FALSE(result);
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  channel.Stop();

  reader.join();
  EXPECT_TRUE(read_returned);
}

TEST_F(IpcProcessChannelSyncTest, WriteBlocking) {
  IpcProcessChannel channel(shm_helper_->GetMetaPtr(), shm_helper_->GetDataPtr(), 2, 0, false, true);

  for (int i = 0; i < CHANNEL_SLOT_NUM; ++i) {
    EXPECT_TRUE(channel.WaitWritableAndLock());
    EXPECT_TRUE(channel.WriteAndUnlock());
  }

  std::atomic<bool> write_blocked{true};
  std::thread writer([&]() {
    channel.WaitWritableAndLock();
    write_blocked = false;
    channel.WriteAndUnlock();
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  EXPECT_TRUE(write_blocked);

  EXPECT_TRUE(channel.WaitReadableAndLock());
  EXPECT_TRUE(channel.ReadAndUnlock());

  writer.join();
  EXPECT_FALSE(write_blocked);
}

class IpcProcessChannelBroadcastTest : public ::testing::Test {
 protected:
  void SetUp() override {
    shm_helper_ = std::make_unique<SharedMemoryHelper>(CHANNEL_META_SIZE, CHANNEL_DATA_SIZE * CHANNEL_SLOT_NUM);
  }

  void TearDown() override { shm_helper_.reset(); }

  std::unique_ptr<SharedMemoryHelper> shm_helper_;
};

TEST_F(IpcProcessChannelBroadcastTest, BroadcastToMultipleReaders) {
  const size_t executor_num = 3;

  IpcProcessChannel engine_channel(shm_helper_->GetMetaPtr(), shm_helper_->GetDataPtr(), executor_num, 0, true, true);

  EXPECT_TRUE(engine_channel.WaitWritableAndLock());
  int* write_data = static_cast<int*>(engine_channel.GetWritePtr());
  *write_data = 999;
  EXPECT_TRUE(engine_channel.WriteAndUnlock());

  std::vector<std::thread> readers;
  std::atomic<int> success_count{0};

  for (size_t i = 0; i < executor_num; ++i) {
    readers.emplace_back([&, i]() {
      IpcProcessChannel executor_channel(shm_helper_->GetMetaPtr(), shm_helper_->GetDataPtr(), executor_num, i, true,
                                         false);

      if (executor_channel.WaitReadableAndLock()) {
        int* read_data = static_cast<int*>(executor_channel.GetReadPtr());
        if (*read_data == 999) {
          success_count++;
        }
        executor_channel.ReadAndUnlock();
      }
    });
  }

  for (auto& reader : readers) {
    reader.join();
  }

  EXPECT_EQ(success_count, executor_num);
}

TEST_F(IpcProcessChannelBroadcastTest, BroadcastWaitAllReaders) {
  const size_t executor_num = 2;

  IpcProcessChannel engine_channel(shm_helper_->GetMetaPtr(), shm_helper_->GetDataPtr(), executor_num, 0, true, true);

  EXPECT_TRUE(engine_channel.WaitWritableAndLock());
  int* write_data = static_cast<int*>(engine_channel.GetWritePtr());
  *write_data = 100;
  EXPECT_TRUE(engine_channel.WriteAndUnlock());

  // Only one reader executor.
  IpcProcessChannel executor1(shm_helper_->GetMetaPtr(), shm_helper_->GetDataPtr(), executor_num, 0, true, false);
  EXPECT_TRUE(executor1.WaitReadableAndLock());
  EXPECT_TRUE(executor1.ReadAndUnlock());

  // Try the second write, because slot_num of process channel is large than 1, so the write should be succ.
  std::atomic<bool> second_write_done{false};
  std::thread writer([&]() {
    engine_channel.WaitWritableAndLock();
    second_write_done = true;
    engine_channel.WriteAndUnlock();
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  EXPECT_TRUE(second_write_done);

  // The second executor start read.
  IpcProcessChannel executor2(shm_helper_->GetMetaPtr(), shm_helper_->GetDataPtr(), executor_num, 1, true, false);
  EXPECT_TRUE(executor2.WaitReadableAndLock());
  EXPECT_TRUE(executor2.ReadAndUnlock());

  // Wait the second writing finished.
  writer.join();
  EXPECT_TRUE(second_write_done);
}

class IpcProcessChannelMultiProcessTest : public ::testing::Test {
 protected:
  void SetUp() override {
    shm_helper_ = std::make_unique<SharedMemoryHelper>(CHANNEL_META_SIZE, CHANNEL_DATA_SIZE * CHANNEL_SLOT_NUM);
  }

  void TearDown() override { shm_helper_.reset(); }

  std::unique_ptr<SharedMemoryHelper> shm_helper_;
};

TEST_F(IpcProcessChannelMultiProcessTest, MultiProcessWriteRead) {
  const int message_count = 10;

  IpcProcessChannel engine_channel(shm_helper_->GetMetaPtr(), shm_helper_->GetDataPtr(), 2, 0, false, true);

  pid_t pid = fork();

  if (pid == 0) {
    IpcProcessChannel writer_channel(shm_helper_->GetMetaPtr(), shm_helper_->GetDataPtr(), 2, 1, false, false);

    for (int i = 0; i < message_count; ++i) {
      if (writer_channel.WaitWritableAndLock()) {
        int* data = static_cast<int*>(writer_channel.GetWritePtr());
        *data = i;
        writer_channel.WriteAndUnlock();
      }
    }

    exit(0);
  } else if (pid > 0) {
    int success_count = 0;

    for (int i = 0; i < message_count; ++i) {
      if (engine_channel.WaitReadableAndLock()) {
        int* data = static_cast<int*>(engine_channel.GetReadPtr());
        if (*data == i) {
          success_count++;
        }
        engine_channel.ReadAndUnlock();
      }
    }

    int status;
    waitpid(pid, &status, 0);

    EXPECT_EQ(success_count, message_count);
    EXPECT_EQ(WEXITSTATUS(status), 0);
  } else {
    FAIL() << "Fork failed";
  }
}

class IpcProcessChannelBoundaryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    shm_helper_ = std::make_unique<SharedMemoryHelper>(CHANNEL_META_SIZE, CHANNEL_DATA_SIZE * CHANNEL_SLOT_NUM);
  }

  void TearDown() override { shm_helper_.reset(); }

  std::unique_ptr<SharedMemoryHelper> shm_helper_;
};

TEST_F(IpcProcessChannelBoundaryTest, ChannelFull) {
  IpcProcessChannel channel(shm_helper_->GetMetaPtr(), shm_helper_->GetDataPtr(), 2, 0, false, true);

  for (int i = 0; i < CHANNEL_SLOT_NUM; ++i) {
    EXPECT_TRUE(channel.WaitWritableAndLock());
    EXPECT_TRUE(channel.WriteAndUnlock());
  }

  auto* shared_data = reinterpret_cast<IpcProcessChannel::SharedData*>(shm_helper_->GetMetaPtr());
  EXPECT_EQ(shared_data->used_slot_num, CHANNEL_SLOT_NUM);
}

TEST_F(IpcProcessChannelBoundaryTest, ChannelEmpty) {
  IpcProcessChannel channel(shm_helper_->GetMetaPtr(), shm_helper_->GetDataPtr(), 2, 0, false, true);

  auto* shared_data = reinterpret_cast<IpcProcessChannel::SharedData*>(shm_helper_->GetMetaPtr());
  EXPECT_EQ(shared_data->used_slot_num, 0);

  // Should blocked of reading empty channel.
  std::atomic<bool> read_blocked{true};
  std::thread reader([&]() {
    channel.WaitReadableAndLock();
    read_blocked = false;
    channel.ReadAndUnlock();
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  EXPECT_TRUE(read_blocked);

  // Write data, unblocked.
  EXPECT_TRUE(channel.WaitWritableAndLock());
  EXPECT_TRUE(channel.WriteAndUnlock());

  reader.join();
  EXPECT_FALSE(read_blocked);
}

TEST_F(IpcProcessChannelBoundaryTest, OffsetWraparound) {
  IpcProcessChannel channel(shm_helper_->GetMetaPtr(), shm_helper_->GetDataPtr(), 2, 0, false, true);

  // Write and read multiple times, more than CHANNEL_SLOT_NUM.
  const int iterations = CHANNEL_SLOT_NUM * 3;

  for (int i = 0; i < iterations; ++i) {
    EXPECT_TRUE(channel.WaitWritableAndLock());
    int* write_data = static_cast<int*>(channel.GetWritePtr());
    *write_data = i;
    EXPECT_TRUE(channel.WriteAndUnlock());

    EXPECT_TRUE(channel.WaitReadableAndLock());
    int* read_data = static_cast<int*>(channel.GetReadPtr());
    EXPECT_EQ(*read_data, i);
    EXPECT_TRUE(channel.ReadAndUnlock());
  }

  // Verify offset, should be looped.
  auto* shared_data = reinterpret_cast<IpcProcessChannel::SharedData*>(shm_helper_->GetMetaPtr());
  EXPECT_LT(shared_data->write_offset, CHANNEL_SLOT_NUM);
  EXPECT_LT(shared_data->read_offset[0], CHANNEL_SLOT_NUM);
}

TEST_F(IpcProcessChannelBoundaryTest, MaxExecutorsBoundary) {
  // 63 executor should succ.
  ASSERT_NO_THROW(
      { IpcProcessChannel channel(shm_helper_->GetMetaPtr(), shm_helper_->GetDataPtr(), 63, 0, true, true); });

  // Reset shared memory.
  shm_helper_.reset();
  shm_helper_ = std::make_unique<SharedMemoryHelper>(CHANNEL_META_SIZE, CHANNEL_DATA_SIZE * CHANNEL_SLOT_NUM);

  // 64 executor should failed.
  EXPECT_THROW({ IpcProcessChannel channel(shm_helper_->GetMetaPtr(), shm_helper_->GetDataPtr(), 64, 0, true, true); },
               std::runtime_error);
}

class IpcProcessChannelPerformanceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    shm_helper_ = std::make_unique<SharedMemoryHelper>(CHANNEL_META_SIZE, CHANNEL_DATA_SIZE * CHANNEL_SLOT_NUM);
  }

  void TearDown() override { shm_helper_.reset(); }

  std::unique_ptr<SharedMemoryHelper> shm_helper_;
};

TEST_F(IpcProcessChannelPerformanceTest, Throughput) {
  IpcProcessChannel channel(shm_helper_->GetMetaPtr(), shm_helper_->GetDataPtr(), 2, 0, false, true);

  const int message_count = 10000;
  std::atomic<int> write_count{0};
  std::atomic<int> read_count{0};

  auto start_time = std::chrono::high_resolution_clock::now();

  std::thread writer([&]() {
    for (int i = 0; i < message_count; ++i) {
      if (channel.WaitWritableAndLock()) {
        int* data = static_cast<int*>(channel.GetWritePtr());
        *data = i;
        channel.WriteAndUnlock();
        write_count++;
      }
    }
  });

  std::thread reader([&]() {
    for (int i = 0; i < message_count; ++i) {
      if (channel.WaitReadableAndLock()) {
        channel.ReadAndUnlock();
        read_count++;
      }
    }
  });

  writer.join();
  reader.join();

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

  EXPECT_EQ(write_count, message_count);
  EXPECT_EQ(read_count, message_count);
}

TEST_F(IpcProcessChannelPerformanceTest, Latency) {
  IpcProcessChannel channel(shm_helper_->GetMetaPtr(), shm_helper_->GetDataPtr(), 2, 0, false, true);

  const int iterations = 1000;
  std::vector<double> latencies;
  latencies.reserve(iterations);

  for (int i = 0; i < iterations; ++i) {
    auto start = std::chrono::high_resolution_clock::now();

    // 写入
    EXPECT_TRUE(channel.WaitWritableAndLock());
    EXPECT_TRUE(channel.WriteAndUnlock());

    // 读取
    EXPECT_TRUE(channel.WaitReadableAndLock());
    EXPECT_TRUE(channel.ReadAndUnlock());

    auto end = std::chrono::high_resolution_clock::now();
    auto latency = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    latencies.push_back(latency);
  }

  double sum = 0;
  double min_latency = latencies[0];
  double max_latency = latencies[0];

  for (double latency : latencies) {
    sum += latency;
    min_latency = std::min(min_latency, latency);
    max_latency = std::max(max_latency, latency);
  }

  // Should less than 1 second.
  double avg_latency = sum / iterations;
  EXPECT_LT(avg_latency, 1000.0);
}

TEST_F(IpcProcessChannelPerformanceTest, ConcurrentPerformance) {
  IpcProcessChannel channel(shm_helper_->GetMetaPtr(), shm_helper_->GetDataPtr(), 2, 0, false, true);

  const int thread_count = 4;
  const int messages_per_thread = 1000;
  std::vector<std::thread> threads;
  std::atomic<int> total_processed{0};

  auto start_time = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < thread_count / 2; ++i) {
    threads.emplace_back([&]() {
      for (int j = 0; j < messages_per_thread; ++j) {
        if (channel.WaitWritableAndLock()) {
          channel.WriteAndUnlock();
          total_processed++;
        }
      }
    });

    threads.emplace_back([&]() {
      for (int j = 0; j < messages_per_thread; ++j) {
        if (channel.WaitReadableAndLock()) {
          channel.ReadAndUnlock();
        }
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }
}

class IpcProcessChannelStressTest : public ::testing::Test {
 protected:
  void SetUp() override {
    shm_helper_ = std::make_unique<SharedMemoryHelper>(CHANNEL_META_SIZE, CHANNEL_DATA_SIZE * CHANNEL_SLOT_NUM);
  }

  void TearDown() override { shm_helper_.reset(); }

  std::unique_ptr<SharedMemoryHelper> shm_helper_;
};

TEST_F(IpcProcessChannelStressTest, LongRunning) {
  IpcProcessChannel channel(shm_helper_->GetMetaPtr(), shm_helper_->GetDataPtr(), 2, 0, false, true);

  const int duration_seconds = 5;
  const auto end_time = std::chrono::steady_clock::now() + std::chrono::seconds(duration_seconds);

  std::atomic<bool> should_stop{false};
  std::atomic<int> write_count{0};
  std::atomic<int> read_count{0};

  std::thread writer([&]() {
    while (!should_stop) {
      if (channel.WaitWritableAndLock()) {
        channel.WriteAndUnlock();
        write_count++;
      }
    }
  });

  std::thread reader([&]() {
    while (!should_stop) {
      if (channel.WaitReadableAndLock()) {
        channel.ReadAndUnlock();
        read_count++;
      }
    }
  });

  std::this_thread::sleep_until(end_time);
  should_stop = true;

  writer.join();
  reader.join();

  EXPECT_GT(write_count, 0);
  EXPECT_GT(read_count, 0);
}

TEST_F(IpcProcessChannelStressTest, RandomStopRestart) {
  IpcProcessChannel channel(shm_helper_->GetMetaPtr(), shm_helper_->GetDataPtr(), 2, 0, false, true);

  std::atomic<bool> test_running{true};
  std::atomic<int> successful_operations{0};

  std::thread worker([&]() {
    while (test_running) {
      if (channel.WaitWritableAndLock()) {
        if (channel.WriteAndUnlock()) {
          successful_operations++;
        }
      }

      if (channel.WaitReadableAndLock()) {
        if (channel.ReadAndUnlock()) {
          successful_operations++;
        }
      }
    }
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  channel.Stop();

  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  test_running = false;

  worker.join();
}

}  // namespace ksana_llm
