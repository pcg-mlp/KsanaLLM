// Copyright 2024 Tencent Inc.  All rights reserved.

#include <numeric>
#include <thread>

#include "ksana_llm/utils/channel.h"
#include "test.h"

namespace ksana_llm {

using testing::ElementsAreArray;

class ChannelTest : public testing::Test {
 protected:
  void SetUp() override { chan_ = Channel<int32_t>(capacity_, "TestChan"); }
  void TearDown() override {}

  Channel<int32_t> chan_;
  size_t capacity_{100ul};
};

TEST_F(ChannelTest, SingleWriteRead) {
  ASSERT_EQ(chan_.Size(), 0ul);
  std::vector<int32_t> data = {0, 1, 2};
  for (int32_t x : data) {
    chan_.Write(std::move(x));
  }
  ASSERT_EQ(chan_.Size(), data.size());
  for (int32_t expected : data) {
    int32_t readed = 0;
    ASSERT_TRUE(chan_.Read(&readed));
    ASSERT_EQ(readed, expected);
  }
  ASSERT_EQ(chan_.Size(), 0ul);
}

TEST_F(ChannelTest, BatchWriteRead) {
  std::vector<int32_t> origin_data(capacity_);
  std::iota(origin_data.begin(), origin_data.end(), 0);
  ASSERT_EQ(chan_.WriteMovable(origin_data.data(), origin_data.size()), capacity_);
  ASSERT_EQ(chan_.Size(), capacity_);
  std::vector<int32_t> readed_data(capacity_);
  ASSERT_EQ(chan_.Read(readed_data.data(), readed_data.size()), capacity_);
  ASSERT_THAT(origin_data, ElementsAreArray(readed_data));
  chan_.PrintWaitRate();
}

void MultiThreadsReadWrite(size_t capacity, size_t data_size) {
  FOR_RANGE(int32_t, repeat_i, 0, 33) {
    Channel<int32_t> chan(capacity, "TestChan");
    std::vector<int32_t> origin_data(data_size);
    std::iota(origin_data.begin(), origin_data.end(), 0);
    std::vector<int32_t> readed_data(data_size, -1);
    // readers
    std::vector<std::thread> readers(5ul);
    FOR_RANGE(size_t, reader_i, 0ul, readers.size()) {
      readers[reader_i] = std::thread([&, reader_i]() {
        FOR_RANGE(size_t, i, 0ul, data_size) {
          if (i % readers.size() == reader_i) {
            int32_t x = -1;
            ASSERT_TRUE(chan.Read(&x));
            ASSERT_LT(x, data_size);
            readed_data[x] = x;
          }
        }
      });
    }
    // writers
    std::vector<std::thread> writers(6ul);
    FOR_RANGE(size_t, writer_i, 0ul, writers.size()) {
      writers[writer_i] = std::thread([&, writer_i]() {
        FOR_RANGE(size_t, i, 0ul, data_size) {
          if (i % writers.size() == writer_i) {
            ASSERT_TRUE(chan.Write(i));
          }
        }
      });
    }
    // join
    for (std::thread& t : readers) {
      t.join();
    }
    for (std::thread& t : writers) {
      t.join();
    }
    chan.Close();
    ASSERT_THAT(origin_data, ElementsAreArray(readed_data));
    chan.PrintWaitRate();
  }
}

TEST_F(ChannelTest, MultiThreadsReadWrite) { MultiThreadsReadWrite(/*capacity=*/capacity_, /*data_size=*/capacity_); }

TEST_F(ChannelTest, ZeroCapacity) { MultiThreadsReadWrite(/*capacity=*/0ul, /*data_size=*/capacity_); }

}  // namespace ksana_llm
