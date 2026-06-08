/* Copyright 2021 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <atomic>
#include <memory>
#include <thread>

#include "test.h"
#include "utils/waiter.h"

namespace ksana_llm {

TEST(WaiterTest, Waiter) {
  std::vector<std::thread> threads;
  int total_threads = 50;
  std::atomic<int> cnt{0};
  Waiter w(total_threads);
  for (int i = 0; i < total_threads; ++i) {
    threads.emplace_back([&w, &cnt]() {
      ++cnt;
      w.Wait();
      EXPECT_EQ(cnt, 0) << "Should never happen, Debug!";
    });
  }
  while (cnt < total_threads) {
  }

  for (int i = 0; i < total_threads; ++i) {
    --cnt;
    w.Notify();
  }

  for (auto& t : threads) {
    t.join();
  }
  EXPECT_EQ(cnt, 0) << "Should never happen, Debug!";
}

TEST(WaiterTest, AtomicCounter) {
  std::vector<std::thread> threads;
  int total_threads = 30;
  std::atomic<int> cnt{0};
  AtomicCounter w(total_threads);
  for (int i = 0; i < total_threads; ++i) {
    threads.emplace_back([&w, &cnt]() {
      if (w.DecAndIsZero()) ++cnt;
    });
  }

  for (auto& t : threads) {
    t.join();
  }
  EXPECT_TRUE(w.IsZero());
  EXPECT_EQ(cnt, 1);
}

}  // namespace ksana_llm
