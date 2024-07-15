/* Copyright 2021 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <atomic>
#include <memory>
#include <thread>

#include "ksana_llm/utils/waiter.h"
#include "test.h"

namespace ksana_llm {

TEST(WaitGroupTest, AddDone) {
  static const int s_iter_count = 33;
  static const int s_add_count = 100;
  WaitGroup wg;
  for (int iter = 0; iter < s_iter_count; iter++) {
    wg.Wait();
    EXPECT_EQ(wg.Count(), 0);

    wg.Add();
    wg.Add(s_add_count);
    EXPECT_EQ(wg.Count(), 1 + s_add_count);

    wg.Done();
    wg.Done(s_add_count);

    wg.Wait();
    EXPECT_EQ(wg.Count(), 0);
  }
}

TEST(WaitGroupTest, MultiThreadAddDone) {
  static const int s_thread_num = 3;
  static const int s_iter_count = 33;
  static const int s_each_iter_add_count = 100;

  auto add_func = [](WaitGroup* data_wg, int iter_count, int each_iter_add_count, WaitGroup* thread_wg) {
    for (int iter = 0; iter < iter_count; iter++) {
      data_wg->Add(each_iter_add_count);
    }
    thread_wg->Done();
  };

  auto done_func = [](WaitGroup* data_wg, int iter_count, int each_iter_done_count, WaitGroup* thread_wg) {
    for (int iter = 0; iter < iter_count; iter++) {
      data_wg->Done(each_iter_done_count);
    }
    thread_wg->Done();
  };

  WaitGroup data_wg;
  WaitGroup thread_wg;

  for (int i = 0; i < s_thread_num; i++) {
    thread_wg.Add();
    std::thread add_one_thread(add_func, &data_wg, s_iter_count, 1, &thread_wg);
    add_one_thread.detach();

    thread_wg.Add();
    std::thread add_many_thread(add_func, &data_wg, s_iter_count, s_each_iter_add_count, &thread_wg);
    add_many_thread.detach();
  }
  thread_wg.Wait();
  EXPECT_EQ(thread_wg.Count(), 0);
  EXPECT_EQ(data_wg.Count(), s_thread_num * s_iter_count * (1 + s_each_iter_add_count));

  for (int i = 0; i < s_thread_num; i++) {
    thread_wg.Add();
    std::thread done_one_thread(done_func, &data_wg, s_iter_count, 1, &thread_wg);
    done_one_thread.detach();

    thread_wg.Add();
    std::thread done_many_thread(done_func, &data_wg, s_iter_count, s_each_iter_add_count, &thread_wg);
    done_many_thread.detach();
  }

  thread_wg.Wait();
  EXPECT_EQ(thread_wg.Count(), 0);
  EXPECT_EQ(data_wg.Count(), 0);
}

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
