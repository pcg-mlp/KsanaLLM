/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <time.h>

#include "ksana_llm/runtime/threadpool.h"
#include "test.h"

namespace ksana_llm {

TEST(ThreadPool, ThreadPoolTest) {
  ThreadPool threadpool(10);
  threadpool.Start();

  // calc 0 + 1 + 2 + 3 + 4, expect 10
  time_t begin = time(NULL);
  std::vector<std::future<int>> results;
  for (int i = 0; i < 5; ++i) {
    results.push_back(threadpool.Submit([=]() -> int {
      std::this_thread::sleep_for(std::chrono::seconds(i));
      return i;
    }));
  }
  time_t end = time(NULL);

  int sum = 0;
  for (auto&& t : results) {
    int ret = t.get();
    sum += ret;
  }

  EXPECT_LT(end - begin, 5);
  EXPECT_EQ(sum, 10);

  threadpool.Stop();
}

}  // namespace ksana_llm
