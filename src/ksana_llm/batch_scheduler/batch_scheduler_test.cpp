/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/batch_scheduler/batch_scheduler.h"

#include <exception>
#include <memory>
#include <thread>

#include "ksana_llm/batch_scheduler/batch_scheduler_test_client.h"
#include "ksana_llm/batch_scheduler/batch_scheduler_test_helper.h"

#include "ksana_llm/cache_manager/direct_cache_manager.h"
#include "ksana_llm/cache_manager/prefix_cache_manager.h"

#include "ksana_llm/profiler/timer.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/singleton.h"
#include "test.h"

using namespace ksana_llm;

// 定义一个 BatchSchedulerTest 类，继承自 testing::Test
class BatchSchedulerTest : public testing::Test {
 protected:
  static void SetUpTestSuite() { InitLoguru(); }

  void CommonSetUp() {
    // Init BatchSchedulerEnvironmentSimulator and BatchScheduler
    InitDefaultConfig();
    env_simulator = new BatchSchedulerEnvironmentSimulator(block_manager_config, tp_num);
    batch_scheduler = new BatchScheduler(batch_scheduler_config, tp_num);

    cache_manager = std::make_shared<PrefixCacheManager>(cache_manager_config);
    cache_manager->InitializeCachedBlocks();
    batch_scheduler->SetCacheManager(cache_manager);
  }

  void TearDown() override {
    delete batch_scheduler;
    delete env_simulator;
  }

 protected:
  void InitDefaultConfig() {
    tp_num = 4;
    int device_block_num = 100;
    block_manager_config.host_allocator_config.blocks_num = device_block_num * tp_num * 2;
    block_manager_config.device_allocator_config.blocks_num = device_block_num;
    block_manager_config.device_allocator_config.block_token_num = 6;

    batch_scheduler_config.schedule_strategy = static_cast<ScheduleStrategy>(0);
    batch_scheduler_config.waiting_timeout_in_ms = 600000;
    batch_scheduler_config.max_waiting_queue_len = 256;
    batch_scheduler_config.max_step_tokens = 4096;
    batch_scheduler_config.max_batch_size = 8;
    batch_scheduler_config.max_token_len = 1024;
    batch_scheduler_config.swapout_block_threshold = 1.0;
    batch_scheduler_config.swapin_block_threshold = 2.0;
    batch_scheduler_config.launch_block_threshold = 2.0;
    batch_scheduler_config.preempt_mode = static_cast<PreemptMode>(0);

    cache_manager_config.block_token_num = block_manager_config.device_allocator_config.block_token_num;
    cache_manager_config.tensor_para_size = tp_num;
    cache_manager_config.swap_threadpool_size = 8;
    cache_manager_config.enable_prefix_caching = false;
  }

 protected:
  // 定义一个 BlockManager 指针，用于在测试用例中使用
  BatchSchedulerEnvironmentSimulator* env_simulator = nullptr;
  BatchSchedulerInterface* batch_scheduler = nullptr;
  std::shared_ptr<CacheManagerInterface> cache_manager = nullptr;

  BlockManagerConfig block_manager_config;
  BatchSchedulerConfig batch_scheduler_config;
  CacheManagerConfig cache_manager_config;
  int tp_num;
};

TEST_F(BatchSchedulerTest, BasicTokenGenerationTest) {
  CommonSetUp();
  ParallelTester tester(batch_scheduler, env_simulator);

  std::vector<ParallelTester::ExeHookInterface*> hooks;
  ParallelTester::DefaultResultCheckHook default_hook(env_simulator);
  hooks.push_back(&default_hook);

  // Run requests one by one
  int request_num = 100;
  int client_num = 1;
  int max_expect_output_num = 100;
  int max_input_num = 400;
  std::vector<ParallelTester::RequestInfo> req_list;
  tester.GenerateRequests(request_num, 1, max_expect_output_num, 0, max_input_num, req_list);
  tester.InitRequestInfoListByDefault(req_list);
  tester.DoParallelRequestAndCheck(client_num, req_list, hooks);

  auto& stat = env_simulator->GetBlockManagerStat();
  EXPECT_EQ(stat.swapout_succ_num, 0);
  EXPECT_EQ(stat.swapout_fail_num, 0);
  EXPECT_EQ(stat.swapin_succ_num, 0);
  EXPECT_EQ(stat.swapin_fail_num, 0);
}

TEST_F(BatchSchedulerTest, SwapOutInNotTriggeredPressTest) {
  CommonSetUp();
  ParallelTester tester(batch_scheduler, env_simulator);

  std::vector<ParallelTester::ExeHookInterface*> hooks;
  ParallelTester::DefaultResultCheckHook default_hook(env_simulator);
  hooks.push_back(&default_hook);

  // Run requests in parallel
  // input and max output token are limited, SwapOut/In are not triggered.
  int request_num = 100;
  int client_num = 10;
  int max_expect_output_num = 40;
  int max_input_num = 60;
  std::vector<ParallelTester::RequestInfo> req_list;
  tester.GenerateRequests(request_num, 1, max_expect_output_num, 0, max_input_num, req_list);
  tester.InitRequestInfoListByDefault(req_list);
  tester.DoParallelRequestAndCheck(client_num, req_list, hooks);

  auto& stat = env_simulator->GetBlockManagerStat();
  EXPECT_EQ(stat.swapout_succ_num, 0);
  EXPECT_EQ(stat.swapout_fail_num, 0);
  EXPECT_EQ(stat.swapin_succ_num, 0);
  EXPECT_EQ(stat.swapin_fail_num, 0);
}

TEST_F(BatchSchedulerTest, SwapOutInTriggeredPressTest) {
  CommonSetUp();
  ParallelTester tester(batch_scheduler, env_simulator);

  std::vector<ParallelTester::ExeHookInterface*> hooks;
  ParallelTester::DefaultResultCheckHook default_hook(env_simulator);
  hooks.push_back(&default_hook);

  // Run requests in parallel
  // max output token are large, SwapOut/In will be triggered when there are multiple requests.
  int request_num = 100;
  int client_num = 10;
  int min_expect_output_num = 1;
  int max_expect_output_num = 400;
  int min_input_num = 0;
  int max_input_num = 100;
  std::vector<ParallelTester::RequestInfo> req_list;
  tester.GenerateRequests(request_num, min_expect_output_num, max_expect_output_num, min_input_num, max_input_num,
                          req_list);
  tester.InitRequestInfoListByDefault(req_list);
  tester.DoParallelRequestAndCheck(client_num, req_list, hooks, 10);

  auto& stat = env_simulator->GetBlockManagerStat();
  EXPECT_GT(stat.swapout_succ_num, 0);
  EXPECT_EQ(stat.swapout_fail_num, 0);
  EXPECT_GT(stat.swapin_succ_num, 0);
  EXPECT_EQ(stat.swapin_fail_num, 0);
}

TEST_F(BatchSchedulerTest, FixPrefixCacheNoSwapTriggeredTest) {
  CommonSetUp();

  int prefix_block_num = 3;
  int block_token_num = 6;
  int device_num = 4;
  FixPrefixTestCase test_case(prefix_block_num, block_token_num, device_num, false);
  test_case.SetBatchScheduler(batch_scheduler);
  test_case.SetEnvSimulator(env_simulator);
  test_case.RunTestNoSwapTriggered();
}

TEST_F(BatchSchedulerTest, FixPrefixCacheSwapTriggeredTest) {
  CommonSetUp();

  int prefix_block_num = 30;
  int block_token_num = 6;
  int device_num = 4;
  FixPrefixTestCase test_case(prefix_block_num, block_token_num, device_num, false);
  test_case.SetBatchScheduler(batch_scheduler);
  test_case.SetEnvSimulator(env_simulator);
  test_case.RunTestSwapTriggered();
}
