/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/batch_manager/batch_scheduler/batch_scheduler.h"
#include "ksana_llm/batch_manager/batch_scheduler/batch_scheduler_test_client.h"
#include "ksana_llm/batch_manager/batch_scheduler/batch_scheduler_test_helper.h"

#include <exception>
#include <memory>
#include <thread>

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
    // Init BatchSchedulerEvironmentSimulator and BatchScheduler
    InitDefaultConfig();
    env_simulator = new BatchSchedulerEvironmentSimulator(block_manager_config, tp_num);
    batch_scheduler = new BatchScheduler(batch_scheduler_config, tp_num);
  }

  void TearDown() override {
    delete batch_scheduler;
    delete env_simulator;
  }

 protected:
  void GenerateRequests(int request_num, int min_expect_output_num, int max_expect_output_num, int min_input_num,
                        int max_input_num, std::vector<ParallelTester::RequestInfo>& reqs) {
    NLLM_CHECK_WITH_INFO(
        min_expect_output_num < max_expect_output_num,
        FormatStr("min_expect_output_num % should be larger than 0 and less than max_expect_output_num %d.",
                  min_expect_output_num, max_expect_output_num));

    NLLM_CHECK_WITH_INFO(
        min_input_num < max_input_num,
        FormatStr("min_input_num % should be less than max_input_num %d.", min_input_num, max_input_num));

    std::srand(10);
    for (int i = 0; i < request_num; i++) {
      ParallelTester::RequestInfo info;
      info.req_id = i;
      info.expect_output_token_num =
          std::rand() % (max_expect_output_num - min_expect_output_num) + min_expect_output_num;
      info.input_token_num = std::rand() % (max_input_num - min_input_num) + min_input_num;
      reqs.push_back(info);
    }
  }

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
    batch_scheduler_config.swap_threadpool_size = 8;
    batch_scheduler_config.preempt_mode = static_cast<PreemptMode>(0);
  }

 protected:
  // 定义一个 BlockManager 指针，用于在测试用例中使用
  BatchSchedulerEvironmentSimulator* env_simulator;
  BatchScheduler* batch_scheduler;

  BlockManagerConfig block_manager_config;
  BatchSchedulerConfig batch_scheduler_config;
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
  GenerateRequests(request_num, 1, max_expect_output_num, 0, max_input_num, req_list);
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
  GenerateRequests(request_num, 1, max_expect_output_num, 0, max_input_num, req_list);
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
  int max_expect_output_num = 400;
  int max_input_num = 100;
  std::vector<ParallelTester::RequestInfo> req_list;
  GenerateRequests(request_num, 1, max_expect_output_num, 0, max_input_num, req_list);
  tester.InitRequestInfoListByDefault(req_list);
  tester.DoParallelRequestAndCheck(client_num, req_list, hooks);

  auto& stat = env_simulator->GetBlockManagerStat();
  EXPECT_GT(stat.swapout_succ_num, 0);
  EXPECT_EQ(stat.swapout_fail_num, 0);
  EXPECT_GT(stat.swapin_succ_num, 0);
  EXPECT_EQ(stat.swapin_fail_num, 0);
}
