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
  static void SetUpTestSuite() {
    InitLoguru();
  }
  // 在每个测试用例执行之前调用的函数
  void SetUp() override {
    // 创建一个 BlockManagerConfig 对象，用于配置 BatchSchedulerEvironmentSimulator
    tp_num = 4;
    int device_block_num = 100;
    block_manager_config.host_allocator_config.blocks_num = device_block_num * tp_num * 2;
    block_manager_config.device_allocator_config.blocks_num = device_block_num;
    block_manager_config.device_allocator_config.block_token_num = 6;

    // 使用配置创建一个 BlockManagerSimulator 对象
    env_simulator = new BatchSchedulerEvironmentSimulator(block_manager_config, tp_num);

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
    // 创建BatchScheduler对象
    batch_scheduler = new BatchScheduler(batch_scheduler_config, tp_num);
  }

  // 在每个测试用例执行之后调用的函数
  void TearDown() override {
    // 删除 BatchScheduler 对象
    delete batch_scheduler;
    delete env_simulator;
  }

 protected:
  struct RequestInfo {
    int req_id;
    int expect_output_token_num;
    int input_token_num;
    std::shared_ptr<Request> req;
    std::vector<std::shared_ptr<InferRequest>> infer_req_group;
  };

  void GenerateRequests(int request_num, int max_expect_output_num, int max_input_num, std::vector<RequestInfo>& reqs) {
    std::srand(10);
    for (int i = 0; i < request_num; i++) {
      RequestInfo info;
      info.req_id = i;
      info.expect_output_token_num = std::rand() % max_expect_output_num + 1;
      info.input_token_num = std::rand() % max_input_num + 1;
      reqs.push_back(info);
    }
  }

  void DoParallelRequestAndCheck(int client_num, std::vector<RequestInfo>& reqs, int timeout = 5) {
    NLLM_LOG_INFO << "DoParallelRequestAndCheck start.  client_num=" << client_num << ", request_num=" << reqs.size();
    time_t start_time = ProfileTimer::GetCurrentTime();
    ClientSimulator client_simulator(client_num, batch_scheduler);
    for (auto& info : reqs) {
      info.infer_req_group = env_simulator->InitRequest(info.req_id, info.req_id, info.input_token_num,
                                                        info.expect_output_token_num, info.req);
      client_simulator.AddInferRequests(info.req_id, info.infer_req_group);
    }

    // Wait for request enqueue
    std::this_thread::sleep_for(std::chrono::microseconds(1));
    // schedule and generate tokens
    int i = 0;
    while (true) {
      std::vector<std::shared_ptr<InferRequest>> scheduled_reqs;
      scheduled_reqs = batch_scheduler->Schedule();
      if (scheduled_reqs.empty()) {
        if (client_simulator.IsAllRequestFinished()) {
          NLLM_LOG_INFO << "All requests finished";
          break;
        }
        time_t cur_time = ProfileTimer::GetCurrentTime();
        if((cur_time - start_time)> timeout){
          NLLM_LOG_INFO << "Test Timeout. timeout="<< timeout<< " seconds";
          break;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(1));
        continue;
      }
      env_simulator->RunAStep(scheduled_reqs);

      for (auto req : scheduled_reqs) {
        NLLM_LOG_DEBUG << "Step " << i << ": req_id:" << req->req_id
                      << ", output_token.size()=" << req->output_tokens.size()
                      << ", last output token= " << req->output_tokens.back();
      }
      i++;
    }

    // Check request results
    for (auto& info : reqs) {
      for (auto& infer_req : info.infer_req_group) {
        env_simulator->CheckRequestOutput(infer_req);
      }
    }
    NLLM_LOG_INFO << "DoParallelRequestAndCheck finished";
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
  // Run requests one by one
  int request_num = 100;
  int client_num = 1;
  int max_expect_output_num = 100;
  int max_input_num = 400;
  std::vector<RequestInfo> req_list;
  GenerateRequests(request_num, max_expect_output_num, max_input_num, req_list);
  DoParallelRequestAndCheck(client_num, req_list);

  auto& stat = env_simulator->GetBlockManagerStat();
  EXPECT_EQ(stat.swapout_succ_num, 0);
  EXPECT_EQ(stat.swapout_fail_num, 0);
  EXPECT_EQ(stat.swapin_succ_num, 0);
  EXPECT_EQ(stat.swapin_fail_num, 0);
}

TEST_F(BatchSchedulerTest, SwapOutInNotTriggeredPressTest) {
  // Run requests in parallel
  // input and max output token are limited, SwapOut/In are not triggered.
  int request_num = 100;
  int client_num = 10;
  int max_expect_output_num = 40;
  int max_input_num = 60;
  std::vector<RequestInfo> req_list;
  GenerateRequests(request_num, max_expect_output_num, max_input_num, req_list);
  DoParallelRequestAndCheck(client_num, req_list);

  auto& stat = env_simulator->GetBlockManagerStat();
  EXPECT_EQ(stat.swapout_succ_num, 0);
  EXPECT_EQ(stat.swapout_fail_num, 0);
  EXPECT_EQ(stat.swapin_succ_num, 0);
  EXPECT_EQ(stat.swapin_fail_num, 0);
}

TEST_F(BatchSchedulerTest, SwapOutInTriggeredPressTest) {
  // Run requests in parallel
  // max output token are large, SwapOut/In will be triggered when there are multiple requests.
  int request_num = 100;
  int client_num = 10;
  int max_expect_output_num = 400;
  int max_input_num = 100;
  std::vector<RequestInfo> req_list;
  GenerateRequests(request_num, max_expect_output_num, max_input_num, req_list);
  DoParallelRequestAndCheck(client_num, req_list);

  auto& stat = env_simulator->GetBlockManagerStat();
  EXPECT_GT(stat.swapout_succ_num, 0);
  EXPECT_EQ(stat.swapout_fail_num, 0);
  EXPECT_GT(stat.swapin_succ_num, 0);
  EXPECT_EQ(stat.swapin_fail_num, 0);
}

