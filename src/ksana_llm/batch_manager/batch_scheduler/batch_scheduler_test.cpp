/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/batch_manager/batch_scheduler/batch_scheduler.h"
#include "ksana_llm/batch_manager/batch_scheduler/batch_scheduler_test_helper.h"

#include <memory>
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/singleton.h"
#include "test.h"

using namespace ksana_llm;

// 定义一个 BatchSchedulerTest 类，继承自 testing::Test
class BatchSchedulerTest : public testing::Test {
 protected:
  // 在每个测试用例执行之前调用的函数
  void SetUp() override {
    // 创建一个 BlockManagerConfig 对象，用于配置 BatchSchedulerEvironmentSimulator
    block_manager_config.host_allocator_config.blocks_num = 100;
    block_manager_config.device_allocator_config.blocks_num = 100;
    block_manager_config.device_allocator_config.block_token_num = 6;

    tp_num = 4;

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
  // 定义一个 BlockManager 指针，用于在测试用例中使用
  BatchSchedulerEvironmentSimulator* env_simulator;
  BatchScheduler* batch_scheduler;

  BlockManagerConfig block_manager_config;
  BatchSchedulerConfig batch_scheduler_config;
  int tp_num;
};

TEST_F(BatchSchedulerTest, BasicTokenGenerationTest) {
  // 创建两个请求
  int expected_output_token_num1 = 100;
  int input_token_num1 = 10;
  int expected_output_token_num2 = 2;
  int input_token_num2 = 30;
  std::shared_ptr<Request> req1, req2;
  std::vector<std::shared_ptr<InferRequest>> infer_req_list1 =
      env_simulator->InitRequest(1, 1, input_token_num1, expected_output_token_num1, req1);
  std::vector<std::shared_ptr<InferRequest>> infer_req_list2 =
      env_simulator->InitRequest(2, 2, input_token_num2, expected_output_token_num2, req2);
  std::shared_ptr<InferRequest> infer_req1 = infer_req_list1[0];
  std::shared_ptr<InferRequest> infer_req2 = infer_req_list2[0];

  // Add requests to scheduler
  batch_scheduler->AddInferRequest(infer_req_list1);
  batch_scheduler->AddInferRequest(infer_req_list2);

  // schedule and generate tokens
  int max_output_step = std::max(expected_output_token_num1, expected_output_token_num2) + 1;
  for (int i = 0; i < max_output_step; i++) {
    std::vector<std::shared_ptr<InferRequest>> scheduled_reqs;
    scheduled_reqs = batch_scheduler->Schedule();
    if (scheduled_reqs.empty()) {
      break;
    }
    env_simulator->RunAStep(scheduled_reqs);
    for (auto req : scheduled_reqs) {
      NLLM_LOG_INFO << "Step " << i << ": req_id:" << req->req_id
                    << ", output_token.size()=" << req->output_tokens.size()
                    << ", last output token= " << req->output_tokens.back();
    }
  }

  // Check request results
  env_simulator->CheckRequestOutput(infer_req1);
  env_simulator->CheckRequestOutput(infer_req2);
}
