/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/batch_manager/batch_scheduler/batch_scheduler_test_helper.h"

#include <memory>
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/singleton.h"
#include "test.h"

using namespace ksana_llm;

// 定义一个 BlockSchedulerTest 类，用于测试BatchSchedulerEvironmentSimulator
class BatchSchedulerEvironmentSimulatorTest : public testing::Test {
 protected:
  // 在每个测试用例执行之前调用的函数
  void SetUp() override {
    // 创建一个 BlockManagerConfig 对象，用于配置 BatchSchedulerEvironmentSimulator
    block_manager_config.host_allocator_config.blocks_num = 100;
    block_manager_config.device_allocator_config.blocks_num = 100;
    block_manager_config.device_allocator_config.block_token_num = 6;
    device_num = 2;

    // 使用配置创建一个 BlockManagerSimulator 对象
    env_simulator = new BatchSchedulerEvironmentSimulator(block_manager_config, device_num);
    NLLM_LOG_INFO << "Simulator start";
  }

  // 在每个测试用例执行之后调用的函数
  void TearDown() override {
    // 删除 BatchScheduler 对象
    delete env_simulator;
  }

  void InitRequestBlock(std::shared_ptr<InferRequest>& req, int output_token_num) {
    // Allocate all blocks at the beginning;
    int block_token_num = block_manager_config.device_allocator_config.block_token_num;
    int total_block_num = (req->input_tokens.size() + output_token_num + block_token_num - 1) / block_token_num;
    NLLM_LOG_INFO << "Start init req " << req->req_id << ", block num =" << total_block_num;
    NLLM_CHECK_WITH_INFO(req->kv_cache_blocks.size() == device_num,
                         FormatStr("req->kv_cache_blocks.size()=%d", req->kv_cache_blocks.size()));
    for (int i = 0; i < device_num; i++) {
      std::vector<int> blocks;
      GetBlockManager()->SetDeviceId(i);
      GetBlockManager()->AllocateBlocks(total_block_num, blocks);
      req->kv_cache_blocks[i].insert(req->kv_cache_blocks[i].end(), blocks.begin(), blocks.end());
      NLLM_LOG_INFO << "req " << req->req_id << ", kv_cache_blocks[" << i
                    << "].size()=" << req->kv_cache_blocks[i].size();
    }
    NLLM_LOG_INFO << "Init infer request " << req->req_id << " block=" << total_block_num;
  }

 protected:
  // 定义一个 BatchSchedulerEvironmentSimulator 指针，用于在测试用例中使用
  BatchSchedulerEvironmentSimulator* env_simulator;

  BlockManagerConfig block_manager_config;
  int device_num;
};

TEST_F(BatchSchedulerEvironmentSimulatorTest, BasicTokenGenerationTest) {
  int expected_output_token_num1 = 3;
  int input_token_num1 = 2;
  int expected_output_token_num2 = 2;
  int input_token_num2 = 3;
  std::shared_ptr<Request> req1, req2;
  std::vector<std::shared_ptr<InferRequest>> infer_req_list1 =
    env_simulator->InitRequest(1, 1, input_token_num1, expected_output_token_num1, req1);
  std::vector<std::shared_ptr<InferRequest>> infer_req_list2 =
    env_simulator->InitRequest(2, 2, input_token_num2, expected_output_token_num2, req2);
  std::shared_ptr<InferRequest> infer_req1 = infer_req_list1[0];
  std::shared_ptr<InferRequest> infer_req2 = infer_req_list2[0];

  // Simple memory strategy, init all blocks at the beginning
  InitRequestBlock(infer_req1, expected_output_token_num1);
  InitRequestBlock(infer_req2, expected_output_token_num2);

  int max_output_step = std::max(expected_output_token_num1, expected_output_token_num2) + 1;
  for (int i = 0; i < max_output_step; i++) {
    std::vector<std::shared_ptr<InferRequest>> scheduled_reqs;
    if (!env_simulator->IsRequestFinished(infer_req1)) {
      scheduled_reqs.push_back(infer_req1);
    }
    if (!env_simulator->IsRequestFinished(infer_req2)) {
      scheduled_reqs.push_back(infer_req2);
    }
    if (scheduled_reqs.empty()) break;
    NLLM_LOG_INFO << "Step " << i << ": scheduled_reqs.size(): " << scheduled_reqs.size();
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

TEST_F(BatchSchedulerEvironmentSimulatorTest, SwapTokenGenerationTest) {
  // 创建两个请求
  int expected_output_token_num1 = 23;
  int input_token_num1 = 20;
  int expected_output_token_num2 = 12;
  int input_token_num2 = 30;
  std::shared_ptr<Request> req1, req2;
  std::vector<std::shared_ptr<InferRequest>> infer_req_list1 =
    env_simulator->InitRequest(1, 1, input_token_num1, expected_output_token_num1, req1);
  std::vector<std::shared_ptr<InferRequest>> infer_req_list2 =
    env_simulator->InitRequest(2, 2, input_token_num2, expected_output_token_num2, req2);
  std::shared_ptr<InferRequest> infer_req1 = infer_req_list1[0];
  std::shared_ptr<InferRequest> infer_req2 = infer_req_list2[0];

  // Simple memory strategy, init all blocks at the beginning
  InitRequestBlock(infer_req1, expected_output_token_num1);
  InitRequestBlock(infer_req2, expected_output_token_num2);

  int max_output_step = 2 * std::max(expected_output_token_num1, expected_output_token_num2) + 1;

  for (int i = 0; i < max_output_step; i++) {
    std::vector<std::shared_ptr<InferRequest>> scheduled_reqs;
    if (!env_simulator->IsRequestFinished(infer_req1)) {
      if (i >= 4 && i < 10) {
        if (i % 2 == 0) {
          infer_req1->SwapOutAsync(0);
        } else {
          infer_req1->SwapInAsync();
          scheduled_reqs.push_back(infer_req1);
        }
      } else {
        scheduled_reqs.push_back(infer_req1);
      }
    }
    if (!env_simulator->IsRequestFinished(infer_req2)) {
      scheduled_reqs.push_back(infer_req2);
    }
    if (scheduled_reqs.empty()) break;
    NLLM_LOG_INFO << "Step " << i << ": scheduled_reqs.size(): " << scheduled_reqs.size();
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
