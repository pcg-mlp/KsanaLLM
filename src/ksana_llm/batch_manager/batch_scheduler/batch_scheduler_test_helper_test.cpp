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
  static void SetUpTestSuite() { InitLoguru(); }

  // 在每个测试用例执行之前调用的函数
  void SetUp() override {
    // 创建一个 BlockManagerConfig 对象，用于配置 BatchSchedulerEvironmentSimulator
    block_manager_config.host_allocator_config.blocks_num = 100;
    block_manager_config.device_allocator_config.blocks_num = 100;
    block_manager_config.device_allocator_config.block_token_num = 6;
    device_num = 2;

    // 使用配置创建一个 BlockManagerSimulator 对象
    env_simulator = new BatchSchedulerEvironmentSimulator(block_manager_config, device_num);
    KLLM_LOG_INFO << "Simulator start";
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
    KLLM_LOG_INFO << "Start init req " << req->req_id << ", block num =" << total_block_num;
    KLLM_CHECK_WITH_INFO(req->kv_cache_blocks.size() == (size_t)device_num,
                         FormatStr("req->kv_cache_blocks.size()=%d", req->kv_cache_blocks.size()));
    for (int i = 0; i < device_num; i++) {
      std::vector<int> blocks;
      GetBlockManager()->SetDeviceId(i);
      GetBlockManager()->AllocateBlocks(total_block_num, blocks);
      req->kv_cache_blocks[i].insert(req->kv_cache_blocks[i].end(), blocks.begin(), blocks.end());
      KLLM_LOG_INFO << "req " << req->req_id << ", kv_cache_blocks[" << i
                    << "].size()=" << req->kv_cache_blocks[i].size();
    }
    KLLM_LOG_INFO << "Init infer request " << req->req_id << " block=" << total_block_num;
  }

 protected:
  // 定义一个 BatchSchedulerEvironmentSimulator 指针，用于在测试用例中使用
  BatchSchedulerEvironmentSimulator* env_simulator;

  BlockManagerConfig block_manager_config;
  int device_num;
};

TEST_F(BatchSchedulerEvironmentSimulatorTest, BasicTokenGenerationTest) {
  int expected_output_token_num1 = 30;
  int input_token_num1 = 20;

  int seed_1_token_num = 20;
  int seed_2_token_num = 30;
  int seed_0 = 20;
  int seed_1 = 30;
  int seed_2 = 40;
  int expected_output_token_num2 = seed_1_token_num + seed_2_token_num;
  int input_token_num2 = 30;

  std::shared_ptr<Request> req1, req2, req2_same_seed, req2_diff_seed, req2_same20_diff30;
  std::shared_ptr<InferRequest> infer_req1, infer_req2, infer_req2_same_seed, infer_req2_diff_seed,
      infer_req2_same20_diff30;

  std::vector<std::pair<int, int>> seeds;
  seeds.push_back(std::make_pair(0, 1));
  // Init req1
  std::vector<std::shared_ptr<InferRequest>> infer_req_list =
      env_simulator->InitRequest(1, input_token_num1, expected_output_token_num1, req1, seeds);
  infer_req1 = infer_req_list[0];
  InitRequestBlock(infer_req1, expected_output_token_num1);

  // Init req2
  seeds[0].second = seed_0;
  infer_req_list = env_simulator->InitRequest(2, input_token_num2, expected_output_token_num2, req2, seeds);
  infer_req2 = infer_req_list[0];
  InitRequestBlock(infer_req2, expected_output_token_num2);

  // Init req2_same_seed
  seeds.push_back(std::make_pair(input_token_num2, seed_0));
  infer_req_list = env_simulator->InitRequest(3, input_token_num2, expected_output_token_num2, req2_same_seed, seeds);
  infer_req2_same_seed = infer_req_list[0];
  InitRequestBlock(infer_req2_same_seed, expected_output_token_num2);

  // Init req2_diff_seed
  seeds[1].second = seed_1;
  infer_req_list = env_simulator->InitRequest(4, input_token_num2, expected_output_token_num2, req2_diff_seed, seeds);
  infer_req2_diff_seed = infer_req_list[0];
  InitRequestBlock(infer_req2_diff_seed, expected_output_token_num2);

  // Init req2_same20_diff30
  seeds.push_back(std::make_pair(input_token_num2 + seed_1_token_num, seed_2));
  infer_req_list =
      env_simulator->InitRequest(5, input_token_num2, expected_output_token_num2, req2_same20_diff30, seeds);
  infer_req2_same20_diff30 = infer_req_list[0];
  InitRequestBlock(infer_req2_same20_diff30, expected_output_token_num2);

  std::vector<std::shared_ptr<InferRequest>> infer_reqs;
  infer_reqs.push_back(infer_req1);
  infer_reqs.push_back(infer_req2);
  infer_reqs.push_back(infer_req2_same_seed);
  infer_reqs.push_back(infer_req2_diff_seed);
  infer_reqs.push_back(infer_req2_same20_diff30);

  int max_output_step = std::max(expected_output_token_num1, expected_output_token_num2) + 1;
  for (int i = 0; i < max_output_step; i++) {
    std::vector<std::shared_ptr<InferRequest>> scheduled_reqs;
    for (auto& req : infer_reqs) {
      if (!env_simulator->IsRequestFinished(req)) {
        scheduled_reqs.push_back(req);
      }
    }

    if (scheduled_reqs.empty()) break;
    KLLM_LOG_DEBUG << "Step " << i << ": scheduled_reqs.size(): " << scheduled_reqs.size();
    env_simulator->RunAStep(scheduled_reqs);
    for (auto req : scheduled_reqs) {
      KLLM_LOG_DEBUG << "Step " << i << ": req_id:" << req->req_id
                     << ", output_token.size()=" << req->output_tokens.size()
                     << ", last output token= " << req->output_tokens.back();
    }
  }

  // Check request results
  for (auto& req : infer_reqs) {
    env_simulator->CheckRequestOutput(req);
  }

  // Check seed generation results
  // input token should be same
  for (int i = 0; i < input_token_num2; i++) {
    int input_token = infer_req2->input_tokens[i];
    EXPECT_EQ(input_token, infer_req2_same_seed->input_tokens[i]);
    EXPECT_EQ(input_token, infer_req2_diff_seed->input_tokens[i]);
    EXPECT_EQ(input_token, infer_req2_same20_diff30->input_tokens[i]);
  }

  for (int i = 0; i < expected_output_token_num2 - 1; i++) {
    int offset = input_token_num2 + i;
    int output2 = infer_req2->output_tokens[offset];
    int output2_same = infer_req2_same_seed->output_tokens[offset];
    int output2_diff = infer_req2_diff_seed->output_tokens[offset];
    int output2_same20_diff30 = infer_req2_same20_diff30->output_tokens[offset];
    if (i < seed_1_token_num) {
      EXPECT_EQ(output2, output2_same);                // same seed_0
      EXPECT_EQ(output2_diff, output2_same20_diff30);  // same seed_1
      EXPECT_NE(output2, output2_diff);                // seed_0 vs seed_1
    } else {
      EXPECT_EQ(output2, output2_same);                // same seed_0
      EXPECT_NE(output2_diff, output2_same20_diff30);  // seed_1 vs seed_2
      EXPECT_NE(output2, output2_diff);                // seed_0 vs seed_1
      EXPECT_NE(output2, output2_same20_diff30);       // seed_0 vs seed_2
    }
  }
}

TEST_F(BatchSchedulerEvironmentSimulatorTest, SwapTokenGenerationTest) {
  // 创建两个请求
  int expected_output_token_num1 = 23;
  int input_token_num1 = 20;
  int expected_output_token_num2 = 12;
  int input_token_num2 = 30;
  std::shared_ptr<Request> req1, req2;

  std::vector<std::pair<int, int>> seeds;
  seeds.resize(1);
  seeds[0].first = 0;
  seeds[0].second = 1;
  std::vector<std::shared_ptr<InferRequest>> infer_req_list1 =
      env_simulator->InitRequest(1, input_token_num1, expected_output_token_num1, req1, seeds);
  seeds[0].second = 2;
  std::vector<std::shared_ptr<InferRequest>> infer_req_list2 =
      env_simulator->InitRequest(2, input_token_num2, expected_output_token_num2, req2, seeds);

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
    KLLM_LOG_DEBUG << "Step " << i << ": scheduled_reqs.size(): " << scheduled_reqs.size();
    env_simulator->RunAStep(scheduled_reqs);
    for (auto req : scheduled_reqs) {
      KLLM_LOG_DEBUG << "Step " << i << ": req_id:" << req->req_id
                     << ", output_token.size()=" << req->output_tokens.size()
                     << ", last output token= " << req->output_tokens.back();
    }
  }

  // Check request results
  env_simulator->CheckRequestOutput(infer_req1);
  env_simulator->CheckRequestOutput(infer_req2);
}
