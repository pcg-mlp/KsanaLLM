/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <exception>
#include <memory>
#include <thread>

#include "batch_scheduler/batch_scheduler_test.h"

#include "batch_scheduler/event_driven_scheduler/continuous_batching_backend.h"
#include "batch_scheduler/event_driven_scheduler/event_driven_schedule_processor.h"
#include "batch_scheduler/event_driven_scheduler/schedule_hook_manager.h"
#include "batch_scheduler/event_driven_scheduler/workload_balance/basic_workload_balancer.h"

#include "profiler/timer.h"
#include "utils/grammar_backend.h"
#include "utils/logger.h"
#include "utils/singleton.h"

using namespace ksana_llm;

class EventDrivenSchedulerTest : public BaseSchedulerTest {
 protected:
  void InitScheduleProcessor() override {
    const size_t attn_dp_size = 1;
    auto event_driven_schedule_processor =
        std::make_shared<EventDrivenScheduleProcessor>(attn_dp_size, batch_scheduler_config_.max_pp_batch_num);

    event_driven_schedule_processor->SetScheduleBackend(
        std::make_shared<ContinuousBatchingBackend>(batch_scheduler_config_));

    event_driven_schedule_processor->SetCacheManagers({cache_manager});
    event_driven_schedule_processor->SetScheduleHookManager(std::make_shared<ScheduleHookManager>());

    event_driven_schedule_processor->SetWorkloadBalancer(std::make_shared<BasicWorkloadBalancer>(
        attn_dp_size, static_cast<size_t>(batch_scheduler_config_.max_batch_size),
        static_cast<size_t>(batch_scheduler_config_.max_decode_tokens_per_req),
        static_cast<size_t>(batch_scheduler_config_.max_step_token_num)));

    schedule_processor_ = event_driven_schedule_processor;
    schedule_processor_->Initialize(nullptr);
  }
};

TEST_F(EventDrivenSchedulerTest, BasicTokenGenerationTest) {
  CommonSetUp();
  ParallelTester tester(schedule_processor_, env_simulator_);

  std::vector<ParallelTester::ExeHookInterface*> hooks;
  ParallelTester::DefaultResultCheckHook default_hook(env_simulator_);
  hooks.push_back(&default_hook);

  // Run requests one by one
  int request_num = 30;
  int client_num = 1;
  int max_expect_output_num = 100;
  int max_input_num = 400;
  std::vector<ParallelTester::RequestInfo> req_list;
  tester.GenerateRequests(request_num, 1, max_expect_output_num, 1, max_input_num, req_list);
  tester.InitRequestInfoListByDefault(req_list);
  tester.DoParallelRequestAndCheck(client_num, req_list, hooks);

  auto& stat = env_simulator_->GetBlockManagerStat();
  EXPECT_EQ(stat.swapout_succ_num, 0);
  EXPECT_EQ(stat.swapout_fail_num, 0);
  EXPECT_EQ(stat.swapin_succ_num, 0);
  EXPECT_EQ(stat.swapin_fail_num, 0);
}

TEST_F(EventDrivenSchedulerTest, SwapOutInNotTriggeredPressTest) {
  CommonSetUp();
  ParallelTester tester(schedule_processor_, env_simulator_);

  std::vector<ParallelTester::ExeHookInterface*> hooks;
  ParallelTester::DefaultResultCheckHook default_hook(env_simulator_);
  hooks.push_back(&default_hook);

  // Run requests in parallel
  // input and max output token are limited, SwapOut/In are not triggered.
  int request_num = 100;
  int client_num = 10;
  int max_expect_output_num = 40;
  int max_input_num = 60;
  std::vector<ParallelTester::RequestInfo> req_list;
  tester.GenerateRequests(request_num, 1, max_expect_output_num, 1, max_input_num, req_list);
  tester.InitRequestInfoListByDefault(req_list);
  tester.DoParallelRequestAndCheck(client_num, req_list, hooks);

  auto& stat = env_simulator_->GetBlockManagerStat();
  EXPECT_EQ(stat.swapout_succ_num, 0);
  EXPECT_EQ(stat.swapout_fail_num, 0);
  EXPECT_EQ(stat.swapin_succ_num, 0);
  EXPECT_EQ(stat.swapin_fail_num, 0);
}

TEST_F(EventDrivenSchedulerTest, SwapOutInTriggeredPressTest) {
  CommonSetUp();
  ParallelTester tester(schedule_processor_, env_simulator_);

  std::vector<ParallelTester::ExeHookInterface*> hooks;
  ParallelTester::DefaultResultCheckHook default_hook(env_simulator_);
  hooks.push_back(&default_hook);

  // Run requests in parallel
  // max output token are large, SwapOut/In will be triggered when there are multiple requests.
  // exceed cache size, not exceed_batch size and max_step_size
  int request_num = 10;
  int client_num = 10;
  int min_expect_output_num = 1;
  int max_expect_output_num = 50;
  int min_input_num = 230;
  int max_input_num = 300;
  std::vector<ParallelTester::RequestInfo> req_list;
  tester.GenerateRequests(request_num, min_expect_output_num, max_expect_output_num, min_input_num, max_input_num,
                          req_list);
  tester.InitRequestInfoListByDefault(req_list);
  tester.DoParallelRequestAndCheck(client_num, req_list, hooks, 60);

  auto& stat = env_simulator_->GetBlockManagerStat();
  EXPECT_GT(stat.swapout_succ_num, 0);
  EXPECT_EQ(stat.swapout_fail_num, 0);
  EXPECT_GT(stat.swapin_succ_num, 0);
  EXPECT_EQ(stat.swapin_fail_num, 0);
}

TEST_F(EventDrivenSchedulerTest, SplitFusePressTest) {
  KLLM_LOG_INFO << "SplitFusePressTest start";
  split_fuse_token_num_ = 16;
  CommonSetUp();
  ParallelTester tester(schedule_processor_, env_simulator_);

  std::vector<ParallelTester::ExeHookInterface*> hooks;
  ParallelTester::DefaultResultCheckHook default_hook(env_simulator_);
  ParallelTester::SplitFuseCheckHook split_fuse_check_hook(split_fuse_token_num_);
  hooks.push_back(&default_hook);
  hooks.push_back(&split_fuse_check_hook);

  // Run requests in parallel
  // max output token are large, SwapOut/In will be triggered when there are multiple requests.
  // exceed cache size, not exceed_batch size and max_step_size
  int request_num = 10;
  int client_num = 10;
  int min_expect_output_num = 1;
  int max_expect_output_num = 40;
  int min_input_num = 17;
  int max_input_num = 60;
  std::vector<ParallelTester::RequestInfo> req_list;
  tester.GenerateRequests(request_num, min_expect_output_num, max_expect_output_num, min_input_num, max_input_num,
                          req_list);
  tester.InitRequestInfoListByDefault(req_list);
  tester.DoParallelRequestAndCheck(client_num, req_list, hooks, 60);

  auto& stat = env_simulator_->GetBlockManagerStat();
  EXPECT_EQ(stat.swapout_succ_num, 0);  // Recomputed, no swap
  EXPECT_EQ(stat.swapout_fail_num, 0);
  EXPECT_EQ(stat.swapin_succ_num, 0);  // Recomputed, no swap
  EXPECT_EQ(stat.swapin_fail_num, 0);
}

TEST_F(EventDrivenSchedulerTest, FixPrefixCacheNoSwapTriggeredTest) {
  enable_prefix_cache_ = true;
  CommonSetUp();

  int prefix_block_num = 3;
  int block_token_num = 6;
  int device_num = 4;
  FixPrefixTestCase test_case(prefix_block_num, block_token_num, device_num, split_fuse_token_num_, true, false);
  test_case.SetScheduleProcessor(schedule_processor_);
  test_case.SetEnvSimulator(env_simulator_);
  test_case.RunTestNoSwapTriggered();
}

TEST_F(EventDrivenSchedulerTest, FixPrefixCacheNoSwapTriggeredSplitfuseTest) {
  enable_prefix_cache_ = true;
  split_fuse_token_num_ = 6;
  CommonSetUp();

  int prefix_block_num = 3;
  int block_token_num = 6;
  int device_num = 4;
  FixPrefixTestCase test_case(prefix_block_num, block_token_num, device_num, split_fuse_token_num_, true, false);
  test_case.SetScheduleProcessor(schedule_processor_);
  test_case.SetEnvSimulator(env_simulator_);
  test_case.RunTestNoSwapTriggered();
}

TEST_F(EventDrivenSchedulerTest, FixPrefixCacheSwapTriggeredTest) {
  enable_prefix_cache_ = true;
  FixPrefixCacheBlockLimitTriggeredTest();
}

TEST_F(EventDrivenSchedulerTest, FixPrefixCacheRecomputeTriggeredTest) {
  enable_prefix_cache_ = true;
  enable_swap_ = false;
  FixPrefixCacheBlockLimitTriggeredTest();
}
/*
TEST_F(EventDrivenSchedulerTest, MultiBatchFixPrefixCacheRecomputeTriggeredTest) {
  enable_prefix_cache_ = true;
  enable_swap_ = false;
  max_pp_batch_num_ = 2;
  FixPrefixCacheBlockLimitTriggeredTest();
}
*/
TEST_F(EventDrivenSchedulerTest, CheckRequestTimeoutTest) {
  waiting_timeout_in_ms_ = 1000;
  CommonSetUp();
  size_t timeout_in_ms = waiting_timeout_in_ms_;
  ParallelTester tester(schedule_processor_, env_simulator_);

  std::vector<ParallelTester::ExeHookInterface*> hooks;

  class TimeoutHook : public ParallelTester::ExeHookInterface {
   public:
    explicit TimeoutHook(size_t timeout_in_ms, size_t step_req_num)
        : timeout_in_ms_(timeout_in_ms), step_req_num_(step_req_num) {}
    ~TimeoutHook() {
      KLLM_LOG_INFO << "~TimeoutHook, after_exe_num=" << after_exe_num;
      EXPECT_GT(before_step_num, 0);
      EXPECT_EQ(req_steps_.size(), step_req_num_);
    }

    void CheckRequestsBeforeAStep(const std::vector<std::shared_ptr<InferRequest>>& reqs) {
      before_step_num++;
      std::this_thread::sleep_for(std::chrono::milliseconds(timeout_in_ms_));
      for (auto& req : reqs) {
        req_steps_[req->req_id]++;
      }
    }

   private:
    size_t timeout_in_ms_;
    size_t step_req_num_;
    std::unordered_map<int, int> req_steps_;
  };
  TimeoutHook timeout_hook(timeout_in_ms + 10, 1);
  hooks.push_back(&timeout_hook);

  // Run requests one by one
  int request_num = 2;
  int client_num = 1;
  int max_expect_output_num = 100;
  int max_input_num = 400;
  std::vector<ParallelTester::RequestInfo> req_list;
  tester.GenerateRequests(request_num, 1, max_expect_output_num, 0, max_input_num, req_list);
  tester.InitRequestInfoListByDefault(req_list);

  req_list[0].infer_req_group[0]->timestamp_in_us = 1;  // will timeout in waiting queue
  tester.DoParallelRequestAndCheck(client_num, req_list, hooks, 3);

  EXPECT_EQ(req_list[0].req->finish_status.GetCode(), RET_REQUEST_TIMEOUT);
}

TEST_F(EventDrivenSchedulerTest, FlexibleCacheTaskTest) {
  enable_prefix_cache_ = true;
  enable_flexible_cache_ = true;

  int dp_num = 1;
  int tp_num = 1;
  int ep_world_size = 1;
  CommonSetUp(dp_num, tp_num, ep_world_size);
  ParallelTester tester(schedule_processor_, env_simulator_);

  std::vector<ParallelTester::ExeHookInterface*> hooks;
  ParallelTester::DefaultResultCheckHook default_hook(env_simulator_);
  hooks.push_back(&default_hook);

  int request_num = 2;
  int client_num = 1;
  int max_expect_output_num = 2;
  int input_num = 300;
  int del_token_idx = 40;
  std::vector<int> input_tokens(input_num);
  std::iota(input_tokens.begin(), input_tokens.end(), 1);

  std::vector<ParallelTester::RequestInfo> req_list;
  tester.GenerateRequests(request_num, 1, max_expect_output_num, input_num, input_num + 1, req_list);
  tester.InitRequestInfoListByDefault(req_list);

  const auto& req_0 = req_list[0].infer_req_group[0];
  req_0->input_tokens.assign(input_tokens.begin(), input_tokens.end());
  req_0->return_cache_stat = true;

  const auto& req_1 = req_list[1].infer_req_group[0];
  req_1->input_tokens.assign(input_tokens.begin(), input_tokens.end());
  req_1->input_tokens.erase(req_1->input_tokens.begin() + del_token_idx);
  req_1->return_cache_stat = true;
  tester.DoParallelRequestAndCheck(client_num, req_list, hooks);

  std::vector<std::pair<size_t, size_t>> cache_stat_0{{0, 0}};
  EXPECT_EQ(req_0->cache_stat, cache_stat_0);

  size_t prefix_tokens =
      DivRoundDown(del_token_idx, cache_manager_config.block_token_num) * cache_manager_config.block_token_num;
  size_t flexible_tokens = input_num - 1 - prefix_tokens - cache_manager_config.block_token_num;
  EXPECT_EQ(req_1->flexible_cache_len, flexible_tokens);
  std::vector<std::pair<size_t, size_t>> cache_stat_1{{prefix_tokens, prefix_tokens},
                                                      {prefix_tokens, prefix_tokens + flexible_tokens}};
  EXPECT_EQ(req_1->cache_stat, cache_stat_1);
}
