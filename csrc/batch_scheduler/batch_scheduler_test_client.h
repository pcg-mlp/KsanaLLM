/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include "batch_scheduler/batch_scheduler_test_helper.h"
#include "batch_scheduler/schedule_output_process.h"
#include "batch_scheduler/schedule_processor.h"
#include "profiler/reporter.h"
#include "runtime/threadpool.h"

namespace ksana_llm {

class ClientSimulator {
 public:
  ClientSimulator(int thread_num, std::shared_ptr<ScheduleProcessorInterface> schedule_processor)
      : schedule_processor_(schedule_processor),
        thread_num_(thread_num),
        threadpool_(thread_num),
        is_destroying_(false) {
    threadpool_.Start();
  }

  ~ClientSimulator() {
    KLLM_LOG_INFO << "~ClientSimulator IsAllRequestFinished=" << IsAllRequestFinished();
    is_destroying_ = true;
    if (!IsAllRequestFinished()) {
      for (auto& it : client_req_map_) {
        it.second.req_group[0]->waiter->Notify();
        KLLM_LOG_INFO << "Notify unfinished req: " << it.second.req_group[0]->req_id;
      }
    }
    threadpool_.Stop();
  }

  void AddInferRequests(int req_group_id, std::vector<std::shared_ptr<InferRequest>>& infer_reqs) {
    std::unordered_map<int, ClientRequest>::iterator iter;
    KLLM_CHECK_WITH_INFO(infer_reqs.size() > 0, FormatStr("infer_reqs.size()==%d, must >0.", infer_reqs.size()));
    {
      std::lock_guard<std::mutex> guard(mux_);
      KLLM_CHECK_WITH_INFO(client_req_map_.find(req_group_id) == client_req_map_.end(),
                           FormatStr("req_group_id %d exists.", req_group_id));
      ClientRequest dummy_req;
      client_req_map_[req_group_id] = dummy_req;
      iter = client_req_map_.find(req_group_id);
      iter->second.req_group = infer_reqs;
    }
    threadpool_.Submit([=]() -> int {
      if (is_destroying_) {
        return 0;
      }
      ClientRequest& req = iter->second;
      req.enqueue_status = schedule_processor_->AddInferRequest(req.req_group);
      // all requests in req_group come from same request.
      req.req_group[0]->waiter->Wait();
      req.is_finished = true;
      return 0;
    });
  }

  void AddAnInferRequest(std::shared_ptr<InferRequest>& infer_req) {
    std::vector<std::shared_ptr<InferRequest>> reqs;
    reqs.push_back(infer_req);
    AddInferRequests(infer_req->req_id, reqs);
  }

  bool IsAllRequestStopped() {
    std::lock_guard<std::mutex> guard(mux_);
    for (auto& it : client_req_map_) {
      if (!it.second.req_group[0]->IsStopped()) {
        return false;
      }
    }
    return true;
  }

  bool IsAllRequestFinished() {
    for (auto& it : client_req_map_) {
      if (!it.second.is_finished) {
        return false;
      }
    }
    return true;
  }

 private:
  struct ClientRequest {
    std::vector<std::shared_ptr<InferRequest>> req_group;
    bool is_finished = false;
    Status enqueue_status;
  };

  std::shared_ptr<ScheduleProcessorInterface> schedule_processor_;
  int thread_num_;
  ThreadPool threadpool_;
  bool is_destroying_;

  std::unordered_map<int, ClientRequest> client_req_map_;
  std::mutex mux_;
};

class ParallelTester {
 public:
  ParallelTester(std::shared_ptr<ScheduleProcessorInterface> schedule_processor,
                 BatchSchedulerEnvironmentSimulator* env_simulator)
      : schedule_processor_(schedule_processor), env_simulator_(env_simulator) {}

  struct RequestInfo {
    int64_t req_id;
    int expect_output_token_num;
    int input_token_num;
    std::shared_ptr<Request> req;
    std::vector<std::shared_ptr<InferRequest>> infer_req_group;
  };

  // hooks used during execution.
  class ExeHookInterface {
   public:
    virtual void CheckRequestsBeforeAStep(const std::vector<std::shared_ptr<InferRequest>>& reqs) {}

    virtual void CheckScheduleResult(const std::shared_ptr<ScheduleResult>& schedule_result) {}

    virtual void CheckRequestsAfterExecution(const std::vector<RequestInfo>& reqs) {}

   public:
    int before_step_num = 0;
    int after_exe_num = 0;
  };

  // This hook checks results when  all requests are finished as expected.
  class DefaultResultCheckHook : public ExeHookInterface {
   public:
    explicit DefaultResultCheckHook(BatchSchedulerEnvironmentSimulator* env_simulator)
        : env_simulator_(env_simulator) {}
    ~DefaultResultCheckHook() {
      KLLM_LOG_INFO << "~DefaultResultCheckHook, after_exe_num=" << after_exe_num;
      EXPECT_GT(after_exe_num,
                0);  // CheckRequestsAfterExecution must be invoked. Maybe this hook is not added to hook list.
    }

    void CheckRequestsAfterExecution(const std::vector<RequestInfo>& reqs) override {
      after_exe_num++;
      for (auto& info : reqs) {
        for (auto& infer_req : info.infer_req_group) {
          env_simulator_->CheckRequestOutput(infer_req);
        }
      }
    }

   private:
    BatchSchedulerEnvironmentSimulator* env_simulator_;
  };

  // Check split fuse
  class SplitFuseCheckHook : public ExeHookInterface {
   public:
    explicit SplitFuseCheckHook(size_t split_fuse_token_num, size_t prefix_token_num = 0)
        : split_fuse_token_num_(split_fuse_token_num), prefix_token_num_(prefix_token_num) {}
    ~SplitFuseCheckHook() {
      KLLM_LOG_INFO << "~SplitFuseCheckHook, before_step_num=" << before_step_num
                    << ", after_exe_num=" << after_exe_num;
      EXPECT_GT(before_step_num, 0);
      EXPECT_GT(after_exe_num, 0);
    }

    void CheckRequestsBeforeAStep(const std::vector<std::shared_ptr<InferRequest>>& reqs) override {
      before_step_num++;
      for (auto req : reqs) {
        if (split_fuse_token_num_ > 0 && req->forwarding_tokens.size() < req->output_tokens.size()) {
          EXPECT_EQ((req->forwarding_tokens.size() - req->kv_cached_token_num), split_fuse_token_num_);
          req_split_step_[req->req_id]++;
        }
      }
    }

    void CheckRequestsAfterExecution(const std::vector<RequestInfo>& reqs) override {
      after_exe_num++;
      for (auto req : reqs) {
        if ((split_fuse_token_num_ > 0) && ((req.input_token_num - prefix_token_num_) > split_fuse_token_num_)) {
          EXPECT_GE(req_split_step_[req.infer_req_group[0]->req_id],
                    (req.input_token_num - prefix_token_num_) / split_fuse_token_num_);
        }
      }
    }

   private:
    std::unordered_map<int, int> req_split_step_;
    size_t split_fuse_token_num_;
    size_t prefix_token_num_;
  };

  class SwapCheckHook : public ExeHookInterface {
   public:
    explicit SwapCheckHook(bool expect_swap) : expect_swap_(expect_swap) {}
    ~SwapCheckHook() {
      if (expect_swap_) {
        // all swapout and swapin requests are merged and same number
        EXPECT_GT(swapout_req_num_, 0);
        EXPECT_EQ(swapout_req_num_, swapin_req_num_);
        EXPECT_EQ(swapout_req_num_, merged_swapout_req_num_);
        EXPECT_EQ(swapout_req_num_, merged_swapin_req_num_);
      } else {
        EXPECT_EQ(swapout_req_num_, 0);
        EXPECT_EQ(swapin_req_num_, 0);
        EXPECT_EQ(merged_swapout_req_num_, 0);
        EXPECT_EQ(merged_swapin_req_num_, 0);
      }
    }

    void CheckScheduleResult(const std::shared_ptr<ScheduleResult>& schedule_result) override {
      if (!schedule_result || !schedule_result->schedule_output) {
        return;
      }
      const auto& schedule_output = schedule_result->schedule_output;
      merged_swapout_req_num_ += CountMergedReqs(schedule_output->merged_swapout_req_ids);
      merged_swapin_req_num_ += CountMergedReqs(schedule_output->merged_swapin_req_ids);
      swapout_req_num_ += CountSwapReqs(schedule_output->swapout_req_block_ids);
      swapin_req_num_ += CountSwapReqs(schedule_output->swapin_req_block_ids);
    }

   private:
    static size_t CountMergedReqs(const std::vector<std::vector<int64_t>>& merged_req_ids) {
      size_t total = 0;
      for (const auto& ids : merged_req_ids) {
        total += ids.size();
      }
      return total;
    }

    static size_t CountSwapReqs(const std::vector<std::unordered_map<int64_t, SwapReqBlockInfo>>& swap_req_block_ids) {
      size_t total = 0;
      for (const auto& req_map : swap_req_block_ids) {
        total += req_map.size();
      }
      return total;
    }

   private:
    const bool expect_swap_;
    size_t merged_swapout_req_num_ = 0;
    size_t merged_swapin_req_num_ = 0;
    size_t swapout_req_num_ = 0;
    size_t swapin_req_num_ = 0;
  };

  // Generate RequestInfos
  void GenerateRequests(int request_num, int min_expect_output_num, int max_expect_output_num, int min_input_num,
                        int max_input_num, std::vector<ParallelTester::RequestInfo>& reqs) {
    KLLM_CHECK_WITH_INFO(
        min_expect_output_num < max_expect_output_num,
        FormatStr("min_expect_output_num %d should be larger than 0 and less than max_expect_output_num %d.",
                  min_expect_output_num, max_expect_output_num));

    KLLM_CHECK_WITH_INFO(
        min_input_num < max_input_num,
        FormatStr("min_input_num %d should be less than max_input_num %d.", min_input_num, max_input_num));

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

  // Add RequestInfo by single pair seeds
  void InitRequestInfoListByDefault(std::vector<RequestInfo>& reqs) {
    std::vector<std::pair<int, int>> seeds;
    seeds.resize(1);
    seeds[0].first = 0;
    for (auto& info : reqs) {
      seeds[0].second = info.req_id;
      info.infer_req_group =
          env_simulator_->InitRequest(info.req_id, info.input_token_num, info.expect_output_token_num, info.req, seeds);
    }
  }

  // Run requests in parallel and check
  void DoParallelRequestAndCheck(int client_num, std::vector<RequestInfo>& reqs, std::vector<ExeHookInterface*>& hooks,
                                 int timeout = 10) {
    KLLM_LOG_INFO << "DoParallelRequestAndCheck start.  client_num=" << client_num << ", request_num=" << reqs.size();
    KLLM_CHECK_WITH_INFO(hooks.size() > 0, "There must be some hooks");
    KLLM_CHECK_WITH_INFO(round < 1,
                         "DoParallelRequestAndCheck can be invoke once, because schedule_processor is stopped");
    round++;

    time_t start_time = ProfileTimer::GetCurrentTime();
    ClientSimulator client_simulator(client_num, schedule_processor_);

    // Create timeout monitoring thread
    std::atomic<bool> is_stopped(false);

    std::thread timeout_thread([this, &client_simulator, &is_stopped, start_time, timeout]() {
      auto deadline = start_time + timeout;
      KLLM_LOG_SCHEDULER << "timeout checking, timeout=" << timeout << "s";
      while (!is_stopped.load() && ProfileTimer::GetCurrentTime() < deadline) {
        if (client_simulator.IsAllRequestStopped()) {
          KLLM_LOG_SCHEDULER << "timeout checking. all requests stopped";
          break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
      }

      schedule_processor_->Stop();

      // Only call Stop() if actually timed out (not stopped normally)
      if (!is_stopped.load() && !client_simulator.IsAllRequestStopped()) {
        KLLM_LOG_SCHEDULER << "Not all requests stopped, exit because of timeout " << timeout << "s";
      } else {
        KLLM_LOG_SCHEDULER << "All requests completed normally";
      }
    });

    for (auto& info : reqs) {
      client_simulator.AddInferRequests(info.req_id, info.infer_req_group);
    }

    // Wait for request enqueue
    std::this_thread::sleep_for(std::chrono::microseconds(10));
    // schedule and generate tokens
    std::unordered_set<size_t> seen_schedule_ids;
    while (true) {
      std::shared_ptr<ScheduleResult> schedule_result = schedule_processor_->GetNextScheduleResult();
      if (!schedule_result) {
        // Only happen during stopping
        break;
      }

      // Check that schedule_id is always unique (never repeated)
      size_t current_schedule_id = schedule_result->schedule_output->schedule_id;
      EXPECT_EQ(seen_schedule_ids.find(current_schedule_id), seen_schedule_ids.end())
          << "Duplicate schedule_id detected: " << current_schedule_id;
      seen_schedule_ids.insert(current_schedule_id);

      schedule_processor_->Lock();
      for (auto hook : hooks) {
        hook->CheckScheduleResult(schedule_result);
      }

      auto& scheduled_reqs = schedule_result->schedule_output->running_reqs;
      for (auto hook : hooks) {
        hook->CheckRequestsBeforeAStep(scheduled_reqs);
      }

      env_simulator_->RunAStep(scheduled_reqs);
      schedule_processor_->Unlock();

      schedule_processor_->UpdateWithGenerationResult(schedule_result->generation_output_group);
      if (client_simulator.IsAllRequestStopped()) {
        KLLM_LOG_INFO << "All requests eos generated";
        break;
      }
    }

    // Signal timeout thread to stop
    is_stopped.store(true);
    if (timeout_thread.joinable()) {
      timeout_thread.join();
    }

    // Check request results
    for (auto hook : hooks) {
      KLLM_LOG_INFO << "CheckRequestsAfterExecution";
      hook->CheckRequestsAfterExecution(reqs);
    }
    KLLM_LOG_INFO << "DoParallelRequestAndCheck finished";
  }

 private:
  std::shared_ptr<ScheduleProcessorInterface> schedule_processor_;
  BatchSchedulerEnvironmentSimulator* env_simulator_;
  int round = 0;
};

class PrintStepHook : public ParallelTester::ExeHookInterface {
 public:
  explicit PrintStepHook(bool print_all_blocks = false) : print_all_blocks_(print_all_blocks) {}
  ~PrintStepHook() {
    KLLM_LOG_INFO << "~PrintStepHook, before_step_num=" << before_step_num;
    EXPECT_GT(before_step_num, 0);
  }

  void CheckRequestsBeforeAStep(const std::vector<std::shared_ptr<InferRequest>>& reqs) override {
    before_step_num++;
    for (auto& req : reqs) {
      std::ostringstream ss;
      ss << "BeforeRunStep " << before_step_num << req << ", output_tokens.size()=" << req->output_tokens.size()
         << ", sampling_result_tokens.size()=" << req->sampling_result_tokens.size();
      ss << ", tokens={ ";
      for (size_t i = 0; i < req->output_tokens.size(); i++) {
        ss << req->output_tokens[i] << ", ";
      }
      ss << "} ";
      KLLM_LOG_INFO << ss.str() << req->PrintKVBlockIds(print_all_blocks_);
    }
  }

 private:
  bool print_all_blocks_;
};

class FixPrefixTestCase {
 public:
  FixPrefixTestCase(int prefix_block_num, int block_token_num, int device_num, size_t splitfuse_token_num,
                    bool enable_swap = true, bool init_env_simulator = true)
      : prefix_block_num_(prefix_block_num),
        device_num_(device_num),
        splitfuse_token_num_(splitfuse_token_num),
        enable_swap_(enable_swap) {
    // 创建一个 BlockManagerConfig 对象，用于配置 BatchSchedulerEnvironmentSimulator
    if (enable_swap_) {
      block_manager_config_.host_allocator_config.blocks_num = 100;
    } else {
      block_manager_config_.host_allocator_config.blocks_num = 0;
    }
    block_manager_config_.device_allocator_config.blocks_num = 100;
    block_manager_config_.device_allocator_config.block_token_num = block_token_num;

    init_env_simulator_ = init_env_simulator;

    // 使用配置创建一个 BlockManagerSimulator 对象
    if (init_env_simulator_) {
      std::shared_ptr<FakedBlockAllocatorGroup> block_allocator_group =
          std::make_shared<FakedBlockAllocatorGroup>(block_manager_config_, device_num_);
      env_simulator_ =
          new BatchSchedulerEnvironmentSimulator(block_manager_config_, device_num_, block_allocator_group);
      KLLM_LOG_INFO << "Simulator start";
    }
  }

  ~FixPrefixTestCase() {
    if (init_env_simulator_) {
      delete env_simulator_;
    }
  }

  const BlockManagerConfig& GetBlockManagerConfig() const { return block_manager_config_; }

  void SetScheduleProcessor(std::shared_ptr<ScheduleProcessorInterface> schedule_processor) {
    schedule_processor_ = schedule_processor;
  }

  void SetEnvSimulator(BatchSchedulerEnvironmentSimulator* env_simulator) { env_simulator_ = env_simulator; }

  BatchSchedulerEnvironmentSimulator* GetEnvSimulator() { return env_simulator_; }

  void RunTestNoSwapTriggered() {
    int block_token_num = block_manager_config_.device_allocator_config.block_token_num;
    int block_num = block_manager_config_.device_allocator_config.blocks_num;

    int prefix_token_num = prefix_block_num_ * block_manager_config_.device_allocator_config.block_token_num;

    int request_num = 10;
    int client_num = 3;
    int min_expect_output_num = 3;
    int max_expect_output_num = min_expect_output_num + 3;
    int min_input_num = prefix_token_num;
    int max_input_num = 10 + prefix_token_num;

    // check request token num can not trigger swapout
    ASSERT_LT(client_num * (max_input_num + max_expect_output_num), block_num * block_token_num);

    RunTest(request_num, client_num, min_expect_output_num, max_expect_output_num, min_input_num, max_input_num,
            prefix_token_num, false);

    auto& stat = env_simulator_->GetBlockManagerStat();
    EXPECT_EQ(stat.swapout_succ_num, 0);
    EXPECT_EQ(stat.swapout_fail_num, 0);

    EXPECT_EQ(stat.swapin_succ_num, 0);
    EXPECT_EQ(stat.swapin_fail_num, 0);
  }

  void RunTestSwapTriggered() {
    int block_token_num = block_manager_config_.device_allocator_config.block_token_num;
    int block_num = block_manager_config_.device_allocator_config.blocks_num;

    int prefix_token_num = prefix_block_num_ * block_manager_config_.device_allocator_config.block_token_num;

    int request_num = 50;
    int client_num = 10;
    int min_expect_output_num = 3;
    int max_expect_output_num = min_expect_output_num + 150;
    int min_input_num = prefix_token_num;
    int max_input_num = 100 + prefix_token_num;

    // check request token num can trigger swapout
    ASSERT_LT((max_input_num + max_expect_output_num), block_num * block_token_num);
    ASSERT_GT(client_num * (min_input_num + min_expect_output_num), block_num * block_token_num);
    ASSERT_GT(request_num, client_num);

    RunTest(request_num, client_num, min_expect_output_num, max_expect_output_num, min_input_num, max_input_num,
            prefix_token_num, enable_swap_);

    auto& stat = env_simulator_->GetBlockManagerStat();
    if (enable_swap_) {
      EXPECT_GT(stat.swapout_succ_num, 0);
      EXPECT_EQ(stat.swapout_fail_num, 0);
      EXPECT_GT(stat.swapin_succ_num, 0);
      EXPECT_EQ(stat.swapin_fail_num, 0);
    } else {
      EXPECT_EQ(stat.swapout_succ_num, 0);  // recomputed
      EXPECT_EQ(stat.swapout_fail_num, 0);
      EXPECT_EQ(stat.swapin_succ_num, 0);  // recomputed
      EXPECT_EQ(stat.swapin_fail_num, 0);
    }
  }

 private:
  // In current implementation, requests will not share blocks in first step.
  // Test: all requests have same prefix token
  class SamePrefixCacheNoMergeBeforeFirstBatchCheckHook : public ParallelTester::ExeHookInterface {
   public:
    SamePrefixCacheNoMergeBeforeFirstBatchCheckHook(int prefix_block_num, int block_token_num, int tp_num)
        : prefix_block_num_(prefix_block_num), block_token_num_(block_token_num), tp_num_(tp_num) {
      prefix_blocks_.resize(prefix_block_num_);
    }

    ~SamePrefixCacheNoMergeBeforeFirstBatchCheckHook() {
      KLLM_LOG_INFO << "~SamePrefixCacheNoMergeBeforeFirstBatchCheckHook, before_step_num=" << before_step_num;
      EXPECT_GT(before_step_num, 0);
      EXPECT_GT(cache_hit_num_, 0);
    }

    void CheckRequestsBeforeAStep(const std::vector<std::shared_ptr<InferRequest>>& reqs) override {
      before_step_num++;
      ASSERT_GT(reqs.size(), 0);
      if (reqs[0]->forwarding_tokens.size() < prefix_block_num_ * block_token_num_) {
        return;
      }

      if (!is_cache_set_) {
        if (reqs_in_first_step_.size() == 0) {
          // record requests in first step. Their kv cache will be reused.
          for (auto& req : reqs) {
            reqs_in_first_step_[req->req_id] = req;
          }

          // Requests should not share same prefix cache blocks
          for (size_t i = 1; i < reqs.size(); i++) {
            EXPECT_FALSE(CheckBlocksSame(reqs[0]->kv_cache_blocks, reqs[i]->kv_cache_blocks, prefix_block_num_));
          }
          return;
        }

        // Blocks should be merged in second step.
        for (auto& req : reqs) {
          if (reqs_in_first_step_.find(req->req_id) != reqs_in_first_step_.end()) {
            if (!is_cache_set_) {
              StorePrefixCache(req);
            } else {
              EXPECT_TRUE(IsRequestPrefixCacheHit(req))
                  << "Cache setting batch error: " << req->ScheduleStateToStr()
                  << ", kv_caches=" << Vector2Str(req->kv_cache_blocks) << ", prefix_block_num_=" << prefix_block_num_
                  << ", prefix_blocks_=" << Vector2Str(prefix_blocks_);
            }
          }
        }
        return;
      }

      // All request should use same prefix caching after cache is set
      for (auto& req : reqs) {
        EXPECT_TRUE(IsRequestPrefixCacheHit(req))
            << "Later round error: " << req->ScheduleStateToStr() << ", kv_caches=" << Vector2Str(req->kv_cache_blocks)
            << ", prefix_block_num_=" << prefix_block_num_ << ", prefix_blocks_=" << Vector2Str(prefix_blocks_);
        cache_hit_num_++;
      }
    }

   private:
    bool IsRequestPrefixCacheHit(const std::shared_ptr<InferRequest>& req) {
      return CheckBlocksSame(req->kv_cache_blocks, prefix_blocks_, prefix_block_num_);
    }

    bool CheckBlocksSame(const std::vector<int>& this_blocks, const std::vector<int>& that_blocks, int block_num) {
      for (int j = 0; j < block_num; j++) {
        if (this_blocks[j] != that_blocks[j]) {
          return false;
        }
      }
      return true;
    }

    void StorePrefixCache(const std::shared_ptr<InferRequest>& req) {
      ASSERT_FALSE(is_cache_set_);
      auto& kv_cache_blocks = req->kv_cache_blocks;
      std::copy(kv_cache_blocks.begin(), kv_cache_blocks.begin() + prefix_block_num_, prefix_blocks_.begin());
      is_cache_set_ = true;
    }

   private:
    size_t prefix_block_num_;
    size_t block_token_num_;
    int tp_num_;
    std::vector<int> prefix_blocks_;

    bool is_cache_set_ = false;

    std::unordered_map<int, std::shared_ptr<InferRequest>> reqs_in_first_step_;

    int cache_hit_num_ = 0;
  };

  void RunTest(int request_num, int client_num, int min_expect_output_num, int max_expect_output_num, int min_input_num,
               int max_input_num, int prefix_token_num, bool expect_swap) {
    ParallelTester tester(schedule_processor_, env_simulator_);

    std::vector<ParallelTester::ExeHookInterface*> hooks;
    ParallelTester::DefaultResultCheckHook default_hook(env_simulator_);
    ParallelTester::SplitFuseCheckHook splitfuse_check_hook(splitfuse_token_num_, prefix_token_num);
    ParallelTester::SwapCheckHook swap_check_hook(expect_swap);
    //    PrintStepHook print_hook(true);
    SamePrefixCacheNoMergeBeforeFirstBatchCheckHook prefix_check_hook(
        prefix_token_num / block_manager_config_.device_allocator_config.block_token_num,
        block_manager_config_.device_allocator_config.block_token_num, device_num_);
    hooks.push_back(&default_hook);
    hooks.push_back(&splitfuse_check_hook);
    hooks.push_back(&swap_check_hook);
    //    hooks.push_back(&print_hook);
    hooks.push_back(&prefix_check_hook);

    std::vector<ParallelTester::RequestInfo> req_list;
    tester.GenerateRequests(request_num, min_expect_output_num, max_expect_output_num, prefix_token_num, max_input_num,
                            req_list);

    // Init requests with same prefix tokens.
    std::vector<std::pair<int, int>> seeds;
    seeds.resize(2);
    seeds[0].first = 0;
    seeds[0].second = 1;
    seeds[1].first = prefix_token_num;
    for (auto& info : req_list) {
      seeds[1].second = info.req_id;
      info.infer_req_group =
          env_simulator_->InitRequest(info.req_id, info.input_token_num, info.expect_output_token_num, info.req, seeds);
    }

    tester.DoParallelRequestAndCheck(client_num, req_list, hooks, 60);
  }

 private:
  int prefix_block_num_;
  int device_num_;
  size_t splitfuse_token_num_;
  bool enable_swap_;

  std::shared_ptr<ScheduleProcessorInterface> schedule_processor_;
  BatchSchedulerEnvironmentSimulator* env_simulator_ = nullptr;
  BlockManagerConfig block_manager_config_;

  bool init_env_simulator_ = true;
};

}  // namespace ksana_llm
