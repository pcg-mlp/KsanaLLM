/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "batch_scheduler/batch_scheduler_test_helper.h"
#include "batch_scheduler/batch_scheduler_test_client.h"

#include <memory>
#include <numeric>
#include "test.h"
#include "utils/logger.h"
#include "utils/singleton.h"

using namespace ksana_llm;

// 定义一个 BlockSchedulerTest 类，用于测试BatchSchedulerEnvironmentSimulator
class BatchSchedulerEnvironmentSimulatorTest : public testing::Test {
 protected:
  static void SetUpTestSuite() { InitLoguru(); }

  // 在每个测试用例执行之前调用的函数
  void CommonSetUp() {
    // 创建一个 BlockManagerConfig 对象，用于配置 BatchSchedulerEnvironmentSimulator
    block_manager_config.host_allocator_config.blocks_num = 100;
    block_manager_config.device_allocator_config.blocks_num = 100;
    block_manager_config.device_allocator_config.block_token_num = 6;
    device_num = 2;

    std::shared_ptr<FakedBlockAllocatorGroup> block_allocator_group =
        std::make_shared<FakedBlockAllocatorGroup>(block_manager_config, device_num);

    // 使用配置创建一个 BlockManagerSimulator 对象
    env_simulator = new BatchSchedulerEnvironmentSimulator(block_manager_config, device_num, block_allocator_group);
    KLLM_LOG_INFO << "Simulator start";
  }

  // 在每个测试用例执行之后调用的函数
  void TearDown() override {
    // 删除 BatchScheduler 对象
    if (env_simulator) {
      delete env_simulator;
    }
  }

  void InitRequestBlock(std::shared_ptr<InferRequest>& req, int output_token_num) {
    // Allocate all blocks at the beginning;
    int block_token_num = block_manager_config.device_allocator_config.block_token_num;
    int total_block_num = (req->input_tokens.size() + output_token_num + block_token_num - 1) / block_token_num;
    KLLM_LOG_INFO << "Start init req " << req->req_id << ", block num =" << total_block_num;
    std::vector<int> blocks;
    env_simulator->GetBlockAllocatorGroup()->GetDeviceBlockAllocator(0)->AllocateBlocks(total_block_num, blocks);
    req->kv_cache_blocks.insert(req->kv_cache_blocks.end(), blocks.begin(), blocks.end());
    KLLM_LOG_INFO << "req " << req->req_id << ", kv_cache_blocks size=" << req->kv_cache_blocks.size();
    KLLM_LOG_INFO << "Init infer request " << req->req_id << " block=" << total_block_num;
  }

 protected:
  // 定义一个 BatchSchedulerEnvironmentSimulator 指针，用于在测试用例中使用
  BatchSchedulerEnvironmentSimulator* env_simulator = nullptr;

  BlockManagerConfig block_manager_config;
  int device_num;
};

TEST_F(BatchSchedulerEnvironmentSimulatorTest, BasicTokenGenerationTest) {
  CommonSetUp();
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

  for (auto req : infer_reqs) {
    req->ResetPrefillingTokens();
  }

  int max_output_step = std::max(expected_output_token_num1, expected_output_token_num2) + 10;
  for (int i = 0; i < max_output_step; i++) {
    std::vector<std::shared_ptr<InferRequest>> running_reqs;
    KLLM_LOG_DEBUG << "Scheduling step=" << i;
    for (auto& req : infer_reqs) {
      if (req->finished) {
        continue;
      }
      req->SetPlanningTask();
      req->LaunchPlanningTask();
      running_reqs.push_back(req);
    }

    if (running_reqs.empty()) break;
    KLLM_LOG_DEBUG << "Step " << i << ": running_reqs.size(): " << running_reqs.size();
    env_simulator->RunAStep(running_reqs);

    KLLM_LOG_DEBUG << "After step " << i;
    for (auto req : running_reqs) {
      req->UpdateAfterInflightTaskFinished();
      req->ResetInflightTask();
      if (req->IsEosGenerated()) {
        req->finished = true;
      }
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

std::string KvCaches2Str(const std::vector<int>& kv_cache_blocks) {
  std::ostringstream ss;
  ss << "blocks={ ";
  for (auto blk_id : kv_cache_blocks) {
    ss << blk_id << ", ";
  }
  return ss.str();
}

// All request must have same prefix tokens
// Requests should not trigger swapout or rerun
class FixPrefixBatchScheduler : public BatchSchedulerInterface {
 public:
  FixPrefixBatchScheduler(BatchSchedulerEnvironmentSimulator* env_simulator, int prefix_token_num, int tp_num)
      : env_simulator_(env_simulator), prefix_token_num_(prefix_token_num), tp_num_(tp_num), schedule_id_(0) {
    block_token_num_ = env_simulator_->GetBlockManagerConfig().device_allocator_config.block_token_num;
    KLLM_CHECK_WITH_INFO(
        prefix_token_num % block_token_num_ == 0,
        FormatStr("prefix_token_num(%d) is not aligned with block_token_num(%d)", prefix_token_num, block_token_num_));
    prefix_block_num_ = prefix_token_num_ / block_token_num_;

    PreparePrefixCacheBlocks();

    schedule_output = new ScheduleOutput();
  }

  virtual ~FixPrefixBatchScheduler() {
    if (schedule_output != nullptr) {
      free(schedule_output);
      schedule_output = nullptr;
    }
  }

  std::shared_ptr<CacheManagerInterface>& GetCacheManager(int attn_dp_idx) { return dummy_cache_mgr_; }

  bool TryToLaunchPlannedScheduleOutput(size_t multi_batch_id, ScheduleOutput& merged_schedule_output,
                                        std::vector<std::shared_ptr<InferRequest>>& stopped_reqs) override {
    assert(merged_schedule_output.IsLaunchable());
    merged_schedule_output.LaunchScheduleOutput();
    return true;
  }

  void UpdateWithGenerationResult(size_t multi_batch_id, const GenerationOutputGroup& generation_output) override {
    assert(generation_output.reqs.size() == 1);                       // only support one dp
    assert(generation_output.reqs[0].size() == running_reqs.size());  //

    for (auto req : generation_output.reqs[0]) {
      req->UpdateAfterInflightTaskFinished();
      req->ResetInflightTask();
      if (req->IsEosGenerated()) {
        ClearRequestAfterFinished(req);
        auto it = std::find(running_reqs.begin(), running_reqs.end(), req);
        KLLM_CHECK(it != running_reqs.end());
        KLLM_CHECK((*it)->req_id == req->req_id);
        KLLM_LOG_DEBUG << "Remove request from queue, req id:" << req->req_id;
        running_reqs.erase(it);
        continue;
      }
      ScheduleTaskWorkload planning_workload;
      planning_workload.generated_token_num = req->generated_tokens.size();
      req->SetPlanningWorkload(planning_workload);
    }
  }

  std::shared_ptr<ScheduleOutputGroup> Schedule(size_t multi_batch_id) override {
    assert(multi_batch_id == 0);  // only support one batch
    std::this_thread::sleep_for(std::chrono::microseconds(1));
    KLLM_LOG_DEBUG << " ============= Schedule, step " << step_ << ", running_reqs.size=" << running_reqs.size()
                   << ", waiting_reqs.size=" << waiting_reqs.size();

    std::lock_guard<std::mutex> guard(mux_);
    if (running_reqs.size() > 0) {
      for (auto it = running_reqs.begin(); it != running_reqs.end();) {
        auto& req = *it;
        KLLM_CHECK(!req->HasInflightTask());

        // Need to add block?
        if (req->GetPlanningSequenceLen() > (req->kv_cache_blocks.size() * block_token_num_)) {
          AdjustRequestKvCacheBlocks(req);
        }
        it++;
      }
      if (step_ == 1) {
        // Merge blocks with req[0]
        for (size_t i = 1; i < running_reqs.size(); i++) {
          auto& req = running_reqs[i];
          KLLM_LOG_DEBUG << "Merging blocks. req_id=" << req->req_id
                         << ", kv_cache_blocks=" << KvCaches2Str(req->kv_cache_blocks);
          std::vector<int> blocks_to_drop(prefix_block_num_);
          std::copy(req->kv_cache_blocks.begin(), req->kv_cache_blocks.begin() + prefix_block_num_,
                    blocks_to_drop.begin());
          env_simulator_->GetBlockAllocatorGroup()->GetDeviceBlockAllocator(0)->FreeBlocks(blocks_to_drop);
          FillPrefixCacheBlocks(req);
        }
      }
      step_++;
    }

    // Waiting req starts after running reqs finished.
    if (waiting_reqs.size() > 0 && running_reqs.size() == 0) {
      // Init request and add to running queue;
      if (step_ == 0) {
        // Only first request fill with prefix cache
        for (size_t i = 0; i < waiting_reqs.size(); i++) {
          auto& req = waiting_reqs[i];
          if (i == 0) {
            req->kv_cache_blocks.resize(prefix_block_num_);
            FillPrefixCacheBlocks(req);
          }
          AdjustRequestKvCacheBlocks(req);
        }
      } else {
        // All requests fill with prefix cache
        for (auto& req : waiting_reqs) {
          req->kv_cache_blocks.resize(prefix_block_num_);
          FillPrefixCacheBlocks(req);
          AdjustRequestKvCacheBlocks(req);
        }
      }
      running_reqs.swap(waiting_reqs);
      step_++;
    }
    KLLM_LOG_DEBUG << " ========= Schedule, running_reqs.size = " << running_reqs.size();
    schedule_output->running_reqs = running_reqs;
    schedule_output->SetPlanningTask();

    std::shared_ptr<ScheduleOutputGroup> schedule_output_group = std::make_shared<ScheduleOutputGroup>();
    schedule_output_group->outputs[0] = schedule_output;
    schedule_output_group->schedule_id = schedule_id_++;
    return schedule_output_group;
  }

  virtual Status AddInferRequest(std::vector<std::shared_ptr<InferRequest>>& infer_request_group) override {
    for (auto& req : infer_request_group) {
      AddAnInferRequest(req);
    }
    return Status();
  }

  virtual void SetCacheManager(std::shared_ptr<CacheManagerInterface> cache_manager, int idx) override {}

  virtual bool IsIdle(size_t multi_batch_id) override { return false; }

  virtual void WaitUntilHaveReqs(size_t multi_batch_id) override {}

  void Stop() override {}

 private:
  Status AddAnInferRequest(std::shared_ptr<InferRequest>& infer_req) {
    std::lock_guard<std::mutex> guard(mux_);
    auto& input_tokens = infer_req->input_tokens;
    if (prefix_cache_tokens_.empty()) {
      // init with the first request
      prefix_cache_tokens_.resize(prefix_token_num_, 0);
      std::copy(input_tokens.begin(), input_tokens.begin() + prefix_token_num_, prefix_cache_tokens_.begin());
    }
    // set prefill info and planning workload
    infer_req->output_tokens = infer_req->input_tokens;
    infer_req->ResetPrefillingTokens();
    infer_req->SetPlanningWorkload(infer_req->GetRemainingWorkload());

    waiting_reqs.push_back(infer_req);
    return Status();
  }

  void ClearRequestAfterFinished(std::shared_ptr<InferRequest>& infer_req) {
    ASSERT_TRUE(env_simulator_->IsRequestFinished(infer_req));
    KLLM_LOG_DEBUG << "ClearRequestAfterFinished. req_id=" << infer_req->req_id
                   << ", output_tokens.size=" << infer_req->output_tokens.size()
                   << ", kv_cache_blocks=" << KvCaches2Str(infer_req->kv_cache_blocks);
    infer_req->finished = true;
    infer_req->Notify();
    infer_req->Stop();
    return;
    // Free blocks after prefix blocks
    auto& kv_cache_blocks = infer_req->kv_cache_blocks;
    std::vector<int> free_blocks(kv_cache_blocks.size() - prefix_block_num_);
    std::copy(kv_cache_blocks.begin() + prefix_block_num_, kv_cache_blocks.end(), free_blocks.begin());
    env_simulator_->GetBlockAllocatorGroup()->GetDeviceBlockAllocator(0)->FreeBlocks(free_blocks);
  }

  void PreparePrefixCacheBlocks() {
    env_simulator_->GetBlockAllocatorGroup()->GetDeviceBlockAllocator(0)->AllocateBlocks(prefix_block_num_,
                                                                                         prefix_cache_blocks_);
    KLLM_LOG_INFO << "set prefix_cache_blocks =" << KvCaches2Str(prefix_cache_blocks_);
  }

  bool CheckReqIsValidForPrefixCache(const std::vector<int>& input_tokens) {
    for (int token_idx = 0; token_idx < prefix_token_num_; ++token_idx) {
      if (prefix_cache_tokens_[token_idx] != input_tokens[token_idx]) {
        return false;
      }
    }
    return true;
  }

  void FillPrefixCacheBlocks(std::shared_ptr<InferRequest>& req) {
    auto& kv_cache_blocks = req->kv_cache_blocks;
    for (int i = 0; i < prefix_block_num_; i++) {
      kv_cache_blocks[i] = prefix_cache_blocks_[i];
    }
    KLLM_LOG_DEBUG << " Fill prefix, req_id= " << req->req_id << ", after fill = " << KvCaches2Str(kv_cache_blocks);
  }

  void AdjustRequestKvCacheBlocks(const std::shared_ptr<InferRequest>& req) {
    auto& kv_cache_blocks = req->kv_cache_blocks;
    int adding_block_num =
        (req->GetPlanningSequenceLen() + block_token_num_ - 1) / block_token_num_ - kv_cache_blocks.size();
    std::vector<int> new_blocks;
    env_simulator_->GetBlockAllocatorGroup()->GetDeviceBlockAllocator(0)->AllocateBlocks(adding_block_num, new_blocks);
    kv_cache_blocks.insert(kv_cache_blocks.end(), new_blocks.begin(), new_blocks.end());
    KLLM_LOG_DEBUG << "AdjustRequestKvCache, req_id= " << req->req_id
                   << ", output_tokens.size()=" << req->output_tokens.size() << ", add " << adding_block_num
                   << " blocks, kv_cache=" << KvCaches2Str(req->kv_cache_blocks);
  }

 private:
  BatchSchedulerEnvironmentSimulator* env_simulator_;
  int prefix_token_num_;
  int tp_num_;
  int block_token_num_;
  size_t schedule_id_;

  int step_ = 0;
  std::vector<std::shared_ptr<InferRequest>> waiting_reqs;
  std::vector<std::shared_ptr<InferRequest>> running_reqs;

  int prefix_block_num_;
  std::vector<int> prefix_cache_blocks_;
  std::vector<int> prefix_cache_tokens_;

  std::mutex mux_;
  std::shared_ptr<CacheManagerInterface> dummy_cache_mgr_ = nullptr;

  ScheduleOutput* schedule_output = nullptr;
};

TEST_F(BatchSchedulerEnvironmentSimulatorTest, SwaBasicTokenGenerationTest) {
  // Parameters per plan section 3.4
  const int sliding_window_size = 12;  // = 2 * block_token_num
  const int block_token_num = 6;
  const int device_num = 2;
  const int input_token_num = 20;
  const int expected_output_token_num = 30;

  // Set up block manager config with the specified block_token_num
  block_manager_config.host_allocator_config.blocks_num = 100;
  block_manager_config.device_allocator_config.blocks_num = 100;
  block_manager_config.device_allocator_config.block_token_num = block_token_num;
  this->device_num = device_num;

  // Create main KV cache block allocator group
  std::shared_ptr<FakedBlockAllocatorGroup> main_block_allocator_group =
      std::make_shared<FakedBlockAllocatorGroup>(block_manager_config, device_num);

  // Create SWA block allocator group (no host blocks needed for SWA)
  BlockManagerConfig swa_block_manager_config = block_manager_config;
  swa_block_manager_config.host_allocator_config.blocks_num = 0;
  std::shared_ptr<FakedBlockAllocatorGroup> swa_block_allocator_group =
      std::make_shared<FakedBlockAllocatorGroup>(swa_block_manager_config, device_num);

  // Create simulator with SWA enabled
  if (env_simulator) {
    delete env_simulator;
    env_simulator = nullptr;
  }
  env_simulator =
      new BatchSchedulerEnvironmentSimulator(block_manager_config, device_num, main_block_allocator_group,
                                             static_cast<size_t>(sliding_window_size), swa_block_allocator_group);

  // Init request
  std::shared_ptr<Request> req;
  std::vector<std::pair<int, int>> seeds;
  seeds.push_back(std::make_pair(0, 42));
  std::vector<std::shared_ptr<InferRequest>> infer_req_list =
      env_simulator->InitRequest(1, input_token_num, expected_output_token_num, req, seeds);
  std::shared_ptr<InferRequest> infer_req = infer_req_list[0];

  // Allocate main KV cache blocks (enough for input + output)
  int total_token_num = input_token_num + expected_output_token_num;
  int total_block_num = (total_token_num + block_token_num - 1) / block_token_num;
  std::vector<int> main_blocks;
  main_block_allocator_group->GetDeviceBlockAllocator(0)->AllocateBlocks(total_block_num, main_blocks);
  infer_req->kv_cache_blocks.insert(infer_req->kv_cache_blocks.end(), main_blocks.begin(), main_blocks.end());

  // Allocate SWA blocks: only need to cover sliding_window_size tokens
  int swa_block_num = sliding_window_size / block_token_num;
  std::vector<int> swa_blocks;
  swa_block_allocator_group->GetDeviceBlockAllocator(0)->AllocateBlocks(swa_block_num, swa_blocks);
  infer_req->swa_kv_cache_blocks.insert(infer_req->swa_kv_cache_blocks.end(), swa_blocks.begin(), swa_blocks.end());
  infer_req->swa_kv_cache_block_idx_offsets.resize(swa_blocks.size());
  std::iota(infer_req->swa_kv_cache_block_idx_offsets.begin(), infer_req->swa_kv_cache_block_idx_offsets.end(),
            static_cast<size_t>(0));

  // Run simulation steps
  infer_req->ResetPrefillingTokens();

  int max_output_step = expected_output_token_num + 10;
  for (int step = 0; step < max_output_step; step++) {
    if (infer_req->finished) break;

    infer_req->SetPlanningTask();
    infer_req->LaunchPlanningTask();

    std::vector<std::shared_ptr<InferRequest>> running_reqs = {infer_req};
    env_simulator->RunAStep(running_reqs);

    infer_req->UpdateAfterInflightTaskFinished();
    infer_req->ResetInflightTask();
    if (infer_req->IsEosGenerated()) {
      infer_req->finished = true;
    }
  }

  // Verify: the output token sequence length should match expected
  EXPECT_EQ(infer_req->output_tokens.size(), static_cast<size_t>(input_token_num + expected_output_token_num));

  // Verify SWA block contents: after simulation, the SWA blocks should contain
  // the tokens from the last sliding_window_size positions of the sequence.
  // The SWA blocks cover [swa_kv_cache_block_idx_offsets[0] * block_token_num, ...).
  // Since swa_kv_cache_block_idx_offsets={0,1} and swa_block_num=2, they cover tokens [0, 12).
  // After the full run, the last written tokens in SWA blocks correspond to
  // the last sliding_window_size tokens of the forwarding sequence.
  // We verify by collecting SWA cache content for the initial window [0, sliding_window_size).
  std::vector<int> swa_contents;
  int swa_left = 0;
  int swa_right = sliding_window_size;
  swa_block_allocator_group->CollectSwaKvCacheContent(infer_req, swa_left, swa_right, swa_contents);
  EXPECT_EQ(static_cast<int>(swa_contents.size()), sliding_window_size);
  // Each SWA content entry should match the corresponding forwarding token
  for (int i = 0; i < sliding_window_size; i++) {
    EXPECT_EQ(swa_contents[i], infer_req->output_tokens[swa_left + i])
        << "SWA content mismatch at token index " << (swa_left + i);
  }
}

TEST_F(BatchSchedulerEnvironmentSimulatorTest, FixPrefixCacheScheduleTest) {
  int prefix_block_num = 3;
  int block_token_num = 6;
  int device_num = 4;
  size_t splitfuse_token_num = 0;
  FixPrefixTestCase test_case(prefix_block_num, block_token_num, device_num, splitfuse_token_num);
  std::shared_ptr<BatchSchedulerInterface> batch_scheduler = std::make_shared<FixPrefixBatchScheduler>(
      test_case.GetEnvSimulator(), prefix_block_num * block_token_num, device_num);
  std::shared_ptr<ScheduleProcessorInterface> schedule_processor = std::make_shared<ScheduleProcessor>(false, 1);
  schedule_processor->Initialize(batch_scheduler);
  test_case.SetScheduleProcessor(schedule_processor);
  test_case.RunTestNoCachePressure();
}
