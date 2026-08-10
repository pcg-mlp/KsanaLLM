/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "batch_scheduler/strategy/continuous_batching.h"

#include <limits>
#include <memory>
#include <random>
#include <unordered_map>
#include <vector>

#include "batch_scheduler/event_driven_scheduler/continuous_batching_backend.h"
#include "batch_scheduler/event_driven_scheduler/event_driven_state.h"
#include "batch_scheduler/event_driven_scheduler/schedule_context.h"
#include "batch_scheduler/event_driven_scheduler/workload_balance/basic_workload_balancer.h"
#include "cache_manager/block_allocator/block_allocator_manager.h"
#include "cache_manager/cache_manager_interface.h"
#include "cache_manager/prefix_cache_manager.h"
#include "profiler/timer.h"
#include "runtime/infer_stage.h"
#include "transfer/transfer_engine.h"
#include "utils/memory_allocator.h"
#include "utils/ret_code.h"
#include "utils/singleton.h"
#include "utils/tokenizer.h"

#include "test.h"

using namespace ksana_llm;

// Minimal cache manager: UpdateRequestTokens / DestroyFinishedRequest succeed so
// UpdateRunningRequests can reach the finish-metrics path.
class FinishPathCacheManager : public CacheManagerInterface {
 public:
  void InitializeCachedBlocks() override {}
  std::shared_ptr<BlockAllocatorGroupInterface> GetBlockAllocatorGroup() const override { return nullptr; }
  size_t GetUsableBlockNumber() override { return 0; }
  size_t GetHostFreeBlockNumber() override { return 0; }
  size_t GetUsedBlockNumber() override { return 0; }
  size_t GetRequestStepBlockNumber(int64_t, size_t, size_t) override { return 0; }
  size_t GetRequestStepBlockNumberForOneNextToken(int64_t) override { return 0; }
  size_t GetRequestUsableBlockNumber(int64_t) override { return 0; }
  Status GetRequestPrefixBlockNumber(int64_t, const std::vector<int>&, size_t, size_t&, size_t&, size_t&) override {
    return Status();
  }
  Status GetRequestPrefixBlockNumberBeforeRemoteCacheLoading(int64_t, const std::vector<int>&, size_t, size_t&, size_t&,
                                                             size_t&, size_t&, size_t&, size_t&) override {
    return Status();
  }
  Status AllocateBlocksForRemoteCache(int64_t, std::vector<int>&, size_t, size_t, std::vector<int>&,
                                      std::vector<int>&) override {
    return Status();
  }
  Status ConfirmRemoteCacheLoadingResult(int64_t, const std::vector<int>&, size_t,
                                         const std::vector<size_t>&) override {
    return Status();
  }
  Status AllocateRequestBlocks(int64_t, size_t, std::vector<int>&, std::vector<int>&, std::vector<size_t>&,
                               int&) override {
    return Status();
  }
  void UpdateFlexibleCache(int64_t, const std::vector<int>&, int, std::vector<FlexibleCachedCopyTask>&,
                           size_t&) override {}
  Status UpdateRequestTokens(int64_t, const std::vector<int>&, size_t, std::vector<int>&, bool&, std::vector<int>&,
                             std::vector<size_t>&) override {
    return Status();
  }
  void DestroyFinishedRequest(int64_t req_id) override { destroyed.push_back(req_id); }
  bool IsPrefixCachingEnabled() override { return false; }

  std::vector<int64_t> destroyed;
};

class ContinuousBatchingStrategyTest : public ContinuousBatchingStrategy {
 public:
  using ContinuousBatchingStrategy::AddTransferMeta;
  using ContinuousBatchingStrategy::ContinuousBatchingStrategy;
  using ContinuousBatchingStrategy::ProcessDecodeTransferQueue;
  using ContinuousBatchingStrategy::ProcessPrefillTransferQueue;
  using ContinuousBatchingStrategy::ProcessWaitingQueue;

  void SetSplitFuseTokenNum(const size_t val) { batch_scheduler_config_.split_fuse_token_num = val; }

  // Set connector role and inject a fresh TransferEngine instance (no longer a singleton).
  void SetConnectorRole(GroupRole role) {
    connector_config_.group_role = role;
    transfer_engine_ = std::make_shared<TransferEngine>();
    transfer_engine_->SetGroupRole(role);
  }

  std::shared_ptr<TransferEngine> GetTransferEngineForTest() { return transfer_engine_; }
};

class ContinuousBatchingTest : public testing::Test {
 protected:
  ContinuousBatchingTest() {
    schedule_output_pool_ = std::make_shared<ScheduleOutputPool>();

    constexpr int kTpNum = 2;
    constexpr int kAttnDpNum = 2;
    constexpr int kMultiBatchNum = 1;
    BatchSchedulerConfig batch_scheduler_config;
    batch_scheduler_config.split_fuse_token_num = 256;
    const auto* test_info = ::testing::UnitTest::GetInstance()->current_test_info();
    const std::string test_name = test_info->name();
    if (test_name.find("ProcessDecodeTransferQueue") != std::string::npos ||
        test_name.find("ProcessMTPDecodeTransferQueueTest") != std::string::npos) {
      auto env = Singleton<Environment>::GetInstance();
      ConnectorConfig connector_config;
      connector_config.group_role = GroupRole::DECODE;
      connector_config.router_addr = "127.0.0.1:13579";
      env->SetConnectorConfigs(connector_config);
    }
    RuntimeConfig runtime_config;
    if (test_name.find("ProcessMTPDecodeTransferQueueTest") != std::string::npos) {
      runtime_config.mtp_step_num = 1;
    }

    continuous_batching_strategy_ =
        std::make_shared<ContinuousBatchingStrategyTest>(batch_scheduler_config, runtime_config);
    size_t multi_batch_id = 0;
    continuous_batching_strategy_->SetBatchState(
        std::make_shared<BatchState>(multi_batch_id, batch_scheduler_config, schedule_output_pool_));

    BlockAllocatorGroupConfig group_1_config;
    group_1_config.devices = {0, 1};
    group_1_config.device_block_num = 100;
    group_1_config.host_block_num = 100;
    group_1_config.block_size = 16 * 1024 * 1024;

    BlockAllocatorManagerConfig block_allocator_manager_config;
    block_allocator_manager_config[1] = group_1_config;

    std::shared_ptr<Context> context = std::make_shared<Context>();
    (void)kAttnDpNum;
    (void)kMultiBatchNum;
    std::shared_ptr<MemoryAllocator> memory_allocator_ = std::make_shared<MemoryAllocator>();
    BlockAllocatorManager block_allocator_manager(/*is_engine=*/true, block_allocator_manager_config, memory_allocator_,
                                                  context);

    std::shared_ptr<BlockAllocatorGroupInterface> block_allocator_group =
        block_allocator_manager.GetBlockAllocatorGroup(1);

    CacheManagerConfig cache_manager_config;
    cache_manager_config.tensor_para_size = kTpNum;
    cache_manager_config.enable_prefix_caching = false;
    auto cache_manager = std::make_shared<PrefixCacheManager>(cache_manager_config, block_allocator_group);
    cache_manager->InitializeCachedBlocks();
    continuous_batching_strategy_->SetCacheManager(cache_manager);

    // 初始化调度同步器与共享计数器，避免 SchedulerTickTok::Barrier 未初始化导致的测试崩溃
    auto scheduler_shared_counter = std::make_shared<SchedulerSharedCounter>(1);
    auto scheduler_ticktok = std::make_shared<SchedulerTickTok>(1);
    continuous_batching_strategy_->SetSharedCounter(scheduler_shared_counter);
    continuous_batching_strategy_->SetSchedulerTickTok(scheduler_ticktok);
    continuous_batching_strategy_->SetDataParaGroupId(0);

    pybind11::initialize_interpreter();
  }

  ~ContinuousBatchingTest() {
    schedule_output_pool_.reset();

    if (auto te = continuous_batching_strategy_ ? continuous_batching_strategy_->GetTransferEngineForTest() : nullptr) {
      te->CleanupTransferMeta(123);
    }
    const auto* test_info = ::testing::UnitTest::GetInstance()->current_test_info();
    const std::string test_name = test_info->name();
    if (test_name.find("ProcessDecodeTransferQueueTest") != std::string::npos) {
      auto env = Singleton<Environment>::GetInstance();
      ConnectorConfig connector_config;
      connector_config.group_role = GroupRole::NONE;
      connector_config.router_addr = "127.0.0.1:13579";
      env->SetConnectorConfigs(connector_config);
    }
    pybind11::finalize_interpreter();
  }

  void SetUp() {}

 protected:
  std::shared_ptr<ScheduleOutputPool> schedule_output_pool_ = nullptr;
  std::shared_ptr<ContinuousBatchingStrategyTest> continuous_batching_strategy_ = nullptr;
};
/*
// test function: ProcessSplitFuseTokenTest, check the split size
TEST_F(ContinuousBatchingTest, ProcessSplitFuseTokenTest) {
  auto ksana_python_input = std::make_shared<KsanaPythonInput>();
  auto req_ctx = std::make_shared<std::unordered_map<std::string, std::string>>();
  auto request = std::make_shared<Request>(ksana_python_input, req_ctx);
  auto req = std::make_shared<InferRequest>(request, 0);
  req->block_token_num = 16;

  bool stop_schedule = false;
  size_t shared_block_num, unique_block_num;    // not use in this test
  size_t shared_token_num = 4;                  // random init
  size_t decode_request_num = 1;                // have decode request
  size_t step_token_without_kv_cache_num = 10;  // random init
  size_t forwarding_len;
  size_t split_fuse_token_num;

  constexpr size_t kMinLen = std::numeric_limits<size_t>::min();
  constexpr size_t kMaxLen = 16384;  // avoid overflow
  constexpr size_t kMultiTestTimes = 30;

  auto RandomNum = [](const size_t min, const size_t max) {
    static std::default_random_engine random_engine;
    return std::uniform_int_distribution<size_t>(min, max)(random_engine);
  };

  // disable, forwarding not change
  split_fuse_token_num = 0;  // disable split fuse
  continuous_batching_strategy_->SetSplitFuseTokenNum(split_fuse_token_num);
  forwarding_len = RandomNum(kMinLen, kMaxLen);  // a random number, any len will not change
  req->forwarding_tokens.resize(forwarding_len);
  stop_schedule = continuous_batching_strategy_->ProcessSplitFuseToken(
      req, shared_block_num, unique_block_num, shared_token_num, step_token_without_kv_cache_num, decode_request_num);
  EXPECT_FALSE(stop_schedule);
  EXPECT_EQ(req->forwarding_tokens.size(), forwarding_len);

  // enable split fuse
  split_fuse_token_num = 256;
  continuous_batching_strategy_->SetSplitFuseTokenNum(split_fuse_token_num);

  // Current step already meets quota
  step_token_without_kv_cache_num = RandomNum(split_fuse_token_num, kMaxLen);  // a number >= split_fuse_token_num
  forwarding_len = RandomNum(kMinLen, kMaxLen);  // a random number, any len will not change
  req->forwarding_tokens.resize(forwarding_len);
  stop_schedule = continuous_batching_strategy_->ProcessSplitFuseToken(
      req, shared_block_num, unique_block_num, shared_token_num, step_token_without_kv_cache_num, decode_request_num);
  EXPECT_FALSE(stop_schedule);
  EXPECT_EQ(req->forwarding_tokens.size(), forwarding_len);

  // no decode request
  decode_request_num = 0;                                         // no decode request
  step_token_without_kv_cache_num = RandomNum(kMinLen, kMaxLen);  // a random number, no impact in this case
  for (size_t i = 0; i < kMultiTestTimes; ++i) {
    forwarding_len = RandomNum(kMinLen,
                               kMaxLen);  // a random number, any len will not change
    req->forwarding_tokens.resize(forwarding_len);
    stop_schedule = continuous_batching_strategy_->ProcessSplitFuseToken(
        req, shared_block_num, unique_block_num, shared_token_num, step_token_without_kv_cache_num, decode_request_num);
    EXPECT_FALSE(stop_schedule);
    EXPECT_EQ(req->forwarding_tokens.size(), forwarding_len);
  }
  decode_request_num = 1;  // exist decode request, and number > 0

  // forwading < split_fuse && have decode
  step_token_without_kv_cache_num = RandomNum(1, split_fuse_token_num);  // a number < split_fuse_token_num
  shared_token_num = 0;                                                  // no kv_cache in forwarding_tokens
  // step_token_without_kv_cache_num + forwarding_len <= split_fuse_token_num, len will not change
  forwarding_len = RandomNum(1, split_fuse_token_num - step_token_without_kv_cache_num);
  req->forwarding_tokens.resize(forwarding_len);
  stop_schedule = continuous_batching_strategy_->ProcessSplitFuseToken(
      req, shared_block_num, unique_block_num, shared_token_num, step_token_without_kv_cache_num, decode_request_num);
  EXPECT_FALSE(stop_schedule);
  EXPECT_EQ(req->forwarding_tokens.size(), forwarding_len);

  // forwading < block > split_fuse_remain && have decode
  // reamin split fuse len less than a block
  step_token_without_kv_cache_num = split_fuse_token_num - req->block_token_num / 2;  // reamin less then a block
  shared_token_num = 0;  // no kv_cache in forwarding_tokens
  forwarding_len = RandomNum(split_fuse_token_num - step_token_without_kv_cache_num + 1,
                             kMaxLen);  // large than reamin
  req->forwarding_tokens.resize(forwarding_len);
  stop_schedule = continuous_batching_strategy_->ProcessSplitFuseToken(
      req, shared_block_num, unique_block_num, shared_token_num, step_token_without_kv_cache_num, decode_request_num);
  EXPECT_TRUE(stop_schedule);
  EXPECT_EQ(req->forwarding_tokens.size(), forwarding_len);

  // split fuse with prefix
  step_token_without_kv_cache_num = RandomNum(
      kMinLen, split_fuse_token_num - req->block_token_num);  // a random number < split_fuse_token_num - block_num
  size_t prefix_num = 64;                                     // input have some prefix cache
  shared_token_num = prefix_num;
  forwarding_len =
      shared_token_num + RandomNum(split_fuse_token_num, kMaxLen);  // a large input, exceed split fuse remain len
  req->forwarding_tokens.resize(forwarding_len);
  stop_schedule = continuous_batching_strategy_->ProcessSplitFuseToken(
      req, shared_block_num, unique_block_num, shared_token_num, step_token_without_kv_cache_num, decode_request_num);
  EXPECT_FALSE(stop_schedule);
  EXPECT_EQ(req->forwarding_tokens.size(), prefix_num + (split_fuse_token_num - step_token_without_kv_cache_num) /
                                                            req->block_token_num * req->block_token_num);

  // split fuse
  decode_request_num = 1;  // have decode request, a number > 0
  for (size_t i = 0; i < kMultiTestTimes; ++i) {
    step_token_without_kv_cache_num =
        RandomNum(kMinLen, split_fuse_token_num - req->block_token_num);  // remain >= 1 block
    shared_token_num = RandomNum(kMinLen, split_fuse_token_num / req->block_token_num) * req->block_token_num;
    forwarding_len = shared_token_num + RandomNum(kMinLen, kMaxLen);  // random len
    req->forwarding_tokens.resize(forwarding_len);

    size_t expect_len;
    const size_t remain = split_fuse_token_num - step_token_without_kv_cache_num;
    if (forwarding_len - shared_token_num <= remain) {
      expect_len = forwarding_len;
    } else {
      expect_len = shared_token_num + (remain / req->block_token_num * req->block_token_num);
    }

    stop_schedule = continuous_batching_strategy_->ProcessSplitFuseToken(
        req, shared_block_num, unique_block_num, shared_token_num, step_token_without_kv_cache_num, decode_request_num);
    EXPECT_FALSE(stop_schedule);
    EXPECT_EQ(req->forwarding_tokens.size(), expect_len);
  }
}
*/
// 测试ProcessDecodeTransferQueue函数 - 使用 PREFILL role 避免 connector 依赖
// 仅测试 IsRecvDone 路径（无 transfer meta 时返回 -1，请求保留在 transfer_queue）
TEST_F(ContinuousBatchingTest, ProcessDecodeTransferQueueTest) {
  // 使用 PREFILL 节点避免 SendConfigToPrefill 触发 connector_ 空指针
  continuous_batching_strategy_->SetConnectorRole(GroupRole::PREFILL);

  // 创建测试请求
  auto ksana_python_input = std::make_shared<KsanaPythonInput>();
  auto req_ctx = std::make_shared<std::unordered_map<std::string, std::string>>();
  auto request = std::make_shared<Request>(ksana_python_input, req_ctx);
  auto req = std::make_shared<InferRequest>(request, 0);
  req->req_id = 123;
  req->cache_manager = continuous_batching_strategy_->GetCacheManager();

  // 设置block_token_num，防止添加元数据时报错
  req->block_token_num = 16;

  // 设置request调度状态
  req->input_tokens = {1, 2, 3, 4, 5};
  req->output_tokens = req->input_tokens;
  req->ResetPrefillingTokens();

  // 将请求添加到传输队列
  continuous_batching_strategy_->batch_state_->transfer_queue.push_back(req);

  // 调用ProcessDecodeTransferQueue函数
  continuous_batching_strategy_->ProcessDecodeTransferQueue();

  // 验证结果
  // 由于TransferEngine::IsRecvDone返回-1（未完成），请求应该仍在传输队列中
  ASSERT_EQ(continuous_batching_strategy_->batch_state_->transfer_queue.size(), 1);

  // 清理
  continuous_batching_strategy_->batch_state_->transfer_queue.clear();
}

// DISABLED: needs real connector_ for SendConfigToPrefill in DECODE mode
TEST_F(ContinuousBatchingTest, DISABLED_ProcessDecodeTransferQueueMaxBatchStopTest) {
  // 设置为DECODE节点
  continuous_batching_strategy_->SetConnectorRole(GroupRole::DECODE);

  // 预先填充 running_reqs 到达到最大 decode batch size
  const size_t max_decode_bs = continuous_batching_strategy_->batch_state_->batch_scheduler_config_.max_batch_size;

  auto ksana_python_input = std::make_shared<KsanaPythonInput>();
  auto req_ctx = std::make_shared<std::unordered_map<std::string, std::string>>();
  auto request = std::make_shared<Request>(ksana_python_input, req_ctx);

  for (size_t i = 0; i < max_decode_bs; ++i) {
    auto dummy_req = std::make_shared<InferRequest>(request, 0);
    continuous_batching_strategy_->batch_state_->decoding_queue.push_back(dummy_req);
  }
  size_t transfer_req_num = 20;
  continuous_batching_strategy_->batch_state_->transfer_queue.clear();
  // 构造若干 transfer_queue 请求，若处理应进入 running_reqs，但由于达到上限，应被阻止
  for (size_t i = 0; i < transfer_req_num; ++i) {
    auto req = std::make_shared<InferRequest>(request, 0);
    req->cache_manager = continuous_batching_strategy_->GetCacheManager();

    // 设置KV缓存块
    req->kv_cache_blocks.resize(3);
    for (size_t j = 0; j < 3; ++j) {
      req->kv_cache_blocks[j] = j;
    }
    req->req_id = 123 + i;
    // 设置block_token_num，防止添加元数据时报错
    req->block_token_num = 16;

    // 设置request调度状态
    req->input_tokens = {1, 2, 3, 4, 5};
    req->output_tokens = req->input_tokens;
    req->ResetPrefillingTokens();

    std::vector<std::shared_ptr<InferRequest>> queue;
    queue.push_back(req);

    // 调用AddTransferMeta函数
    continuous_batching_strategy_->AddTransferMeta(queue);
    continuous_batching_strategy_->batch_state_->transfer_queue.push_back(req);
  }

  // 调用处理函数
  continuous_batching_strategy_->ProcessDecodeTransferQueue();

  // 验证：running_reqs 未超过最大 decode batch size，上限触发后不再增加
  ASSERT_EQ(continuous_batching_strategy_->batch_state_->decoding_queue.size(), max_decode_bs);
  ASSERT_EQ(continuous_batching_strategy_->batch_state_->transfer_queue.size(), transfer_req_num);

  // 清理传输元数据
  for (size_t i = 0; i < transfer_req_num; ++i) {
    continuous_batching_strategy_->GetTransferEngineForTest()->CleanupTransferMeta(123 + i);
  }
}

// DISABLED: needs real connector_ for SendConfigToPrefill in DECODE mode
TEST_F(ContinuousBatchingTest, DISABLED_ProcessMTPDecodeTransferQueueTest) {
  // 设置为DECODE节点
  continuous_batching_strategy_->SetConnectorRole(GroupRole::DECODE);

  // 创建测试请求
  auto ksana_python_input = std::make_shared<KsanaPythonInput>();
  auto req_ctx = std::make_shared<std::unordered_map<std::string, std::string>>();
  auto request = std::make_shared<Request>(ksana_python_input, req_ctx);

  for (int i = 0; i < 10; ++i) {
    auto req = std::make_shared<InferRequest>(request, 0);
    req->cache_manager = continuous_batching_strategy_->GetCacheManager();

    // 设置KV缓存块
    req->kv_cache_blocks.resize(3);
    for (size_t j = 0; j < 3; ++j) {
      req->kv_cache_blocks[j] = j;
    }
    req->req_id = 123 + i;
    // 设置block_token_num，防止添加元数据时报错
    req->block_token_num = 16;

    // 设置request调度状态
    req->input_tokens = {1, 2, 3};
    req->output_tokens = req->input_tokens;
    req->ResetPrefillingTokens();

    std::vector<std::shared_ptr<InferRequest>> queue;
    queue.push_back(req);
    // 调用AddTransferMeta函数
    continuous_batching_strategy_->AddTransferMeta(queue);
    continuous_batching_strategy_->batch_state_->transfer_queue.push_back(req);
  }

  // 调用ProcessDecodeTransferQueue函数
  continuous_batching_strategy_->ProcessDecodeTransferQueue();
  for (auto running_req : continuous_batching_strategy_->batch_state_->decoding_queue) {
    ASSERT_EQ(running_req->forwarding_tokens.size(), 3);
    ASSERT_EQ(running_req->draft_tokens.GetDraftTokens()[0], 1);
    ASSERT_EQ(running_req->generated_tokens[0], 1);
    ScheduleTaskWorkload planning_workload = running_req->GetPlanningWorkload();
    EXPECT_EQ(planning_workload.prefill_token_num, 0);
    EXPECT_EQ(planning_workload.generated_token_num, 1);
    EXPECT_EQ(planning_workload.draft_token_num, 1);
  }

  for (int i = 0; i < 10; ++i) {
    continuous_batching_strategy_->GetTransferEngineForTest()->CleanupTransferMeta(123 + i);
  }
}

// 测试AddTransferMeta函数
TEST_F(ContinuousBatchingTest, AddTransferMetaTest) {
  // 设置为PREFILL节点
  continuous_batching_strategy_->SetConnectorRole(GroupRole::PREFILL);

  // 创建测试请求
  auto ksana_python_input = std::make_shared<KsanaPythonInput>();
  auto req_ctx = std::make_shared<std::unordered_map<std::string, std::string>>();
  auto request = std::make_shared<Request>(ksana_python_input, req_ctx);
  auto req = std::make_shared<InferRequest>(request, 0);
  req->req_id = 123;  // 设置请求ID
  req->cache_manager = continuous_batching_strategy_->GetCacheManager();

  // 设置KV缓存块
  req->kv_cache_blocks.resize(3);
  for (size_t j = 0; j < 3; ++j) {
    req->kv_cache_blocks[j] = j;
  }

  // 设置kv_cached_token_num和block_token_num，防止添加元数据时报错
  req->kv_cached_token_num = 16;
  req->block_token_num = 16;

  // 创建请求队列
  std::vector<std::shared_ptr<InferRequest>> queue;
  queue.push_back(req);

  // 调用AddTransferMeta函数
  continuous_batching_strategy_->AddTransferMeta(queue);

  // 验证传输元数据是否已添加
  auto transfer_engine = continuous_batching_strategy_->GetTransferEngineForTest();
  auto meta = transfer_engine->GetTransferMeta(req->req_id);
  ASSERT_NE(meta, nullptr);
}

// 测试ProcessPrefillTransferQueue函数
TEST_F(ContinuousBatchingTest, ProcessPrefillTransferQueueTest) {
  // 设置为PREFILL节点
  continuous_batching_strategy_->SetConnectorRole(GroupRole::PREFILL);

  // 创建测试请求
  auto ksana_python_input = std::make_shared<KsanaPythonInput>();
  auto req_ctx = std::make_shared<std::unordered_map<std::string, std::string>>();
  auto request = std::make_shared<Request>(ksana_python_input, req_ctx);
  auto req = std::make_shared<InferRequest>(request, 0);
  req->req_id = 123;
  req->cache_manager = continuous_batching_strategy_->GetCacheManager();

  // 设置kv_cached_token_num和block_token_num，防止添加元数据时报错
  req->kv_cached_token_num = 0;
  req->block_token_num = 16;

  // 将请求添加到传输队列
  continuous_batching_strategy_->batch_state_->transfer_queue.push_back(req);

  // 调用ProcessPrefillTransferQueue函数
  continuous_batching_strategy_->ProcessPrefillTransferQueue();

  // 验证结果
  // 由于TransferEngine::IsSendDone返回false（未完成），请求应该仍在传输队列中
  ASSERT_EQ(continuous_batching_strategy_->batch_state_->transfer_queue.size(), 1);

  // 清理
  continuous_batching_strategy_->batch_state_->transfer_queue.clear();
}

// 测试ProcessTransferQueue函数（DECODE节点）
TEST_F(ContinuousBatchingTest, ProcessTransferQueueDecodeTest) {
  // 设置为DECODE节点
  continuous_batching_strategy_->SetConnectorRole(GroupRole::DECODE);

  // 创建测试请求
  auto ksana_python_input = std::make_shared<KsanaPythonInput>();
  auto req_ctx = std::make_shared<std::unordered_map<std::string, std::string>>();
  auto request = std::make_shared<Request>(ksana_python_input, req_ctx);
  auto req = std::make_shared<InferRequest>(request, 0);
  req->req_id = 123;
  req->cache_manager = continuous_batching_strategy_->GetCacheManager();

  // 设置block_token_num，防止添加元数据时报错
  req->block_token_num = 16;

  // 设置request调度状态
  req->input_tokens = {1, 2, 3, 4, 5};
  req->output_tokens = req->input_tokens;
  req->ResetPrefillingTokens();

  // 将请求添加到传输队列
  continuous_batching_strategy_->batch_state_->transfer_queue.push_back(req);

  // 调用ProcessTransferQueue函数
  continuous_batching_strategy_->ProcessTransferQueue();

  // 验证结果
  // 由于TransferEngine::IsRecvDone返回-1（未完成），请求应该仍在传输队列中
  ASSERT_EQ(continuous_batching_strategy_->batch_state_->transfer_queue.size(), 1);
  ASSERT_EQ(continuous_batching_strategy_->batch_state_->schedule_output->running_reqs.size(), 0);

  // 清理
  continuous_batching_strategy_->batch_state_->transfer_queue.clear();
}

// 测试ProcessTransferQueue函数（PREFILL节点）
TEST_F(ContinuousBatchingTest, ProcessTransferQueuePrefillTest) {
  // 设置为PREFILL节点
  continuous_batching_strategy_->SetConnectorRole(GroupRole::PREFILL);

  // 创建测试请求
  auto ksana_python_input = std::make_shared<KsanaPythonInput>();
  auto req_ctx = std::make_shared<std::unordered_map<std::string, std::string>>();
  auto request = std::make_shared<Request>(ksana_python_input, req_ctx);
  auto req = std::make_shared<InferRequest>(request, 0);
  req->req_id = 123;
  req->cache_manager = continuous_batching_strategy_->GetCacheManager();

  // 设置kv_cached_token_num和block_token_num，防止添加元数据时报错
  req->kv_cached_token_num = 0;
  req->block_token_num = 16;

  // 将请求添加到传输队列
  continuous_batching_strategy_->batch_state_->transfer_queue.push_back(req);

  // 调用ProcessTransferQueue函数
  continuous_batching_strategy_->ProcessTransferQueue();

  // 验证结果
  // 由于TransferEngine::IsSendDone返回false（未完成），请求应该仍在传输队列中
  ASSERT_EQ(continuous_batching_strategy_->batch_state_->transfer_queue.size(), 1);

  // 清理
  continuous_batching_strategy_->batch_state_->transfer_queue.clear();
}

// Draft-token validation in ProcessWaitingQueue: speculative decoding disabled,
// user-supplied draft_tokens.spec must be cleared entirely.
TEST_F(ContinuousBatchingTest, ProcessWaitingQueueDraftClearedWhenSpecDisabled) {
  continuous_batching_strategy_->batch_scheduler_config_.enable_speculative_decoding = false;

  auto ksana_python_input = std::make_shared<KsanaPythonInput>();
  auto req_ctx = std::make_shared<std::unordered_map<std::string, std::string>>();
  auto request = std::make_shared<Request>(ksana_python_input, req_ctx);
  auto req = std::make_shared<InferRequest>(request, 0);
  req->cache_manager = continuous_batching_strategy_->GetCacheManager();
  req->block_token_num = 16;
  req->draft_tokens.spec = {10, 20, 30};

  // Empty input_tokens: request short-circuits right after the draft filter,
  // isolating the tested logic from block allocation.
  req->input_tokens = {};
  req->output_tokens = {};
  req->ResetPrefillingTokens();

  continuous_batching_strategy_->batch_state_->ResetInfoBeforeSchedule();
  continuous_batching_strategy_->batch_state_->waiting_queue.push_back(req);
  continuous_batching_strategy_->ProcessWaitingQueue();

  ASSERT_TRUE(req->draft_tokens.spec.empty());
}

// Draft-token validation: invalid ids (negative or >= max_vocab_size) are dropped,
// valid ones preserved in order.
TEST_F(ContinuousBatchingTest, ProcessWaitingQueueDraftDropsInvalidIds) {
  continuous_batching_strategy_->batch_scheduler_config_.enable_speculative_decoding = true;
  continuous_batching_strategy_->batch_scheduler_config_.max_vocab_size = 1000;
  continuous_batching_strategy_->dp_max_logits_num_ = 64;

  auto ksana_python_input = std::make_shared<KsanaPythonInput>();
  auto req_ctx = std::make_shared<std::unordered_map<std::string, std::string>>();
  auto request = std::make_shared<Request>(ksana_python_input, req_ctx);
  auto req = std::make_shared<InferRequest>(request, 0);
  req->cache_manager = continuous_batching_strategy_->GetCacheManager();
  req->block_token_num = 16;
  req->draft_tokens.spec = {5, -1, 42, 1000, 999, 2000};

  req->input_tokens = {};
  req->output_tokens = {};
  req->ResetPrefillingTokens();

  continuous_batching_strategy_->batch_state_->ResetInfoBeforeSchedule();
  continuous_batching_strategy_->batch_state_->waiting_queue.push_back(req);
  continuous_batching_strategy_->ProcessWaitingQueue();

  const std::vector<int> expected = {5, 42, 999};
  ASSERT_EQ(req->draft_tokens.spec, expected);
}

// Draft-token validation: length capped so filtered.size()+kStepGenerateTokenNum < dp_max_logits_num_.
TEST_F(ContinuousBatchingTest, ProcessWaitingQueueDraftCapsLength) {
  continuous_batching_strategy_->batch_scheduler_config_.enable_speculative_decoding = true;
  continuous_batching_strategy_->batch_scheduler_config_.max_vocab_size = 32000;
  // dp_max_logits_num_=5, kStepGenerateTokenNum=1 → cap at 4 tokens.
  continuous_batching_strategy_->dp_max_logits_num_ = 5;

  auto ksana_python_input = std::make_shared<KsanaPythonInput>();
  auto req_ctx = std::make_shared<std::unordered_map<std::string, std::string>>();
  auto request = std::make_shared<Request>(ksana_python_input, req_ctx);
  auto req = std::make_shared<InferRequest>(request, 0);
  req->cache_manager = continuous_batching_strategy_->GetCacheManager();
  req->block_token_num = 16;
  req->draft_tokens.spec = {1, 2, 3, 4, 5, 6, 7, 8};

  req->input_tokens = {};
  req->output_tokens = {};
  req->ResetPrefillingTokens();

  continuous_batching_strategy_->batch_state_->ResetInfoBeforeSchedule();
  continuous_batching_strategy_->batch_state_->waiting_queue.push_back(req);
  continuous_batching_strategy_->ProcessWaitingQueue();

  const std::vector<int> expected = {1, 2, 3, 4};
  ASSERT_EQ(req->draft_tokens.spec, expected);
}

// Test that empty request (input_tokens=[]) is rejected instead of crashing.
TEST_F(ContinuousBatchingTest, EmptyRequestAbortTest) {
  // Create an empty request with no input tokens.
  auto ksana_python_input = std::make_shared<KsanaPythonInput>();
  auto req_ctx = std::make_shared<std::unordered_map<std::string, std::string>>();
  auto request = std::make_shared<Request>(ksana_python_input, req_ctx);
  auto req = std::make_shared<InferRequest>(request, 0);
  req->cache_manager = continuous_batching_strategy_->GetCacheManager();
  req->block_token_num = 16;

  // Set empty input: input_tokens is empty, output_tokens is empty.
  req->input_tokens = {};
  req->output_tokens = {};
  req->ResetPrefillingTokens();

  // Verify the request has 0 tokens.
  ASSERT_EQ(req->GetPlanningSequenceLen(), 0);

  // Add the empty request to the waiting queue.
  continuous_batching_strategy_->batch_state_->waiting_queue.push_back(req);
  ASSERT_EQ(continuous_batching_strategy_->batch_state_->waiting_queue.size(), 1);

  // Process the waiting queue — should reject the empty request without crashing.
  continuous_batching_strategy_->ProcessWaitingQueue();

  // The empty request should be removed from waiting queue (aborted).
  ASSERT_EQ(continuous_batching_strategy_->batch_state_->waiting_queue.size(), 0);

  // The request should be marked as finished with error.
  ASSERT_TRUE(req->finished);
}

// Cover waiting_latency_us / ttft_us / tpot_us reporting in UpdateRunningRequests.
TEST_F(ContinuousBatchingTest, UpdateRunningRequestsReportsAccessLogLatencyMetrics) {
  continuous_batching_strategy_->SetConnectorRole(GroupRole::NONE);
  auto cache = std::make_shared<FinishPathCacheManager>();
  continuous_batching_strategy_->SetCacheManager(cache);

  auto ksana_python_input = std::make_shared<KsanaPythonInput>();
  auto req_ctx = std::make_shared<std::unordered_map<std::string, std::string>>();
  auto request = std::make_shared<Request>(ksana_python_input, req_ctx);
  auto req = std::make_shared<InferRequest>(request, 0);
  req->req_id = 4242;
  req->block_token_num = 16;
  req->attn_dp_group_id = 0;
  req->input_tokens = {1, 2, 3};
  req->output_tokens = {1, 2, 3, 10, 11};  // 2 generated tokens
  req->sampling_config.max_new_tokens = 2;
  req->finish_status = Status(RET_SUCCESS);
  req->ResetPrefillingTokens();

  // Prefill already done: remaining prefill=0 so IsRequestFinished can return true.
  ScheduleTaskWorkload remain;
  remain.prefill_token_num = 0;
  remain.prefill_start_offset = req->output_tokens.size();
  req->SetRemainingWorkload(remain);

  // One decode inflight so UpdateRunningRequests can retire it.
  req->forwarding_tokens = {1, 2, 3, 10};
  req->SetKvCachedTokenNum(3);
  req->generated_tokens = {11};
  ScheduleTaskWorkload plan;
  plan.generated_token_num = 1;
  plan.sampling_token_num = 1;
  req->SetPlanningWorkload(plan);
  req->SetPlanningTask();
  req->LaunchPlanningTask();

  const uint64_t arrive = req->timestamp_in_us;
  ASSERT_GT(arrive, 0u);
  req->scheduled_time_us.store(arrive + 100);
  req->first_token_time_us.store(arrive + 200);
  // finish_time_us is read before StopRequest stamps it; seed it for tpot_us.
  req->finish_time_us.store(arrive + 500);

  continuous_batching_strategy_->batch_state_->decoding_queue.push_back(req);

  continuous_batching_strategy_->UpdateRunningRequests({req});

  EXPECT_TRUE(req->finished);
  EXPECT_TRUE(continuous_batching_strategy_->batch_state_->decoding_queue.empty());
  ASSERT_EQ(cache->destroyed.size(), 1u);
  EXPECT_EQ(cache->destroyed[0], 4242);
}

// Controllable cache manager for ContinuousBatchingBackend KV-pressure paths.
class ControllableCacheManager : public CacheManagerInterface {
 public:
  void InitializeCachedBlocks() override {}
  std::shared_ptr<BlockAllocatorGroupInterface> GetBlockAllocatorGroup() const override { return nullptr; }
  size_t GetUsableBlockNumber() override { return usable_blocks; }
  size_t GetHostFreeBlockNumber() override { return 0; }
  size_t GetUsedBlockNumber() override { return 0; }
  size_t GetRequestStepBlockNumber(int64_t, size_t, size_t) override { return step_blocks; }
  size_t GetRequestStepBlockNumberForOneNextToken(int64_t) override { return 1; }
  size_t GetRequestUsableBlockNumber(int64_t) override { return usable_blocks; }
  Status GetRequestPrefixBlockNumber(int64_t, const std::vector<int>&, size_t, size_t&, size_t&, size_t&) override {
    return Status();
  }
  Status GetRequestPrefixBlockNumberBeforeRemoteCacheLoading(int64_t, const std::vector<int>&, size_t, size_t&, size_t&,
                                                             size_t&, size_t&, size_t&, size_t&) override {
    return Status();
  }
  Status AllocateBlocksForRemoteCache(int64_t, std::vector<int>&, size_t, size_t, std::vector<int>&,
                                      std::vector<int>&) override {
    return Status();
  }
  Status ConfirmRemoteCacheLoadingResult(int64_t, const std::vector<int>&, size_t,
                                         const std::vector<size_t>&) override {
    return Status();
  }
  Status AllocateRequestBlocks(int64_t, size_t, std::vector<int>&, std::vector<int>&, std::vector<size_t>&,
                               int&) override {
    ++allocate_calls;
    if (!allocate_ok) {
      return Status(RET_OUT_OF_DEVICE_MEMORY, "oom");
    }
    return Status();
  }
  void UpdateFlexibleCache(int64_t, const std::vector<int>&, int, std::vector<FlexibleCachedCopyTask>&,
                           size_t&) override {}
  Status UpdateRequestTokens(int64_t, const std::vector<int>&, size_t, std::vector<int>&, bool&, std::vector<int>&,
                             std::vector<size_t>&) override {
    return Status();
  }
  void DestroyFinishedRequest(int64_t) override {}
  bool IsPrefixCachingEnabled() override { return false; }

  size_t usable_blocks = 0;
  size_t step_blocks = 1;
  bool allocate_ok = false;
  size_t allocate_calls = 0;
};

class ContinuousBatchingBackendKvPressureTest : public testing::Test {
 protected:
  void SetUp() override {
    InitLoguru();
    cfg_.max_pp_batch_num = 1;
    cfg_.max_batch_size = 8;
    cfg_.max_step_token_num = 4096;
    cfg_.max_decode_tokens_per_req = 4;
    cfg_.split_fuse_token_num = 256;
    cfg_.launch_block_threshold = 0;

    cache_manager_ = std::make_shared<ControllableCacheManager>();
    dp_cache_managers_ = {cache_manager_};

    schedule_state_.schedule_states.assign(1, std::vector<DpScheduleState>(1));
    schedule_context_.SetScheduleState(&schedule_state_);
    schedule_context_.SetDpCacheManagers(&dp_cache_managers_);
    schedule_context_.SetCurrentMultiBatchId(0);

    backend_ = std::make_shared<ContinuousBatchingBackend>(cfg_);
    backend_->SetWorkloadBalancer(std::make_shared<BasicWorkloadBalancer>(
        /*dp_num=*/1, static_cast<size_t>(cfg_.max_batch_size), static_cast<size_t>(cfg_.max_decode_tokens_per_req),
        static_cast<size_t>(cfg_.max_step_token_num)));

    planning_batch_ = std::make_shared<PlanningBatch>();
    planning_batch_->dp_batches.resize(1);
    planning_batches_ = {planning_batch_};
  }

  std::shared_ptr<InferRequest> MakeRequest(int64_t req_id, size_t token_len) {
    auto input = std::make_shared<KsanaPythonInput>();
    input->input_tokens.assign(token_len, 1);
    auto req_ctx = std::make_shared<std::unordered_map<std::string, std::string>>();
    auto request = std::make_shared<Request>(input, req_ctx);
    input_holders_.push_back(input);
    request_holders_.push_back(request);

    auto infer_req = std::make_shared<InferRequest>(request, 0);
    infer_req->req_id = req_id;
    infer_req->block_token_num = 16;
    infer_req->attn_dp_group_id = 0;
    infer_req->sampling_token_num = 1;
    infer_req->output_tokens.assign(token_len, 1);
    infer_req->ResetPrefillingTokens();
    return infer_req;
  }

  void ArmPartialPrefill(const std::shared_ptr<InferRequest>& req, size_t kv_blocks, size_t prefill_start_offset,
                         size_t remaining_prefill_tokens) {
    req->infer_stage = InferStage::kContext;
    req->kv_cache_blocks.assign(kv_blocks, 1);
    ScheduleTaskWorkload wl;
    wl.prefill_start_offset = prefill_start_offset;
    wl.prefill_token_num = remaining_prefill_tokens;
    wl.sampling_token_num = 1;
    req->SetRemainingWorkload(wl);
    req->SetPlanningWorkload(wl);
  }

  void ArmDecode(const std::shared_ptr<InferRequest>& req) {
    req->infer_stage = InferStage::kDecode;
    ScheduleTaskWorkload wl;
    wl.prefill_token_num = 0;
    wl.prefill_start_offset = 0;
    wl.sampling_token_num = 1;
    req->SetRemainingWorkload(wl);
    req->SetPlanningWorkload(wl);
  }

  static std::vector<ScheduleBackendEvent> FilterRecompute(const std::vector<ScheduleBackendEvent>& events) {
    std::vector<ScheduleBackendEvent> out;
    for (const auto& ev : events) {
      if (ev.type == ScheduleBackendEventType::kRecomputeRequest) {
        out.push_back(ev);
      }
    }
    return out;
  }

  BatchSchedulerConfig cfg_;
  std::shared_ptr<ControllableCacheManager> cache_manager_;
  std::vector<std::shared_ptr<CacheManagerInterface>> dp_cache_managers_;
  ScheduleState schedule_state_;
  ScheduleContext schedule_context_;
  std::shared_ptr<ContinuousBatchingBackend> backend_;
  std::shared_ptr<PlanningBatch> planning_batch_;
  std::vector<std::shared_ptr<PlanningBatch>> planning_batches_;
  std::vector<std::shared_ptr<KsanaPythonInput>> input_holders_;
  std::vector<std::shared_ptr<Request>> request_holders_;
};

// 1) Decode cannot allocate → recompute Prefill victim, skip Prefill, keep split-fuse.
TEST_F(ContinuousBatchingBackendKvPressureTest, DecodeBlockedSkipsPrefillKeepsSplitFuse) {
  cache_manager_->allocate_ok = false;
  cache_manager_->usable_blocks = 0;
  cache_manager_->step_blocks = 1;

  auto decode_req = MakeRequest(/*req_id=*/1001, /*token_len=*/32);
  ArmDecode(decode_req);
  auto prefill_req = MakeRequest(/*req_id=*/2001, /*token_len=*/128);
  ArmPartialPrefill(prefill_req, /*kv_blocks=*/8, /*prefill_start_offset=*/64, /*remaining_prefill_tokens=*/64);

  auto& dp_state = schedule_state_.schedule_states[0][0];
  dp_state.decoding_reqs.push_back(decode_req);
  dp_state.prefilling_reqs.push_back(prefill_req);

  std::vector<ScheduleBackendEvent> events;
  backend_->Schedule(schedule_context_, planning_batches_, events);

  EXPECT_EQ(backend_->GetBatchSchedulerConfig().split_fuse_token_num, 256u);
  EXPECT_TRUE(planning_batch_->dp_batches[0].running_reqs.empty());
  EXPECT_GT(cache_manager_->allocate_calls, 0u);

  const auto recomputes = FilterRecompute(events);
  ASSERT_EQ(recomputes.size(), 1u);
  ASSERT_NE(recomputes[0].req_ptr, nullptr);
  EXPECT_EQ(recomputes[0].req_ptr->req_id, prefill_req->req_id);
}

// Decode blocked with no Prefill/Decode victim: skip Prefill only, no recompute event.
TEST_F(ContinuousBatchingBackendKvPressureTest, DecodeBlockedNoVictimSkipsPrefillWithoutRecompute) {
  cache_manager_->allocate_ok = false;
  cache_manager_->usable_blocks = 0;
  cache_manager_->step_blocks = 1;

  auto decode_req = MakeRequest(/*req_id=*/1101, /*token_len=*/32);
  ArmDecode(decode_req);
  // Prefill present but holds no KV, so it cannot be a recompute victim.
  auto waiting_prefill = MakeRequest(/*req_id=*/2101, /*token_len=*/64);
  waiting_prefill->infer_stage = InferStage::kContext;
  ScheduleTaskWorkload wl;
  wl.prefill_token_num = 64;
  wl.sampling_token_num = 1;
  waiting_prefill->SetRemainingWorkload(wl);
  waiting_prefill->SetPlanningWorkload(wl);

  auto& dp_state = schedule_state_.schedule_states[0][0];
  dp_state.decoding_reqs.push_back(decode_req);
  dp_state.prefilling_reqs.push_back(waiting_prefill);

  std::vector<ScheduleBackendEvent> events;
  backend_->Schedule(schedule_context_, planning_batches_, events);

  EXPECT_EQ(backend_->GetBatchSchedulerConfig().split_fuse_token_num, 256u);
  EXPECT_TRUE(planning_batch_->dp_batches[0].running_reqs.empty());
  EXPECT_TRUE(FilterRecompute(events).empty());
}

// 2) Prefill-only: keep survivor (more KV), recompute the other holder on Allocate OOM.
TEST_F(ContinuousBatchingBackendKvPressureTest, PrefillRecomputeNonSurvivor) {
  // Pass threshold, fail at AllocateRequestBlocks (covers allocation-failure recovery).
  cache_manager_->allocate_ok = false;
  cache_manager_->usable_blocks = 1024;
  cache_manager_->step_blocks = 4;

  auto survivor = MakeRequest(/*req_id=*/3001, /*token_len=*/256);
  ArmPartialPrefill(survivor, /*kv_blocks=*/10, /*prefill_start_offset=*/128, /*remaining_prefill_tokens=*/128);
  auto victim = MakeRequest(/*req_id=*/3002, /*token_len=*/256);
  ArmPartialPrefill(victim, /*kv_blocks=*/3, /*prefill_start_offset=*/48, /*remaining_prefill_tokens=*/128);

  auto& dp_state = schedule_state_.schedule_states[0][0];
  dp_state.prefilling_reqs.push_back(survivor);
  dp_state.prefilling_reqs.push_back(victim);

  std::vector<ScheduleBackendEvent> events;
  backend_->Schedule(schedule_context_, planning_batches_, events);

  EXPECT_EQ(backend_->GetBatchSchedulerConfig().split_fuse_token_num, 256u);
  EXPECT_GE(cache_manager_->allocate_calls, 2u);
  const auto recomputes = FilterRecompute(events);
  ASSERT_EQ(recomputes.size(), 1u);
  ASSERT_NE(recomputes[0].req_ptr, nullptr);
  EXPECT_EQ(recomputes[0].req_ptr->req_id, victim->req_id);
}

// 3) Prefill-only: sole KV holder self-recomputes as last resort on Allocate OOM.
TEST_F(ContinuousBatchingBackendKvPressureTest, PrefillSelfRecomputeSoleSurvivor) {
  cache_manager_->allocate_ok = false;
  cache_manager_->usable_blocks = 1024;
  cache_manager_->step_blocks = 4;

  auto sole = MakeRequest(/*req_id=*/4001, /*token_len=*/256);
  ArmPartialPrefill(sole, /*kv_blocks=*/12, /*prefill_start_offset=*/128, /*remaining_prefill_tokens=*/128);
  // Another Prefill with empty KV should not become victim.
  auto waiting = MakeRequest(/*req_id=*/4002, /*token_len=*/64);
  waiting->infer_stage = InferStage::kContext;
  ScheduleTaskWorkload wl;
  wl.prefill_token_num = 64;
  wl.sampling_token_num = 1;
  waiting->SetRemainingWorkload(wl);
  waiting->SetPlanningWorkload(wl);

  auto& dp_state = schedule_state_.schedule_states[0][0];
  dp_state.prefilling_reqs.push_back(sole);
  dp_state.prefilling_reqs.push_back(waiting);

  std::vector<ScheduleBackendEvent> events;
  backend_->Schedule(schedule_context_, planning_batches_, events);

  EXPECT_EQ(backend_->GetBatchSchedulerConfig().split_fuse_token_num, 256u);
  EXPECT_GE(cache_manager_->allocate_calls, 1u);
  const auto recomputes = FilterRecompute(events);
  ASSERT_EQ(recomputes.size(), 1u);
  ASSERT_NE(recomputes[0].req_ptr, nullptr);
  EXPECT_EQ(recomputes[0].req_ptr->req_id, sole->req_id);

  // Lightweight "Apply" simulation (HandleRecomputeRequest lives in processor): free KV and
  // clear in-plan ops, then confirm the next Schedule can admit Prefill instead of livelocking.
  sole->kv_cache_blocks.clear();
  sole->output_tokens.assign(64, 1);
  sole->ResetPrefillingTokens();  // keeps remaining/planning consistent with prefilling_tokens_
  dp_state.in_plan_schedule_ops = DpScheduleOpsInPlan{};
  planning_batch_->Reset();
  cache_manager_->allocate_ok = true;
  cache_manager_->allocate_calls = 0;

  events.clear();
  backend_->Schedule(schedule_context_, planning_batches_, events);
  EXPECT_TRUE(FilterRecompute(events).empty());
  EXPECT_FALSE(planning_batch_->dp_batches[0].running_reqs.empty());
  EXPECT_EQ(planning_batch_->dp_batches[0].running_reqs.count(sole->req_id), 1u);
}
