/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "batch_scheduler/batch_scheduler.h"

#include "batch_scheduler/batch_scheduler_test_client.h"
#include "batch_scheduler/batch_scheduler_test_helper.h"
#include "cache_manager/cache_manager_factory.h"
#include "helpers/environment_test_helper.h"

#include "test.h"

namespace ksana_llm {

// 定义一个 BaseSchedulerTest 类，继承自 testing::Test
class BaseSchedulerTest : public testing::Test {
 protected:
  static void SetUpTestSuite() { InitLoguru(); }

  void SetUp() override {
    enable_prefix_cache_ = false;
    enable_flexible_cache_ = false;
    split_fuse_token_num_ = 0;
    enable_async_ = false;
    waiting_timeout_in_ms_ = 600000;
    max_pp_batch_num_ = 1;
    sliding_window_size_ = 0;
  }

  void CommonSetUp(int dp_num = 1, int tp_num = 4, int ep_world_size = 1) {
    runtime_config_.parallel_basic_config.tensor_parallel_size = tp_num;
    runtime_config_.parallel_basic_config.attn_data_parallel_size = dp_num;
    runtime_config_.parallel_basic_config.attn_tensor_parallel_size = tp_num / dp_num;
    ep_world_size_ = ep_world_size;

    // Init BatchSchedulerEnvironmentSimulator and BatchScheduler
    InitDefaultConfig();

    block_allocator_group = std::make_shared<FakedBlockAllocatorGroup>(block_manager_config_, tp_num);

    // When SWA is enabled, create a separate SWA block allocator group.
    // SWA blocks do not need host blocks, so host_allocator_config.blocks_num = 0.
    if (sliding_window_size_ > 0) {
      BlockManagerConfig swa_block_manager_config = block_manager_config_;
      swa_block_manager_config.host_allocator_config.blocks_num = 0;
      swa_block_allocator_group = std::make_shared<FakedBlockAllocatorGroup>(swa_block_manager_config, tp_num);
    }

    env_simulator_ = new BatchSchedulerEnvironmentSimulator(block_manager_config_, tp_num, block_allocator_group,
                                                            sliding_window_size_, swa_block_allocator_group);

    // Create Context
    std::string config_file = GetTestConfigFile();
    Singleton<Environment>::GetInstance()->ParseConfig(config_file);

    schedule_output_pool_ = std::make_shared<ScheduleOutputPool>();

    KLLM_LOG_INFO << "enable_prefix_cache=" << cache_manager_config.enable_prefix_caching
                  << ", split_fuse_num=" << batch_scheduler_config_.split_fuse_token_num
                  << ", sliding_window_size=" << sliding_window_size_;

    cache_manager =
        CacheManagerFactory::CreateCacheManager(cache_manager_config, block_allocator_group, swa_block_allocator_group);
    cache_manager->InitializeCachedBlocks();

    InitScheduleProcessor();
    schedule_processor_->Start();
  }

  virtual void InitScheduleProcessor() = 0;

  void FixPrefixCacheBlockLimitTriggeredTest() {
    CommonSetUp();

    int prefix_block_num = 30;
    int block_token_num = 6;
    int device_num = 4;
    FixPrefixTestCase test_case(prefix_block_num, block_token_num, device_num, split_fuse_token_num_, false, false);
    test_case.SetScheduleProcessor(schedule_processor_);
    test_case.SetEnvSimulator(env_simulator_);
    test_case.RunTestRecomputeTriggered();
  }

  void TearDown() override {
    delete env_simulator_;
    schedule_output_pool_ = nullptr;
  }

 protected:
  void InitDefaultConfig() {
    int device_block_num = 100;
    block_manager_config_.host_allocator_config.blocks_num = 0;
    block_manager_config_.device_allocator_config.blocks_num = device_block_num;
    block_manager_config_.device_allocator_config.block_token_num = 6;

    batch_scheduler_config_.schedule_strategy = static_cast<ScheduleStrategy>(0);
    batch_scheduler_config_.waiting_timeout_in_ms = waiting_timeout_in_ms_;
    batch_scheduler_config_.max_waiting_queue_len = 256;
    batch_scheduler_config_.max_step_token_num = 4096;
    batch_scheduler_config_.max_batch_size = 8;
    batch_scheduler_config_.max_token_len = 1024;
    batch_scheduler_config_.launch_block_threshold = 4.0;
    batch_scheduler_config_.split_fuse_token_num = split_fuse_token_num_;
    batch_scheduler_config_.enable_async = enable_async_;
    batch_scheduler_config_.max_pp_batch_num = max_pp_batch_num_;

    cache_manager_config.block_token_num = block_manager_config_.device_allocator_config.block_token_num;
    cache_manager_config.tensor_para_size = runtime_config_.parallel_basic_config.tensor_parallel_size;
    cache_manager_config.enable_prefix_caching = enable_prefix_cache_;
    if (enable_flexible_cache_) {
      cache_manager_config.min_flexible_cache_num = cache_manager_config.block_token_num * 4;
    }
    cache_manager_config.sliding_window_size = sliding_window_size_;
  }

 protected:
  // 定义一个 BlockManager 指针，用于在测试用例中使用
  BatchSchedulerEnvironmentSimulator* env_simulator_ = nullptr;

  std::shared_ptr<ScheduleOutputPool> schedule_output_pool_ = nullptr;
  std::shared_ptr<ScheduleProcessorInterface> schedule_processor_ = nullptr;

  std::shared_ptr<FakedBlockAllocatorGroup> block_allocator_group = nullptr;
  std::shared_ptr<CacheManagerInterface> cache_manager = nullptr;

  BlockManagerConfig block_manager_config_;
  BatchSchedulerConfig batch_scheduler_config_;
  CacheManagerConfig cache_manager_config;
  RuntimeConfig runtime_config_;
  int ep_world_size_;

  bool enable_prefix_cache_;
  bool enable_flexible_cache_;
  size_t split_fuse_token_num_;
  bool enable_async_;
  size_t waiting_timeout_in_ms_;
  size_t max_pp_batch_num_;
  // SWA (Sliding Window Attention) configuration; 0 means SWA disabled.
  size_t sliding_window_size_ = 0;
  std::shared_ptr<FakedBlockAllocatorGroup> swa_block_allocator_group = nullptr;
};

}  // namespace ksana_llm
