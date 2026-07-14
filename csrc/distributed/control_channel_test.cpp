/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <gtest/gtest.h>

#include <arpa/inet.h>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "distributed/control_channel.h"
#include "runtime/sampling_output.h"
#include "runtime/schedule_output.h"

#include "test.h"

#include "configure/context.h"
#include "configure/environment.h"
#include "configure/global_context.h"
#include "distributed/socket_util.h"
#include "utils/memory_utils.h"
#include "utils/singleton.h"
#include "utils/status.h"

#include "helpers/environment_test_helper.h"

using namespace ksana_llm;

class ControlChannelTest : public testing::Test {
 protected:
  void SetUp() override {
    master_env_ = std::make_shared<Environment>();
    worker_env_ = std::make_shared<Environment>();

    // Set model config.
    std::string config_file = GetTestConfigFile();
    Singleton<Environment>::GetInstance()->ParseConfig(config_file);

    master_env_->ParseConfig(config_file);
    worker_env_->ParseConfig(config_file);

    BlockManagerConfig master_block_manager_config;
    master_env_->InitializeBlockManagerConfig();
    master_env_->GetBlockManagerConfig(master_block_manager_config);
    master_block_manager_config.device_allocator_config.blocks_num = 10;
    master_block_manager_config.host_allocator_config.blocks_num = 8;
    master_env_->SetBlockManagerConfig(master_block_manager_config);

    BlockManagerConfig worker_block_manager_config;
    worker_env_->InitializeBlockManagerConfig();
    worker_env_->GetBlockManagerConfig(worker_block_manager_config);
    worker_block_manager_config.device_allocator_config.blocks_num = 6;
    worker_block_manager_config.host_allocator_config.blocks_num = 4;
    worker_env_->SetBlockManagerConfig(worker_block_manager_config);

    master_schedule_output_pool_ = std::make_shared<ScheduleOutputPool>();
    worker_schedule_output_pool_ = std::make_shared<ScheduleOutputPool>();

    master_sampling_output_pool_ = std::make_shared<SamplingOutputPool>();
    worker_sampling_output_pool_ = std::make_shared<SamplingOutputPool>();

    std::string interface;
    GetAvailableInterfaceAndIP(interface, master_host_);
    GetAvailablePort(master_port_);

    // Set pipeline config for master (node_rank = 0)
    PipelineConfig master_pipeline_config;
    master_pipeline_config.master_host = master_host_;
    master_pipeline_config.master_port = master_port_;
    master_pipeline_config.world_size = world_size_;
    master_pipeline_config.node_rank = 0;
    master_env_->SetPipelineConfig(master_pipeline_config);

    // Set pipeline config for worker (node_rank = 1)
    PipelineConfig worker_pipeline_config;
    worker_pipeline_config.master_host = master_host_;
    worker_pipeline_config.master_port = master_port_;
    worker_pipeline_config.world_size = world_size_;
    worker_pipeline_config.node_rank = 1;
    worker_env_->SetPipelineConfig(worker_pipeline_config);

    // Create GlobalContext for master
    master_global_context_ = std::make_shared<GlobalContext>(1, 1, 1, master_env_);
    master_global_context_->InitDistributedConfig();

    // Create GlobalContext for worker
    worker_global_context_ = std::make_shared<GlobalContext>(1, 1, 1, worker_env_);
    worker_global_context_->InitDistributedConfig();

    // Create Context for master
    master_context_ = std::make_shared<Context>();
    master_context_->global = master_global_context_;

    // Create Context for worker
    worker_context_ = std::make_shared<Context>();
    worker_context_->global = worker_global_context_;

    ctrl_channel_master_ = std::make_shared<ControlChannel>(GetPacketObject, master_schedule_output_pool_,
                                                            master_sampling_output_pool_, master_context_);
    ctrl_channel_worker_ = std::make_shared<ControlChannel>(GetPacketObject, worker_schedule_output_pool_,
                                                            worker_sampling_output_pool_, worker_context_);
  }

  void TearDown() override {
    ctrl_channel_worker_.reset();
    ctrl_channel_master_.reset();
  }

 protected:
  std::shared_ptr<Environment> master_env_ = nullptr;
  std::shared_ptr<Environment> worker_env_ = nullptr;

  std::shared_ptr<GlobalContext> master_global_context_ = nullptr;
  std::shared_ptr<GlobalContext> worker_global_context_ = nullptr;

  std::shared_ptr<Context> master_context_ = nullptr;
  std::shared_ptr<Context> worker_context_ = nullptr;

  // The schedule output pool.
  std::shared_ptr<ScheduleOutputPool> master_schedule_output_pool_ = nullptr;
  std::shared_ptr<ScheduleOutputPool> worker_schedule_output_pool_ = nullptr;

  // The sampling output pool.
  std::shared_ptr<SamplingOutputPool> master_sampling_output_pool_ = nullptr;
  std::shared_ptr<SamplingOutputPool> worker_sampling_output_pool_ = nullptr;

  std::shared_ptr<ControlChannel> ctrl_channel_master_ = nullptr;
  std::shared_ptr<ControlChannel> ctrl_channel_worker_ = nullptr;

  std::string master_host_;
  uint16_t master_port_;

  size_t world_size_ = 2;
};

TEST_F(ControlChannelTest, TestControlChannel) {
  size_t master_device_block_num;
  size_t master_host_block_num;
  size_t worker_device_block_num;
  size_t worker_host_block_num;

  size_t sampling_node_offload_layer_num = 1;

  int16_t master_lower_layer_idx, master_upper_layer_idx, master_nextn_lower_layer_idx, master_nextn_upper_layer_idx;
  int16_t worker_lower_layer_idx, worker_upper_layer_idx, worker_nextn_lower_layer_idx, worker_nextn_upper_layer_idx;

  // Get model config for layer info
  ModelConfig model_config;
  master_env_->GetModelConfig(model_config);
  int16_t num_layer = model_config.num_layer;
  int16_t num_nextn_predict_layers = 0;  // No nextn predict layers in this test
  size_t mtp_step_num = 0;

  // master node.
  auto master_fn = [&]() {
    // Start master
    ctrl_channel_master_->Listen();

    // Wait all workers connected.
    ctrl_channel_master_->Barrier();

    // synchronize layers.
    std::vector<int16_t> layer_nums = {num_layer, num_nextn_predict_layers};
    std::vector<int16_t> master_layer_indexes;
    ctrl_channel_master_->SynchronizeNodeLayers(layer_nums, mtp_step_num, sampling_node_offload_layer_num,
                                                master_layer_indexes);

    // synchronize cache block num.
    BlockManagerConfig master_block_config;
    master_env_->GetBlockManagerConfig(master_block_config);
    std::vector<size_t> master_block_nums = {master_block_config.device_allocator_config.blocks_num,
                                             master_block_config.host_allocator_config.blocks_num};
    std::vector<size_t> master_node_block_nums;
    ctrl_channel_master_->SynchronizeCacheBlockNum(master_block_nums, master_node_block_nums);

    // Close master.
    ctrl_channel_master_->Close();

    master_device_block_num = master_node_block_nums[0];
    master_host_block_num = master_node_block_nums[1];

    master_lower_layer_idx = master_layer_indexes[0];
    master_upper_layer_idx = master_layer_indexes[1];
    master_nextn_lower_layer_idx = master_layer_indexes[2];
    master_nextn_upper_layer_idx = master_layer_indexes[3];
  };
  std::thread master_thread = std::thread(master_fn);

  // worker node.
  auto worker_fn = [&]() {
    // Start worker
    ctrl_channel_worker_->Connect();

    // Add worker to cluster.
    ctrl_channel_worker_->AddNode();

    // Wait all workers connected.
    ctrl_channel_worker_->Barrier();

    // Wait layer result.
    std::vector<int16_t> layer_nums = {num_layer, num_nextn_predict_layers};
    std::vector<int16_t> worker_layer_indexes;
    ctrl_channel_worker_->SynchronizeNodeLayers(layer_nums, mtp_step_num, sampling_node_offload_layer_num,
                                                worker_layer_indexes);

    // synchronize cache block num.
    BlockManagerConfig worker_block_config;
    worker_env_->GetBlockManagerConfig(worker_block_config);
    std::vector<size_t> worker_block_nums = {worker_block_config.device_allocator_config.blocks_num,
                                             worker_block_config.host_allocator_config.blocks_num};
    std::vector<size_t> worker_node_block_nums;
    ctrl_channel_worker_->SynchronizeCacheBlockNum(worker_block_nums, worker_node_block_nums);

    // Disconnect from master.
    ctrl_channel_worker_->Disconnect();

    worker_device_block_num = worker_node_block_nums[0];
    worker_host_block_num = worker_node_block_nums[1];

    worker_lower_layer_idx = worker_layer_indexes[0];
    worker_upper_layer_idx = worker_layer_indexes[1];
    worker_nextn_lower_layer_idx = worker_layer_indexes[2];
    worker_nextn_upper_layer_idx = worker_layer_indexes[3];
  };
  std::thread worker_thread = std::thread(worker_fn);

  master_thread.join();
  worker_thread.join();

  // Check layer range.
  // With sampling_node_offload_layer_num, the sampling node (worker, node_rank=1) has fewer layers
  // Master (node_rank=0) gets more layers: 0 to 16
  // Worker/sampling node (node_rank=1) gets fewer layers: 17 to 31 (15 layers instead of 16)
  EXPECT_EQ(master_lower_layer_idx, 0);
  EXPECT_EQ(master_upper_layer_idx, 16 + sampling_node_offload_layer_num - 1);
  EXPECT_EQ(master_nextn_lower_layer_idx, -1);
  EXPECT_EQ(master_nextn_upper_layer_idx, -1);
  EXPECT_EQ(worker_lower_layer_idx, 16 + sampling_node_offload_layer_num);
  EXPECT_EQ(worker_upper_layer_idx, 31);
  EXPECT_EQ(worker_nextn_lower_layer_idx, -1);
  EXPECT_EQ(worker_nextn_upper_layer_idx, -1);

  // Check block num result.
  EXPECT_EQ(master_device_block_num, 6);
  EXPECT_EQ(master_host_block_num, 4);
  EXPECT_EQ(worker_device_block_num, 6);
  EXPECT_EQ(worker_host_block_num, 4);
}
