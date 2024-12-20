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

#include "ksana_llm/distributed/distributed_coordinator.h"
#include "ksana_llm/distributed/distributed_test_helper.h"
#include "ksana_llm/distributed/packet_util.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/memory_utils.h"
#include "ksana_llm/utils/singleton.h"
#include "ksana_llm/utils/socket_util.h"
#include "ksana_llm/utils/status.h"
#include "test.h"

#include "ksana_llm/helpers/block_manager_test_helper.h"
#include "ksana_llm/helpers/environment_test_helper.h"

using namespace ksana_llm;

class DistributedCoordinatorTest : public testing::Test {
 protected:
  void SetUp() override {
    master_env_ = std::make_shared<Environment>();
    worker_env_ = std::make_shared<Environment>();

    uint16_t master_port;
    std::string master_host;
    std::string master_interface;

    GetAvailableInterfaceAndIP(master_interface, master_host);
    GetAvailablePort(master_port);

    // Set model config.
    std::string config_file = GetTestConfigFile();
    Singleton<Environment>::GetInstance()->ParseConfig(config_file);
    master_env_->ParseConfig(config_file);
    worker_env_->ParseConfig(config_file);

    // default config.
    PipelineConfig default_pipeline_config;
    Singleton<Environment>::GetInstance()->GetPipelineConfig(default_pipeline_config);

    // Set master config.
    PipelineConfig master_pipeline_config;
    master_env_->GetPipelineConfig(master_pipeline_config);
    master_pipeline_config.master_host = master_host;
    master_pipeline_config.master_port = master_port;
    master_pipeline_config.world_size = 2;
    master_pipeline_config.node_rank = 0;
    master_env_->SetPipelineConfig(master_pipeline_config);

    int master_tp_para = master_env_->GetTensorParallelSize();
    Singleton<Environment>::GetInstance()->SetPipelineConfig(master_pipeline_config);
    master_context_ = std::make_shared<Context>(master_tp_para, 1);

    // Set worker config.
    PipelineConfig worker_pipeline_config;
    worker_env_->GetPipelineConfig(worker_pipeline_config);
    worker_pipeline_config.master_host = master_host;
    worker_pipeline_config.master_port = master_port;
    worker_pipeline_config.world_size = 2;
    worker_pipeline_config.node_rank = 1;
    worker_env_->SetPipelineConfig(worker_pipeline_config);

    int worker_tp_para = worker_env_->GetTensorParallelSize();
    Singleton<Environment>::GetInstance()->SetPipelineConfig(worker_pipeline_config);
    worker_context_ = std::make_shared<Context>(worker_tp_para, 1);

    // Restore pipeline config.
    Singleton<Environment>::GetInstance()->SetPipelineConfig(default_pipeline_config);

    // Set block manager.
    InitTestBlockManager(Singleton<Environment>::GetInstance().get());

    // Must initialized before create data channel instance.
    master_hidden_unit_buffer_pool_ = new HiddenUnitBufferPool();
    worker_hidden_unit_buffer_pool_ = new HiddenUnitBufferPool();

    master_schedule_output_pool_ = new ScheduleOutputPool();
    worker_schedule_output_pool_ = new ScheduleOutputPool();

    // The packet creation function.
    auto master_packet_creation_fn = [&](PacketType packet_type, size_t body_size) -> Packet* {
      if (packet_type == PacketType::DATA_REQ_HIDDEN_UNIT) {
        Packet* packet = master_hidden_unit_buffer_pool_->GetHostBuffer();
        packet->size = master_hidden_unit_buffer_pool_->GetHostPacketSize(packet);
        packet->type = packet_type;
        return packet;
      }

      return GetPacketObject(packet_type, body_size);
    };

    auto worker_packet_creation_fn = [&](PacketType packet_type, size_t body_size) -> Packet* {
      if (packet_type == PacketType::DATA_REQ_HIDDEN_UNIT) {
        Packet* packet = worker_hidden_unit_buffer_pool_->GetHostBuffer();
        packet->size = worker_hidden_unit_buffer_pool_->GetHostPacketSize(packet);
        packet->type = packet_type;
        return packet;
      }

      return GetPacketObject(packet_type, body_size);
    };

    master_distributed_coordinator_ = std::make_shared<DistributedCoordinator>(
        master_context_, master_packet_creation_fn, master_schedule_output_pool_, master_hidden_unit_buffer_pool_,
        master_env_);
    worker_distributed_coordinator_ = std::make_shared<DistributedCoordinator>(
        worker_context_, worker_packet_creation_fn, worker_schedule_output_pool_, worker_hidden_unit_buffer_pool_,
        worker_env_);
  }

  void TearDown() override {
    master_distributed_coordinator_.reset();
    worker_distributed_coordinator_.reset();

    delete master_hidden_unit_buffer_pool_;
    delete worker_hidden_unit_buffer_pool_;

    delete master_schedule_output_pool_;
    delete worker_schedule_output_pool_;
  }

 protected:
  std::shared_ptr<Context> master_context_ = nullptr;
  std::shared_ptr<Context> worker_context_ = nullptr;

  std::shared_ptr<Environment> master_env_ = nullptr;
  std::shared_ptr<Environment> worker_env_ = nullptr;

  HiddenUnitBufferPool* master_hidden_unit_buffer_pool_ = nullptr;
  HiddenUnitBufferPool* worker_hidden_unit_buffer_pool_ = nullptr;

  // The schedule output pool.
  ScheduleOutputPool* master_schedule_output_pool_ = nullptr;
  ScheduleOutputPool* worker_schedule_output_pool_ = nullptr;

  std::shared_ptr<DistributedCoordinator> master_distributed_coordinator_ = nullptr;
  std::shared_ptr<DistributedCoordinator> worker_distributed_coordinator_ = nullptr;
};

TEST_F(DistributedCoordinatorTest, TestDistributedCoordinator) {
  FakedTestBlockManager* test_block_manager = new FakedTestBlockManager();
  SetBlockManager(test_block_manager);

  // Check context.
  EXPECT_TRUE(master_context_->IsChief() == true);
  EXPECT_TRUE(worker_context_->IsChief() == false);

  // master node.
  auto master_fn = [&]() {
    // Set block num for master.
    test_block_manager->SetBlockNumber(10, 8);

    master_distributed_coordinator_->InitializeCluster();
    master_distributed_coordinator_->SynchronizeNodeLayers();
    master_distributed_coordinator_->SynchronizeCacheBlockNum();
    master_distributed_coordinator_->DestroyCluster();
  };
  std::thread master_thread = std::thread(master_fn);

  // worker node.
  auto worker_fn = [&]() {
    // Set block num for worker.
    test_block_manager->SetBlockNumber(6, 4);

    worker_distributed_coordinator_->InitializeCluster();
    worker_distributed_coordinator_->SynchronizeNodeLayers();
    worker_distributed_coordinator_->SynchronizeCacheBlockNum();
    worker_distributed_coordinator_->DestroyCluster();
  };
  std::thread worker_thread = std::thread(worker_fn);

  master_thread.join();
  worker_thread.join();

  // Check layers and block num.
  PipelineConfig master_pipeline_config;
  PipelineConfig worker_pipeline_config;
  master_env_->GetPipelineConfig(master_pipeline_config);
  worker_env_->GetPipelineConfig(worker_pipeline_config);

  EXPECT_EQ(master_pipeline_config.lower_layer_idx, 0);
  EXPECT_EQ(master_pipeline_config.upper_layer_idx, 15);
  EXPECT_EQ(worker_pipeline_config.lower_layer_idx, 16);
  EXPECT_EQ(worker_pipeline_config.upper_layer_idx, 31);

  EXPECT_EQ(master_pipeline_config.device_block_num, 6);
  EXPECT_EQ(master_pipeline_config.host_block_num, 4);
  EXPECT_EQ(worker_pipeline_config.device_block_num, 6);
  EXPECT_EQ(worker_pipeline_config.host_block_num, 4);
}
