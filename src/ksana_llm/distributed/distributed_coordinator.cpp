/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/distributed/distributed_coordinator.h"
#include <stdexcept>
#include "fmt/core.h"
#include "ksana_llm/distributed/control_channel.h"
#include "ksana_llm/utils/device_utils.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/singleton.h"
#include "ksana_llm/utils/status.h"

namespace ksana_llm {

DistributedCoordinator::DistributedCoordinator(std::shared_ptr<Context> context, PacketCreationFunc packet_creation_fn,
                                               ScheduleOutputPool* schedule_output_pool,
                                               HiddenUnitBufferPool* hidden_unit_buffer_pool,
                                               std::shared_ptr<Environment> env) {
  context_ = context;
  env_ = env ? env : Singleton<Environment>::GetInstance();

  env_->GetPipelineConfig(pipeline_config_);
  control_channel_ = std::make_shared<ControlChannel>(pipeline_config_.master_host, pipeline_config_.master_port,
                                                      pipeline_config_.world_size, pipeline_config_.node_rank,
                                                      packet_creation_fn, schedule_output_pool, env_);
  data_channel_ = std::make_shared<DataChannel>(packet_creation_fn, hidden_unit_buffer_pool, env_);
}

DistributedCoordinator::~DistributedCoordinator() {
  control_channel_.reset();
  data_channel_.reset();
}

Status DistributedCoordinator::InitializeCluster() {
  // Must invoke first, the add node method will report data port to master.
  Status status = data_channel_->Listen();
  if (!status.OK()) {
    throw std::runtime_error(fmt::format("Listen data port error: {}", status.GetMessage()));
  }

  if (context_->IsChief()) {
    status = control_channel_->Listen();
    if (!status.OK()) {
      throw std::runtime_error(fmt::format("Listen on {}:{} error: {}", pipeline_config_.master_host,
                                           pipeline_config_.master_port, status.GetMessage()));
    }
  } else {
    // Server maybe not ready, try connection at most 600 seconds.
    int try_times = 600;
    while (--try_times >= 0) {
      status = control_channel_->Connect();
      if (status.OK()) {
        break;
      }

      if (try_times == 0 && !status.OK()) {
        throw std::runtime_error(fmt::format("Connect to {}:{} error: {}", pipeline_config_.master_host,
                                             pipeline_config_.master_port, status.GetMessage()));
      }

      KLLM_LOG_INFO << "DistributedCoordinator control channel connect failed, try again.";
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    status = control_channel_->AddNode();
    if (!status.OK()) {
      throw std::runtime_error(
          fmt::format("Register node rank {} to master error: {}", pipeline_config_.node_rank, status.GetMessage()));
    }
  }

  // Wait until all nodes connected.
  return control_channel_->Barrier();
}

Status DistributedCoordinator::SynchronizeNodeLayers() {
  // This method tell the downstream data port of every node.
  control_channel_->SynchronizeNodeLayers();

  // Connect downstream node.
  int try_times = 600;
  while (--try_times >= 0) {
    Status status = data_channel_->Connect();
    if (status.OK()) {
      break;
    }

    if (try_times == 0 && !status.OK()) {
      throw std::runtime_error(fmt::format("Connect to {}:{} error: {}", pipeline_config_.master_host,
                                           pipeline_config_.master_port, status.GetMessage()));
    }

    KLLM_LOG_INFO << "DistributedCoordinator data channel connect failed, try again.";
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }

  return Status();
}

Status DistributedCoordinator::SynchronizeCacheBlockNum() { return control_channel_->SynchronizeCacheBlockNum(); }

Status DistributedCoordinator::DestroyCluster() {
  // Close all data channels.
  data_channel_->Disconnect();
  data_channel_->Close();

  if (context_->IsChief()) {
    control_channel_->Close();
  } else {
    control_channel_->Disconnect();
  }

  return Status();
}

}  // namespace ksana_llm
