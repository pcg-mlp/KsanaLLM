/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "distributed/distributed_control_coordinator.h"

#include <stdexcept>

#include "fmt/core.h"
#include "utils/logger.h"

namespace ksana_llm {

DistributedControlCoordinator::DistributedControlCoordinator(PacketCreationFunc packet_creation_fn,
                                                             std::shared_ptr<ScheduleOutputPool> schedule_output_pool,
                                                             std::shared_ptr<SamplingOutputPool> sampling_output_pool,
                                                             std::shared_ptr<Context> context) {
  context_ = context;
  control_channel_ =
      std::make_shared<ControlChannel>(packet_creation_fn, schedule_output_pool, sampling_output_pool, context_);
}

DistributedControlCoordinator::~DistributedControlCoordinator() { control_channel_.reset(); }

// Extract function for pipeline parallel later.
Status DistributedControlCoordinator::InitializeCluster() {
  // Determine if current node is chief/master:
  // For PP mode (world_size > 1): use IsDistributedWorker()
  // For EP-only mode (world_size == 1, expert_world_size > 1): use IsExpertParallelChief()
  bool is_chief = context_->global->IsStandalone() ? context_->global->IsExpertParallelChief()
                                                   : !context_->global->IsDistributedWorker();

  if (is_chief) {
    // Port maybe not released yet by previous process, try listen at most 600 seconds.
    const int max_try_times = 600;
    int try_times = max_try_times;
    while (--try_times >= 0) {
      Status status = control_channel_->Listen();
      if (status.OK()) {
        break;
      }

      if (try_times == max_try_times - 1) {
        KLLM_LOG_WARNING << "Listen on " << context_->global->GetMasterHost() << ":"
                         << context_->global->GetMasterPort() << " failed: " << status.GetMessage()
                         << ", will retry in " << max_try_times << " seconds.";
      }

      if (try_times == 0) {
        KLLM_LOG_ERROR << "Listen on " << context_->global->GetMasterHost() << ":" << context_->global->GetMasterPort()
                       << " failed after " << max_try_times << " retries: " << status.GetMessage();
        throw std::runtime_error(fmt::format("Listen on {}:{} error: {}", context_->global->GetMasterHost(),
                                             context_->global->GetMasterPort(), status.GetMessage()));
      }
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
  } else {
    // Server maybe not ready, try connection at most 600 seconds.
    const int max_try_times = 600;
    int try_times = max_try_times;
    while (--try_times >= 0) {
      Status status = control_channel_->Connect();
      if (status.OK()) {
        break;
      }

      if (try_times == max_try_times - 1) {
        KLLM_LOG_WARNING << "Connect to " << context_->global->GetMasterHost() << ":"
                         << context_->global->GetMasterPort() << " failed: " << status.GetMessage()
                         << ", will retry in " << max_try_times << " seconds.";
      }

      if (try_times == 0) {
        KLLM_LOG_ERROR << "Connect to " << context_->global->GetMasterHost() << ":" << context_->global->GetMasterPort()
                       << " failed after " << max_try_times << " retries: " << status.GetMessage();
        throw std::runtime_error(fmt::format("Connect to {}:{} error: {}", context_->global->GetMasterHost(),
                                             context_->global->GetMasterPort(), status.GetMessage()));
      }
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    Status status = control_channel_->AddNode();
    if (!status.OK()) {
      throw std::runtime_error(
          fmt::format("Register node rank {} error: {}", context_->global->GetNodeRank(), status.GetMessage()));
    }
  }

  // Wait until all nodes connected.
  control_channel_->Barrier();

  return Status();
}

Status DistributedControlCoordinator::SynchronizeTcpDataPort(const std::vector<HostPort>& executor_host_ports,
                                                             std::vector<HostPort>& executor_downstream_host_ports) {
  return control_channel_->SynchronizeTcpDataPort(executor_host_ports, executor_downstream_host_ports);
}

#ifdef ENABLE_CUDA
Status DistributedControlCoordinator::SynchronizeNcclUniqueId(ncclUniqueId& nccl_unique_id) {
  return control_channel_->SynchronizeNcclUniqueId(nccl_unique_id);
}
#endif

Status DistributedControlCoordinator::SynchronizeEPCommunicationConfig(
    const std::vector<EPCommunicationConfig>& executor_comm_configs,
    std::vector<EPCommunicationConfig>& aggregated_comm_configs) {
  return control_channel_->SynchronizeEPCommunicationConfig(executor_comm_configs, aggregated_comm_configs);
}

Status DistributedControlCoordinator::SynchronizeNodeLayers(const std::vector<int16_t>& layer_nums, size_t mtp_step_num,
                                                            size_t sampling_node_offload_layer_num,
                                                            std::vector<int16_t>& layer_indexes) {
  return control_channel_->SynchronizeNodeLayers(layer_nums, mtp_step_num, sampling_node_offload_layer_num,
                                                 layer_indexes);
}

Status DistributedControlCoordinator::SynchronizeCacheBlockNum(const std::vector<size_t>& block_nums,
                                                               std::vector<size_t>& node_block_nums) {
  return control_channel_->SynchronizeCacheBlockNum(block_nums, node_block_nums);
}

Status DistributedControlCoordinator::InitializeResultReporter() {
  return control_channel_->InitializeResultReporter();
}

void DistributedControlCoordinator::Frozen() {
  if (control_channel_) {
    control_channel_->Frozen();
  }
}

Status DistributedControlCoordinator::Barrier() { return control_channel_->Barrier(); }

void DistributedControlCoordinator::DestroyCluster() {
  if (!context_->global->IsDistributedWorker()) {
    control_channel_->Close();
  } else {
    control_channel_->Disconnect();
  }
}

}  // namespace ksana_llm
