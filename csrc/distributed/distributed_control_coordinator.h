/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>

#include "distributed/control_channel.h"

#include "configure/context.h"
#include "utils/status.h"

namespace ksana_llm {

// manage all the distributed pipeline nodes.
class DistributedControlCoordinator {
 public:
  DistributedControlCoordinator(PacketCreationFunc packet_creation_fn,
                                std::shared_ptr<ScheduleOutputPool> schedule_output_pool,
                                std::shared_ptr<SamplingOutputPool> sampling_output_pool,
                                std::shared_ptr<Context> context);
  ~DistributedControlCoordinator();

  // Initialize and destroy the distributed cluster.
  Status InitializeCluster();
  void DestroyCluster();

  // Synchronize layers and block num.
  // If sampling_node_offload_layer_num > 0, offload that many layers from sampling node (node_rank == world_size - 1)
  // to other nodes evenly input:  [num_layer, num_nextn_predict_layers] output: [lower_layer_idx, upper_layer_idx,
  // lower_nextn_layer_idx, upper_nextn_layer_idx]
  Status SynchronizeNodeLayers(const std::vector<int16_t>& layer_nums, size_t mtp_step_num,
                               size_t sampling_node_offload_layer_num, std::vector<int16_t>& layer_indexes);

  // Synchronize block num.
  // input:  [device_block_num, host_block_num]
  // output: [device_block_num, host_block_num]
  Status SynchronizeCacheBlockNum(const std::vector<size_t>& block_nums, std::vector<size_t>& node_block_nums);

  // Synchronize data channel on tcp mode of all executors.
  Status SynchronizeTcpDataPort(const std::vector<HostPort>& executor_host_ports,
                                std::vector<HostPort>& executor_downstream_host_ports);

#ifdef ENABLE_CUDA
  // Synchronize nccl data channel on nccl mode of all executors.
  Status SynchronizeNcclUniqueId(ncclUniqueId& nccl_unique_id);
#endif

  // Synchronize EP communicator config across all nodes.
  // Engine gets EPCommunicationConfig from executors, sends to master.
  // Master aggregates config from all nodes and broadcasts back.
  // executor_comm_configs: input vector of EPCommunicationConfig from each executor
  // aggregated_comm_configs: output aggregated config from all nodes
  Status SynchronizeEPCommunicationConfig(const std::vector<EPCommunicationConfig>& executor_comm_configs,
                                          std::vector<EPCommunicationConfig>& aggregated_comm_configs);

  // Create control channel used for result report.
  Status InitializeResultReporter();

  // Stop to accept any new connection.
  void Frozen();

  // Stop to accept any new connection.
  Status Barrier();

 private:
  // Global context.
  std::shared_ptr<Context> context_ = nullptr;

  // control communication, for engine process only.
  std::shared_ptr<ControlChannel> control_channel_ = nullptr;
};

}  // namespace ksana_llm
