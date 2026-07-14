/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <map>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

#include "configure/environment.h"
#include "distributed/control_message.h"
#include "distributed/node_info.h"
#include "distributed/raw_packet.h"
#include "distributed/raw_socket.h"
#include "runtime/sampling_output.h"
#include "runtime/schedule_output.h"
#include "utils/status.h"

#ifdef ENABLE_CUDA
#  include <nccl.h>
#endif

namespace ksana_llm {

// Used to send & recv control message.
class ControlChannel {
 public:
  ControlChannel(PacketCreationFunc packet_creation_fn, std::shared_ptr<ScheduleOutputPool> schedule_output_pool,
                 std::shared_ptr<SamplingOutputPool> sampling_output_pool, std::shared_ptr<Context> context);
  ~ControlChannel();

  // For master node only.
  virtual Status Listen();

  // Close open port.
  virtual Status Close();

  // For slave node only.
  virtual Status Connect();

  // disconnect from master.
  virtual Status Disconnect();

  // Wait until all nodes arrive same location.
  virtual Status Barrier();

  // Stop to accept any new connection.
  virtual Status Frozen();

  // Add node to cluster.
  virtual Status AddNode();

  // Synchronize data channel on tcp mode of all executors.
  Status SynchronizeTcpDataPort(const std::vector<HostPort>& executor_host_ports,
                                std::vector<HostPort>& executor_downstream_host_ports);

#ifdef ENABLE_CUDA
  // Synchronize nccl data channel on nccl mode of all executors.
  Status SynchronizeNcclUniqueId(ncclUniqueId& nccl_unique_id);
#endif

  // Synchronize EP communicator config across all nodes.
  // Engine gets EPCommunicationConfig from executors, sends to master.
  // Master aggregates data from all nodes and broadcasts back.
  // executor_config: input vector of EPCommunicationConfig from each executor
  // aggregated_configs: output aggregated data from all nodes
  Status SynchronizeEPCommunicationConfig(const std::vector<EPCommunicationConfig>& executor_config,
                                          std::vector<EPCommunicationConfig>& aggregated_configs);

  // Synchronize model layers for every node.
  // If sampling_node_offload_layer_num > 0, offload that many layers from sampling node (node_rank == world_size - 1)
  // to other nodes evenly input:  [num_layer, num_nextn_predict_layers] output: [lower_layer_idx, upper_layer_idx,
  // lower_nextn_layer_idx, upper_nextn_layer_idx]
  Status SynchronizeNodeLayers(const std::vector<int16_t>& layer_nums, size_t mtp_step_num,
                               size_t sampling_node_offload_layer_num, std::vector<int16_t>& layer_indexes);

  // Synchronize block num.
  // input:  [device_block_num, host_block_num]
  // output: [device_block_num, host_block_num]
  Status SynchronizeCacheBlockNum(const std::vector<size_t>& block_nums, std::vector<size_t>& node_block_nums);

  // Create control channel used for result report.
  Status InitializeResultReporter();

  // Shutdown the pipeline cluster.
  virtual Status ShutdownCluster();

 protected:
  // worker to master.
  Status HandlePacketFromWorker(NodeInfo* node_info, Packet* packet);

  // master to woker
  Status HandlePacketFromMaster(NodeInfo* node_info, Packet* packet);

 private:
  // Add node.
  virtual Status ProcessAddNodeRequest(NodeInfo* node_info, Packet* req_packet);
  virtual Status ProcessAddNodeResponse(NodeInfo* node_info, Packet* rsp_packet);

  // Add reporter.
  virtual Status ProcessAddReporterRequest(NodeInfo* node_info, Packet* req_packet);
  virtual Status ProcessAddReporterResponse(NodeInfo* node_info, Packet* rsp_packet);

  // tcp data
  virtual Status ProcessTcpDataRequest(NodeInfo* node_info, Packet* req_packet);
  virtual Status ProcessTcpDataResponse(NodeInfo* node_info, Packet* rsp_packet);

  // nccl data
  virtual Status ProcessNcclDataRequest(NodeInfo* node_info, Packet* req_packet);
  virtual Status ProcessNcclDataResponse(NodeInfo* node_info, Packet* rsp_packet);

  // EP communicator config
  virtual Status ProcessEPCommunicationConfigRequest(NodeInfo* node_info, Packet* req_packet);
  virtual Status ProcessEPCommunicationConfigResponse(NodeInfo* node_info, Packet* rsp_packet);

  // heartbeat
  virtual Status ProcessHeartbeatRequest(NodeInfo* node_info, Packet* req_packet);
  virtual Status ProcessHeartbeatResponse(NodeInfo* node_info, Packet* rsp_packet);

  // Barrier
  virtual Status ProcessBarrierRequest(NodeInfo* node_info, Packet* req_packet);
  virtual Status ProcessBarrierResponse(NodeInfo* node_info, Packet* rsp_packet);

  // Layers
  virtual Status ProcessLayerRequest(NodeInfo* node_info, Packet* req_packet);
  virtual Status ProcessLayerResponse(NodeInfo* node_info, Packet* rsp_packet);

  // cache block num
  Status ProcessBlockNumRequest(NodeInfo* node_info, Packet* req_packet);
  Status ProcessBlockNumResponse(NodeInfo* node_info, Packet* rsp_packet);

  // Schedule output, from master to worker.
  Status ProcessScheduleRequest(NodeInfo* node_info, Packet* req_packet);
  Status ProcessScheduleResponse(NodeInfo* node_info, Packet* rsp_packet);

  // Sampling output, from worker to master.
  Status ProcessSamplingOutputRequest(NodeInfo* node_info, Packet* req_packet);
  Status ProcessSamplingOutputResponse(NodeInfo* node_info, Packet* rsp_packet);

  // Process shutdown message.
  Status ProcessShutdownRequest(NodeInfo* node_info, Packet* req_packet);
  Status ProcessShutdownResponse(NodeInfo* node_info, Packet* rsp_packet);

  // heartbeat thread handle.
  virtual Status ProcessHeartbeatLoop();

  // send schedule output to workers.
  virtual Status ProcessSendScheduleOutputLoop();
  virtual Status ProcessSendSamplingOutputLoop();

  virtual Status ProcessExpertParallelResponse(NodeInfo* node_info, Packet* req_packet) { return Status(); }
  virtual Status ProcessExpertParallelRequest(NodeInfo* node_info, Packet* req_packet) { return Status(); }

  // Generate layer distribution for all nodes
  // If sampling_node_offload_layer_num > 0, offload that many layers from sampling node (node_rank == world_size - 1)
  // to other nodes evenly
  std::vector<AllocateLayerRequest> GenerateLayerDistribution(const size_t num_layer,
                                                              const size_t num_nextn_predict_layers,
                                                              const size_t mtp_step_num,
                                                              const size_t sampling_node_offload_layer_num);

 private:
  std::shared_ptr<RawSocket> raw_socket_ = nullptr;

  // The context.
  std::shared_ptr<Context> context_ = nullptr;

  // The buffer pool.
  std::shared_ptr<ScheduleOutputPool> schedule_output_pool_ = nullptr;
  std::shared_ptr<SamplingOutputPool> sampling_output_pool_ = nullptr;

  std::string master_host_;
  uint16_t master_port_;

  size_t world_size_;
  size_t expert_world_size_;
  int node_rank_;

  // Used for barrier.
  int barrier_clock_idx_ = 0;

  // Whether the control channl is terminated..
  bool terminated_ = false;

  // The node ranks that has sent barrier message.
  std::unordered_map<int, std::unordered_set<int>> barrier_req_ranks_;

  // whether some operations have been finished.
  bool layer_allocated_ = false;
  bool tcp_data_allocated_ = false;
  bool nccl_data_allocated_ = false;
  bool ep_communication_config_synchronized_ = false;
  bool block_num_synchronized_ = false;
  bool reporter_initialized_ = false;

  // The barrier clocks that has checked by master.
  std::unordered_set<int> barrier_rsp_clocks_;

  // The shutdown nodes.
  std::unordered_set<int> shutdown_nodes_;

  // Notify for barrier utility.
  std::condition_variable barrier_cv_;
  std::condition_variable shutdown_cv_;
  std::condition_variable tcp_data_cv_;
  std::condition_variable nccl_data_cv_;
  std::condition_variable ep_communication_config_cv_;
  std::condition_variable layer_allocation_cv_;
  std::condition_variable block_num_cv_;
  std::condition_variable reporter_cv_;

  // rank to nodes and vice versa.
  std::unordered_map<int, NodeInfo> rank_nodes_;
  std::unordered_map<NodeInfo, int, NodeInfoHash, NodeInfoEqual> node_ranks_;

  // rank to tcp data node.
  std::unordered_map<int, std::vector<HostPort>> rank_tcp_data_nodes_;

  // The result state of current node.
  std::vector<HostPort> executor_downstream_host_ports_;

#ifdef ENABLE_CUDA
  ncclUniqueId nccl_unique_id_;
#endif

  // rank to EP communicator config from all executors
  std::unordered_map<int, std::vector<EPCommunicationConfig>> rank_ep_communication_configs_;

  // Aggregated EP communicator config result
  std::vector<EPCommunicationConfig> aggregated_ep_communication_configs_;

  // [lower_layer_idx, upper_layer_idx, lower_nextn_layer_idx, upper_nextn_layer_idx]
  std::vector<int16_t> node_layer_indexes_;

  // [device_block_num, host_block_num]
  std::vector<size_t> node_block_nums_;

  // rank to {device_block_num, host_block_num}
  std::unordered_map<int, std::pair<size_t, size_t>> rank_cache_block_num_;

  // The heartbeat thread & timeout.
  std::shared_ptr<std::thread> heartbeat_thread_ = nullptr;
  size_t heartbeat_timeout_secs_ = 120;
  size_t heartbeat_interval_secs_ = 30;

  // rank to timestamp.
  std::unordered_map<int, time_t> node_heartbeat_timestamp_;

  // Send packet threads.
  std::shared_ptr<std::thread> send_schedule_output_thread_ = nullptr;
  std::shared_ptr<std::thread> send_sampling_output_thread_ = nullptr;

  // Protect multi-thread receive handles.
  std::mutex mutex_;
};

}  // namespace ksana_llm
