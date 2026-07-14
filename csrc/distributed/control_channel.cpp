/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "distributed/control_channel.h"

#include <cassert>
#include <chrono>
#include <complex>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_set>
#include <utility>

#include "configure/environment.h"
#include "distributed/control_message.h"
#include "distributed/node_info.h"
#include "distributed/packet_util.h"
#include "distributed/raw_packet.h"
#include "distributed/raw_socket.h"
#include "utils/logger.h"
#include "utils/memory_utils.h"
#include "utils/ret_code.h"
#include "utils/service_utils.h"
#include "utils/status.h"

namespace ksana_llm {

ControlChannel::ControlChannel(PacketCreationFunc packet_creation_fn,
                               std::shared_ptr<ScheduleOutputPool> schedule_output_pool,
                               std::shared_ptr<SamplingOutputPool> sampling_output_pool,
                               std::shared_ptr<Context> context) {
  context_ = context;

  master_host_ = context_->global->GetMasterHost();
  master_port_ = context_->global->GetMasterPort();

  world_size_ = context->global->GetWorldSize();
  expert_world_size_ = context->global->GetExpertParallelWorldSize();

  // For EP-only mode (world_size == 1, expert_world_size > 1), use expert_node_rank
  // For PP mode (world_size > 1), use node_rank
  node_rank_ = context->global->IsStandalone() ? context->global->GetExpertParallelExpertNodeRank()
                                               : context->global->GetNodeRank();

  if (!context_->global->IsStandalone() || !context_->global->IsExpertParallelStandalone()) {
    raw_socket_ = std::make_shared<RawSocket>(packet_creation_fn);

    // Start assisant threads.
    heartbeat_thread_ = std::unique_ptr<std::thread>(new std::thread(&ControlChannel::ProcessHeartbeatLoop, this));
  }

  if (!context_->global->IsStandalone()) {
    schedule_output_pool_ = schedule_output_pool;
    if (schedule_output_pool_) {
      send_schedule_output_thread_ =
          std::unique_ptr<std::thread>(new std::thread(&ControlChannel::ProcessSendScheduleOutputLoop, this));
    }

    sampling_output_pool_ = sampling_output_pool;
    if (sampling_output_pool_) {
      send_sampling_output_thread_ =
          std::unique_ptr<std::thread>(new std::thread(&ControlChannel::ProcessSendSamplingOutputLoop, this));
    }
  }
}

ControlChannel::~ControlChannel() {
  if (!context_->global->IsStandalone()) {
    terminated_ = true;

    if (schedule_output_pool_) {
      schedule_output_pool_->Stop();
    }

    if (sampling_output_pool_) {
      sampling_output_pool_->Stop();
    }

    if (heartbeat_thread_) {
      heartbeat_thread_->join();
    }

    if (send_schedule_output_thread_) {
      send_schedule_output_thread_->join();
    }

    if (send_sampling_output_thread_) {
      send_sampling_output_thread_->join();
    }
  }
}

Status ControlChannel::ProcessHeartbeatLoop() {
  while (!terminated_) {
    time_t curr_time_stamp = GetCurrentTime();

    {
      std::unique_lock<std::mutex> lock(mutex_);

      // For master and worker.
      for (auto it = node_heartbeat_timestamp_.begin(); it != node_heartbeat_timestamp_.end(); ++it) {
        time_t last_time_stamp = it->second;
        if (curr_time_stamp > last_time_stamp + heartbeat_timeout_secs_) {
          KLLM_LOG_ERROR << "Heartbeat timeout, cluster exited.";

          if (node_rank_ == 0) {
            // For master node, stop whole cluster.
            ShutdownCluster();
          } else {
            // For worker node, stop current service.
            GetServiceLifetimeManager()->ShutdownService();
          }
        }

        // For worker node.
        if (node_rank_ > 0) {
          if (raw_socket_->IsConnected() && curr_time_stamp > last_time_stamp + heartbeat_interval_secs_) {
            // Send heartbeat to master.
            Packet* packet = GetPacketObject(PacketType::CONTROL_REQ_HEARTBEAT, 0);
            if (packet == nullptr) {
              throw std::runtime_error("ControlChannel::ProcessHeartbeatLoop allocate memory error.");
            }

            HeartbeatRequest* heartbeat_req = reinterpret_cast<HeartbeatRequest*>(packet->body);
            heartbeat_req->node_rank = node_rank_;

            Status status = raw_socket_->Send({master_host_, master_port_}, packet);
            free(packet);

            if (!status.OK()) {
              KLLM_LOG_ERROR << "ControlChannel heartbeat error, send packet failed, info:" << status.GetMessage();
            }
          }
        }
      }
    }
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }

  return Status();
}

Status ControlChannel::ProcessSendScheduleOutputLoop() {
  while (!terminated_) {
    Packet* packet = schedule_output_pool_->GetSendSerializedPacket();
    if (!packet) {
      KLLM_LOG_WARNING << "ProcessSendScheduleLoop empty packet from send queue, break..";
      break;
    }

    // Send to all workers
    for (auto& [rank_id, node_info] : rank_nodes_) {
      Status status = raw_socket_->Send(node_info, packet);
      if (!status.OK()) {
        KLLM_LOG_ERROR << "ControlChannel broadcast schedule output error, send packet failed, info:"
                       << status.GetMessage();
      }
    }

    free(packet);
  }

  return Status();
}

Status ControlChannel::ProcessSendSamplingOutputLoop() {
  while (!terminated_) {
    Packet* packet = sampling_output_pool_->GetSendSerializedPacket();
    if (!packet) {
      KLLM_LOG_WARNING << "ProcessSendSamplingOutputLoop empty packet from send queue, break..";
      break;
    }

    // Send to master.
    Status status = raw_socket_->Send({master_host_, master_port_}, packet);
    if (!status.OK()) {
      KLLM_LOG_ERROR << "ControlChannel send sampling output error, send packet failed, info:" << status.GetMessage();
    }

    free(packet);
  }

  return Status();
}

Status ControlChannel::Listen() {
  auto listen_fn = [this](NodeInfo* node_info, Packet* packet) -> Status {
    return HandlePacketFromWorker(node_info, packet);
  };

  KLLM_LOG_INFO << "ControlChannel listen on " << master_host_ << ":" << master_port_ << ".";
  Status status = raw_socket_->Listen(master_host_, master_port_, listen_fn);
  if (!status.OK()) {
    KLLM_LOG_ERROR << "Listen control channel error:" << status.GetMessage();
  }

  return status;
}

Status ControlChannel::Close() { return raw_socket_->Close(); }

Status ControlChannel::Connect() {
  auto connect_fn = [this](NodeInfo* node_info, Packet* packet) -> Status {
    return HandlePacketFromMaster(node_info, packet);
  };

  KLLM_LOG_INFO << "ControlChannel connect to " << master_host_ << ":" << master_port_ << ".";
  Status status = raw_socket_->Connect(master_host_, master_port_, connect_fn);
  if (!status.OK()) {
    KLLM_LOG_ERROR << "Connect control channel error:" << status.GetMessage();
  } else {
    KLLM_LOG_INFO << "ControlChannel connect to " << master_host_ << ":" << master_port_ << " succeed.";
  }

  return status;
}

Status ControlChannel::Disconnect() { return raw_socket_->Disconnect(); }

Status ControlChannel::ProcessAddNodeRequest(NodeInfo* node_info, Packet* req_packet) {
  auto it = node_ranks_.find(*node_info);
  if (it != node_ranks_.end()) {
    return Status(RET_RUNTIME_FAILED, fmt::format("Duplicated node {}:{}", node_info->host, node_info->port));
  }

  AddNodeRequest* add_node_req = reinterpret_cast<AddNodeRequest*>(req_packet->body);

  int node_rank = add_node_req->node_rank;
  node_ranks_[*node_info] = node_rank;
  rank_nodes_[node_rank] = *node_info;

  Packet* rsp_packet = GetPacketObject(PacketType::CONTROL_RSP_ADD_NODE, 0);
  if (rsp_packet == nullptr) {
    throw std::runtime_error("ControlChannel::ProcessAddNodeRequest allocate memory error.");
  }

  Status status = raw_socket_->Send(*node_info, rsp_packet);
  free(rsp_packet);

  if (!status.OK()) {
    KLLM_LOG_ERROR << "ControlChannel process the add node reqeust error, send packet failed, info:"
                   << status.GetMessage();
  }

  free(req_packet);
  return status;
}

Status ControlChannel::ProcessAddNodeResponse(NodeInfo* node_info, Packet* rsp_packet) {
  free(rsp_packet);
  return Status();
}

Status ControlChannel::ProcessAddReporterRequest(NodeInfo* node_info, Packet* req_packet) {
  Packet* rsp_packet = GetPacketObject(PacketType::CONTROL_RSP_ADD_REPORTER, 0);
  if (rsp_packet == nullptr) {
    throw std::runtime_error("ControlChannel::ProcessAddReporterRequest allocate memory error.");
  }

  Status status = raw_socket_->Send(*node_info, rsp_packet);
  free(rsp_packet);

  if (!status.OK()) {
    KLLM_LOG_ERROR << "ControlChannel process the add reporter reqeust error, send packet failed, info:"
                   << status.GetMessage();
  }

  std::unique_lock<std::mutex> lock(mutex_);
  reporter_initialized_ = true;
  reporter_cv_.notify_all();

  free(req_packet);
  return status;
}

Status ControlChannel::ProcessAddReporterResponse(NodeInfo* node_info, Packet* rsp_packet) {
  std::unique_lock<std::mutex> lock(mutex_);
  reporter_initialized_ = true;
  reporter_cv_.notify_all();

  free(rsp_packet);
  return Status();
}

Status ControlChannel::ProcessTcpDataRequest(NodeInfo* node_info, Packet* req_packet) {
  TcpDataRequest* tcp_data_req = reinterpret_cast<TcpDataRequest*>(req_packet->body);

  int node_rank = tcp_data_req->node_rank;
  size_t executor_num = context_->global->GetTensorParallelSize();
  rank_tcp_data_nodes_[node_rank].resize(executor_num);

  for (size_t executor_id = 0; executor_id < executor_num; ++executor_id) {
    HostPort host_port = tcp_data_req->executor_host_ports[executor_id];
    rank_tcp_data_nodes_[node_rank][executor_id] = host_port;
  }

  // notify if all worker arrived.
  if (rank_tcp_data_nodes_.size() == world_size_) {
    std::unique_lock<std::mutex> lock(mutex_);
    tcp_data_cv_.notify_all();
  }

  free(req_packet);
  return Status();
}

Status ControlChannel::ProcessTcpDataResponse(NodeInfo* node_info, Packet* rsp_packet) {
  TcpDataResponse* tcp_data_rsp = reinterpret_cast<TcpDataResponse*>(rsp_packet->body);

  size_t executor_num = context_->global->GetTensorParallelSize();
  executor_downstream_host_ports_.resize(executor_num);

  for (size_t executor_id = 0; executor_id < executor_num; ++executor_id) {
    HostPort host_port = tcp_data_rsp->executor_downstream_host_ports[executor_id];
    executor_downstream_host_ports_[executor_id] = host_port;
  }

  {
    std::unique_lock<std::mutex> lock(mutex_);
    tcp_data_allocated_ = true;
    tcp_data_cv_.notify_all();
  }

  free(rsp_packet);
  return Status();
}

#ifdef ENABLE_CUDA
Status ControlChannel::ProcessNcclDataRequest(NodeInfo* node_info, Packet* req_packet) {
  NcclDataRequest* nccl_data_req = reinterpret_cast<NcclDataRequest*>(req_packet->body);

  std::string s = nccl_data_req->nccl_unique_id;
  memcpy(nccl_unique_id_.internal, nccl_data_req->nccl_unique_id, sizeof(ncclUniqueId));

  {
    std::unique_lock<std::mutex> lock(mutex_);
    nccl_data_allocated_ = true;
    nccl_data_cv_.notify_all();
  }

  free(req_packet);
  return Status();
}
#endif

#ifdef ENABLE_CUDA
Status ControlChannel::ProcessNcclDataResponse(NodeInfo* node_info, Packet* rsp_packet) {
  free(rsp_packet);
  return Status();
}
#endif

Status ControlChannel::ProcessEPCommunicationConfigRequest(NodeInfo* node_info, Packet* req_packet) {
  EPCommunicationConfigRequest* ep_config_req = reinterpret_cast<EPCommunicationConfigRequest*>(req_packet->body);

  int node_rank = ep_config_req->node_rank;
  size_t num_structs = ep_config_req->data_size / sizeof(EPCommunicationConfig);

  // Store data from this worker node
  {
    std::unique_lock<std::mutex> lock(mutex_);
    rank_ep_communication_configs_[node_rank].resize(num_structs);
    memcpy(rank_ep_communication_configs_[node_rank].data(), ep_config_req->ep_comm_configs, ep_config_req->data_size);

    // Notify if all workers have arrived
    if (rank_ep_communication_configs_.size() == expert_world_size_) {
      ep_communication_config_cv_.notify_all();
    }
  }

  free(req_packet);
  return Status();
}

Status ControlChannel::ProcessEPCommunicationConfigResponse(NodeInfo* node_info, Packet* rsp_packet) {
  EPCommunicationConfigResponse* ep_config_rsp = reinterpret_cast<EPCommunicationConfigResponse*>(rsp_packet->body);

  size_t num_structs = ep_config_rsp->data_size / sizeof(EPCommunicationConfig);

  // Copy aggregated data from master
  {
    std::unique_lock<std::mutex> lock(mutex_);
    aggregated_ep_communication_configs_.resize(num_structs);
    memcpy(aggregated_ep_communication_configs_.data(), ep_config_rsp->aggregated_ep_comm_configs,
           ep_config_rsp->data_size);
    ep_communication_config_synchronized_ = true;
    ep_communication_config_cv_.notify_all();
  }

  free(rsp_packet);
  return Status();
}

Status ControlChannel::ProcessHeartbeatRequest(NodeInfo* node_info, Packet* req_packet) {
  std::unique_lock<std::mutex> lock(mutex_);

  HeartbeatRequest* heartbeat_req = reinterpret_cast<HeartbeatRequest*>(req_packet->body);
  int node_rank = heartbeat_req->node_rank;
  node_heartbeat_timestamp_[node_rank] = GetCurrentTime();

  // Send response.
  Packet* packet = GetPacketObject(PacketType::CONTROL_RSP_HEARTBEAT, 0);
  if (packet == nullptr) {
    throw std::runtime_error("ControlChannel::ProcessHeartbeatRequest allocate memory error.");
  }

  HeartbeatResponse* heartbeat_rsp = reinterpret_cast<HeartbeatResponse*>(packet->body);
  heartbeat_rsp->node_rank = node_rank_;

  Status status = raw_socket_->Send({master_host_, master_port_}, packet);
  free(packet);

  if (!status.OK()) {
    KLLM_LOG_ERROR << "ControlChannel process heartbeat reqeust error, send packet failed, info:"
                   << status.GetMessage();
  }

  free(req_packet);
  return Status();
}

Status ControlChannel::ProcessHeartbeatResponse(NodeInfo* node_info, Packet* rsp_packet) {
  std::unique_lock<std::mutex> lock(mutex_);

  HeartbeatResponse* heartbeat_rsp = reinterpret_cast<HeartbeatResponse*>(rsp_packet->body);
  int node_rank = heartbeat_rsp->node_rank;
  node_heartbeat_timestamp_[node_rank] = GetCurrentTime();

  free(rsp_packet);
  return Status();
}

Status ControlChannel::ProcessBarrierRequest(NodeInfo* node_info, Packet* req_packet) {
  BarrierRequest* barrier_req = reinterpret_cast<BarrierRequest*>(req_packet->body);

  int clock_idx = barrier_req->clock_idx;
  if (barrier_req_ranks_.find(clock_idx) == barrier_req_ranks_.end()) {
    barrier_req_ranks_.insert(std::make_pair(clock_idx, std::unordered_set<int>()));
  }

  int node_rank = barrier_req->node_rank;
  barrier_req_ranks_[clock_idx].insert(node_rank);

  // For EP-only mode (world_size==1), use expert_world_size
  // For PP mode (world_size>1), use world_size
  size_t expected_node_count = (world_size_ > 1) ? (world_size_ - 1) : (expert_world_size_ - 1);

  // Notify if all nodes arrives.
  if (barrier_req_ranks_[clock_idx].size() == expected_node_count) {
    std::unique_lock<std::mutex> lock(mutex_);
    barrier_cv_.notify_all();
  }

  free(req_packet);
  return Status();
}

Status ControlChannel::ProcessBarrierResponse(NodeInfo* node_info, Packet* rsp_packet) {
  BarrierResponse* barrier_rsp = reinterpret_cast<BarrierResponse*>(rsp_packet->body);

  int clock_idx = barrier_rsp->clock_idx;
  if (barrier_rsp_clocks_.find(clock_idx) == barrier_rsp_clocks_.end()) {
    barrier_rsp_clocks_.insert(clock_idx);
  }

  // Notity thread to continue.
  {
    std::unique_lock<std::mutex> lock(mutex_);
    barrier_cv_.notify_all();
  }

  free(rsp_packet);
  return Status();
}

Status ControlChannel::ProcessLayerRequest(NodeInfo* node_info, Packet* req_packet) {
  AllocateLayerRequest* layer_req = reinterpret_cast<AllocateLayerRequest*>(req_packet->body);

  node_layer_indexes_.resize(4, -1);
  node_layer_indexes_[0] = layer_req->lower_layer_idx;
  node_layer_indexes_[1] = layer_req->upper_layer_idx;
  node_layer_indexes_[2] = layer_req->lower_nextn_layer_idx;
  node_layer_indexes_[3] = layer_req->upper_nextn_layer_idx;

  {
    std::unique_lock<std::mutex> lock(mutex_);
    layer_allocated_ = true;
    layer_allocation_cv_.notify_all();
  }

  // Send response.
  Packet* rsp_packet = GetPacketObject(PacketType::CONTROL_RSP_LAYER, 0);
  if (rsp_packet == nullptr) {
    throw std::runtime_error("ControlChannel::ProcessLayerRequest allocate memory error.");
  }

  Status status = raw_socket_->Send({master_host_, master_port_}, rsp_packet);
  free(rsp_packet);

  if (!status.OK()) {
    KLLM_LOG_ERROR << "ControlChannel process allocate layer reqeust error, send packet failed, info:"
                   << status.GetMessage();
  }

  free(req_packet);
  return status;
}

Status ControlChannel::ProcessLayerResponse(NodeInfo* node_info, Packet* rsp_packet) {
  free(rsp_packet);
  return Status();
}

Status ControlChannel::ProcessBlockNumRequest(NodeInfo* node_info, Packet* req_packet) {
  CacheBlockNumRequest* block_req = reinterpret_cast<CacheBlockNumRequest*>(req_packet->body);

  int node_rank = block_req->node_rank;
  size_t device_block_num = block_req->device_block_num;
  size_t host_block_num = block_req->host_block_num;
  rank_cache_block_num_[node_rank] = std::make_pair(device_block_num, host_block_num);

  if (rank_cache_block_num_.size() == world_size_) {
    std::unique_lock<std::mutex> lock(mutex_);
    block_num_cv_.notify_all();
  }

  free(req_packet);
  return Status();
}

Status ControlChannel::ProcessBlockNumResponse(NodeInfo* node_info, Packet* rsp_packet) {
  CacheBlockNumResponse* block_rsp = reinterpret_cast<CacheBlockNumResponse*>(rsp_packet->body);

  node_block_nums_.resize(2);
  node_block_nums_[0] = block_rsp->device_block_num;
  node_block_nums_[1] = block_rsp->host_block_num;

  {
    std::unique_lock<std::mutex> lock(mutex_);
    block_num_synchronized_ = true;
    block_num_cv_.notify_all();
  }

  free(rsp_packet);
  return Status();
}

Status ControlChannel::ProcessScheduleRequest(NodeInfo* node_info, Packet* req_packet) {
  if (schedule_output_pool_ == nullptr) {
    throw std::runtime_error("ProcessScheduleRequest error, the schedule_output_pool is nullptr.");
  }

  // DO NOT free packet, the following module will do that.
  schedule_output_pool_->PutRecvSerializedPacket(req_packet);

  return Status();
}

Status ControlChannel::ProcessScheduleResponse(NodeInfo* node_info, Packet* rsp_packet) {
  free(rsp_packet);
  return Status();
}

Status ControlChannel::ProcessSamplingOutputRequest(NodeInfo* node_info, Packet* req_packet) {
  if (sampling_output_pool_ == nullptr) {
    throw std::runtime_error("ProcessSamplingOutputRequest error, the sampling_output_pool is nullptr.");
  }

  SamplingOutput* sampling_output = sampling_output_pool_->GetFreeSamplingOutput();
  if (sampling_output == nullptr) {
    return Status(RET_SERVICE_TERMINATED);
  }

  SamplingOutputParser::DeserializeSamplingOutput(reinterpret_cast<void*>(req_packet->body), sampling_output);
  sampling_output_pool_->PutRecvSamplingOutput(sampling_output);

  free(req_packet);
  return Status();
}

Status ControlChannel::ProcessSamplingOutputResponse(NodeInfo* node_info, Packet* rsp_packet) {
  free(rsp_packet);
  return Status();
}

Status ControlChannel::ProcessShutdownRequest(NodeInfo* node_info, Packet* req_packet) {
  // Send response.
  Packet* rsp_packet = GetPacketObject(PacketType::CONTROL_RSP_SHUTDOWN, 0);
  if (rsp_packet == nullptr) {
    throw std::runtime_error("ControlChannel::ProcessShutdownRequest allocate memory error.");
  }

  Status status = raw_socket_->Send({master_host_, master_port_}, rsp_packet);
  free(rsp_packet);

  if (!status.OK()) {
    KLLM_LOG_ERROR << "ControlChannel process shutdown reqeust error, send packet failed, info:" << status.GetMessage();
  }

  GetServiceLifetimeManager()->ShutdownService();

  free(req_packet);
  return status;
}

Status ControlChannel::ProcessShutdownResponse(NodeInfo* node_info, Packet* rsp_packet) {
  std::unique_lock<std::mutex> lock(mutex_);

  auto it = node_ranks_.find(*node_info);
  if (it == node_ranks_.end()) {
    return Status(RET_RUNTIME_FAILED, "Unknown node received.");
  }

  int node_rank = it->second;
  shutdown_nodes_.insert(node_rank);

  if (shutdown_nodes_.size() == world_size_) {
    shutdown_cv_.notify_all();
  }

  free(rsp_packet);
  return Status();
}

Status ControlChannel::Barrier() {
  ++barrier_clock_idx_;

  // The master does not send request to itself.
  if (node_rank_ > 0) {
    Packet* req_packet = GetPacketObject(PacketType::CONTROL_REQ_BARRIER);
    if (req_packet == nullptr) {
      throw std::runtime_error("ControlChannel::Barrier allocate memory error.");
    }

    BarrierRequest* barrier_req = reinterpret_cast<BarrierRequest*>(req_packet->body);
    barrier_req->node_rank = node_rank_;
    barrier_req->clock_idx = barrier_clock_idx_;

    Status status = raw_socket_->Send({master_host_, master_port_}, req_packet);
    free(req_packet);

    if (!status.OK()) {
      KLLM_LOG_ERROR << "ControlChannel barrier error, send packet failed, info:" << status.GetMessage();
      return status;
    }
  }

  if (node_rank_ == 0) {
    // Wait until all nodes
    std::unique_lock<std::mutex> lock(mutex_);

    // For EP-only mode (world_size==1), use expert_world_size
    // For PP mode (world_size>1), use world_size
    size_t expected_node_count = (world_size_ > 1) ? (world_size_ - 1) : (expert_world_size_ - 1);
    barrier_cv_.wait(lock, [this, expected_node_count]() -> bool {
      return (node_ranks_.size() == expected_node_count) &&
             (barrier_req_ranks_[barrier_clock_idx_].size() == expected_node_count);
    });

    // Send response to all nodes.
    for (auto it = node_ranks_.begin(); it != node_ranks_.end(); ++it) {
      NodeInfo node_info = it->first;

      Packet* rsp_packet = GetPacketObject(PacketType::CONTROL_RSP_BARRIER, 0);
      if (rsp_packet == nullptr) {
        throw std::runtime_error("ControlChannel::Barrier allocate memory error.");
      }

      BarrierResponse* barrier_rsp = reinterpret_cast<BarrierResponse*>(rsp_packet->body);
      barrier_rsp->clock_idx = barrier_clock_idx_;

      Status status = raw_socket_->Send(node_info, rsp_packet);
      free(rsp_packet);

      if (!status.OK()) {
        KLLM_LOG_ERROR << "ControlChannel barrier error, send packet failed, info:" << status.GetMessage();
      }
    }

  } else {
    // Wait master response
    std::unique_lock<std::mutex> lock(mutex_);

    barrier_cv_.wait(
        lock, [this]() -> bool { return (barrier_rsp_clocks_.find(barrier_clock_idx_) != barrier_rsp_clocks_.end()); });
  }
  return Status();
}

Status ControlChannel::Frozen() { return raw_socket_->Frozen(); }

Status ControlChannel::AddNode() {
  Packet* req_packet = GetPacketObject(PacketType::CONTROL_REQ_ADD_NODE, 0);
  if (req_packet == nullptr) {
    throw std::runtime_error("ControlChannel::AddNode allocate memory error.");
  }

  AddNodeRequest* add_node_req = reinterpret_cast<AddNodeRequest*>(req_packet->body);
  add_node_req->node_rank = node_rank_;

  KLLM_LOG_INFO << "ControlChannel add node " << node_rank_;
  Status status = raw_socket_->Send({master_host_, master_port_}, req_packet);
  free(req_packet);

  if (!status.OK()) {
    KLLM_LOG_ERROR << "ControlChannel add node error, send packet failed, info:" << status.GetMessage();
  }

  return status;
}

std::vector<AllocateLayerRequest> ControlChannel::GenerateLayerDistribution(
    const size_t num_layer, const size_t num_nextn_predict_layers, const size_t mtp_step_num,
    const size_t sampling_node_offload_layer_num) {
  std::vector<AllocateLayerRequest> layer_distribution(world_size_);

  // Fast path for single node case
  if (world_size_ <= 1) {
    layer_distribution[0].lower_layer_idx = 0;
    layer_distribution[0].upper_layer_idx = num_layer - 1;
    if (num_nextn_predict_layers != 0 && mtp_step_num > 0) {
      layer_distribution[0].lower_nextn_layer_idx = num_layer;
      layer_distribution[0].upper_nextn_layer_idx = num_layer + num_nextn_predict_layers - 1;
    }
    return layer_distribution;
  }

  int quotient = num_layer / world_size_;
  int remainder = num_layer % world_size_;

  // sampling_node is the last node (node_rank == world_size_ - 1)
  // Calculate how many layers the sampling node has
  int sampling_node_layers = quotient;  // reminders are assigned to other workers

  // If offloading layers, assert that we're not trying to offload more layers than the sampling node has (minus 1)
  KLLM_CHECK_WITH_INFO(sampling_node_offload_layer_num < static_cast<size_t>(sampling_node_layers),
                       "Cannot offload more layers than the sampling node has (minus 1)");

  // Adjust sampling node layers with offloading
  sampling_node_layers = sampling_node_layers - static_cast<int>(sampling_node_offload_layer_num);

  // Recompute layer distribution for nodes before sampling node
  const size_t nodes_before_sampling = world_size_ - 1;
  int num_layer_for_others = num_layer - sampling_node_layers;
  int quotient_others = num_layer_for_others / nodes_before_sampling;
  int remainder_others = num_layer_for_others % nodes_before_sampling;

  int current_upper_layer = -1;

  // Distribute layers to nodes before sampling node (node_rank 0 to world_size_ - 2)
  for (size_t node_rank = 0; node_rank < world_size_ - 1; ++node_rank) {
    int remainder_pad = 0;
    if (remainder_others > 0) {
      remainder_pad++;
      remainder_others--;
    }

    const int lower_layer_idx = current_upper_layer + 1;
    const int upper_layer_idx = lower_layer_idx + quotient_others - 1 + remainder_pad;
    current_upper_layer = upper_layer_idx;

    layer_distribution[node_rank].lower_layer_idx = lower_layer_idx;
    layer_distribution[node_rank].upper_layer_idx = upper_layer_idx;
  }

  // Set sampling node's range (last node: node_rank == world_size_ - 1)
  const size_t sampling_node_rank = world_size_ - 1;
  layer_distribution[sampling_node_rank].lower_layer_idx = current_upper_layer + 1;
  layer_distribution[sampling_node_rank].upper_layer_idx = current_upper_layer + sampling_node_layers;

  // only sampling node (last worker node) do nextn predict
  if (num_nextn_predict_layers != 0 && mtp_step_num > 0) {
    layer_distribution[sampling_node_rank].lower_nextn_layer_idx = num_layer;
    layer_distribution[sampling_node_rank].upper_nextn_layer_idx = num_layer + num_nextn_predict_layers - 1;
  }

  return layer_distribution;
}

Status ControlChannel::SynchronizeTcpDataPort(const std::vector<HostPort>& executor_host_ports,
                                              std::vector<HostPort>& executor_downstream_host_ports) {
  if (node_rank_ != 0) {
    // for worker node, send req to master.
    Packet* req_packet = GetPacketObject(PacketType::CONTROL_REQ_TCP_DATA, 0);
    if (req_packet == nullptr) {
      throw std::runtime_error("ControlChannel::SynchronizeTcpDataPort allocate memory error.");
    }

    TcpDataRequest* tcp_req = reinterpret_cast<TcpDataRequest*>(req_packet->body);
    tcp_req->node_rank = node_rank_;
    for (size_t i = 0; i < executor_host_ports.size(); ++i) {
      tcp_req->executor_host_ports[i] = executor_host_ports[i];
    }

    Status status = raw_socket_->Send({master_host_, master_port_}, req_packet);
    free(req_packet);

    if (!status.OK()) {
      KLLM_LOG_ERROR << "ControlChannel sync tcp_port error, send packet failed, info:" << status.GetMessage();
    }

    // wait master's result
    std::unique_lock<std::mutex> lock(mutex_);
    tcp_data_cv_.wait(lock, [this]() -> bool { return tcp_data_allocated_; });
  } else {
    // For master node.
    size_t executor_num = context_->global->GetTensorParallelSize();
    rank_tcp_data_nodes_[0].resize(executor_num);
    rank_tcp_data_nodes_[0] = executor_host_ports;

    // Wait for all workers.
    std::unique_lock<std::mutex> lock(mutex_);
    tcp_data_cv_.wait(lock, [this]() -> bool { return rank_tcp_data_nodes_.size() == world_size_; });

    // the downstream of master is node 1.
    executor_downstream_host_ports_ = rank_tcp_data_nodes_[1];

    // Send to every worker node.
    for (size_t node_rank = 1; node_rank < world_size_; ++node_rank) {
      Packet* req_packet = GetPacketObject(PacketType::CONTROL_RSP_TCP_DATA, 0);
      if (req_packet == nullptr) {
        throw std::runtime_error("ControlChannel::SynchronizeTcpDataPort allocate memory error.");
      }

      TcpDataResponse* tcp_rsp = reinterpret_cast<TcpDataResponse*>(req_packet->body);
      tcp_rsp->node_rank = node_rank;

      size_t downstream_node_rank = (node_rank == world_size_ - 1) ? 0 : (node_rank + 1);
      for (size_t executor_id = 0; executor_id < executor_num; ++executor_id) {
        tcp_rsp->executor_downstream_host_ports[executor_id] = rank_tcp_data_nodes_[downstream_node_rank][executor_id];
      }

      // Broadcast to for all nodes.
      Status status = raw_socket_->Send(rank_nodes_[node_rank], req_packet);
      free(req_packet);

      if (!status.OK()) {
        KLLM_LOG_ERROR << "ControlChannel sync node layers error, send packet failed, info:" << status.GetMessage();
      }
    }
  }

  executor_downstream_host_ports = executor_downstream_host_ports_;
  return Status();
}

#ifdef ENABLE_CUDA
Status ControlChannel::SynchronizeNcclUniqueId(ncclUniqueId& nccl_unique_id) {
  if (node_rank_ != 0) {
    // for worker node, just wait master's result.
    std::unique_lock<std::mutex> lock(mutex_);
    nccl_data_cv_.wait(lock, [this]() -> bool { return nccl_data_allocated_; });
  } else {
    // For master node.
    NCCL_CHECK(ncclGetUniqueId(&nccl_unique_id_));

    // Send to every worker node.
    for (int node_rank = 1; node_rank < world_size_; ++node_rank) {
      Packet* req_packet = GetPacketObject(PacketType::CONTROL_REQ_NCCL_DATA, 0);
      if (req_packet == nullptr) {
        throw std::runtime_error("ControlChannel::SynchronizeNodeLayers allocate memory error.");
      }

      NcclDataRequest* nccl_req = reinterpret_cast<NcclDataRequest*>(req_packet->body);
      nccl_req->node_rank = node_rank;
      memcpy(nccl_req->nccl_unique_id, reinterpret_cast<const char*>(&nccl_unique_id_), sizeof(ncclUniqueId));

      Status status = raw_socket_->Send(rank_nodes_[node_rank], req_packet);
      free(req_packet);

      if (!status.OK()) {
        KLLM_LOG_ERROR << "ControlChannel sync nccl_unique_id_ error, send packet failed, info:" << status.GetMessage();
        return status;
      }
    }
  }

  nccl_unique_id = nccl_unique_id_;
  return Status();
}
#endif

Status ControlChannel::SynchronizeEPCommunicationConfig(const std::vector<EPCommunicationConfig>& executor_configs,
                                                        std::vector<EPCommunicationConfig>& aggregated_configs) {
  // Calculate total data size from all executors
  size_t total_data_size = executor_configs.size() * sizeof(EPCommunicationConfig);
  if (total_data_size > kMaxEPConfigPayloadSize) {
    return Status(RET_INVALID_ARGUMENT, FormatStr("Total data size %zu exceeds maximum allowed size %d",
                                                  total_data_size, kMaxEPConfigPayloadSize));
  }

  if (node_rank_ != 0) {
    // For worker node, send data to master
    Packet* req_packet = GetPacketObject(PacketType::CONTROL_REQ_EP_COMMUNICATION_CONFIG, 0);
    if (req_packet == nullptr) {
      KLLM_THROW("ControlChannel::SynchronizeEPCommunicationConfig allocate memory error.");
    }

    EPCommunicationConfigRequest* ep_config_req = reinterpret_cast<EPCommunicationConfigRequest*>(req_packet->body);
    ep_config_req->node_rank = node_rank_;
    ep_config_req->data_size = total_data_size;

    // Pack all executor data into the request buffer
    memcpy(ep_config_req->ep_comm_configs, executor_configs.data(), total_data_size);

    Status status = raw_socket_->Send({master_host_, master_port_}, req_packet);
    free(req_packet);

    if (!status.OK()) {
      KLLM_LOG_ERROR << "ControlChannel sync EP communicator config error, send packet failed, info:"
                     << status.GetMessage();
      return status;
    }

    // Wait for master's aggregated result
    std::unique_lock<std::mutex> lock(mutex_);
    ep_communication_config_cv_.wait(lock, [this]() -> bool { return ep_communication_config_synchronized_; });
    // Copy aggregated data to output
    aggregated_configs = aggregated_ep_communication_configs_;
  } else {
    // For master node, collect data from this node first
    rank_ep_communication_configs_[0] = executor_configs;

    // Wait for all workers to send their data
    std::unique_lock<std::mutex> lock(mutex_);
    ep_communication_config_cv_.wait(
        lock, [this]() -> bool { return rank_ep_communication_configs_.size() == expert_world_size_; });

    // Aggregate data from all nodes
    aggregated_ep_communication_configs_.clear();
    for (size_t node_rank = 0; node_rank < expert_world_size_; ++node_rank) {
      const auto& node_data = rank_ep_communication_configs_[node_rank];
      aggregated_ep_communication_configs_.insert(aggregated_ep_communication_configs_.end(), node_data.begin(),
                                                  node_data.end());
    }

    // Broadcast aggregated data to all worker nodes
    for (size_t node_rank = 1; node_rank < expert_world_size_; ++node_rank) {
      Packet* rsp_packet = GetPacketObject(PacketType::CONTROL_RSP_EP_COMMUNICATION_CONFIG, 0);
      if (rsp_packet == nullptr) {
        KLLM_THROW("ControlChannel::SynchronizeEPCommunicationConfig allocate memory error.");
      }

      EPCommunicationConfigResponse* ep_config_rsp = reinterpret_cast<EPCommunicationConfigResponse*>(rsp_packet->body);
      ep_config_rsp->node_rank = node_rank;
      ep_config_rsp->data_size = aggregated_ep_communication_configs_.size() * sizeof(EPCommunicationConfig);
      memcpy(ep_config_rsp->aggregated_ep_comm_configs, aggregated_ep_communication_configs_.data(),
             ep_config_rsp->data_size);

      Status status = raw_socket_->Send(rank_nodes_[node_rank], rsp_packet);
      free(rsp_packet);

      if (!status.OK()) {
        KLLM_LOG_ERROR << "ControlChannel sync EP communicator config error, send packet failed, info:"
                       << status.GetMessage();
        return status;
      }
    }

    // Set output for master node
    aggregated_configs = aggregated_ep_communication_configs_;
  }
  return Status();
}

Status ControlChannel::SynchronizeNodeLayers(const std::vector<int16_t>& layer_nums, size_t mtp_step_num,
                                             size_t sampling_node_offload_layer_num,
                                             std::vector<int16_t>& layer_indexes) {
  if (node_rank_ != 0) {
    // for worker node, just wait master's result.
    std::unique_lock<std::mutex> lock(mutex_);
    layer_allocation_cv_.wait(lock, [this]() -> bool { return layer_allocated_; });
    layer_indexes = node_layer_indexes_;
    return Status();
  }

  // For master node.
  if (layer_nums.size() != 2) {
    return Status(RET_INVALID_ARGUMENT, "The layer_nums of ControlChannel::SynchronizeNodeLayers must be 2.");
  }

  // Generate layer distribution for all nodes
  const auto layer_distribution =
      GenerateLayerDistribution(layer_nums[0], layer_nums[1], mtp_step_num, sampling_node_offload_layer_num);

  node_layer_indexes_.assign(4, -1);
  node_layer_indexes_[0] = layer_distribution[0].lower_layer_idx;
  node_layer_indexes_[1] = layer_distribution[0].upper_layer_idx;
  node_layer_indexes_[2] = layer_distribution[0].lower_nextn_layer_idx;
  node_layer_indexes_[3] = layer_distribution[0].upper_nextn_layer_idx;
  KLLM_LOG_INFO << "ControlChannel set master node " << node_rank_ << ", layer range [" << node_layer_indexes_[0]
                << ", " << node_layer_indexes_[1] << "], nextn layer range [" << node_layer_indexes_[2] << ", "
                << node_layer_indexes_[2] << "].";

  // Send to every worker node.
  for (size_t node_rank = 1; node_rank < world_size_; ++node_rank) {
    Packet* const req_packet = GetPacketObject(PacketType::CONTROL_REQ_LAYER, 0);
    if (req_packet == nullptr) {
      throw std::runtime_error("ControlChannel::SynchronizeNodeLayers allocate memory error.");
    }

    AllocateLayerRequest& layer_req = *reinterpret_cast<AllocateLayerRequest*>(req_packet->body);
    layer_req = layer_distribution[node_rank];
    KLLM_LOG_INFO << "ControlChannel set worker node " << node_rank << ", layer range [" << layer_req.lower_layer_idx
                  << ", " << layer_req.upper_layer_idx << "], nextn layer range [" << layer_req.lower_nextn_layer_idx
                  << ", " << layer_req.upper_nextn_layer_idx << "].";
    const Status status = raw_socket_->Send(rank_nodes_[node_rank], req_packet);
    free(req_packet);
    if (!status.OK()) {
      KLLM_LOG_ERROR << "ControlChannel sync node layers error, send packet failed, info:" << status.GetMessage();
    }
  }

  layer_indexes = node_layer_indexes_;
  return Status();
}

Status ControlChannel::SynchronizeCacheBlockNum(const std::vector<size_t>& block_nums,
                                                std::vector<size_t>& node_block_nums) {
  if (block_nums.size() != 2) {
    return Status(RET_INVALID_ARGUMENT, "The block_nums of ControlChannel::SynchronizeCacheBlockNum must be 2.");
  }

  size_t device_block_num = block_nums[0];
  size_t host_block_num = block_nums[1];

  // For master
  if (node_rank_ == 0) {
    rank_cache_block_num_[0] = std::make_pair(device_block_num, host_block_num);

    // Wait for all workers.
    std::unique_lock<std::mutex> lock(mutex_);

    block_num_cv_.wait(lock, [this]() -> bool { return rank_cache_block_num_.size() == world_size_; });

    // Get minimum block num.
    size_t real_device_block_num = std::numeric_limits<size_t>::max();
    size_t real_host_block_num = std::numeric_limits<size_t>::max();
    size_t i = 0;
    std::stringstream ss;
    for (auto pair : rank_cache_block_num_) {
      if (real_device_block_num > pair.second.first) {
        real_device_block_num = pair.second.first;
      }
      if (real_host_block_num > pair.second.second) {
        real_host_block_num = pair.second.second;
      }
      ss << "[" << i << "] device_block_num=" << pair.second.first << ", host_block_num=" << pair.second.second;
      i++;
    }
    KLLM_LOG_INFO << "Reset block nums after sync. real_device_block_num=" << real_device_block_num
                  << ", real_host_block_num=" << real_host_block_num << ", node block infos: " << ss.str();

    node_block_nums_.resize(2);
    node_block_nums_[0] = real_device_block_num;
    node_block_nums_[1] = real_host_block_num;

    // Send response to all nodes.
    for (auto it = node_ranks_.begin(); it != node_ranks_.end(); ++it) {
      NodeInfo node_info = it->first;

      Packet* rsp_packet = GetPacketObject(PacketType::CONTROL_RSP_BLOCKNUM, 0);
      if (rsp_packet == nullptr) {
        throw std::runtime_error("ControlChannel::SynchronizeCacheBlockNum allocate memory error.");
      }

      CacheBlockNumResponse* block_rsp = reinterpret_cast<CacheBlockNumResponse*>(rsp_packet->body);
      block_rsp->device_block_num = real_device_block_num;
      block_rsp->host_block_num = real_host_block_num;

      Status status = raw_socket_->Send(node_info, rsp_packet);
      free(rsp_packet);

      if (!status.OK()) {
        KLLM_LOG_ERROR << "ControlChannel synchronize cache block num error, send packet failed, info:"
                       << status.GetMessage();
        return status;
      }
    }
  } else {
    // For worker
    Packet* req_packet = GetPacketObject(PacketType::CONTROL_REQ_BLOCKNUM);
    if (req_packet == nullptr) {
      throw std::runtime_error("ControlChannel::SynchronizeCacheBlockNum allocate memory error.");
    }

    CacheBlockNumRequest* block_req = reinterpret_cast<CacheBlockNumRequest*>(req_packet->body);
    block_req->device_block_num = device_block_num;
    block_req->host_block_num = host_block_num;
    block_req->node_rank = node_rank_;

    Status status = raw_socket_->Send({master_host_, master_port_}, req_packet);
    free(req_packet);

    if (!status.OK()) {
      KLLM_LOG_ERROR << "ControlChannel synchronize cache block num error, send packet failed, info:"
                     << status.GetMessage();
      return status;
    }

    std::unique_lock<std::mutex> lock(mutex_);
    block_num_cv_.wait(lock, [this]() -> bool { return block_num_synchronized_; });
  }

  if (!node_block_nums_.empty()) {
    node_block_nums = node_block_nums_;
  }
  return Status();
}

Status ControlChannel::InitializeResultReporter() {
  if (context_->global->IsDistributedMaster()) {
    // For master, wait util reporter connected.
    std::unique_lock<std::mutex> lock(mutex_);
    reporter_cv_.wait(lock, [this]() -> bool { return reporter_initialized_; });
  } else if (context_->global->IsDistributedResultReporter() && !context_->global->IsEngine()) {
    // For reporter, connect to master and wait until response arrived.
    Status status = Connect();
    if (!status.OK()) {
      throw std::runtime_error("ControlChannel::InitializeResultReporter connect master error.");
    }

    Packet* req_packet = GetPacketObject(PacketType::CONTROL_REQ_ADD_REPORTER, 0);
    if (req_packet == nullptr) {
      throw std::runtime_error("ControlChannel::InitializeResultReporter allocate memory error.");
    }

    AddReporterRequest* add_reporter_req = reinterpret_cast<AddReporterRequest*>(req_packet->body);
    add_reporter_req->node_rank = context_->global->GetWorldSize();

    KLLM_LOG_INFO << "ControlChannel add reporter " << add_reporter_req->node_rank;
    status = raw_socket_->Send({master_host_, master_port_}, req_packet);
    free(req_packet);

    if (!status.OK()) {
      KLLM_LOG_ERROR << "ControlChannel add reporter error, send packet failed, info:" << status.GetMessage();
    }

    std::unique_lock<std::mutex> lock(mutex_);
    reporter_cv_.wait(lock, [this]() -> bool { return reporter_initialized_; });

    return status;
  }

  // For any other node, return directly.
  return Status();
}

Status ControlChannel::ShutdownCluster() {
  // Only master can call shutdown.
  if (node_rank_ == 0) {
    for (size_t node_rank = 1; node_rank < world_size_; ++node_rank) {
      Packet* req_packet = GetPacketObject(PacketType::CONTROL_REQ_SHUTDOWN, 0);
      if (req_packet == nullptr) {
        throw std::runtime_error("ControlChannel::ShutdownCluster allocate memory error.");
      }

      Status status = raw_socket_->Send(rank_nodes_[node_rank], req_packet);
      free(req_packet);

      if (!status.OK()) {
        KLLM_LOG_ERROR << "ControlChannel shutdown error, send packet failed, info:" << status.GetMessage();
      }
    }

    // Wait for all workers.
    std::unique_lock<std::mutex> lock(mutex_);
    shutdown_nodes_.insert(0);

    // Wait at most 5 seconds.
    size_t timeout = 5;
    shutdown_cv_.wait_for(lock, std::chrono::seconds(timeout),
                          [this]() -> bool { return shutdown_nodes_.size() == world_size_; });

    // Shutdown master node finally.
    GetServiceLifetimeManager()->ShutdownService();
  }

  return Status();
}

Status ControlChannel::HandlePacketFromWorker(NodeInfo* node_info, Packet* packet) {
  switch (packet->type) {
    case PacketType::CONTROL_REQ_ADD_NODE: {
      return ProcessAddNodeRequest(node_info, packet);
    }
    case PacketType::CONTROL_REQ_ADD_REPORTER: {
      return ProcessAddReporterRequest(node_info, packet);
    }
    case PacketType::CONTROL_REQ_TCP_DATA: {
      return ProcessTcpDataRequest(node_info, packet);
    }
#ifdef ENABLE_CUDA
    case PacketType::CONTROL_RSP_NCCL_DATA: {
      return ProcessNcclDataResponse(node_info, packet);
    }
#endif
    case PacketType::CONTROL_REQ_EP_COMMUNICATION_CONFIG: {
      return ProcessEPCommunicationConfigRequest(node_info, packet);
    }
    case PacketType::CONTROL_RSP_EP_COMMUNICATION_CONFIG: {
      return ProcessEPCommunicationConfigResponse(node_info, packet);
    }
    case PacketType::CONTROL_REQ_BARRIER: {
      return ProcessBarrierRequest(node_info, packet);
    }
    case PacketType::CONTROL_RSP_LAYER: {
      return ProcessLayerResponse(node_info, packet);
    }
    case PacketType::CONTROL_REQ_BLOCKNUM: {
      return ProcessBlockNumRequest(node_info, packet);
    }
    case PacketType::CONTROL_RSP_SCHEDULE: {
      return ProcessScheduleResponse(node_info, packet);
    }
    case PacketType::CONTROL_REQ_SAMPLING_OUTPUT: {
      return ProcessSamplingOutputRequest(node_info, packet);
    }
    case PacketType::CONTROL_RSP_SHUTDOWN: {
      return ProcessShutdownResponse(node_info, packet);
    }
    case PacketType::CONTROL_RSP_EXPERT_PARALLEL: {
      return ProcessExpertParallelResponse(node_info, packet);
    }
    default: {
      KLLM_LOG_ERROR << "Not supported packet type:" << packet->type;
      return Status(RET_RUNTIME_FAILED, FormatStr("Not supported packet type %d", packet->type));
    }
  }

  return Status();
}

Status ControlChannel::HandlePacketFromMaster(NodeInfo* node_info, Packet* packet) {
  switch (packet->type) {
    case PacketType::CONTROL_RSP_ADD_NODE: {
      return ProcessAddNodeResponse(node_info, packet);
    }
    case PacketType::CONTROL_RSP_ADD_REPORTER: {
      return ProcessAddReporterResponse(node_info, packet);
    }
    case PacketType::CONTROL_RSP_TCP_DATA: {
      return ProcessTcpDataResponse(node_info, packet);
    }
#ifdef ENABLE_CUDA
    case PacketType::CONTROL_REQ_NCCL_DATA: {
      return ProcessNcclDataRequest(node_info, packet);
    }
#endif
    case PacketType::CONTROL_RSP_EP_COMMUNICATION_CONFIG: {
      return ProcessEPCommunicationConfigResponse(node_info, packet);
    }
    case PacketType::CONTROL_RSP_BARRIER: {
      return ProcessBarrierResponse(node_info, packet);
    }
    case PacketType::CONTROL_REQ_LAYER: {
      return ProcessLayerRequest(node_info, packet);
    }
    case PacketType::CONTROL_RSP_BLOCKNUM: {
      return ProcessBlockNumResponse(node_info, packet);
    }
    case PacketType::CONTROL_REQ_SCHEDULE: {
      return ProcessScheduleRequest(node_info, packet);
    }
    case PacketType::CONTROL_RSP_SAMPLING_OUTPUT: {
      return ProcessSamplingOutputResponse(node_info, packet);
    }
    case PacketType::CONTROL_REQ_SHUTDOWN: {
      return ProcessShutdownRequest(node_info, packet);
    }
    case PacketType::CONTROL_REQ_EXPERT_PARALLEL: {
      return ProcessExpertParallelRequest(node_info, packet);
    }
    default: {
      KLLM_LOG_ERROR << "Not supported packet type:" << packet->type;
      return Status(RET_RUNTIME_FAILED, FormatStr("Not supported packet type %d", packet->type));
    }
  }

  return Status();
}

}  // namespace ksana_llm
