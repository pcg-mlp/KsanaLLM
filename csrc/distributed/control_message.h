/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <stddef.h>

#include <array>
#include <cstdint>
#include <vector>

#include "distributed/packet_type.h"

namespace ksana_llm {

// Maximum number of ranks for DeepEP networking
constexpr int kMaxNumRanks = 256;
constexpr int kNcclUniqueIdSize = 128;
constexpr int kNvshmemUniqueIdSize = 128;
constexpr int kIpcHandlesSize = 64;
constexpr int kMaxExecutorNum = 8;
constexpr int kMaxIpAddressSize = 40;
constexpr int kMaxEPConfigPayloadSize = 65536;  // Maximum payload size for transferring EP communicator configs

// A ip and port.
struct HostPort {
  char host[kMaxIpAddressSize];
  uint16_t port;
};

// Expert Parallel Communicator configuration data
struct EPCommunicationConfig {
  char nvshmem_id[kNvshmemUniqueIdSize];
  char ipc_handles[kIpcHandlesSize];
};

// For barrier
struct BarrierRequest {
  int node_rank;
  int clock_idx;
};

struct BarrierResponse {
  int clock_idx;
};

// for layer allocation, master to worker.
struct AllocateLayerRequest {
  int16_t lower_layer_idx = -1;
  int16_t upper_layer_idx = -1;
  int16_t lower_nextn_layer_idx = -1;
  int16_t upper_nextn_layer_idx = -1;
};

// for layer allocation, master to worker.
struct AllocateExpertRequest {
  char downstream_host[kMaxNumRanks];
  uint16_t downstream_port;

  // Used to broadcast nccl unique_id to all workers.
  // Because ncclUniqueId is platform dependency, so char[] is used.
  char nccl_unique_id[kNcclUniqueIdSize];
  std::vector<std::array<char, kNcclUniqueIdSize> > nccl_unique_ids;
};

// add node, worker to master.
struct AddNodeRequest {
  std::size_t node_rank;
};

// add reporter, worker to master.
struct AddReporterRequest {
  std::size_t node_rank;
};

// for data channel communications.
struct TcpDataRequest {
  size_t node_rank;
  HostPort executor_host_ports[kMaxExecutorNum];
};

struct TcpDataResponse {
  size_t node_rank;
  HostPort executor_downstream_host_ports[kMaxExecutorNum];
};

// for nccl data channel communications.
struct NcclDataRequest {
  size_t node_rank;

  // Used to broadcast nccl unique_id to all workers.
  // Because ncclUniqueId is platform dependency, so char[] is used.
  char nccl_unique_id[kNcclUniqueIdSize];
};

struct NcclDataResponse {
  size_t node_rank;
};

// del node
struct DelNodeRequest {
  size_t node_rank;
};

// make sure cache block num.
struct CacheBlockNumRequest {
  size_t node_rank;
  size_t device_block_num;
  size_t host_block_num;
};

// cache block num.
struct CacheBlockNumResponse {
  size_t device_block_num;
  size_t host_block_num;
};

// heartbeat
struct HeartbeatRequest {
  size_t node_rank;
};

// Same as req
struct HeartbeatResponse {
  size_t node_rank;
};

// for EP communicator config communications.
struct EPCommunicationConfigRequest {
  size_t node_rank;
  size_t data_size;  // Actual size of EP communicator config

  // EP communicator config buffer (can store vector<EPCommunicationConfig> data)
  char ep_comm_configs[kMaxEPConfigPayloadSize];
};

struct EPCommunicationConfigResponse {
  size_t node_rank;
  size_t data_size;  // Actual size of aggregated config from all nodes

  // Aggregated EP communicator config from all nodes
  char aggregated_ep_comm_configs[kMaxEPConfigPayloadSize];
};

}  // namespace ksana_llm
