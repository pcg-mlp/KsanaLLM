/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <stddef.h>
#include <cstdint>

#include "ksana_llm/distributed/packet_type.h"

namespace ksana_llm {

// For barrier
struct BarrierRequest {
  int node_rank;
  int clock_idx;
};

struct BarrierResponse {
  int clock_idx;
};

// for layer allocation.
struct AllocateLayerRequest {
  uint16_t lower_layer_idx;
  uint16_t upper_layer_idx;

  char downstream_host[16];
  uint16_t downstream_port;
};

// add node
struct AddNodeRequest {
  std::size_t node_rank;

  char data_host[16];
  uint16_t data_port;
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

}  // namespace ksana_llm
