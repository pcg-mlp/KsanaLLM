/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>

#include "distributed/control_message.h"
#include "distributed/packet_util.h"
#include "distributed/tcp_data_channel.h"
#include "runtime/hidden_unit_buffer.h"
#include "configure/context.h"
#include "configure/environment.h"
#include "utils/status.h"

#ifdef ENABLE_CUDA
#  include "distributed/nvidia/nccl_data_channel.h"
#endif

namespace ksana_llm {

// manage all the distributed pipeline nodes.
class DistributedDataCoordinator {
 public:
  DistributedDataCoordinator(PacketCreationFunc packet_creation_fn, std::shared_ptr<Environment> env,
                             std::shared_ptr<Context> context);
  ~DistributedDataCoordinator();

  // Initialize and destroy the distributed cluster.
  void DestroyCluster();

  // Initialize data channel.
  Status InitializeTcpDataChannel();
#ifdef ENABLE_CUDA
  Status InitializeNcclDataChannl();
#endif

  // For tcp data channel.
  Status Listen(std::string& node_host, uint16_t& node_port);
  Status Connect(const std::string& downstream_host, uint16_t downstream_port);

#ifdef ENABLE_CUDA
  // For nccl data channel.
  Status InitializeNcclCluster(ncclUniqueId nccl_unique_id);
#endif

  // Send & recv data.
  Status SendData(HiddenUnitDeviceBuffer* hidden_unit_dev, const std::vector<size_t>& shape, DataType data_type);
  Status RecvData(HiddenUnitDeviceBuffer* hidden_unit_dev, const std::vector<size_t>& shape, DataType data_type);

  // Stop to accept any new connection.
  void Frozen();

 private:
  // The environment & context.
  std::shared_ptr<Environment> env_ = nullptr;
  std::shared_ptr<Context> context_ = nullptr;

  PacketCreationFunc packet_creation_fn_;

  // data communication, for executor process only.
  std::shared_ptr<TcpDataChannel> tcp_data_channel_ = nullptr;
#ifdef ENABLE_CUDA
  std::shared_ptr<NcclDataChannel> nccl_data_channel_ = nullptr;
#endif
};

}  // namespace ksana_llm
