/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "distributed/distributed_data_coordinator.h"

#include <stdexcept>

#include "device/device_utils.h"
#include "fmt/core.h"
#include "utils/logger.h"
#include "utils/singleton.h"

namespace ksana_llm {

DistributedDataCoordinator::DistributedDataCoordinator(PacketCreationFunc packet_creation_fn,
                                                       std::shared_ptr<Environment> env,
                                                       std::shared_ptr<Context> context) {
  env_ = env;
  context_ = context;
  packet_creation_fn_ = packet_creation_fn;
}

DistributedDataCoordinator::~DistributedDataCoordinator() { tcp_data_channel_.reset(); }

Status DistributedDataCoordinator::InitializeTcpDataChannel() {
  tcp_data_channel_ = std::make_shared<TcpDataChannel>(packet_creation_fn_, env_);
  return Status();
}

#ifdef ENABLE_CUDA
Status DistributedDataCoordinator::InitializeNcclDataChannl() {
  nccl_data_channel_ = std::make_shared<NcclDataChannel>(context_);
  return Status();
}
#endif

Status DistributedDataCoordinator::Listen(std::string& node_host, uint16_t& node_port) {
  return tcp_data_channel_->Listen(node_host, node_port);
}

Status DistributedDataCoordinator::Connect(const std::string& downstream_host, uint16_t downstream_port) {
  return tcp_data_channel_->Connect(downstream_host, downstream_port);
}

#ifdef ENABLE_CUDA
Status DistributedDataCoordinator::InitializeNcclCluster(ncclUniqueId nccl_unique_id) {
  return nccl_data_channel_->InitializeNcclCluster(nccl_unique_id);
}
#endif

Status DistributedDataCoordinator::SendData(HiddenUnitDeviceBuffer* hidden_unit_dev, const std::vector<size_t>& shape,
                                            DataType data_type) {
  if (tcp_data_channel_) {
    return tcp_data_channel_->SendData(hidden_unit_dev, shape, data_type);
#ifdef ENABLE_CUDA
  } else if (nccl_data_channel_) {
    return nccl_data_channel_->SendData(hidden_unit_dev, shape, data_type);
#endif
  }
  return Status();
}

Status DistributedDataCoordinator::RecvData(HiddenUnitDeviceBuffer* hidden_unit_dev, const std::vector<size_t>& shape,
                                            DataType data_type) {
  if (tcp_data_channel_) {
    return tcp_data_channel_->RecvData(hidden_unit_dev, shape, data_type);
#ifdef ENABLE_CUDA
  } else if (nccl_data_channel_) {
    return nccl_data_channel_->RecvData(hidden_unit_dev, shape, data_type);
#endif
  }
  return Status();
}

void DistributedDataCoordinator::Frozen() {
  if (tcp_data_channel_) {
    tcp_data_channel_->Frozen();
  }
}

void DistributedDataCoordinator::DestroyCluster() {
  if (tcp_data_channel_) {
    tcp_data_channel_->Disconnect();
    tcp_data_channel_->Close();
  }

#ifdef ENABLE_CUDA
  if (nccl_data_channel_) {
    nccl_data_channel_->DestroyNcclCluster();
  }
#endif
}

}  // namespace ksana_llm
