/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "distributed/tcp_data_channel.h"

#include <stdexcept>
#include "fmt/core.h"

#include "configure/environment.h"
#include "distributed/raw_packet.h"
#include "distributed/socket_util.h"
#include "runtime/hidden_unit_buffer.h"
#include "utils/logger.h"
#include "utils/ret_code.h"
#include "utils/singleton.h"
#include "utils/status.h"
#include "utils/string_utils.h"

namespace ksana_llm {

TcpDataChannel::TcpDataChannel(PacketCreationFunc packet_creation_fn, std::shared_ptr<Environment> env) {
  env_ = env;
  hidden_unit_buffer_helper_ = std::make_shared<HiddenUnitBufferHelper>(env_);

  server_raw_socket_ = std::make_shared<RawSocket>(packet_creation_fn);
  client_raw_socket_ = std::make_shared<RawSocket>(packet_creation_fn);

  send_packet_thread_ = std::unique_ptr<std::thread>(new std::thread(&TcpDataChannel::ProcessSendPacketLoop, this));

  constexpr size_t TMP_HOST_BUFFER_SIZE = 2;
  for (size_t i = 0; i < TMP_HOST_BUFFER_SIZE; ++i) {
    free_host_buffer_.Put(hidden_unit_buffer_helper_->CreateHostBuffer());
  }
}

TcpDataChannel::~TcpDataChannel() {
  terminated_ = true;

  if (send_packet_thread_) {
    send_packet_thread_->join();
  }
}

Status TcpDataChannel::Listen(std::string& node_host, uint16_t& node_port) {
  std::string interface;
  Status status = GetAvailableInterfaceAndIP(interface, data_host_);
  if (!status.OK()) {
    throw std::runtime_error(fmt::format("Get data ip error: {}", status.GetMessage()));
  }

  status = GetAvailablePort(data_port_);
  if (!status.OK()) {
    throw std::runtime_error(fmt::format("Get data port error: {}", status.GetMessage()));
  }

  // Listen data port.
  auto listen_fn = [this](NodeInfo* node_info, Packet* packet) -> Status {
    return HandleServerPacket(node_info, packet);
  };

  KLLM_LOG_INFO << "TcpDataChannel Listen on " << data_host_ << ":" << data_port_;
  status = server_raw_socket_->Listen(data_host_, data_port_, listen_fn);
  if (!status.OK()) {
    KLLM_LOG_ERROR << "Listen data channel error:" << status.GetMessage();
  }

  node_host = data_host_;
  node_port = data_port_;

  return status;
}

Status TcpDataChannel::Close() {
  terminated_ = true;
  return server_raw_socket_->Close();
}

Status TcpDataChannel::Connect(const std::string& downstream_host, uint16_t downstream_port) {
  auto connect_fn = [this](NodeInfo* node_info, Packet* packet) -> Status {
    return HandleClientPacket(node_info, packet);
  };

  downstream_host_ = downstream_host;
  downstream_port_ = downstream_port;

  KLLM_LOG_INFO << "TcpDataChannel connect to " << downstream_host << ":" << downstream_port;
  Status status = client_raw_socket_->Connect(downstream_host, downstream_port, connect_fn);
  if (!status.OK()) {
    KLLM_LOG_ERROR << "TcpDataChannel connect error:" << status.GetMessage();
  }

  return status;
}

Status TcpDataChannel::Disconnect() { return client_raw_socket_->Disconnect(); }

Status TcpDataChannel::Frozen() {
  server_raw_socket_->Frozen();
  client_raw_socket_->Frozen();
  return Status();
}

Status TcpDataChannel::ProcessHiddenUnitRequest(NodeInfo* node_info, Packet* req_packet) {
  recv_host_buffer_.Put(req_packet);
  return Status();
}

Status TcpDataChannel::ProcessHiddenUnitResponse(NodeInfo* node_info, Packet* rsp_packet) { return Status(); }

Status TcpDataChannel::HandleServerPacket(NodeInfo* node_info, Packet* packet) {
  switch (packet->type) {
    case PacketType::DATA_REQ_HIDDEN_UNIT: {
      return ProcessHiddenUnitRequest(node_info, packet);
    }
    default: {
      KLLM_LOG_ERROR << "Not supported packet type:" << packet->type;
      return Status(RET_RUNTIME_FAILED, FormatStr("Not supported packet type %d", packet->type));
    }
  }

  return Status();
}

Status TcpDataChannel::HandleClientPacket(NodeInfo* node_info, Packet* packet) {
  switch (packet->type) {
    case PacketType::DATA_RSP_HIDDEN_UNIT: {
      return ProcessHiddenUnitResponse(node_info, packet);
    }
    default: {
      KLLM_LOG_ERROR << "Not supported packet type:" << packet->type;
      return Status(RET_RUNTIME_FAILED, FormatStr("Not supported packet type %d", packet->type));
    }
  }

  return Status();
}

Status TcpDataChannel::ProcessSendPacketLoop() {
  while (!terminated_) {
    Packet* packet = send_host_buffer_.Get();

    Status status = client_raw_socket_->Send({downstream_host_, downstream_port_}, packet);
    if (!status.OK()) {
      KLLM_LOG_ERROR << "TcpDataChannel ProcessSendPacketLoop error, send packet failed:" << status.GetMessage();
    }

    // Resue the packet buffer
    free_host_buffer_.Put(packet);
  }

  return Status();
}

Status TcpDataChannel::SendData(HiddenUnitDeviceBuffer* hidden_unit_dev, const std::vector<size_t>& shape,
                                DataType data_type) {
  if (!hidden_unit_dev) {
    return Status(RET_INVALID_ARGUMENT, "SendData hidden_unit_dev could not be nullptr");
  }

  // Pick a temporarily buffer.
  Packet* packet = free_host_buffer_.Get();
  if (!packet) {
    return Status(RET_RUNTIME_FAILED, "SendData got nullptr from free buffer.");
  }

  // Force reset shape and dtype.
  hidden_unit_dev->tensor.shape = shape;
  hidden_unit_dev->tensor.dtype = data_type;

  HiddenUnitHostBuffer* hidden_unit_host = reinterpret_cast<HiddenUnitHostBuffer*>(packet->body);
  hidden_unit_buffer_helper_->ConvertDeviceBufferToHost(hidden_unit_host, hidden_unit_dev);

  packet->size = hidden_unit_buffer_helper_->GetHostBufferSize(packet);
  send_host_buffer_.Put(packet);

  return Status();
}

Status TcpDataChannel::RecvData(HiddenUnitDeviceBuffer* hidden_unit_dev, const std::vector<size_t>& shape,
                                DataType data_type) {
  if (!hidden_unit_dev) {
    return Status(RET_INVALID_ARGUMENT, "RecvData hidden_unit_dev could not be nullptr.");
  }

  // Just read from recv buffer.
  Packet* packet = recv_host_buffer_.Get();
  if (!packet) {
    return Status(RET_RUNTIME_FAILED, "RecvData got nullptr from recv buffer.");
  }

  HiddenUnitHostBuffer* hidden_unit_host = reinterpret_cast<HiddenUnitHostBuffer*>(packet->body);
  hidden_unit_buffer_helper_->ConvertHostBufferToDevice(hidden_unit_dev, hidden_unit_host);

  // Reset type and shape.
  hidden_unit_dev->tensor.shape = shape;
  hidden_unit_dev->tensor.dtype = data_type;

  return Status();
}

}  // namespace ksana_llm
