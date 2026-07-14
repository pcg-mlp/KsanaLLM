/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>

#include "distributed/raw_socket.h"
#include "runtime/hidden_unit_buffer.h"
#include "configure/context.h"
#include "configure/environment.h"
#include "utils/status.h"

namespace ksana_llm {

// Used to send & recv data message.
class TcpDataChannel {
 public:
  TcpDataChannel(PacketCreationFunc packet_creation_fn, std::shared_ptr<Environment> env);

  ~TcpDataChannel();

  // Listen and return the tcp host and port.
  Status Listen(std::string& node_host, uint16_t& node_port);

  // Close open port.
  Status Close();

  // For normal node only.
  Status Connect(const std::string& downstream_host, uint16_t downstream_port);

  // disconnect from master.
  Status Disconnect();

  // Stop to accept any new connection.
  Status Frozen();

  // Send & recv data.
  Status SendData(HiddenUnitDeviceBuffer* hidden_unit_dev, const std::vector<size_t>& shape, DataType data_type);
  Status RecvData(HiddenUnitDeviceBuffer* hidden_unit_dev, const std::vector<size_t>& shape, DataType data_type);

 public:
  // Invoked when data arrives.
  Status HandleServerPacket(NodeInfo* node_info, Packet* packet);
  Status HandleClientPacket(NodeInfo* node_info, Packet* packet);

  // Process hidden units.
  Status ProcessHiddenUnitRequest(NodeInfo* node_info, Packet* req_packet);
  Status ProcessHiddenUnitResponse(NodeInfo* node_info, Packet* rsp_packet);

  // Send data to downstream node.
  Status ProcessSendPacketLoop();

  // Copy received packet to device if device buffer is free.
  Status ProcessHostToDeviceLoop();

 private:
  std::shared_ptr<Environment> env_ = nullptr;
  std::shared_ptr<HiddenUnitBufferHelper> hidden_unit_buffer_helper_ = nullptr;

  std::shared_ptr<RawSocket> server_raw_socket_ = nullptr;
  std::shared_ptr<RawSocket> client_raw_socket_ = nullptr;

  // The send and recv host buffer.
  BlockingQueue<Packet*> send_host_buffer_;
  BlockingQueue<Packet*> recv_host_buffer_;

  // The free host buffer, used as intermediate buffer.
  BlockingQueue<Packet*> free_host_buffer_;

  // Send data buffers to remote.
  std::shared_ptr<std::thread> send_packet_thread_ = nullptr;

  // data ip and port.
  std::string data_host_;
  uint16_t data_port_;

  // downstream data ip and port.
  std::string downstream_host_;
  uint16_t downstream_port_;

  // Whether channel is terminated.
  bool terminated_ = false;
};

}  // namespace ksana_llm
