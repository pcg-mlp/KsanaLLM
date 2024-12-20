/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <gtest/gtest.h>

#include <arpa/inet.h>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include "ksana_llm/distributed/packet_util.h"
#include "ksana_llm/distributed/raw_packet.h"
#include "ksana_llm/distributed/raw_socket.h"
#include "ksana_llm/utils/socket_util.h"

#include "ksana_llm/utils/status.h"
#include "test.h"

using namespace ksana_llm;

class RawSocketTest : public testing::Test {
 protected:
  void SetUp() override {
    raw_socket_srv_ = std::make_shared<RawSocket>(GetPacketObject);
    raw_socket_cli_ = std::make_shared<RawSocket>(GetPacketObject);
    raw_socket_cli_2_ = std::make_shared<RawSocket>(GetPacketObject);
  }

  void TearDown() override {}

  // Whether a ip address is avlid.
  bool CheckValidIp(const std::string& ip) {
    struct sockaddr_in sa;
    return inet_pton(AF_INET, ip.c_str(), &(sa.sin_addr)) != 0;
  }

 protected:
  std::shared_ptr<RawSocket> raw_socket_srv_ = nullptr;
  std::shared_ptr<RawSocket> raw_socket_cli_ = nullptr;
  std::shared_ptr<RawSocket> raw_socket_cli_2_ = nullptr;
};

TEST_F(RawSocketTest, TestSocketUtil) {
  std::string interface;
  std::string ip;

  Status status = GetAvailableInterfaceAndIP(interface, ip);
  EXPECT_TRUE(status.OK());
  EXPECT_TRUE(CheckValidIp(ip));

  uint16_t port;
  status = GetAvailablePort(port);
  EXPECT_TRUE(status.OK());
  EXPECT_TRUE(port >= 1 && port <= 65535);
}

TEST_F(RawSocketTest, TestSocke) {
  uint16_t port;
  std::string ip;
  std::string interface;

  GetAvailableInterfaceAndIP(interface, ip);
  GetAvailablePort(port);

  char client_send[8] = "0123456";
  char server_recv[8];
  char server_send[8] = "6543210";
  char client_recv[8];

  // Start server.
  auto listen_fn = [&](NodeInfo* node_info, Packet* packet) -> Status {
    // Save to buffer
    memcpy(server_recv, packet->body, 8);

    // send back.
    Packet* rsp_packet = GetRawPacket(8);
    memcpy(rsp_packet->body, server_send, 8);

    raw_socket_srv_->Send(*node_info, rsp_packet);
    return Status();
  };

  raw_socket_srv_->Listen(ip, port, listen_fn);
  EXPECT_TRUE(raw_socket_srv_->IsConnected());

  // Start client.
  auto connect_fn = [&](NodeInfo* node_info, Packet* packet) -> Status {
    // Save to buffer
    memcpy(client_recv, packet->body, 8);
    return Status();
  };

  raw_socket_cli_->Connect(ip, port, connect_fn);
  EXPECT_TRUE(raw_socket_cli_->IsConnected());

  // Start client 2.
  raw_socket_cli_2_->Connect(ip, port, connect_fn);

  // Close client 2.
  raw_socket_cli_2_->Disconnect();

  // Send to server and waiting response.
  Packet* packet = GetRawPacket(8);
  memcpy(packet->body, client_send, 8);
  raw_socket_cli_->Send({ip, port}, packet);

  // Waiting server & client.
  std::this_thread::sleep_for(std::chrono::seconds(1));

  // Check both client->server and server->client.
  EXPECT_TRUE(strncmp(client_send, server_recv, 8) == 0);
  EXPECT_TRUE(strncmp(server_send, client_recv, 8) == 0);

  // Change packet data, send again, test multiple packets.
  memcpy(client_send, "abcdefg", 8);
  memcpy(server_send, "gfedcba", 8);

  // Send to server and waiting response.
  Packet* packet2 = GetRawPacket(8);
  memcpy(packet2->body, client_send, 8);
  raw_socket_cli_->Send({ip, port}, packet2);

  // Waiting server & client.
  std::this_thread::sleep_for(std::chrono::seconds(1));

  // Verify again.
  EXPECT_TRUE(strncmp(client_send, server_recv, 8) == 0);
  EXPECT_TRUE(strncmp(server_send, client_recv, 8) == 0);

  // Close client & server.
  raw_socket_cli_->Disconnect();
  raw_socket_srv_->Close();
}
