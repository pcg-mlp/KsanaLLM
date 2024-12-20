/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <functional>

#include "ksana_llm/distributed/node_info.h"
#include "ksana_llm/distributed/raw_packet.h"
#include "ksana_llm/utils/status.h"

namespace ksana_llm {

// Get raw packet for specific size.
inline Packet* GetRawPacket(size_t body_size) {
  Packet* packet = reinterpret_cast<Packet*>(malloc(sizeof(Packet) + body_size));
  if (packet != nullptr) {
    packet->type = PacketType::PACKET_TYPE_UNKNOWN;
    packet->size = body_size;
  }

  return packet;
}

// Get a packet object from type, set body_size to zero as default.
Packet* GetPacketObject(PacketType packet_type, size_t body_size = 0);

// The function used to create packet while something received.
using PacketCreationFunc = std::function<Packet*(PacketType, size_t)>;

// The function used to process received packet.
using PacketProcessFunc = std::function<Status(NodeInfo*, Packet*)>;

}  // namespace ksana_llm
