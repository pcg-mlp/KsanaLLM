/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cstddef>
#include <cstdint>

#include "ksana_llm/distributed/packet_type.h"

namespace ksana_llm {

constexpr int PACKET_MAGIC_NUMBER = 0x2ea9;

// The head information of socket packet.
struct PacketMeta {
  uint64_t timestamp = 0;
};

// Socket Packet format:
//    magic number
//    packet size
//    packet body

// The packet used to send or recv data.
struct Packet {
  // The packet type, control message or data message.
  // Make it as the first field so we can read it at begining.
  PacketType type;

  // Meta information for network transform.
  PacketMeta meta;

  // The bytes of body part.
  size_t size = 0;

  // The really packet struct.
  char body[0];
};

}  // namespace ksana_llm
