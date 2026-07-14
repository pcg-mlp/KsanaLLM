/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "distributed/packet_util.h"

#ifdef ENABLE_CUDA
#  include <nccl.h>
#endif

#include <cstdlib>

#include "configure/environment.h"
#include "distributed/control_message.h"
#include "runtime/hidden_unit_buffer.h"
#include "utils/singleton.h"

namespace ksana_llm {

Packet* GetPacketObject(PacketType packet_type, size_t body_size) {
  Packet* packet = nullptr;
  switch (packet_type) {
    case PacketType::DATA_REQ_HIDDEN_UNIT: {
      packet = HiddenUnitBufferHelper::CreateHostBufferPacket();
      if (packet != nullptr) {
        packet->size = HiddenUnitBufferHelper::GetHostBufferPacketSize(packet);
      }
      break;
    }
    case PacketType::PACKET_TYPE_UNKNOWN:
    case PacketType::CONTROL_REQ_SCHEDULE:
    case PacketType::CONTROL_REQ_SAMPLING_OUTPUT: {
      packet = GetRawPacket(body_size);
      if (packet != nullptr) {
        packet->size = body_size;
      }
      break;
    }
    case PacketType::CONTROL_REQ_BARRIER: {
      packet = reinterpret_cast<Packet*>(malloc(sizeof(Packet) + sizeof(BarrierRequest)));
      if (packet != nullptr) {
        packet->size = sizeof(BarrierRequest);
      }
      break;
    }
    case PacketType::CONTROL_RSP_BARRIER: {
      packet = reinterpret_cast<Packet*>(malloc(sizeof(Packet) + sizeof(BarrierResponse)));
      if (packet != nullptr) {
        packet->size = sizeof(BarrierResponse);
      }
      break;
    }
    case PacketType::CONTROL_REQ_BLOCKNUM: {
      packet = reinterpret_cast<Packet*>(malloc(sizeof(Packet) + sizeof(CacheBlockNumRequest)));
      if (packet != nullptr) {
        packet->size = sizeof(CacheBlockNumRequest);
      }
      break;
    }
    case PacketType::CONTROL_RSP_BLOCKNUM: {
      packet = reinterpret_cast<Packet*>(malloc(sizeof(Packet) + sizeof(CacheBlockNumResponse)));
      if (packet != nullptr) {
        packet->size = sizeof(CacheBlockNumResponse);
      }
      break;
    }
    case PacketType::CONTROL_REQ_LAYER: {
      packet = reinterpret_cast<Packet*>(malloc(sizeof(Packet) + sizeof(AllocateLayerRequest)));
      if (packet != nullptr) {
        packet->size = sizeof(AllocateLayerRequest);
      }
      break;
    }
    case PacketType::CONTROL_REQ_ADD_NODE: {
      packet = reinterpret_cast<Packet*>(malloc(sizeof(Packet) + sizeof(AddNodeRequest)));
      if (packet != nullptr) {
        packet->size = sizeof(AddNodeRequest);
      }
      break;
    }
    case PacketType::CONTROL_REQ_ADD_REPORTER: {
      packet = reinterpret_cast<Packet*>(malloc(sizeof(Packet) + sizeof(AddReporterRequest)));
      if (packet != nullptr) {
        packet->size = sizeof(AddReporterRequest);
      }
      break;
    }
    case PacketType::CONTROL_REQ_TCP_DATA: {
      packet = reinterpret_cast<Packet*>(malloc(sizeof(Packet) + sizeof(TcpDataRequest)));
      if (packet != nullptr) {
        packet->size = sizeof(TcpDataRequest);
      }
      break;
    }
    case PacketType::CONTROL_RSP_TCP_DATA: {
      packet = reinterpret_cast<Packet*>(malloc(sizeof(Packet) + sizeof(TcpDataResponse)));
      if (packet != nullptr) {
        packet->size = sizeof(TcpDataResponse);
      }
      break;
    }
    case PacketType::CONTROL_REQ_NCCL_DATA: {
      packet = reinterpret_cast<Packet*>(malloc(sizeof(Packet) + sizeof(NcclDataRequest)));
      if (packet != nullptr) {
        packet->size = sizeof(NcclDataRequest);
      }
      break;
    }
    case PacketType::CONTROL_RSP_NCCL_DATA: {
      packet = reinterpret_cast<Packet*>(malloc(sizeof(Packet) + sizeof(NcclDataResponse)));
      if (packet != nullptr) {
        packet->size = sizeof(NcclDataResponse);
      }
      break;
    }
    case PacketType::CONTROL_REQ_EP_COMMUNICATION_CONFIG: {
      packet = reinterpret_cast<Packet*>(malloc(sizeof(Packet) + sizeof(EPCommunicationConfigRequest)));
      if (packet != nullptr) {
        packet->size = sizeof(EPCommunicationConfigRequest);
      }
      break;
    }
    case PacketType::CONTROL_RSP_EP_COMMUNICATION_CONFIG: {
      packet = reinterpret_cast<Packet*>(malloc(sizeof(Packet) + sizeof(EPCommunicationConfigResponse)));
      if (packet != nullptr) {
        packet->size = sizeof(EPCommunicationConfigResponse);
      }
      break;
    }
    case PacketType::CONTROL_REQ_HEARTBEAT: {
      packet = reinterpret_cast<Packet*>(malloc(sizeof(Packet) + sizeof(HeartbeatRequest)));
      if (packet != nullptr) {
        packet->size = sizeof(HeartbeatRequest);
      }
      break;
    }
    case PacketType::CONTROL_RSP_HEARTBEAT: {
      packet = reinterpret_cast<Packet*>(malloc(sizeof(Packet) + sizeof(HeartbeatResponse)));
      if (packet != nullptr) {
        packet->size = sizeof(HeartbeatResponse);
      }
      break;
    }
    case PacketType::DATA_RSP_HIDDEN_UNIT:
    case PacketType::CONTROL_RSP_SCHEDULE:
    case PacketType::CONTROL_RSP_SAMPLING_OUTPUT:
    case PacketType::CONTROL_RSP_LAYER:
    case PacketType::CONTROL_RSP_ADD_NODE:
    case PacketType::CONTROL_RSP_ADD_REPORTER:
    case PacketType::CONTROL_REQ_SHUTDOWN:
    case PacketType::CONTROL_RSP_SHUTDOWN: {
      packet = reinterpret_cast<Packet*>(malloc(sizeof(Packet)));
      if (packet != nullptr) {
        packet->size = 0;
      }
      break;
    }
    default:
      return nullptr;
  }

  if (packet != nullptr) {
    packet->type = packet_type;
  }
  return packet;
}

}  // namespace ksana_llm
