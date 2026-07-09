/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "pd_v2/wire/prefill_request_wire.h"

#include <cstdint>

namespace ksana_llm {
namespace pd_v2 {

namespace {

inline void AppendU32LE(std::vector<uint8_t>* out, uint32_t v) {
  for (int i = 0; i < 4; ++i) {
    out->push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xFFu));
  }
}

inline uint32_t ReadU32LE(const uint8_t* in) {
  return static_cast<uint32_t>(in[0]) | (static_cast<uint32_t>(in[1]) << 8) | (static_cast<uint32_t>(in[2]) << 16) |
         (static_cast<uint32_t>(in[3]) << 24);
}

}  // namespace

// Wire envelope: [u32 msg_type LE][u32 body_len LE][msgpack body...].
// The msgpack body is self-describing (length is implicit in the
// encoding) but we keep an explicit body_len so a stream-mode receiver
// can know how many bytes to buffer before handing to the msgpack parser.
void WrapFrame(PdV2WireMsgType type, const std::vector<uint8_t>& body, std::vector<uint8_t>* out) {
  AppendU32LE(out, static_cast<uint32_t>(type));
  AppendU32LE(out, static_cast<uint32_t>(body.size()));
  out->insert(out->end(), body.begin(), body.end());
}

bool PeekFrameHeader(const uint8_t* in, size_t len, PdV2WireMsgType* type, uint32_t* body_len) {
  if (in == nullptr || type == nullptr || body_len == nullptr) return false;
  if (len < kPdV2WireHeaderSize) return false;
  *type = static_cast<PdV2WireMsgType>(ReadU32LE(in));
  *body_len = ReadU32LE(in + 4);
  return true;
}

}  // namespace pd_v2
}  // namespace ksana_llm
