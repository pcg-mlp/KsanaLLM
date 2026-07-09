/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <chrono>
#include <cstdint>

// Shared internals for the MooncakeEngineConnector implementation, which is
// split across several translation units by role:
//   mooncake_engine_connector.cpp            lifecycle + boot
//   mooncake_engine_connector_decode.cpp     Decode request/completion path
//   mooncake_engine_connector_prefill.cpp    Prefill reception path
//   mooncake_engine_connector_discovery.cpp  poller + heartbeat/keepalive +
//                                            snapshot send/receive
//
// Only symbols referenced by more than one of those units live here (the
// notify channel names and the millisecond clock). Helpers scoped to a
// single unit stay in that unit's anonymous namespace.
namespace ksana_llm {
namespace pd_v2 {
namespace mooncake_backend {

// Notify channel names. The send side stamps them onto Notification::name;
// DispatchOne routes incoming notifications by comparing against them.
inline constexpr const char* kChPrefillReq = "pd_v2/prefill_req";
inline constexpr const char* kChPrefillComplete = "pd_v2/prefill_complete";
inline constexpr const char* kChPrefillSnapshot = "pd_v2/prefill_snapshot";
inline constexpr const char* kChStageAck = "pd_v2/stage_ack";

inline uint64_t NowMs() {
  using clock = std::chrono::system_clock;
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(clock::now().time_since_epoch()).count());
}

}  // namespace mooncake_backend
}  // namespace pd_v2
}  // namespace ksana_llm
