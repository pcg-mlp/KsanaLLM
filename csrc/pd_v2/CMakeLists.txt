# Copyright 2026 Tencent Inc.  All rights reserved.
#
# ==============================================================================

# pd_v2 module: cross-node Prefill/Decode disaggregation v2.
# Layout:
#   public/    pure header ABI (PdV2ConnectorInterface / Config / Types)
#   configure/ process-internal runtime config (PdV2RuntimeConfig)
#   wire/      Engine<->Engine binary serialization (msgpack)
#   decode/    Decode-side business logic (PeerSelector + PdV2DecodeHook)
#   prefill/   KvDispatcher / KvBarrier (intake via endpoints/pd_v2/)
#   mooncake/  libpd_v2_connector_mooncake.so (real Mooncake tent backend)  [C-impl]

add_subdirectory(public)
add_subdirectory(configure)
add_subdirectory(wire)
add_subdirectory(decode)
add_subdirectory(prefill)
# mooncake/ produces libpd_v2_connector_mooncake.so (the real Mooncake tent
# backend). It is the ONLY pd_v2 piece that depends on Mooncake/rdma-core, and
# it is consumed purely via dlopen of its C ABI — no target links it. So it is
# gated on ENABLE_PD_V2_MOONCAKE: on hosts without it (e.g. glibc < 2.35, see
# csrc/CMakeLists.txt) the connector is skipped while the transport-agnostic
# pd_v2 libraries below still build and link normally.
if(ENABLE_PD_V2_MOONCAKE)
  add_subdirectory(mooncake)
endif()

# Aggregator INTERFACE library; downstream modules link against `pd_v2`
# rather than the individual sub-libraries.
add_library(pd_v2 INTERFACE)
target_link_libraries(pd_v2 INTERFACE pd_v2_public pd_v2_configure pd_v2_wire pd_v2_decode pd_v2_prefill)
