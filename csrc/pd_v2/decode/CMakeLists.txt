# Copyright 2026 Tencent Inc.  All rights reserved.
#
# ==============================================================================

# Decode-side business logic for pd_v2:
#   - PeerSelector (snapshot-driven peer routing)
#   - PdV2DecodeHook (slots into EDS ScheduleHookInterface chain at
#     hook_idx=1 after GlobalCacheQueryHook; mediates pd_v2 admission +
#     handles prefill-complete async result + flips infer_stage = kDecode)
#
# WorkloadBalancer intentionally lives in csrc/batch_scheduler/. If
# pd_v2 ever needs peer-aware quota, the right move is to expose a hook
# surface on BasicWorkloadBalancer and inject from here, NOT to subclass
# from this module.
file(GLOB PD_V2_DECODE_SRCS *.cpp)
list(FILTER PD_V2_DECODE_SRCS EXCLUDE REGEX ".*test.cpp")

add_library(pd_v2_decode STATIC ${PD_V2_DECODE_SRCS})
target_include_directories(pd_v2_decode PUBLIC ${PROJECT_SOURCE_DIR}/csrc)
target_link_libraries(pd_v2_decode PUBLIC pd_v2_public pd_v2_wire utils)
# PdV2DecodeHook depends on ScheduleHookInterface (batch_scheduler) and
# InferRequest (runtime). Header-only consumption at compile time via the
# global csrc/ include dir; static link resolves vtable / member-fn symbols.
target_link_libraries(pd_v2_decode PRIVATE batch_scheduler runtime)

if(WITH_STANDALONE_TEST)
    cpp_test(pd_v2_peer_selector_test
        SRCS ${CMAKE_CURRENT_SOURCE_DIR}/peer_selector_test.cpp
        DEPS pd_v2_decode)

    cpp_test(pd_v2_local_load_test
        SRCS ${CMAKE_CURRENT_SOURCE_DIR}/local_load_test.cpp
        DEPS pd_v2_decode batch_scheduler runtime)

    cpp_test(pd_v2_linucb_math_test
        SRCS ${CMAKE_CURRENT_SOURCE_DIR}/linucb_math_test.cpp
        DEPS pd_v2_decode)

    cpp_test(pd_v2_linucb_router_test
        SRCS ${CMAKE_CURRENT_SOURCE_DIR}/linucb_router_test.cpp
        DEPS pd_v2_decode)

    # PdV2DecodeHook unit test: drives Process / OnPrefillComplete /
    # UpdateAsyncResult against a mock PdV2ConnectorInterface + locally
    # constructed InferRequests. Links pd_v2_decode (where the hook lives,
    # so its coverage counters are exercised) + runtime/batch_scheduler for
    # InferRequest / ScheduleContext.
    cpp_test(pd_v2_decode_hook_test
        SRCS ${CMAKE_CURRENT_SOURCE_DIR}/pd_v2_decode_hook_test.cpp
        DEPS pd_v2_decode batch_scheduler runtime)
endif()
