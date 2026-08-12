# Copyright 2026 Tencent Inc.  All rights reserved.
#
# ==============================================================================

# Prefill-side business logic for pd_v2.
#
# Two libs:
#   pd_v2_layer_event_tracker  — tiny, no runtime deps. Pulled in by
#                                csrc/layers/ so per-layer hook in the
#                                attention layer compiles + links cleanly
#                                even in test binaries that don't have
#                                the heavier intake wrapper.
#   pd_v2_prefill              — KvDispatcher / KvBarrier (intake in endpoints/pd_v2/).
#                                Depends on runtime + batch_scheduler;
#                                only linked by the engine binary.

# Tracker — kept thin so anyone can link it.
add_library(pd_v2_layer_event_tracker STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/pd_v2_layer_event_tracker.cpp)
target_include_directories(pd_v2_layer_event_tracker PUBLIC ${PROJECT_SOURCE_DIR}/csrc)
target_link_libraries(pd_v2_layer_event_tracker PUBLIC utils device)

# Heavier prefill business logic — uses runtime/InferRequest etc. Like
# pd_v2_decode, pulls runtime/infer_request.h transitively (utils/request.h
# pulls pybind11). Engine binary already links Python so this is fine.
file(GLOB PD_V2_PREFILL_SRCS *.cpp)
list(FILTER PD_V2_PREFILL_SRCS EXCLUDE REGEX ".*test.cpp")
list(FILTER PD_V2_PREFILL_SRCS EXCLUDE REGEX ".*pd_v2_layer_event_tracker.cpp")

if(PD_V2_PREFILL_SRCS)
    add_library(pd_v2_prefill STATIC ${PD_V2_PREFILL_SRCS})
    target_include_directories(pd_v2_prefill PUBLIC ${PROJECT_SOURCE_DIR}/csrc)
    target_link_libraries(pd_v2_prefill PUBLIC pd_v2_public pd_v2_wire pd_v2_layer_event_tracker utils)
    target_link_libraries(pd_v2_prefill PRIVATE batch_scheduler runtime)
endif()

# Pure-formula spec test for the design-doc §6 device_id routing rule.
# Doesn't instantiate PdV2KvDispatcher (avoids pulling
# MooncakeExecutorConnector + Mooncake transport into the test binary),
# so it stays a thin lock-in for the matrix policy.
if(WITH_STANDALONE_TEST)
    cpp_test(pd_v2_kv_dispatcher_routing_test
        SRCS ${CMAKE_CURRENT_SOURCE_DIR}/pd_v2_kv_dispatcher_routing_test.cpp
        DEPS pd_v2_public)

    # PdV2KvBarrier unit test. Compiles pd_v2_kv_barrier.cpp directly with a
    # fake PdV2ConnectorInterface and a stubbed IpcFuncWrapper symbol, so it
    # stays isolated from the heavy pd_v2_prefill link (runtime +
    # batch_scheduler). Pure poll-loop logic, no Mooncake transport.
    cpp_test(pd_v2_kv_barrier_test
        SRCS
            ${CMAKE_CURRENT_SOURCE_DIR}/pd_v2_kv_barrier_test.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/pd_v2_kv_barrier.cpp
        DEPS pd_v2_public pd_v2_wire utils)
    if(TARGET pd_v2_kv_barrier_test)
        target_include_directories(pd_v2_kv_barrier_test PRIVATE
            ${PROJECT_SOURCE_DIR}/csrc)
    endif()

    cpp_test(pd_v2_kv_dispatcher_epoch_test
        SRCS
            ${CMAKE_CURRENT_SOURCE_DIR}/pd_v2_kv_dispatcher_epoch_test.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/pd_v2_kv_dispatcher.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/pd_v2_kv_barrier.cpp
        DEPS pd_v2_public pd_v2_wire pd_v2_layer_event_tracker runtime device utils)
    if(TARGET pd_v2_kv_dispatcher_epoch_test)
        target_include_directories(pd_v2_kv_dispatcher_epoch_test PRIVATE
            ${PROJECT_SOURCE_DIR}/csrc)
    endif()

    # PdV2LayerEventTracker unit test. Links the thin tracker lib + device
    # runtime; exercises the per-batch EpochSlot lifecycle (BeginBatch /
    # RecordLayerProgress / ResetState force-fire / monitor callbacks) on a
    # real GPU stream. Requires a visible CUDA device.
    cpp_test(pd_v2_layer_event_tracker_test
        SRCS ${CMAKE_CURRENT_SOURCE_DIR}/pd_v2_layer_event_tracker_test.cpp
        DEPS pd_v2_layer_event_tracker device utils)
    if(TARGET pd_v2_layer_event_tracker_test)
        target_include_directories(pd_v2_layer_event_tracker_test PRIVATE
            ${PROJECT_SOURCE_DIR}/csrc)
    endif()

endif()
