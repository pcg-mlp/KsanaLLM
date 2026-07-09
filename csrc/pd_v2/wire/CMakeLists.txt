# Copyright 2026 Tencent Inc.  All rights reserved.
#
# ==============================================================================

# Wire format: Engine<->Engine PrefillRequest/Complete/Snapshot/StageAck
# binary serialization, plus the Mooncake Notify packed codec.
file(GLOB PD_V2_WIRE_SRCS *.cpp)
list(FILTER PD_V2_WIRE_SRCS EXCLUDE REGEX ".*test.cpp")

add_library(pd_v2_wire STATIC ${PD_V2_WIRE_SRCS})
target_include_directories(pd_v2_wire PUBLIC
    ${PROJECT_SOURCE_DIR}/csrc
)
target_link_libraries(pd_v2_wire PUBLIC pd_v2_public)

if(WITH_STANDALONE_TEST)
    cpp_test(pd_v2_wire_test
        SRCS ${CMAKE_CURRENT_SOURCE_DIR}/prefill_request_wire_test.cpp
        DEPS pd_v2_wire)
endif()
