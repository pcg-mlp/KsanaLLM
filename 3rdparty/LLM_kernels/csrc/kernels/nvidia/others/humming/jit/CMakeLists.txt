# Copyright 2026 Tencent Inc. All rights reserved.
#
# Humming JIT compiler — C++ only (no CUDA), links against CUDA driver API.

set(HUMMING_JIT_SRCS humming_jit_compiler.cpp)

add_library(llm_kernels_nvidia_kernel_humming_jit STATIC ${HUMMING_JIT_SRCS})
set_property(TARGET llm_kernels_nvidia_kernel_humming_jit PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories(llm_kernels_nvidia_kernel_humming_jit PUBLIC
  ${PROJECT_SOURCE_DIR}
)
# nlohmann/json for JSON parsing (replaces hand-written parser).
# The header is fetched by csrc/CMakeLists.txt into ${THIRD_PARTY_PATH}/install/nlohmann_json.
# That include_directories() call is directory-scoped to csrc/ and does not propagate here.
# See fused_moe/CMakeLists.txt for the same pattern.
target_include_directories(llm_kernels_nvidia_kernel_humming_jit PRIVATE
  ${THIRD_PARTY_PATH}/install/nlohmann_json/single_include
)
target_link_libraries(llm_kernels_nvidia_kernel_humming_jit PUBLIC
  -lcuda llm_kernels_nvidia_utils)
