# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================
cmake_minimum_required(VERSION 3.13)

file(GLOB_RECURSE runtime_SRCS
  ${PROJECT_SOURCE_DIR}/csrc/runtime/*.cpp)
list(FILTER runtime_SRCS EXCLUDE REGEX ".*test.cpp")
message(STATUS "runtime_SRCS: ${runtime_SRCS}")

add_library(runtime STATIC ${runtime_SRCS})
add_dependencies(runtime utils fmt gflags)
target_link_libraries(runtime PUBLIC
  -lpthread -ldl gflags ${runtime_nvidia_LIBS} ${runtime_ascend_LIBS}
  model_loader models utils transfer
)

# for test
file(GLOB_RECURSE runtime_test_SRCS
  ${PROJECT_SOURCE_DIR}/csrc/runtime/threadpool_test.cpp
  ${PROJECT_SOURCE_DIR}/csrc/runtime/llm_runtime_test.cpp)
if(WITH_CUDA AND DEFINED SM AND "${SM}" STREQUAL "90a")
  list(APPEND runtime_test_SRCS
  ${PROJECT_SOURCE_DIR}/csrc/runtime/weight_instance_test.cpp)
endif()
message(STATUS "runtime_test_SRCS: ${runtime_test_SRCS}")
#cpp_test(runtime_test SRCS ${runtime_test_SRCS} DEPS runtime)

file(GLOB_RECURSE draft_generator_test_SRCS
  ${PROJECT_SOURCE_DIR}/csrc/runtime/draft_generator/*_test.cpp)
cpp_test(draft_generator_test SRCS ${draft_generator_test_SRCS})

if (WITH_CUDA)
  cpp_test(xgrammar_generator_test SRCS ${PROJECT_SOURCE_DIR}/csrc/runtime/structured_generation/xgrammar/xgrammar_structured_generator_test.cpp DEPS ${MODELS_TEST_DEPS} LIBS ${MODELS_TEST_LIBS})
endif()

cpp_test(generation_controller_test SRCS ${PROJECT_SOURCE_DIR}/csrc/runtime/generation_controller_test.cpp DEPS ${MODELS_TEST_DEPS} ipc LIBS ${MODELS_TEST_LIBS})

cpp_test(schedule_output_test SRCS ${PROJECT_SOURCE_DIR}/csrc/runtime/schedule_output_test.cpp DEPS ${MODELS_TEST_DEPS} LIBS ${MODELS_TEST_LIBS})

cpp_test(sampling_output_test SRCS ${PROJECT_SOURCE_DIR}/csrc/runtime/sampling_output_test.cpp DEPS ${MODELS_TEST_DEPS} LIBS ${MODELS_TEST_LIBS})

cpp_test(infer_request_test SRCS ${PROJECT_SOURCE_DIR}/csrc/runtime/infer_request_test.cpp DEPS ${MODELS_TEST_DEPS} ipc LIBS ${MODELS_TEST_LIBS})

# Fast-path controller 启动 check 单测。
# 虽然 controller 自身只依赖配置层 + ScheduleOutput 头, 但 runtime 库引入了 executor/layer_progress_tracker
# 等组件的 transitive 依赖 (ipc), 因此 test 链接时需要补齐, 与 generation_controller_test 同模式。
cpp_test(fast_path_controller_test
  SRCS ${PROJECT_SOURCE_DIR}/csrc/runtime/fast_path_controller_test.cpp
  DEPS ${MODELS_TEST_DEPS} ipc LIBS ${MODELS_TEST_LIBS})
