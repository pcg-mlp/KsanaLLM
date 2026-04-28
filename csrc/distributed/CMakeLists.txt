# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================
cmake_minimum_required(VERSION 3.8)

file(GLOB_RECURSE distributed_SRCS
  ${PROJECT_SOURCE_DIR}/csrc/distributed/*.cpp)

set(data_channel_nvidia_SRCS "")
set(data_channel_nvidia_LIBS "")

if(WITH_CUDA)
  file(GLOB_RECURSE data_channel_nvidia_SRCS ${PROJECT_SOURCE_DIR}/csrc/distributed/nvidia/*.cpp)
  list(FILTER data_channel_nvidia_SRCS EXCLUDE REGEX ".*test.cpp")
  list(APPEND data_channel_nvidia_LIBS ${NCCL_LIBRARIES} -lcudart -lcublas -lcublasLt)

  # nccl_data_channel_test
  file(GLOB_RECURSE nccl_data_channel_test_SRCS
    ${PROJECT_SOURCE_DIR}/csrc/distributed/nvidia/nccl_data_channel_test.cpp)

  message(STATUS "nccl_data_channel_test_SRCS: ${nccl_data_channel_test_SRCS}")
  # cpp_test(distributed_nccl_data_channel_test SRCS ${nccl_data_channel_test_SRCS} DEPS utils runtime LIBS ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY} ${NCCL_LIBRARIES} -lcudart -lcublas -lcublasLt)
  # 添加 deepep 子目录
  if(WITH_DEEPEP)
    add_subdirectory(expert_parallel/deepep)
  endif()
endif()

# 排除deepep目录的源文件，因为它们在独立的deepep库中编译
list(FILTER distributed_SRCS EXCLUDE REGEX ".*/expert_parallel/deepep/.*")

# 排除所有包含 DeepEP 头文件的源文件，因为它们需要特殊处理以避免符号重复定义
list(FILTER distributed_SRCS EXCLUDE REGEX ".*/expert_parallel/deepep_wrapper\\.cpp$")
list(FILTER distributed_SRCS EXCLUDE REGEX ".*/expert_parallel/deepep_strategy\\.cpp$")
list(FILTER distributed_SRCS EXCLUDE REGEX ".*/expert_parallel/expert_parallel_strategy\\.cpp$")

# 将所有包含 DeepEP 头文件的源文件合并编译以避免符号重复定义
set(deepep_related_SRCS
  ${PROJECT_SOURCE_DIR}/csrc/distributed/expert_parallel/deepep_wrapper.cpp
  ${PROJECT_SOURCE_DIR}/csrc/distributed/expert_parallel/deepep_strategy.cpp
  ${PROJECT_SOURCE_DIR}/csrc/distributed/expert_parallel/expert_parallel_strategy.cpp
)

# 只有在 deepep 可用时才包含 deep_ep.cpp
if(WITH_CUDA AND TARGET deepep)
  list(APPEND deepep_related_SRCS ${PROJECT_SOURCE_DIR}/csrc/distributed/expert_parallel/deepep/deep_ep.cpp)
endif()

if(WITH_ACL)
  # NOTE(karlluo): should not load NCCL in Ascend mode
  list(FILTER distributed_SRCS EXCLUDE REGEX ".*nccl_data_channel.cpp")
endif()

list(FILTER distributed_SRCS EXCLUDE REGEX ".*test.cpp")
message(STATUS "distributed_SRCS: ${distributed_SRCS}")

# 创建一个对象库来编译 deepep 相关的源文件，使用 UNITY_BUILD 避免符号重复定义
add_library(deepep_related_objs OBJECT ${deepep_related_SRCS})
set_target_properties(deepep_related_objs PROPERTIES
  UNITY_BUILD ON
  UNITY_BUILD_BATCH_SIZE 0  # 将所有文件合并到一个编译单元
)

target_include_directories(deepep_related_objs PRIVATE
  ${PROJECT_SOURCE_DIR}/csrc
  ${PROJECT_SOURCE_DIR}/csrc/distributed/expert_parallel/deepep
)

# 如果 deepep 可用，添加相关的包含目录和链接库
if(WITH_CUDA AND TARGET deepep)
  # 从deepep库的属性中获取include目录
  get_target_property(DEEPEP_INCLUDE_DIRS deepep INTERFACE_INCLUDE_DIRECTORIES)
  target_include_directories(deepep_related_objs PRIVATE ${DEEPEP_INCLUDE_DIRS})
  target_link_libraries(deepep_related_objs PRIVATE utils deepep)
  
  # 将对象库添加到 distributed_SRCS
  add_library(distributed STATIC ${distributed_SRCS} ${data_channel_nvidia_SRCS} $<TARGET_OBJECTS:deepep_related_objs>)
  target_link_libraries(distributed PUBLIC utils ${data_channel_nvidia_LIBS} deepep)
  target_include_directories(distributed PRIVATE
    ${PROJECT_SOURCE_DIR}/csrc/distributed/expert_parallel/deepep
    ${DEEPEP_INCLUDE_DIRS}
  )
else()
  target_link_libraries(deepep_related_objs PRIVATE utils)
  add_library(distributed STATIC ${distributed_SRCS} ${data_channel_nvidia_SRCS} $<TARGET_OBJECTS:deepep_related_objs>)
  target_link_libraries(distributed PUBLIC utils ${data_channel_nvidia_LIBS})
endif()

# raw_socket_test
file(GLOB_RECURSE raw_socket_test_SRCS
  ${PROJECT_SOURCE_DIR}/csrc/distributed/raw_socket_test.cpp)

message(STATUS "raw_socket_test_SRCS: ${raw_socket_test_SRCS}")
# cpp_test(distributed_raw_socket_test SRCS ${raw_socket_test_SRCS} DEPS utils runtime LIBS ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})


cpp_test(control_channel_test SRCS ${PROJECT_SOURCE_DIR}/csrc/distributed/control_channel_test.cpp  DEPS utils runtime distributed executor LIBS ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})

if(WITH_CUDA AND TARGET deepep AND WITH_TESTING)
  # DeepEP wrapper 单元测试 - 最小编译单元，不使用 cpp_test 宏以避免引入 ksana_llm_libs 等不必要依赖
  # 注意：不链接 distributed 库（会拉入 GlobalContext、HiddenUnitBufferHelper 等大量传递依赖），
  # 仅链接 deepep_related_objs 对象库（包含 deepep_wrapper.cpp 等必要编译单元）
  add_executable(deepep_wrapper_test
    ${PROJECT_SOURCE_DIR}/csrc/distributed/expert_parallel/deepep_wrapper_test.cpp
  )

  # 仅链接测试所需的最小依赖集
  # 注意：使用 --allow-multiple-definitions 因为 deepep_wrapper_test.cpp 通过
  # deepep_wrapper.h → deep_ep.h → deepep_headers.h 间接包含了 DeepEP 的 config.hpp/event.hpp，
  # 这些头文件中有非 inline 函数定义（get_low_latency_rdma_size_hint, create_event, stream_wait），
  # 与 deepep_related_objs unity build 中的相同符号冲突。两者来自完全相同的头文件，安全合并。
  target_link_libraries(deepep_wrapper_test PRIVATE
    -Wl,--whole-archive
    deepep_related_objs
    utils
    -Wl,--no-whole-archive
    -Wl,-z,muldefs
    deepep
    gtest -pthread
    ${TORCH_LIBRARIES}
    ${NCCL_LIBRARIES}
    -lcudart
  )

  target_include_directories(deepep_wrapper_test PRIVATE
    ${PROJECT_SOURCE_DIR}/csrc
  )

  add_dependencies(deepep_wrapper_test deepep_related_objs utils deepep gtest)

  add_test(NAME deepep_wrapper_test
    COMMAND deepep_wrapper_test
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
endif()

