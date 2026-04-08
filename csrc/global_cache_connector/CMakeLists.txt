# Copyright 2025 Tencent Inc. All rights reserved.
#
# ==============================================================================

file(GLOB GLOBAL_CACHE_CONNECTOR_SRCS *.cpp)
list(FILTER GLOBAL_CACHE_CONNECTOR_SRCS EXCLUDE REGEX ".*test.cpp")
list(FILTER GLOBAL_CACHE_CONNECTOR_SRCS EXCLUDE REGEX ".*test_connector.cpp")
list(FILTER GLOBAL_CACHE_CONNECTOR_SRCS EXCLUDE REGEX ".*bench_connector.cpp")

file(GLOB GLOBAL_CACHE_CONNECTOR_HDRS *.h)
list(FILTER GLOBAL_CACHE_CONNECTOR_HDRS EXCLUDE REGEX ".*test.h")

# 引入第三方 mooncake-mock-store（仅提供 mooncake_store 动态库）
add_subdirectory(${PROJECT_SOURCE_DIR}/3rdparty/mooncake-mock-store
                 ${CMAKE_BINARY_DIR}/3rdparty/mooncake-mock-store)

# ---------- SHARED 库 ----------
# 将 connector 编译为独立 .so，不依赖整个 utils 静态库。
# 仅编译 logger.cpp + status.cpp 满足 Status / KLLM_LOG_* 依赖。
add_library(global_cache_connector SHARED
    ${GLOBAL_CACHE_CONNECTOR_SRCS}
    ${PROJECT_SOURCE_DIR}/csrc/utils/logger.cpp
    ${PROJECT_SOURCE_DIR}/csrc/utils/status.cpp
)

target_include_directories(global_cache_connector
    PUBLIC
        ${PROJECT_SOURCE_DIR}/csrc
        ${PROJECT_SOURCE_DIR}/3rdparty/mooncake-mock-store/include
)

# 基础链接库: loguru(日志) + mooncake_store(mock 后端) + CUDA
target_link_libraries(global_cache_connector
    PRIVATE loguru mooncake_store
)

if(WITH_CUDA)
    target_link_libraries(global_cache_connector PRIVATE -lcudart)
endif()

# 真实 Mooncake Store 后端由 Mooncake 团队提供，当前仅保留 stub 接口。
# 如需启用，请参考 Mooncake 团队的集成文档添加相应的依赖和编译选项。

# ---------- 测试 ----------
# 不使用 cpp_test 宏（它会链接整个 ksana_llm_libs），只依赖 connector 自身 + gtest
if(WITH_TESTING)
    file(GLOB GCC_TEST_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*_test.cpp)
    add_executable(global_cache_connector_test ${GCC_TEST_SRCS})
    target_compile_options(global_cache_connector_test PRIVATE -fno-access-control -g -O0)
    target_link_libraries(global_cache_connector_test
        global_cache_connector gtest_main gtest -pthread)
    if(WITH_CUDA)
        target_link_libraries(global_cache_connector_test -lcudart)
    endif()
    add_test(NAME global_cache_connector_test
        COMMAND global_cache_connector_test
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endif()
