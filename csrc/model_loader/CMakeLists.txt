# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================
cmake_minimum_required(VERSION 3.8)

file(GLOB_RECURSE model_loader_SRCS
     ${PROJECT_SOURCE_DIR}/csrc/model_loader/*.cpp
     ${PROJECT_SOURCE_DIR}/csrc/model_loader/config_parser/*.cpp
     ${PROJECT_SOURCE_DIR}/csrc/model_loader/file_loader/*.cpp
     ${PROJECT_SOURCE_DIR}/csrc/model_loader/model_loader/*.cpp
     ${PROJECT_SOURCE_DIR}/csrc/model_loader/weight_loader/*.cpp)

list(FILTER model_loader_SRCS EXCLUDE REGEX ".*test.cpp")
message(STATUS "model_loader_SRCS: ${model_loader_SRCS}")

add_library(model_loader STATIC ${model_loader_SRCS})
target_link_libraries(model_loader PUBLIC utils models)

if(NOT WITH_LIGHTLY_CI_TEST)
  if(WITH_CUDA)
    # model_loader_test
    file(GLOB_RECURSE model_loader_test_SRCS ${PROJECT_SOURCE_DIR}/csrc/model_loader/model_loader_test.cpp)
    message(STATUS "model_loader_test_SRCS: ${model_loader_test_SRCS}")
    cpp_test(model_loader_test SRCS ${model_loader_test_SRCS} DEPS utils models LIBS ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})

    # llama_model_loader_test
    file(GLOB_RECURSE llama_model_loader_test_SRCS ${PROJECT_SOURCE_DIR}/csrc/model_loader/llama_model_loader_test.cpp)
    message(STATUS "llama_model_loader_test_SRCS: ${llama_model_loader_test_SRCS}")
    cpp_test(llama_model_loader_test SRCS ${llama_model_loader_test_SRCS} DEPS utils models LIBS ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})

    # deepseek_v3_model_loader_test
    file(GLOB_RECURSE deepseek_v3_model_loader_test_SRCS ${PROJECT_SOURCE_DIR}/csrc/model_loader/deepseek_v3_model_loader_test.cpp)
    message(STATUS "deepseek_v3_model_loader_test_SRCS: ${deepseek_v3_model_loader_test_SRCS}")
    cpp_test(deepseek_v3_model_loader_test SRCS ${deepseek_v3_model_loader_test_SRCS} DEPS utils models LIBS ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})
  endif()

  if(WITH_KUNLUN)
    # kunlun_kimi_model_loader_test: Kimi K2.6 MLA W4A8 on Kunlun XPU platform
    file(GLOB_RECURSE kunlun_kimi_model_loader_test_SRCS ${PROJECT_SOURCE_DIR}/csrc/model_loader/kunlun_kimi_model_loader_test.cpp)
    message(STATUS "kunlun_kimi_model_loader_test_SRCS: ${kunlun_kimi_model_loader_test_SRCS}")
    cpp_test(kunlun_kimi_model_loader_test SRCS ${kunlun_kimi_model_loader_test_SRCS} DEPS utils models LIBS ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})
  endif()
endif()
