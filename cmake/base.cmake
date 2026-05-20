# Copyright 2024 Tencent Inc.  All rights reserved.

if(NOT PYTHON_EXECUTABLE)
  find_package(PythonInterp)
  message(STATUS "found python executable: ${PYTHON_EXECUTABLE}")
endif()

function(cpp_test TARGET_NAME)
  if(WITH_TESTING)
    set(oneValueArgs "")
    set(multiValueArgs SRCS DEPS LIBS ARGS)
    add_compile_options(-fno-access-control -g -O0)
    cmake_parse_arguments(cpp_test "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    add_executable(${TARGET_NAME} ${cpp_test_SRCS})

    # Dynamically link against libtorch_serving.so to avoid embedding ~3GB debug info +
    # 493MB CUDA fatbin into every test binary (reduces 4.2GB → ~24MB per test).
    # torch_serving.so already contains all symbols from ksana_llm_libs, so we only
    # need to link the .so. DEPS are kept in add_dependencies for build ordering.
    target_link_libraries(${TARGET_NAME}
      -Wl,--copy-dt-needed-entries -Wl,--allow-shlib-undefined
      torch_serving gtest_main gtest gmock_main gmock -pthread ${cpp_test_LIBS})

    # ksana_llm_libs and DEPS are for build ordering only (not linking - torch_serving provides all symbols)
    add_dependencies(${TARGET_NAME} ${cpp_test_DEPS} ksana_llm_libs torch_serving gtest_main gtest gmock_main gmock)

    # Propagate INTERFACE_INCLUDE_DIRECTORIES from DEPS so test code can find headers
    # from header-only deps (e.g., cppzmq/zmq.hpp) that were previously propagated via linking.
    foreach(_dep ${cpp_test_DEPS})
      if(TARGET ${_dep})
        target_include_directories(${TARGET_NAME} PRIVATE
          $<TARGET_PROPERTY:${_dep},INTERFACE_INCLUDE_DIRECTORIES>)
      endif()
    endforeach()
    set_target_properties(${TARGET_NAME} PROPERTIES
      BUILD_RPATH "${CMAKE_BINARY_DIR}/lib"
      INSTALL_RPATH "$ORIGIN/../lib")
    add_test(NAME ${TARGET_NAME}
      COMMAND ${TARGET_NAME} ${cpp_test_ARGS}
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  endif()
endfunction(cpp_test)

function(python_test TARGET_NAME)
  if(WITH_TESTING)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs SRCS DEPS ARGS ENVS)
    set(ENV{<PYTHONPATH>} ${CMAKE_CURRENT_BINARY_DIR}/lib)
    cmake_parse_arguments(python_test "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(working_dir ${CMAKE_CURRENT_BINARY_DIR})
    file(MAKE_DIRECTORY ${working_dir})
    add_test(NAME ${TARGET_NAME}
      COMMAND ${PYTHON_EXECUTABLE} ${python_test_SRCS} ${python_test_ARGS}
      WORKING_DIRECTORY ${working_dir})
    set_tests_properties(${TARGET_NAME} PROPERTIES PASS_REGULAR_EXPRESSION ".*test PASS")
    message(STATUS "test added ${TARGET_NAME} ${working_dir}/${python_test_SRCS}")
  endif()
endfunction()
