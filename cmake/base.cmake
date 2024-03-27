# Copyright 2023 Tencent Inc.  All rights reserved.

if(NOT PYTHON_EXECUTABLE)
  find_package(PythonInterp)
  message(STATUS "found python executable: ${PYTHON_EXECUTABLE}")
endif()

function(cpp_test TARGET_NAME)
  if(WITH_TESTING)
    set(oneValueArgs "")
    set(multiValueArgs SRCS DEPS ARGS)
    cmake_parse_arguments(cpp_test "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    add_executable(${TARGET_NAME} ${cpp_test_SRCS})
    target_link_libraries(${TARGET_NAME} ${cpp_test_DEPS}
      ksana_llm_libs gtest_main gtest gmock_main gmock -pthread)
    add_dependencies(${TARGET_NAME} ${cpp_test_DEPS} gtest_main gtest gmock_main gmock)
    add_test(NAME ${TARGET_NAME}
      COMMAND ${TARGET_NAME} ${cpp_test_ARGS}
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  endif()
endfunction(cpp_test)

function(py_test TARGET_NAME)
  if(WITH_TESTING)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs SRCS DEPS ARGS ENVS)
    set(ENV{<PYTHONPATH>} ${CMAKE_CURRENT_BINARY_DIR}/lib)
    cmake_parse_arguments(py_test "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(working_dir ${CMAKE_CURRENT_BINARY_DIR})
    file(MAKE_DIRECTORY ${working_dir})
    add_test(NAME ${TARGET_NAME}
      COMMAND ${PYTHON_EXECUTABLE} ${py_test_SRCS} ${py_test_ARGS}
      WORKING_DIRECTORY ${working_dir})
    message(STATUS "test added ${TARGET_NAME} ${working_dir}/${py_test_SRCS}")
  endif()
endfunction()
