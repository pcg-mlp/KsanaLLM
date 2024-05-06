# Copyright 2024 Tencent Inc.  All rights reserved.

set(llm_kernels_ascend_libs "")
set(llm_kernels_ascend_link_libs ${ACL_SHARED_LIBS} llm_kernels_ascend_utils_common)
set(llm_kernels_ascend_libs_deps llm_kernels_ascend_utils_common)

function(add_ascend_kernel_lib)
  set(options "")
  set(oneValueArgs NAME)
  set(multiValueArgs SRCS DEPS_KERNELS ARGS ENVS)
  cmake_parse_arguments(add_ascend_kernel_lib "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  file(GLOB_RECURSE ACL_SRCS *.cc)
  list(FILTER ACL_SRCS EXCLUDE REGEX ".*test.cc")
  list(FILTER ACL_SRCS EXCLUDE REGEX ".*kernel.cc")

  file(GLOB_RECURSE KERNEL_SRCS *kernel.cc)
  set_source_files_properties(${KERNEL_SRCS} PROPERTIES LANGUAGE CCE)

  set(kernel_lib_name llm_kernels_ascend_${add_ascend_kernel_lib_NAME})
  add_library(${kernel_lib_name} STATIC ${ACL_SRCS})
  set_property(TARGET ${kernel_lib_name} PROPERTY POSITION_INDEPENDENT_CODE ON)

  if(KERNEL_SRCS AND NOT KERNEL_SRCS STREQUAL "")
    set(aicore_kernel_lib_name llm_kernels_ascend_core_${add_ascend_kernel_lib_NAME})
    ascendc_library(${aicore_kernel_lib_name} STATIC ${KERNEL_SRCS})
  endif()

  if(NOT "${add_ascend_kernel_lib_DEPS_KERNELS}" STREQUAL "")
    foreach(var ${add_ascend_kernel_lib_DEPS_KERNELS})
      list(APPEND llm_kernels_ascend_libs_deps llm_kernels_ascend_${var})
    endforeach()
  endif()

  set(llm_kernels_ascend_link_libs ${llm_kernels_ascend_link_libs})
  set(llm_kernels_ascend_libs_deps ${llm_kernels_ascend_libs_deps})

  target_link_libraries(${kernel_lib_name} PUBLIC ${llm_kernels_ascend_libs_deps} ${llm_kernels_ascend_link_libs})
  add_dependencies(${kernel_lib_name} ${llm_kernels_ascend_libs_deps})
  set(llm_kernels_ascend_libs "${llm_kernels_ascend_libs} ${kernel_lib_name}" PARENT_SCOPE)
  message(STATUS "Ascend kernel added ${add_ascend_kernel_lib_NAME}")

  # for test
  if(WITH_TESTING)
    file(GLOB_RECURSE KERNEL_TEST_SRCS *test.cc)

    if(KERNEL_TEST_SRCS AND NOT KERNEL_TEST_SRCS STREQUAL "")
      message(STATUS "${kernel_lib_name}_test src: ${KERNEL_TEST_SRCS}")
      cc_test(${kernel_lib_name}_test SRCS ${KERNEL_TEST_SRCS} DEPS ${kernel_lib_name} ${aicore_kernel_lib_name})
    endif()
  endif()
endfunction()

add_subdirectory(activation)
add_subdirectory(argmax)
add_subdirectory(attention)
add_subdirectory(cat)
add_subdirectory(elementwise)
add_subdirectory(embedding)
add_subdirectory(gather)
add_subdirectory(layernorm)
add_subdirectory(matmul)
add_subdirectory(permute)
add_subdirectory(pointwise)
add_subdirectory(reshape)
add_subdirectory(rotary_embedding)
add_subdirectory(slice)
add_subdirectory(transpose)
add_subdirectory(matmul_silu)

string(STRIP "${llm_kernels_ascend_libs}" llm_kernels_ascend_libs)
string(REGEX REPLACE " " ";" llm_kernels_ascend_libs "${llm_kernels_ascend_libs}")
message(STATUS "Ascend kernels: ${llm_kernels_ascend_libs}")
set(llm_kernels_ascend_libs ${llm_kernels_ascend_libs} PARENT_SCOPE)
