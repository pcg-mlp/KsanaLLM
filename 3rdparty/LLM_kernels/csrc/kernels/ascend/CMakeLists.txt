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

  set(aicore_kernel_lib_name "")

  if(KERNEL_SRCS AND NOT KERNEL_SRCS STREQUAL "")
    set(aicore_kernel_lib_name llm_kernels_ascend_core_${add_ascend_kernel_lib_NAME})
    ascendc_library(${aicore_kernel_lib_name} STATIC ${KERNEL_SRCS})
    ascendc_include_directories(${aicore_kernel_lib_name} PRIVATE
      ${PROJECT_SOURCE_DIR} ${ACL_INC_DIRS}
    )
    ascendc_compile_definitions(${aicore_kernel_lib_name} PRIVATE
      $<$<BOOL:$<IN_LIST:${SOC_VERSION},${CUSTOM_ASCEND310P_LIST}>>:CUSTOM_ASCEND310P>
      # NOTE(karlluo): if you want to printf/dump tensor from kernel please uncomment this line
      # -DASCENDC_DUMP
      # -DHAVE_WORKSPACE
    )
  endif()

  if(NOT "${add_ascend_kernel_lib_DEPS_KERNELS}" STREQUAL "")
    foreach(var ${add_ascend_kernel_lib_DEPS_KERNELS})
      list(APPEND llm_kernels_ascend_libs_deps llm_kernels_ascend_${var})
    endforeach()
  endif()

  set(llm_kernels_ascend_link_libs ${llm_kernels_ascend_link_libs})
  set(llm_kernels_ascend_libs_deps ${llm_kernels_ascend_libs_deps})

  target_link_libraries(${kernel_lib_name} PRIVATE
    ${llm_kernels_ascend_libs_deps} ${llm_kernels_ascend_link_libs} ${aicore_kernel_lib_name})

  target_link_libraries(${kernel_lib_name} PUBLIC
    /usr/local/Ascend/ascend-toolkit/latest/lib64/libnnopbase.so
    /usr/local/Ascend/ascend-toolkit/latest/lib64/libprofapi.so
    /usr/local/Ascend/ascend-toolkit/latest/lib64/libmmpa.so
    /usr/local/Ascend/ascend-toolkit/latest/lib64/libc_sec.so
    /usr/local/Ascend/ascend-toolkit/latest/lib64/libgraph.so
    /usr/local/Ascend/ascend-toolkit/latest/lib64/libgraph_base.so
    ${CMAKE_DL_LIBS}
  )

  add_dependencies(${kernel_lib_name} ${llm_kernels_ascend_libs_deps})
  set(llm_kernels_ascend_libs "${llm_kernels_ascend_libs} ${kernel_lib_name}" PARENT_SCOPE)
  message(STATUS "Ascend kernel added ${add_ascend_kernel_lib_NAME}")

  # for test
  if(WITH_TESTING)
    if(NOT WITH_STANDALONE_TEST AND ${add_ascend_kernel_lib_NAME} STREQUAL "rotary_embedding")
      message(STATUS "Skip test ${add_ascend_kernel_lib_NAME} when WITH_STANDALONE_TEST=OFF")
    else()
      file(GLOB_RECURSE KERNEL_TEST_SRCS *test.cc)

      if(KERNEL_TEST_SRCS AND NOT KERNEL_TEST_SRCS STREQUAL "")
        message(STATUS "${kernel_lib_name}_test src: ${KERNEL_TEST_SRCS}")
        cc_test(${kernel_lib_name}_test SRCS ${KERNEL_TEST_SRCS} DEPS ${kernel_lib_name}
          ${aicore_kernel_lib_name} llm_kernels_ascend_utils_common)
      endif()
    endif()
  endif()
endfunction()

add_subdirectory(atb_plugin_operations)
add_subdirectory(attention)

string(STRIP "${llm_kernels_ascend_libs}" llm_kernels_ascend_libs)
string(REGEX REPLACE " " ";" llm_kernels_ascend_libs "${llm_kernels_ascend_libs}")
message(STATUS "Ascend kernels: ${llm_kernels_ascend_libs}")
set(llm_kernels_ascend_libs ${llm_kernels_ascend_libs} PARENT_SCOPE)
