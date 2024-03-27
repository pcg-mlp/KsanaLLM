# Copyright 2023 Tencent Inc.  All rights reserved.
#
# ==============================================================================

include(FetchContent)
include(ExternalProject)
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)

set(FMT_DOWNLOAD_DIR ${THIRD_PARTY_PATH}/download/fmt)
set(FMT_INSTALL_DIR ${THIRD_PARTY_PATH}/install/fmt)

if(NOT TARGET fmt)
  FetchContent_Populate(download_fmt
    GIT_REPOSITORY https://git.woa.com/github-mirrors/fmtlib/fmt.git
    GIT_TAG 4ab01fb1988b70916d52dc1d30f176aebbd543f0
    SOURCE_DIR ${FMT_DOWNLOAD_DIR}
    SUBBUILD_DIR ${THIRD_PARTY_PATH}/tmp
    BINARY_DIR ${THIRD_PARTY_PATH}/tmp
  )
  ExternalProject_Add(extern_fmt
    PREFIX ${THIRD_PARTY_PATH}/fmt
    SOURCE_DIR ${FMT_DOWNLOAD_DIR}
    CMAKE_ARGS
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=0"
    -DCMAKE_INSTALL_PREFIX=${FMT_INSTALL_DIR}
    -DFMT_TEST=OFF
  )
  add_library(fmt STATIC IMPORTED GLOBAL)

  # Compatible with ubuntu.
  execute_process(
    COMMAND bash -c "awk -F= '/^ID=/{print $2}' /etc/os-release |tr -d '\n' | tr -d '\"'"
    OUTPUT_VARIABLE output_os_name)

  if(${output_os_name} MATCHES "ubuntu")
    MESSAGE(STATUS "OS name: ubuntu")
    set_property(TARGET fmt PROPERTY
      IMPORTED_LOCATION ${FMT_INSTALL_DIR}/lib/libfmt.a)
  else()
    MESSAGE(STATUS "OS name: centos")
    set_property(TARGET fmt PROPERTY
      IMPORTED_LOCATION ${FMT_INSTALL_DIR}/lib64/libfmt.a)
  endif()

  add_dependencies(fmt extern_fmt)
endif()

include_directories(${FMT_INSTALL_DIR}/include)
