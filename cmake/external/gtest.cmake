# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

include(FetchContent)

set(GTEST_INSTALL_DIR ${THIRD_PARTY_PATH}/install/gtest)

if(NOT DEFINED GTEST_VER)
  set(GTEST_VER 1.12.1)
endif()

set(GTEST_GIT_URL https://github.com/google/googletest/archive/release-${GTEST_VER}.tar.gz)

FetchContent_Declare(
  googletest
  URL ${GTEST_GIT_URL}
  SOURCE_DIR ${GTEST_INSTALL_DIR}
)

FetchContent_GetProperties(googletest)
FetchContent_GetProperties(com_google_googletest)

if((NOT googletest_POPULATED) AND(NOT com_google_googletest_POPULATED))
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_Populate(googletest)

  # Mark com_google_googletest as populated, to prevent gtest from being imported twice.
  define_property(GLOBAL PROPERTY _FetchContent_com_google_googletest_populated
    BRIEF_DOCS "Indicates whether google test has been imported"
    FULL_DOCS "This is defined by fetch content for google test used by trpc-cpp"
  )
  set_property(GLOBAL PROPERTY _FetchContent_com_google_googletest_populated TRUE)

  add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR})
endif()

message(STATUS "Google test source directory: ${googletest_SOURCE_DIR}")
message(STATUS "Google test binary directory: ${googletest_BINARY_DIR}")

include_directories(
  ${googletest_SOURCE_DIR}/googlemock
  ${googletest_SOURCE_DIR}/googletest
)
