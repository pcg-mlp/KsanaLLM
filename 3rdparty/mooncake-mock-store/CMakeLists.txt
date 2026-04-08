cmake_minimum_required(VERSION 3.14)
project(MooncakeMockStore VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# 让编译产物的 RPATH 指向 .so 所在目录，运行时无需设 LD_LIBRARY_PATH
set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)
set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")
set(CMAKE_BUILD_RPATH "$ORIGIN;${CMAKE_BINARY_DIR}")

find_package(Threads REQUIRED)

# ---- 动态库 ----
add_library(mooncake_store SHARED src/store.cpp)
target_include_directories(mooncake_store PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(mooncake_store PUBLIC Threads::Threads)

install(TARGETS mooncake_store LIBRARY DESTINATION lib)
install(DIRECTORY include/mooncake DESTINATION include)

# ---- 测试程序（仅独立构建时启用，作为子目录引入时跳过）----
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  enable_testing()

  add_executable(test_store tests/test_store.cpp)
  target_link_libraries(test_store PRIVATE mooncake_store)
  add_test(NAME test_store COMMAND test_store)

  add_executable(writer tests/cross_process/writer.cpp)
  target_link_libraries(writer PRIVATE mooncake_store)

  add_executable(reader tests/cross_process/reader.cpp)
  target_link_libraries(reader PRIVATE mooncake_store)

  add_test(NAME test_cross_process
      COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tests/cross_process/run.sh ${CMAKE_CURRENT_BINARY_DIR})
endif()
