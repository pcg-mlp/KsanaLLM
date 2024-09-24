# 确保包含 FetchContent 模块
include(FetchContent)

# 使用 FetchContent 下载和构建 Protobuf
set(protobuf_BUILD_TESTS OFF CACHE BOOL "Disable testing for opentelemetry-cpp" FORCE)
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")

FetchContent_Declare(
  protobuf
  GIT_REPOSITORY https://github.com/protocolbuffers/protobuf.git
  GIT_TAG v3.16.0
  SOURCE_SUBDIR cmake
)

FetchContent_MakeAvailable(protobuf)

# 打印 Protobuf 的构建目录
message(STATUS "Protobuf source directory: ${protobuf_SOURCE_DIR}")
message(STATUS "Protobuf binary directory: ${protobuf_BINARY_DIR}")

# 手动设置 Protobuf 的路径
set(Protobuf_INCLUDE_DIR ${protobuf_SOURCE_DIR}/src)
set(Protobuf_LIBRARIES ${CMAKE_BINARY_DIR})

# 设置变量，确保在 FetchContent_MakeAvailable 之前
set(BUILD_TESTING OFF CACHE BOOL "Disable testing for opentelemetry-cpp" FORCE)
set(BUILD_SHARED_LIBS ON CACHE BOOL "Build shared libraries" FORCE)
set(WITH_OTLP_HTTP ON CACHE BOOL "Build OTLP HTTP exporter" FORCE)
set(WITH_DEPRECATED_SDK_FACTORY OFF CACHE BOOL "Disable deprecated SDK factory" FORCE)
set(PROTOBUF_PROTOC_EXECUTABLE protobuf::protoc)

FetchContent_Declare(
  opentelemetry-cpp
  GIT_REPOSITORY https://github.com/open-telemetry/opentelemetry-cpp.git
  GIT_TAG v1.16.0
  GIT_SUBMODULES ""
)

FetchContent_MakeAvailable(opentelemetry-cpp)

## 设置 opentelemetry-cpp 的路径
set(OPENTELEMETRY_CPP_INCLUDE_DIRS
    ${opentelemetry-cpp_SOURCE_DIR}/api/include
    ${opentelemetry-cpp_SOURCE_DIR}/sdk/include
    ${opentelemetry-cpp_SOURCE_DIR}/exporters/ostream/include
    ${opentelemetry-cpp_SOURCE_DIR}/exporters/otlp/include
    ${opentelemetry-cpp_SOURCE_DIR}/ext/include
)

set(OPENTELEMETRY_CPP_LIBRARY_DIRS
   ${CMAKE_INSTALL_PREFIX}/lib
)

link_directories(${OPENTELEMETRY_CPP_LIBRARY_DIRS})

# 设置库名称
set(OPENTELEMETRY_CPP_LIBRARIES
opentelemetry_exporter_ostream_span
opentelemetry_exporter_ostream_metrics
opentelemetry_exporter_otlp_http
opentelemetry_exporter_otlp_http_metric
opentelemetry_exporter_otlp_http_client
opentelemetry_http_client_curl
opentelemetry_metrics
opentelemetry_otlp_recordable
opentelemetry_resources
opentelemetry_trace
opentelemetry_common
opentelemetry_version
)
