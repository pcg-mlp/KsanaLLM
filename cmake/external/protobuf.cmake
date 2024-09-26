include(FetchContent)

# 使用 FetchContent 下载和构建 Protobuf
set(protobuf_BUILD_TESTS OFF CACHE BOOL "Disable testing for opentelemetry-cpp" FORCE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")

FetchContent_Declare(
  protobuf
  GIT_REPOSITORY https://github.com/protocolbuffers/protobuf.git
  GIT_TAG v3.16.0
  SOURCE_SUBDIR cmake
)

FetchContent_MakeAvailable(protobuf)

## 手动设置 Protobuf 的路径
set(Protobuf_INCLUDE_DIR ${protobuf_SOURCE_DIR}/src)
set(Protobuf_LIBRARIES ${CMAKE_BINARY_DIR})
set(PROTOBUF_PROTOC_EXECUTABLE protobuf::protoc)
