# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

include(ExternalProject)

find_package(OpenSSL REQUIRED)
find_package(Threads REQUIRED)

set(LIBCURL_INSTALL_DIR ${THIRD_PARTY_PATH}/install/libcurl)
set(LIBCURL_INCLUDE_DIR ${LIBCURL_INSTALL_DIR}/include)
set(LIBCURL_LIB_DIR ${LIBCURL_INSTALL_DIR}/lib)
set(LIBCURL_STATIC_LIB ${LIBCURL_LIB_DIR}/libcurl.a)

ExternalProject_Add(
    libcurl_external
    PREFIX ${THIRD_PARTY_PATH}/libcurl
    GIT_REPOSITORY https://github.com/curl/curl.git
    GIT_TAG curl-8_10_1
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=${LIBCURL_INSTALL_DIR}
        -DCMAKE_INSTALL_LIBDIR=lib
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
        -DCMAKE_BUILD_TYPE=Release
        -DBUILD_SHARED_LIBS=OFF
        -DBUILD_CURL_EXE=OFF
        -DBUILD_TESTING=OFF
        -DBUILD_LIBCURL_DOCS=OFF
        -DBUILD_MISC_DOCS=OFF
        -DCURL_USE_OPENSSL=ON
        -DCURL_DISABLE_LDAP=ON
        -DCURL_DISABLE_LDAPS=ON
        -DHTTP_ONLY=ON
        -DCURL_ZLIB=OFF
        -DUSE_LIBIDN2=OFF
        -DUSE_NGHTTP2=OFF
        -DUSE_LIBPSL=OFF
        -DCURL_USE_LIBPSL=OFF
        -DCURL_BROTLI=OFF
        -DCURL_ZSTD=OFF
        -DENABLE_ARES=OFF
        -DENABLE_THREADED_RESOLVER=OFF
    BUILD_BYPRODUCTS ${LIBCURL_STATIC_LIB}
)

file(MAKE_DIRECTORY ${LIBCURL_INCLUDE_DIR})

add_library(CURL::libcurl STATIC IMPORTED GLOBAL)
add_dependencies(CURL::libcurl libcurl_external)
set_target_properties(CURL::libcurl PROPERTIES
    IMPORTED_LOCATION ${LIBCURL_STATIC_LIB}
    INTERFACE_INCLUDE_DIRECTORIES ${LIBCURL_INCLUDE_DIR}
    INTERFACE_LINK_LIBRARIES "OpenSSL::SSL;OpenSSL::Crypto;Threads::Threads;dl"
)

set(CURL_FOUND TRUE CACHE BOOL "libcurl found" FORCE)
set(CURL_INCLUDE_DIR ${LIBCURL_INCLUDE_DIR} CACHE PATH "libcurl include dir" FORCE)
set(CURL_INCLUDE_DIRS ${LIBCURL_INCLUDE_DIR} CACHE PATH "libcurl include dirs" FORCE)
set(CURL_LIBRARY ${LIBCURL_STATIC_LIB} CACHE FILEPATH "libcurl library" FORCE)
set(CURL_LIBRARY_RELEASE ${LIBCURL_STATIC_LIB} CACHE FILEPATH "libcurl release library" FORCE)
set(CURL_LIBRARIES CURL::libcurl CACHE STRING "libcurl libraries" FORCE)

message(STATUS "libcurl include directory: ${LIBCURL_INCLUDE_DIR}")
message(STATUS "libcurl static library: ${LIBCURL_STATIC_LIB}")
