# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fPIC")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fPIC")

# ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
if(NOT "$ENV{ASCEND_HOME_PATH}" STREQUAL "")
  set(ASCEND_PATH $ENV{ASCEND_HOME_PATH})
else()
  set(ASCEND_PATH "/usr/local/Ascend/ascend-toolkit/latest")
endif()

set(ACL_INC_DIRS
  ${ASCEND_PATH}/include
  ${ASCEND_PATH}/include/aclnn
)

set(ACL_LIB_DIRS
  ${ASCEND_PATH}/lib64
)

set(ACL_SHARED_LIBS
  ${ASCEND_PATH}/lib64/libascendcl.so
  ${ASCEND_PATH}/lib64/libnnopbase.so
  ${ASCEND_PATH}/lib64/libopapi.so
)

set(PLATFORM_CONFIG_PATH "${ASCEND_PATH}/compiler/data/platform_config/${ASCEND_PLATFORM_NAME}.ini")

add_definitions("-DENABLE_ACL" "-DPLATFORM_CONFIG_PATH=\"${PLATFORM_CONFIG_PATH}\"")