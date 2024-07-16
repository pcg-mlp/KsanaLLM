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

if(NOT "$ENV{ATB_HOME_PATH}" STREQUAL "")
  set(ATB_HOME_PATH $ENV{ATB_HOME_PATH})
else()
  set(ATB_HOME_PATH "/usr/local/Ascend/mindie/latest/mindie-rt/mindie-atb/atb")
endif()

set(ATB_VER "")

if(EXISTS "${ATB_HOME_PATH}/set_env.sh")
  add_definitions("-DENABLE_ACL_ATB")

  # awk -F: '{ gsub(/ /, "", $0); if ($1 == "Ascend-mindie-atbVersion") print $2 }' /usr/local/Ascend/mindie/latest/mindie-rt/mindie-atb/version.info
  execute_process(COMMAND /bin/sh -c "\"${AWK}\" -F: '\{ gsub(/ /, \"\", $0); if ($1 == \"Ascend-mindie-atbVersion\") print $2 \}' ${ATB_HOME_PATH}/../version.info" OUTPUT_VARIABLE ATB_VER OUTPUT_STRIP_TRAILING_WHITESPACE)
  message(STATUS "Using ATB ${ATB_VER} at ${ATB_HOME_PATH}")
endif()

set(ACL_INC_DIRS
  ${ASCEND_PATH}/include
  ${ASCEND_PATH}/include/aclnn
  ${ATB_HOME_PATH}/include
)

set(ACL_LIB_DIRS
  ${ASCEND_PATH}/lib64
)

set(ACL_SHARED_LIBS
  ${ASCEND_PATH}/lib64/libascendcl.so
  ${ASCEND_PATH}/lib64/libnnopbase.so
  ${ASCEND_PATH}/lib64/libopapi.so
)

if(EXISTS "${ATB_HOME_PATH}/set_env.sh")
  list(APPEND ACL_SHARED_LIBS
    ${ATB_HOME_PATH}/lib/libatb.so
    ${ATB_HOME_PATH}/lib/libasdops.so
    ${ATB_HOME_PATH}/lib/liblcal.so)
endif()

set(PLATFORM_CONFIG_PATH "${ASCEND_PATH}/compiler/data/platform_config/${ASCEND_PLATFORM_NAME}.ini")

add_definitions("-DENABLE_ACL" "-DPLATFORM_CONFIG_PATH=\"${PLATFORM_CONFIG_PATH}\"")