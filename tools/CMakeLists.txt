# Copyright 2025 Tencent Inc.  All rights reserved.
#
# ==============================================================================

add_subdirectory(cpp_client)
add_subdirectory(model_performance)

# import subprojects
if(EXISTS "/usr/local/deepep/")
  add_subdirectory(deepep_wrapper)
endif()

