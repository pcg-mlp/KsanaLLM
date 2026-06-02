# Copyright 2026 Tencent Inc. All rights reserved.
#
# Preprocessing kernels for humming weight format conversion
file(GLOB HUMMING_PREPROCESS_SRCS *.cu)
if(HUMMING_PREPROCESS_SRCS)
  add_library(llm_kernels_nvidia_kernel_humming_preprocess STATIC ${HUMMING_PREPROCESS_SRCS})
  set_property(TARGET llm_kernels_nvidia_kernel_humming_preprocess PROPERTY POSITION_INDEPENDENT_CODE ON)
  set_property(TARGET llm_kernels_nvidia_kernel_humming_preprocess PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  # Resolve Python3 interpreter, prefer find_package result
  find_package(Python3 QUIET COMPONENTS Interpreter)
  if(Python3_FOUND)
    set(_HUMMING_PYTHON "${Python3_EXECUTABLE}")
  else()
    find_program(_HUMMING_PYTHON python3)
  endif()

  if(_HUMMING_PYTHON)
    execute_process(
      COMMAND ${_HUMMING_PYTHON} -c "import humming; import os; print(os.path.join(os.path.dirname(humming.__file__), 'include'))"
      OUTPUT_VARIABLE HUMMING_PIP_INCLUDE_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE HUMMING_PIP_RESULT
      ERROR_VARIABLE HUMMING_PIP_ERROR
    )
  else()
    set(HUMMING_PIP_RESULT 1)
    set(HUMMING_PIP_ERROR "python3 interpreter not found")
  endif()

  if(HUMMING_PIP_RESULT EQUAL 0 AND HUMMING_PIP_INCLUDE_DIR)
    set(HUMMING_INCLUDE_DIR "${HUMMING_PIP_INCLUDE_DIR}")
    message(STATUS "Using humming pip package include: ${HUMMING_INCLUDE_DIR}")
  else()
    message(FATAL_ERROR "humming-kernels pip package not found (${HUMMING_PIP_ERROR}). "
                        "Install with: pip install humming-kernels")
  endif()

  target_include_directories(llm_kernels_nvidia_kernel_humming_preprocess PUBLIC
    ${PROJECT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${HUMMING_INCLUDE_DIR}
  )
  target_link_libraries(llm_kernels_nvidia_kernel_humming_preprocess PUBLIC
    -lcudart llm_kernels_nvidia_utils ${TORCH_LIBRARIES})
endif()
